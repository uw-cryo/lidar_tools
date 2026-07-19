"""
Pre-run staging for an AOI: pin discovery metadata once, resolve source
aliases, index staged-LAZ tiles, and write a site manifest that batch
processing and preprocessing probes consume offline.

Source layers (see sandbox/20260718_lv_nad83_regen/threedep_source_map.md):
indexes are HINTS with independent failure modes — WESM polygons generalize
away real tile coverage, TESM can lag a workunit's LPC publication entirely,
EPT builds lag and carry frozen names. The payload-adjacent links file
(``0_file_download_links.txt``) is the per-tile truth for staged LAZ; the
manifest records every layer plus reconciliation verdicts so disagreement is
visible instead of silent.
"""

import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import shapely
import yaml

from lidar_tools import survey

TESM_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/LPC/"
    "FullExtentSpatialMetadata/LPC_TESM.gpkg"
)

# USGS 1-km staged-LPC tile ids: <zone><band><100km-square><EEE><NNN>,
# indices in 100 m units from the MGRS square origin (verified against LAZ
# headers at 0.00 m residual, zone 11, 2026-07-18)
_GRID_ID_RE = re.compile(
    r"_(\d{1,2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d{3})(\d{3})\.la[sz]$",
    re.IGNORECASE,
)
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"  # 20-letter cycle, 100 km each
_BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"  # 8-degree latitude bands from -80


def parse_grid_id(name: str) -> dict | None:
    """
    Parse a staged-LPC LAZ/LAS filename's national-grid tile id.

    Returns ``{"zone", "band", "square", "e", "n", "gridid"}`` (indices in
    100 m units within the 100-km MGRS square) or None when the filename
    does not carry a grid id.
    """
    m = _GRID_ID_RE.search(name)
    if not m:
        return None
    zone, band, square, e, n = m.groups()
    return {
        "zone": int(zone),
        "band": band.upper(),
        "square": square.upper(),
        "e": int(e),
        "n": int(n),
        "gridid": f"{int(zone)}{band.upper()}{square.upper()}{e}{n}",
    }


def grid_origin(zone: int, square: str, northing_hint: float) -> tuple[float, float]:
    """
    UTM coordinates of an MGRS 100-km square's SW corner (analytic lattice;
    the letter scheme is deterministic per zone). ``northing_hint`` picks
    the correct instance of the 2,000-km row-letter cycle — pass any
    northing near the data (e.g. from the band letter via `band_northing`).

    Validated against LAZ-header-derived origins for all 7 squares of
    NV_Southern_4_D23 (zone 11: MA MB NA NB NV PA PV), 2026-07-18.
    """
    col_sets = {1: "ABCDEFGH", 2: "JKLMNPQR", 0: "STUVWXYZ"}
    cols = col_sets[zone % 3]
    if square[0] not in cols:
        raise ValueError(
            f"column letter {square[0]!r} invalid for zone {zone} (set {cols})"
        )
    easting = (cols.index(square[0]) + 1) * 100_000
    row_idx = _ROW_LETTERS.index(square[1])
    if zone % 2 == 0:  # even zones offset the row lettering by 500 km
        row_idx = (row_idx - 5) % 20
    candidates = [k * 2_000_000 + row_idx * 100_000 for k in range(0, 5)]
    northing = min(candidates, key=lambda c: abs(c - northing_hint))
    return float(easting), float(northing)


def band_northing(band: str) -> float:
    """
    Approximate northing of a latitude band's midpoint (row-cycle hint
    only). Southern bands (C-M) use the UTM-south false northing
    (10,000,000 - |lat|*k) — e.g. band L (American Samoa) sits near
    8.67e6, NOT 1.33e6; getting this wrong silently shifts the decoded
    grid by whole 2,000-km row cycles.
    """
    lat_mid = -80 + _BAND_LETTERS.index(band.upper()) * 8 + 4
    if lat_mid < 0:
        return 10_000_000.0 + lat_mid * 110_946.0
    return lat_mid * 110_946.0


def decode_tile_footprints(urls: list[str], utm_epsg: int = None) -> gpd.GeoDataFrame:
    """
    Build 1-km tile footprints for staged-LPC LAZ URLs from their grid ids.

    The grid ids are ALWAYS in the USGS national grid's own UTM zone
    (encoded in the id), regardless of the workunit's declared
    ``horiz_crs`` — a State-Plane workunit still tiles on UTM.

    Parameters
    ----------
    urls
        Tile URLs or filenames (e.g. from ``0_file_download_links.txt``).
        All parseable ids must share one zone+hemisphere per call (group
        by zone upstream — see `count_links_tiles_in_bbox`).
    utm_epsg
        Projected CRS to stamp on the footprints. Default None derives
        NAD83(2011) UTM north (EPSG 6329+zone) from the grid ids; an
        explicit CRS must be a UTM CRS whose zone and hemisphere MATCH
        the ids (a ``13S...`` id under EPSG:6340 would otherwise decode
        to plausible but wrong zone-11 coordinates — ValueError instead).

    Returns
    -------
    gpd.GeoDataFrame
        One row per parseable URL: ``gridid``, ``url``, box geometry.
        Unparseable names are dropped (count them upstream if needed).
    """
    parsed = [(url, g) for url in urls if (g := parse_grid_id(url))]
    if not parsed:
        crs = f"EPSG:{utm_epsg}" if utm_epsg else None
        return gpd.GeoDataFrame({"gridid": [], "url": []}, geometry=[], crs=crs)

    zones = {g["zone"] for _, g in parsed}
    souths = {g["band"] <= "M" for _, g in parsed}
    if len(zones) > 1 or len(souths) > 1:
        raise ValueError(
            f"grid ids span multiple UTM zones/hemispheres ({sorted(zones)}); "
            "decode each zone separately"
        )
    zone, south = zones.pop(), souths.pop()

    if utm_epsg is None:
        if south:
            raise ValueError(
                "no default CRS for southern-hemisphere grid ids — pass the "
                "workunit's UTM-south EPSG explicitly (e.g. NAD83(PA11) zone 2S)"
            )
        utm_epsg = 6329 + zone  # NAD83(2011) / UTM north
    else:
        from pyproj import CRS as _CRS

        crs_zone = _CRS.from_epsg(utm_epsg).utm_zone  # e.g. '11N', None if not UTM
        if crs_zone is None:
            raise ValueError(
                f"EPSG:{utm_epsg} is not a UTM CRS; staged-LPC grid ids decode "
                "on the UTM lattice only"
            )
        if int(crs_zone[:-1]) != zone or (crs_zone[-1] == "S") != south:
            raise ValueError(
                f"grid ids are UTM zone {zone}{'S' if south else 'N'} but "
                f"EPSG:{utm_epsg} is zone {crs_zone} — refusing to decode "
                "into the wrong zone"
            )

    rows = []
    for url, g in parsed:
        x0, y0 = grid_origin(g["zone"], g["square"], band_northing(g["band"]))
        x0 += g["e"] * 100
        y0 += g["n"] * 100
        rows.append(
            {
                "gridid": g["gridid"],
                "url": url,
                "geometry": shapely.box(x0, y0, x0 + 1000, y0 + 1000),
            }
        )
    return gpd.GeoDataFrame(rows, crs=f"EPSG:{utm_epsg}")


def count_links_tiles_in_bbox(links: list[str], aoi_gdf: gpd.GeoDataFrame) -> int:
    """
    Count links-file tiles whose decoded footprints intersect the AOI's
    bounding box — the same spatial scope as the bbox-filtered TESM read,
    so `reconcile_tile_sources` compares like with like (an AOI-clipped
    TESM count vs a FULL-workunit links count would flag every workunit
    that extends beyond the AOI as \"TESM incomplete\").

    Ids are grouped by UTM zone and each group is decoded in its own zone
    CRS with the AOI bbox transformed to match.
    """
    by_zone: dict[tuple, list] = {}
    for url in links:
        g = parse_grid_id(url)
        if g is None:
            continue
        by_zone.setdefault((g["zone"], g["band"] <= "M"), []).append(url)
    total = 0
    for (_zone, _south), group in by_zone.items():
        fp = decode_tile_footprints(group)  # northern default; southern raises
        bounds = aoi_gdf.to_crs(fp.crs).total_bounds
        total += int(fp.intersects(shapely.box(*bounds)).sum())
    return total


def load_tesm_tiles(aoi_gdf: gpd.GeoDataFrame, tesm_source: str = TESM_URL) -> gpd.GeoDataFrame:
    """
    Read LPC_TESM tile-extent polygons intersecting the AOI bounds (remote
    bbox read over /vsicurl, same pattern as `survey.load_wesm`). TESM rows
    carry ``tile_id/project/project_id/workunit_id`` but NO tile URL, and
    TESM project names drift from WESM's — join by ``workunit_id`` only
    (`attach_workunits`). TESM may entirely lack a recently published
    workunit; treat absence as "index lags", not "no data".
    """
    src = str(tesm_source)
    if src.startswith(("http://", "https://")):
        src = f"/vsicurl/{src}"
    bbox = tuple(aoi_gdf.to_crs("EPSG:4326").total_bounds)
    return gpd.read_file(src, bbox=bbox)


def attach_workunits(tesm_gdf: gpd.GeoDataFrame, wesm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map WESM workunit names onto TESM rows via ``workunit_id`` (never names)."""
    out = tesm_gdf.copy()
    lookup = dict(zip(wesm_gdf["workunit_id"], wesm_gdf["workunit"]))
    out["workunit"] = out["workunit_id"].map(lookup)
    return out


def fetch_links_file(lpc_link: str, opener=None) -> list[str]:
    """
    Fetch a workunit's ``0_file_download_links.txt`` (the staged-LAZ tile
    truth). ``opener(url) -> str`` is injectable for tests/offline use.
    """
    url = str(lpc_link).rstrip("/") + "/0_file_download_links.txt"
    if opener is None:
        def opener(u):
            with urllib.request.urlopen(u, timeout=60) as r:
                return r.read().decode()
    text = opener(url)
    return [line.strip() for line in text.splitlines() if line.strip()]


def reconcile_tile_sources(
    workunit: str,
    tesm_count: int,
    links_count: int | None,
    scope: str = "aoi_bbox",
) -> dict:
    """
    Compare the TESM index against the links-file truth for one workunit,
    over a SINGLE stated spatial scope (both counts must be clipped the
    same way — see `count_links_tiles_in_bbox`).

    ``links_count=None`` means the links file could not be fetched — a
    different fact from "no staged LAZ" (0), and the verdicts keep them
    apart. Returns ``{status, warning, ...}``; ``warning`` is None only
    when the two sources agree.
    """
    verdict = {
        "workunit": workunit,
        "scope": scope,
        "tesm_tiles": int(tesm_count),
        "links_tiles": None if links_count is None else int(links_count),
        "status": "consistent",
        "warning": None,
    }
    if links_count is None:
        verdict["status"] = "links-unavailable"
        verdict["warning"] = (
            "links file could not be fetched — staged-LAZ truth unknown "
            "(NOT the same as 'no staged LAZ'); retry before trusting TESM"
        )
    elif not links_count and not tesm_count:
        verdict["status"] = "no-tiles"
    elif links_count and not tesm_count:
        verdict["status"] = "tesm-missing"
        verdict["warning"] = (
            "TESM has NO tiles for this workunit although staged LAZ exists "
            "(index lags publication — use links-file/grid-decode footprints)"
        )
    elif tesm_count < 0.95 * links_count:
        verdict["status"] = "tesm-incomplete"
        verdict["warning"] = (
            f"TESM tile count ({tesm_count}) well below links file "
            f"({links_count}) in scope {scope} — index incomplete here"
        )
    elif links_count < 0.95 * tesm_count:
        verdict["status"] = "links-behind-tesm"
        verdict["warning"] = (
            f"links file ({links_count}) well below TESM ({tesm_count}) in "
            f"scope {scope} — stale TESM after republication, or a partial "
            "links fetch; verify before selecting tiles"
        )
    return verdict


def build_site_manifest(
    aoi_path: str,
    workunits: list[str],
    wesm_gdf: gpd.GeoDataFrame,
    output_dir: str,
    ept_gdf: gpd.GeoDataFrame = None,
    tesm_counts: dict = None,
    links_counts: dict = None,
) -> dict:
    """
    Assemble the per-AOI site manifest from already-loaded inputs (pure —
    no network; callers/`prepare` do the fetching so tests stay offline).

    Per workunit: pinned WESM record, EPT resolution (tier/name or the
    LookupError message), TESM-vs-links reconciliation, staged-LAZ cache
    convention ``<output_dir>/lpc_cache/<workunit>/``, and empty probe
    slots filled by the staging-time probes (vertical datum, EPT<->LAZ
    single-tile cross-check).
    """
    tesm_counts = tesm_counts or {}
    links_counts = links_counts or {}
    manifest = {
        # bump on any breaking schema change; consumers (rasterize-projects,
        # probes) must check this before reading
        "manifest_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aoi": str(aoi_path),
        "output_dir": str(output_dir),
        "sources": {
            "wesm": survey.WESM_URL,
            "ept_index": survey.EPT_RESOURCES_URL,
            "tesm": TESM_URL,
        },
        "workunits": {},
    }
    for wu in workunits:
        rec = {"wesm": None, "ept": None, "tiles": None, "probes": {}}
        try:
            rec["wesm"] = survey.record_from_wesm(wesm_gdf, wu)
        except ValueError as e:
            rec["wesm"] = {"error": str(e)}
        if ept_gdf is not None:
            try:
                rec["ept"] = survey.resolve_ept_resource(wu, ept_gdf)
            except LookupError as e:
                rec["ept"] = {"error": str(e)}
        if wu in tesm_counts or wu in links_counts:
            # links value None = fetch attempted and failed (distinct from 0)
            rec["tiles"] = reconcile_tile_sources(
                wu, tesm_counts.get(wu, 0), links_counts.get(wu)
            )
        rec["lpc_cache"] = str(Path(output_dir) / "lpc_cache" / wu)
        manifest["workunits"][wu] = rec
    return manifest


def write_site_manifest(manifest: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def load_site_manifest(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare(
    geometry: str,
    output: str,
    workunits: str = None,
) -> dict:
    """
    Stage discovery metadata for an AOI and write
    ``<output>/site_manifest.yaml``: pinned WESM records, EPT name
    resolution per workunit, TESM-vs-links tile reconciliation (all
    counts clipped to the AOI bounding box — one stated scope), and the
    staged-LAZ cache layout — everything the batch and the pre-run probes
    need, fetched once.

    Parameters
    ----------
    geometry
        Path to the AOI polygon.
    output
        Batch output directory (manifest lands next to batch_status.yaml).
    workunits
        Comma-separated WESM workunit names (as for rasterize-projects).
        Default: every workunit whose WESM polygon intersects the AOI.
    """
    aoi = gpd.read_file(geometry)
    wesm = survey.load_wesm(aoi)
    if workunits is None:
        wu_list = sorted(wesm["workunit"].astype(str))
    else:
        wu_list = [w.strip() for w in str(workunits).split(",") if w.strip()]
    if not wu_list:
        raise ValueError("No workunits given and none intersect the AOI")
    ept = survey.load_ept_resources()
    tesm_counts: dict = {}
    links_counts: dict = {}
    try:
        tesm = attach_workunits(load_tesm_tiles(aoi), wesm)
        tesm_counts = tesm.groupby("workunit").size().to_dict()
    except Exception as e:
        print(f"WARNING: TESM read failed ({e}); tile index omitted")
    for wu in wu_list:
        try:
            rec = survey.record_from_wesm(wesm, wu)
        except ValueError:
            continue  # not in WESM: recorded as an error by the manifest
        if not rec.get("lpc_link"):
            continue  # no staged LPC tree at all: nothing to reconcile
        try:
            links = fetch_links_file(rec["lpc_link"])
            # same spatial scope as the bbox-filtered TESM read above —
            # comparing an AOI-clipped TESM count against a full-workunit
            # links count would flag every AOI-spanning workunit
            links_counts[wu] = count_links_tiles_in_bbox(links, aoi)
        except Exception as e:
            links_counts[wu] = None  # attempted and failed != zero tiles
            print(f"WARNING: links file unavailable for {wu} ({e})")
    manifest = build_site_manifest(
        geometry, wu_list, wesm, output,
        ept_gdf=ept, tesm_counts=tesm_counts, links_counts=links_counts,
    )
    out_fn = Path(output) / "site_manifest.yaml"
    write_site_manifest(manifest, out_fn)
    print(f"Site manifest: {out_fn}")
    for wu, rec in manifest["workunits"].items():
        ept_rec = rec.get("ept") or {}
        tag = (
            f"EPT {ept_rec.get('ept_name')} (tier {ept_rec.get('tier')})"
            if ept_rec.get("ept_name")
            else "NO EPT — staged-LAZ path"
        )
        warn = (rec.get("tiles") or {}).get("warning")
        print(f"  {wu}: {tag}" + (f" | {warn}" if warn else ""))
    return manifest
