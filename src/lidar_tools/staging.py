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
import pandas as pd
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
    """Approximate northing of a latitude band's midpoint (cycle hint only)."""
    lat_mid = -80 + _BAND_LETTERS.index(band.upper()) * 8 + 4
    return abs(lat_mid) * 110_946.0


def decode_tile_footprints(urls: list[str], utm_epsg: int) -> gpd.GeoDataFrame:
    """
    Build 1-km tile footprints for staged-LPC LAZ URLs from their grid ids.

    Parameters
    ----------
    urls
        Tile URLs or filenames (e.g. from ``0_file_download_links.txt``).
    utm_epsg
        Projected CRS of the tile grid — use the workunit's declared
        ``horiz_crs`` (WESM), e.g. 6340 for NAD83(2011) / UTM 11N.

    Returns
    -------
    gpd.GeoDataFrame
        One row per parseable URL: ``gridid``, ``url``, box geometry.
        Unparseable names are dropped (count them upstream if needed).
    """
    rows = []
    for url in urls:
        g = parse_grid_id(url)
        if g is None:
            continue
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
    return gpd.GeoDataFrame(rows, crs=f"EPSG:{utm_epsg}") if rows else \
        gpd.GeoDataFrame({"gridid": [], "url": []}, geometry=[], crs=f"EPSG:{utm_epsg}")


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


def reconcile_tile_sources(workunit: str, tesm_count: int, links_count: int) -> dict:
    """
    Compare the TESM index against the links-file truth for one workunit.
    Returns a verdict dict with a human-readable ``warning`` (or None).
    """
    verdict = {
        "workunit": workunit,
        "tesm_tiles": int(tesm_count),
        "links_tiles": int(links_count),
        "warning": None,
    }
    if links_count and not tesm_count:
        verdict["warning"] = (
            "TESM has NO tiles for this workunit although staged LAZ exists "
            "(index lags publication — use links-file/grid-decode footprints)"
        )
    elif links_count and tesm_count and tesm_count < 0.95 * links_count:
        verdict["warning"] = (
            f"TESM tile count ({tesm_count}) well below links file "
            f"({links_count}) — index incomplete for this area"
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
            rec["tiles"] = reconcile_tile_sources(
                wu, tesm_counts.get(wu, 0), links_counts.get(wu, 0)
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
    workunits: str,
    output: str,
) -> dict:
    """
    Stage discovery metadata for an AOI and write
    ``<output>/site_manifest.yaml``: pinned WESM records, EPT name
    resolution per workunit, TESM-vs-links tile reconciliation, and the
    staged-LAZ cache layout — everything the batch and the pre-run probes
    need, fetched once.

    Parameters
    ----------
    geometry
        Path to the AOI polygon.
    workunits
        Comma-separated WESM workunit names (as for rasterize-projects).
    output
        Batch output directory (manifest lands next to batch_status.yaml).
    """
    wu_list = [w.strip() for w in str(workunits).split(",") if w.strip()]
    if not wu_list:
        raise ValueError("No workunits given")
    aoi = gpd.read_file(geometry)
    wesm = survey.load_wesm(aoi)
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
            if rec.get("lpc_link"):
                links_counts[wu] = len(fetch_links_file(rec["lpc_link"]))
        except Exception as e:
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
