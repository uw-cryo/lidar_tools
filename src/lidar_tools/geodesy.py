"""
Geodesy helpers: programmatic CRS construction, datum-transform preflight
checks, and coordinate-epoch stamping for lidar_tools outputs.

3DEP EPT point clouds on AWS carry EPSG:3857 coordinates produced from the
source NAD83(2011) data with a null datum tie, and Z values that pass through
unchanged (NAVD88 orthometric or NAD83(2011) ellipsoidal, per survey). The
builders here construct source CRSs that make those semantics explicit, so
PROJ selects rigorous transformations (geoid grid + time-dependent Helmert)
instead of silently relabeling coordinates between datum realizations.

Note: the pipeline entry point (pdal_pipeline) sets PROJ_NETWORK and
PROJ_ONLY_BEST_DEFAULT environment variables before GDAL/PROJ are imported.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyproj.datadir
import pyproj.network
from osgeo import gdal
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.crs import CompoundCRS, ProjectedCRS
from pyproj.crs.coordinate_operation import UTMConversion
from pyproj.transformer import TransformerGroup

gdal.UseExceptions()

# 3DEP lidar sources are NAD83(2011), epoch-reduced to 2010.0, and the
# NAD83(2011)<->ITRF time-dependent Helmert is evaluated at its 2010.0
# reference epoch, so outputs in a dynamic frame (e.g. the default
# WGS 84 (G2139) target) are coordinates at epoch 2010.0
DEFAULT_COORDINATE_EPOCH = 2010.0

# WGS 84 (G2139) geographic 2D
WGS84_G2139_EPSG = 9755
# NAD83(2011) geographic 2D
NAD83_2011_EPSG = 6318

# geographic 2D bases of the NAD83 family (North America plate): the only
# datums for which the EPT null-tie treatment and the ITRF Helmert at epoch
# 2010.0 are valid. PA11/MA11 (Pacific/Mariana plates) are deliberately
# excluded — a North-America Helmert is the wrong transformation there.
NAD83_FAMILY_GEOGRAPHIC = {
    4269,  # NAD83(1986)
    4152,  # NAD83(HARN)
    4759,  # NAD83(NSRS2007)
    6318,  # NAD83(2011)
    6783,  # NAD83(CORS96)
}

# survey geoid declaration -> PROJ grid-name fragment, for selecting the
# transformation that uses the survey's production geoid model
GEOID_GRID_HINTS = {
    "GEOID18": "g2018",
    "GEOID12B": "g2012b",
    "GEOID12A": "g2012a",
    "GEOID09": "geoid09",
    "GEOID06": "geoid06",
    "GEOID03": "geoid03",
    "GEOID99": "geoid99",
}


def geographic_base_epsg(crs_input) -> int:
    """
    Geographic 2D base EPSG of a survey's declared horizontal CRS, validated
    as NAD83-family (e.g. 7131 NAD83(2011)/SP CA-3 ftUS -> 6318;
    26910 NAD83/UTM 10N -> 4269).

    Parameters
    ----------
    crs_input
        EPSG code (int or numeric string) or anything pyproj accepts.

    Returns
    -------
    int
        Geographic 2D EPSG code of the base datum.

    Raises
    ------
    ValueError
        If the base cannot be identified or is not NAD83-family (including
        Pacific/Mariana-plate PA11/MA11 realizations).
    """
    s = str(crs_input).strip()
    crs = CRS.from_epsg(int(s)) if s.isdigit() else CRS.from_user_input(crs_input)
    base = crs.geodetic_crs
    code = base.to_epsg() if base is not None else None
    if code is not None and code not in NAD83_FAMILY_GEOGRAPHIC:
        # projected CRSs may report a 3D/other variant: try name matching
        if base is not None and base.name.startswith("NAD83") and "PA11" not in base.name and "MA11" not in base.name:
            for cand in NAD83_FAMILY_GEOGRAPHIC:
                if CRS.from_epsg(cand).name == base.name:
                    return cand
    if code in NAD83_FAMILY_GEOGRAPHIC:
        return code
    raise ValueError(
        f"Declared horizontal CRS '{crs.name}' has base "
        f"'{base.name if base is not None else None}' (EPSG:{code}), which is "
        "not a supported NAD83-family (North America plate) datum. The EPT "
        "null-tie treatment and the epoch-2010.0 Helmert do not apply — "
        "handle this survey explicitly."
    )


def geoid_grid_hint(geoid_name) -> "str | None":
    """
    PROJ grid-name fragment for a survey's declared geoid model
    (e.g. 'GEOID12B' -> 'g2012b'), or None when unknown/absent.
    """
    if geoid_name is None:
        return None
    key = str(geoid_name).upper().replace(" ", "")
    return GEOID_GRID_HINTS.get(key)


# Declared geoid model -> PROJ-data grid file(s) per region. Filenames
# verified against the cdn.proj.org us_noaa catalog (pyproj.sync,
# 2026-07-18). GEOID99 CONUS/AK ship as multi-tile sets: vgridshift takes
# the comma list and applies the tile containing each point.
GEOID_GRID_FILES = {
    "GEOID18": {
        "conus": ["us_noaa_g2018u0.tif"],
        "prvi": ["us_noaa_g2018p0.tif"],
    },
    "GEOID12B": {
        "conus": ["us_noaa_g2012bu0.tif"],
        "ak": ["us_noaa_g2012ba0.tif"],
        "prvi": ["us_noaa_g2012bp0.tif"],
        # NGS notes Hawaii had no official vertical datum in the GEOID12
        # era; the g2012bh0 grid exists and is mapped, but an HI survey
        # declaring GEOID12B deserves operator attention (what datum were
        # the heights actually produced against?)
        "hi": ["us_noaa_g2012bh0.tif"],
        "guam": ["us_noaa_g2012bg0.tif"],
        "samoa": ["us_noaa_g2012bs0.tif"],
    },
    "GEOID09": {
        "conus": ["us_noaa_geoid09_conus.tif"],
        "ak": ["us_noaa_geoid09_ak.tif"],
    },
    "GEOID06": {"ak": ["us_noaa_geoid06_ak.tif"]},
    "GEOID03": {"conus": ["us_noaa_geoid03_conus.tif"]},
    "GEOID99": {
        "conus": [f"us_noaa_g1999u0{i}.tif" for i in range(1, 9)],
        "ak": [f"us_noaa_g1999a0{i}.tif" for i in range(1, 5)],
        "hi": ["us_noaa_g1999h01.tif"],
        "prvi": ["us_noaa_g1999p01.tif"],
    },
}

# short forms and case drift observed in WESM 'geoid' declarations
_GEOID_NAME_ALIASES = {"12A": "GEOID12A", "12B": "GEOID12B"}
# declarations that mean "nothing declared", not "declared but unavailable"
_GEOID_NAME_UNDECLARED = {"", "UNKNOWN", "N/A", "NA", "NONE"}

_REGION_BOXES = {
    # (west, south, east, north), checked in order; conus last
    "prvi": (-68.5, 17.0, -64.0, 19.5),
    "hi": (-161.5, 18.0, -154.0, 23.5),
    "guam": (144.0, 13.0, 146.5, 16.0),
    "samoa": (-171.5, -15.0, -169.0, -13.5),
    "ak": (-190.0, 49.5, -129.0, 72.5),
    "conus": (-130.0, 23.0, -66.0, 50.0),
}


def _aoi_region(aoi_bounds: tuple) -> str:
    """NGS geoid-grid region containing the AOI (west, south, east, north)."""
    west, south, east, north = aoi_bounds
    cx, cy = (west + east) / 2.0, (south + north) / 2.0
    for region, (w, s, e, n) in _REGION_BOXES.items():
        if w <= cx <= e and s <= cy <= n:
            return region
    raise ValueError(
        f"AOI {aoi_bounds} is outside every NGS geoid-grid region "
        f"({list(_REGION_BOXES)}); declared-geoid handling needs an explicit "
        "region mapping for this area."
    )


def resolve_declared_geoid(geoid_name, aoi_bounds: tuple) -> "dict | None":
    """
    Resolve a survey's declared geoid model to the exact PROJ grid file(s)
    to REQUIRE for its AOI — the survey's production geoid is part of the
    data definition, and substituting a different model (e.g. GEOID18 for
    a GEOID12B survey) silently shifts heights by the model difference
    (cm-level). Returns None only when nothing was declared
    ('Unknown'/'N/A'/empty); a declared-but-unmappable geoid raises.

    GEOID12A has no grids in PROJ-data. Per the NGS GEOID Team, GEOID12B
    "is EXACTLY the same as the earlier GEOID12A model in the fifty states
    (CONUS); the only change is the portion that covers Puerto Rico and
    the US Virgin Islands" (relayed by the NGS State Geodetic Advisor
    program, 2015; see also the NGS technical summary
    https://geodesy.noaa.gov/research/technical-details/USGG2012-GEOID12B-technical-information.pdf).
    GEOID12A therefore maps to the GEOID12B grids everywhere EXCEPT
    Puerto Rico / USVI, where the models differ and we refuse.

    Returns
    -------
    dict | None
        {"declared", "model", "region", "grids", "substituted_for"} —
        ``grids`` are exact PROJ-data filenames; ``substituted_for`` is
        set when an equivalent model's grids stand in for the declared one.
    """
    if geoid_name is None:
        return None
    key = str(geoid_name).upper().replace(" ", "")
    if key in _GEOID_NAME_UNDECLARED:
        return None
    key = _GEOID_NAME_ALIASES.get(key, key)
    region = _aoi_region(aoi_bounds)

    substituted_for = None
    model = key
    if key == "GEOID12":
        raise ValueError(
            "Survey declares 'GEOID12', which NGS superseded with corrections "
            "(GEOID12A/GEOID12B) and which has no PROJ-data grids. Confirm the "
            "actual production model from the vendor report and handle this "
            "survey explicitly."
        )
    if key == "GEOID12A":
        if region == "prvi":
            raise ValueError(
                "Survey declares GEOID12A over Puerto Rico / USVI — the one "
                "region where GEOID12A and GEOID12B DIFFER (NGS). No "
                "GEOID12A grids exist in PROJ-data; refusing to substitute. "
                "Handle this survey explicitly (NGS distributes legacy "
                "GEOID12A PR/VI grids separately)."
            )
        model = "GEOID12B"
        substituted_for = "GEOID12A"

    files_by_region = GEOID_GRID_FILES.get(model)
    if files_by_region is None:
        raise ValueError(
            f"Survey declares geoid '{geoid_name}', which has no known "
            "PROJ-data grids. Refusing to silently substitute another model; "
            "add the mapping to GEOID_GRID_FILES or use "
            "--geoid-override best-available to consciously accept "
            "substitution."
        )
    grids = files_by_region.get(region)
    if grids is None:
        raise ValueError(
            f"Declared geoid {model} has no grid for region '{region}' "
            f"(AOI {aoi_bounds}; available regions {sorted(files_by_region)}). "
            "The declaration and the AOI disagree — resolve explicitly."
        )
    return {
        "declared": str(geoid_name),
        "model": model,
        "region": region,
        "grids": list(grids),
        "substituted_for": substituted_for,
    }


def _swap_vgridshift_grids(definition: str, grid_files: list) -> str:
    """
    Return a PROJ pipeline with the vgridshift step's grids replaced by
    ``grid_files``. The EPSG registry marks older NAVD88 geoid realizations
    as superseded, so PROJ never even lists them as candidates — the only
    way to honor a declared legacy geoid is to take the ranked pipeline
    (whose structure is correct) and swap the geoid grid it uses.
    """
    # pyproj Transformer.definition drops the '+' prefixes; accept both
    # forms and ANY whitespace. The substitution must use the SAME pattern
    # as the match: a find-then-str.replace pair can silently no-op on a
    # definition whose spacing differs from the reconstruction, which would
    # run the ranked pipeline while claiming the declared one.
    pattern = r"(\+?proj=vgridshift\s+\+?)grids=\S+"
    matches = re.findall(pattern, definition)
    if len(matches) != 1:
        raise ValueError(
            f"Cannot swap the declared geoid into this pipeline: expected "
            f"exactly one vgridshift step, found {len(matches)} "
            f"(pipeline: {definition[:200]})"
        )
    swapped = re.sub(
        pattern,
        lambda m: f"{m.group(1)}grids={','.join(grid_files)}",
        definition,
        count=1,
    )
    if swapped == definition:
        raise ValueError(
            "Declared-geoid grid swap produced no change to the pipeline "
            f"(grids {grid_files}); refusing — an unchanged pipeline would "
            "silently run the ranked geoid while claiming the declared one."
        )
    return swapped


def _grid_locally_available(grid_file: str) -> bool:
    """A datum grid counts as present only as a local file on the PROJ path."""
    dirs = [pyproj.datadir.get_user_data_dir(), pyproj.datadir.get_data_dir()]
    return any((Path(d) / grid_file).exists() for d in dirs if d)


def _download_grid(grid_file: str) -> None:
    """
    Fetch one grid from the PROJ CDN into the user-writable data dir.
    pid-unique temp name + unlink-on-failure so concurrent runs sharing
    the PROJ user dir cannot clobber each other or rename a partial file
    into place; the size is validated against Content-Length before the
    rename publishes it.
    """
    import os
    import urllib.request

    from pyproj.sync import get_proj_endpoint

    dest = Path(pyproj.datadir.get_user_data_dir())
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{get_proj_endpoint()}/{grid_file}"
    tmp = dest / f"{grid_file}.{os.getpid()}.part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            expected = r.headers.get("Content-Length")
            shutil.copyfileobj(r, f)
        if expected is not None and tmp.stat().st_size != int(expected):
            raise OSError(
                f"downloaded {tmp.stat().st_size} bytes for {grid_file}, "
                f"server declared {expected}"
            )
        tmp.rename(dest / grid_file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_grids_local(grid_files: list) -> None:
    """
    Fail fast unless every grid is a LOCAL file: a pipeline that fetches
    grids over PROJ networking at warp time dies hours later on a CDN
    blip. Missing grids are downloaded now (when networking is enabled)
    or reported with the exact remedy.
    """
    for grid_file in grid_files:
        if _grid_locally_available(grid_file):
            continue
        if pyproj.network.is_network_enabled():
            print(f"Downloading datum grid {grid_file} to the PROJ user dir")
            try:
                _download_grid(grid_file)
            except Exception as e:
                raise RuntimeError(
                    f"Datum grid '{grid_file}' is not installed locally and "
                    f"the download failed ({e}). Install it explicitly "
                    f"('pyproj sync --file {grid_file.split('.')[0]}' or the "
                    "conda-forge proj-data package) and re-run."
                ) from e
        if not _grid_locally_available(grid_file):
            raise RuntimeError(
                f"Datum grid '{grid_file}' is not installed locally and PROJ "
                "networking is disabled. Install it explicitly "
                f"('pyproj sync --file {grid_file.split('.')[0]}' or the "
                "conda-forge proj-data package, or set PROJ_NETWORK=ON) "
                "and re-run. Refusing to defer grid access to warp time."
            )


def utm_zone_label(utm_epsg: int) -> str:
    """
    Return the UTM zone label (e.g. '10N' or '19S') for a WGS84 UTM EPSG code.

    Parameters
    ----------
    utm_epsg
        WGS84 UTM EPSG code (326xx for northern zones, 327xx for southern).

    Returns
    -------
    str
        Zone label like '10N'.
    """
    prefix, zone = divmod(utm_epsg, 100)
    if prefix == 326:
        return f"{zone}N"
    if prefix == 327:
        return f"{zone}S"
    raise ValueError(
        f"EPSG:{utm_epsg} is not a WGS84 UTM CRS (expected 326xx or 327xx)"
    )


def build_utm_g2139_3d(utm_epsg: int) -> CRS:
    """
    Build a 3D UTM CRS on the WGS 84 (G2139) realization for a UTM EPSG code.

    Replaces fetching a UTM 10N WKT template over the network and
    text-substituting the zone, which silently kept the northern false
    northing (0) for southern-hemisphere zones.

    Parameters
    ----------
    utm_epsg
        WGS84 UTM EPSG code (e.g. 32610) selecting the zone and hemisphere.

    Returns
    -------
    CRS
        Projected 3D CRS with a dynamic WGS 84 (G2139) datum and
        ellipsoidal-height axis.
    """
    label = utm_zone_label(utm_epsg)
    zone, hemisphere = label[:-1], label[-1]
    return ProjectedCRS(
        conversion=UTMConversion(zone, hemisphere=hemisphere),
        geodetic_crs=CRS.from_epsg(WGS84_G2139_EPSG),
        name=f"WGS 84 (G2139) / UTM zone {label}",
    ).to_3d()


def build_utm_nad83_2011_3d(utm_epsg: int) -> CRS:
    """
    Build a 3D UTM CRS on the static NAD83(2011) datum for a UTM EPSG code.

    NAD83(2011) is the source realization of 3DEP lidar, so a NAD83(2011)
    target is the most native output: no time-dependent Helmert is applied
    (the ellipsoid-source transform is a pure projection change), and the
    frame is static — coordinates carry no epoch (unlike the dynamic WGS 84
    (G2139) default, which is stamped at epoch 2010.0). Heights are
    ellipsoidal on the GRS 1980 ellipsoid.

    Parameters
    ----------
    utm_epsg
        WGS84 UTM EPSG code (e.g. 32610) selecting the zone and hemisphere.
        Only the zone/hemisphere are taken from it; the datum is NAD83(2011).

    Returns
    -------
    CRS
        Projected 3D CRS with a static NAD83(2011) datum and
        ellipsoidal-height axis.
    """
    label = utm_zone_label(utm_epsg)
    zone, hemisphere = label[:-1], label[-1]
    return ProjectedCRS(
        conversion=UTMConversion(zone, hemisphere=hemisphere),
        geodetic_crs=CRS.from_epsg(NAD83_2011_EPSG),
        name=f"NAD83(2011) / UTM zone {label}",
    ).to_3d()


WGS84_G1674_EPSG = 9056
ITRF2020_EPSG = 9990
ITRF2008_EPSG = 8999
ITRF2014_EPSG = 9000


def build_utm_realization_3d(utm_epsg: int, base_epsg: int, base_name: str) -> CRS:
    """
    Build a 3D UTM CRS on an arbitrary geographic realization.

    Generalizes the G2139/NAD83(2011) builders for additional output frames
    (e.g. matching a product delivery's realization for multi-frame
    validation). Same construction: UTM conversion on the given geographic
    2D base, promoted to 3D (ellipsoidal heights).

    Parameters
    ----------
    utm_epsg
        WGS84 UTM EPSG code (e.g. 32611) selecting zone/hemisphere only.
    base_epsg
        Geographic 2D EPSG code of the target realization (e.g. 9056 for
        WGS 84 (G1674), 9990 for ITRF2020).
    base_name
        Realization name used in the CRS name.

    Returns
    -------
    CRS
        Projected 3D CRS on the requested realization.
    """
    label = utm_zone_label(utm_epsg)
    zone, hemisphere = label[:-1], label[-1]
    return ProjectedCRS(
        conversion=UTMConversion(zone, hemisphere=hemisphere),
        geodetic_crs=CRS.from_epsg(base_epsg),
        name=f"{base_name} / UTM zone {label}",
    ).to_3d()


def build_utm_g1674_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on WGS 84 (G1674) (~ITRF2008; dynamic — stamp an epoch)."""
    return build_utm_realization_3d(utm_epsg, WGS84_G1674_EPSG, "WGS 84 (G1674)")


def build_utm_itrf2020_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2020 (dynamic — stamp an epoch)."""
    return build_utm_realization_3d(utm_epsg, ITRF2020_EPSG, "ITRF2020")


def build_utm_itrf2008_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2008 (≡ WGS 84 (G1674) to ~cm; dynamic).

    ⚠ Prefer this over the wgs84_g1674 target when coord_epoch is used:
    with GDAL selecting the operation (no -ct), a WGS84-realization target
    gets the NULL NAD83<->WGS84 tie HORIZONTALLY (~1.2-1.3 m CONUS error),
    while the ITRF alias finds the direct time-dependent
    ITRFxxxx<->NAD83(2011) Helmert (verified empirically, LV T2 2026-07-09).
    """
    return build_utm_realization_3d(utm_epsg, ITRF2008_EPSG, "ITRF2008")


def build_utm_itrf2014_3d(utm_epsg: int) -> CRS:
    """3D UTM CRS on ITRF2014 (≡ WGS 84 (G2139) to ~cm; dynamic).

    ⚠ Same GDAL null-tie caveat as build_utm_itrf2008_3d: use this instead
    of wgs84_g2139 whenever coord_epoch is passed.
    """
    return build_utm_realization_3d(utm_epsg, ITRF2014_EPSG, "ITRF2014")


# Selectable output-datum realizations for the auto-built local-UTM target.
# key -> (3D UTM builder, filename datum label). Arbitrary output CRSs beyond
# these are still supported by passing an explicit dst_crs WKT file.
OUTPUT_DATUM_BUILDERS = {
    "wgs84_g2139": (build_utm_g2139_3d, "WGS84_G2139"),
    "nad83_2011": (build_utm_nad83_2011_3d, "NAD83_2011"),
    "wgs84_g1674": (build_utm_g1674_3d, "WGS84_G1674"),
    "itrf2020": (build_utm_itrf2020_3d, "ITRF2020"),
    "itrf2008": (build_utm_itrf2008_3d, "ITRF2008"),
    "itrf2014": (build_utm_itrf2014_3d, "ITRF2014"),
}


def build_utm_target(utm_epsg: int, output_datum: str = "wgs84_g2139") -> "tuple[CRS, str]":
    """
    Build the auto-target 3D UTM CRS and its canonical WKT filename for a UTM
    zone and a selectable output datum realization.

    Parameters
    ----------
    utm_epsg
        WGS84 UTM EPSG code selecting the zone/hemisphere (e.g. from
        ``gdf.estimate_utm_crs().to_epsg()``).
    output_datum
        Output datum key, one of ``OUTPUT_DATUM_BUILDERS`` ('wgs84_g2139'
        default, or 'nad83_2011').

    Returns
    -------
    tuple[CRS, str]
        The 3D UTM CRS and a canonical basename like
        'UTM_10N_NAD83_2011_3D.wkt' (caller joins it with the run directory).
    """
    try:
        builder, label = OUTPUT_DATUM_BUILDERS[output_datum]
    except KeyError:
        raise ValueError(
            f"Unknown output_datum '{output_datum}'; choose from "
            f"{sorted(OUTPUT_DATUM_BUILDERS)} or pass an explicit dst_crs WKT file."
        )
    return builder(utm_epsg), f"UTM_{utm_zone_label(utm_epsg)}_{label}_3D.wkt"


def build_3857_navd88_compound() -> CRS:
    """
    Build the compound CRS describing geoid-referenced 3DEP EPT data:
    EPSG:3857 horizontal + NAVD88 (EPSG:5703) orthometric heights.

    Used as gdal.Warp source SRS so the geoid-to-ellipsoid shift is applied
    to the elevation values. Equivalent to the SRS_CRS.wkt previously fetched
    from GitHub at runtime.

    Returns
    -------
    CRS
        Compound CRS (EPSG:3857 + EPSG:5703).
    """
    return CompoundCRS(
        name="WGS 84 / Pseudo-Mercator + NAVD88 height",
        components=[CRS.from_epsg(3857), CRS.from_epsg(5703)],
    )


def epoch_pinned_pipeline(src_crs, dst_crs, coord_epoch: float,
                          aoi_bounds=None, require_substrings=()) -> str:
    """
    Resolve ONE explicit PROJ pipeline with the target coordinate epoch baked
    in (``projinfo --t_epoch``), for enforcement via gdalwarp ``-ct``.

    Operation AUTO-selection proved unstable across source datum declarations
    (LV four-frame validation 2026-07-10): GDAL free selection with
    ``-t_coord_epoch`` null-tied the horizontal Helmert for WGS84-realization
    targets, and flipped to null horizontal for ITRF targets when the source
    compound declared its true NAD83(2011) base. The only robust contract is
    an explicit pipeline. projinfo emits the top-ranked operation with
    ``+proj=set +v_4=<epoch>`` bookends, so the time-dependent Helmert is
    evaluated at the requested epoch without 4D input — usable as a static
    ``-ct`` string and recordable as provenance.

    NOTE: general-purpose geodesy with no lidar_tools dependencies — a
    candidate for migration into ``groundcontrol`` when lidar_tools adopts it
    as a dependency (planned refactor).

    Parameters
    ----------
    src_crs, dst_crs
        CRS object, WKT string, or path to a WKT file.
    coord_epoch
        Target coordinate epoch (decimal year).
    aoi_bounds
        Optional (west, south, east, north) degrees — passed as
        ``--bbox`` so area-appropriate operations rank first.
    require_substrings
        Substrings that MUST appear in the selected pipeline (e.g.
        ``["+proj=helmert", "vgridshift"]``) — fail loud on a null or
        wrong-geoid route instead of producing silently shifted rasters.

    Returns
    -------
    str
        The ``+proj=pipeline ...`` string of the top-ranked operation.
    """
    def _wkt(c):
        if isinstance(c, CRS):
            return c.to_wkt()
        p = Path(str(c))
        if p.exists():
            return p.read_text()
        return str(c)

    exe = Path(sys.executable).parent / "projinfo"
    projinfo = str(exe) if exe.exists() else shutil.which("projinfo")
    if projinfo is None:
        raise RuntimeError("projinfo executable not found")
    cmd = [projinfo, "-s", _wkt(src_crs), "-t", _wkt(dst_crs),
           "--t_epoch", str(coord_epoch), "--hide-ballpark",
           "--spatial-test", "intersects", "-o", "PROJ", "--single-line"]
    if aoi_bounds is not None:
        w, s, e, n = aoi_bounds
        cmd += ["--bbox", f"{w},{s},{e},{n}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"projinfo failed: {out.stderr[-500:]}")
    pipelines = [ln.strip() for ln in out.stdout.splitlines()
                 if ln.strip().startswith("+proj=pipeline")]
    if not pipelines:
        raise RuntimeError(
            f"projinfo returned no pipeline for {coord_epoch=}: "
            f"{out.stdout[-500:]}")
    pipe = pipelines[0]
    epoch_tag = f"+proj=set +v_4={coord_epoch:g}"
    missing = [s for s in ([epoch_tag] + list(require_substrings))
               if s not in pipe]
    if missing:
        raise RuntimeError(
            f"selected pipeline lacks required component(s) {missing}: {pipe}")
    return pipe


def build_ept_3857_navd88_compound(base_epsg: int = NAD83_2011_EPSG) -> CRS:
    """
    Compound CRS for geoid-referenced EPT data with the TRUE base datum:
    Pseudo-Mercator on the survey's NAD83-family realization + NAVD88 heights.

    Replaces :func:`build_3857_navd88_compound` (WGS84-based horizontal) as
    the geoid-branch warp source. Declaring the horizontal as generic WGS 84
    forces PROJ to reach NAVD88 through WGS84->NAD83(HARN) chains, and the
    declared-accuracy ranking then prefers GEOID03/NADCON5 routes for some
    targets (empirically 2-16 cm spatially varying vertical + horizontal
    error vs GEOID18, LV validation 2026-07-10). With the truthful
    NAD83(2011)-based horizontal, the NAVD88->ellipsoid candidate set
    collapses to the survey-consistent GEOID18 operation (single candidate
    to NAD83(2011) 3D, top-ranked at 0.015 m to ITRF targets).

    Parameters
    ----------
    base_epsg
        Geographic 2D EPSG code of the survey's true horizontal datum
        (see :func:`build_ept_3857_nad83_2011`).

    Returns
    -------
    CRS
        Compound CRS: Pseudo-Mercator (base-datum based) + NAVD88 height.
    """
    horiz = build_ept_3857_nad83_2011(three_d=False, base_epsg=base_epsg)
    base_name = CRS.from_epsg(base_epsg).name
    return CompoundCRS(
        name=f"Pseudo-Mercator ({base_name} based) + NAVD88 height",
        components=[horiz, CRS.from_epsg(5703)],
    )


def build_ept_3857_nad83_2011(three_d: bool = True, base_epsg: int = NAD83_2011_EPSG) -> CRS:
    """
    Build Pseudo-Mercator on the survey's true NAD83-family base datum,
    describing 3DEP EPT coordinates by what they actually are.

    EPT tiles were projected to EPSG:3857 with a null datum tie, so the
    numbers are values of the source realization relabeled WGS84. Declaring
    the actual datum makes PROJ apply the correct time-dependent Helmert
    when transforming to an ITRF-based output CRS, instead of relabeling
    the coordinates (~1.3 m horizontal / ~0.9 m vertical error in CONUS).

    Parameters
    ----------
    three_d
        If True, add an ellipsoidal-height axis so heights are transformed
        too (for surveys whose EPT Z is already ellipsoidal).
        Use False for rasters whose band values are not heights (intensity).
    base_epsg
        Geographic 2D EPSG code of the survey's true horizontal datum, by
        default NAD83(2011) (EPSG:6318). Older surveys may need e.g.
        NAD83(HARN) (EPSG:4152) or NAD83(NSRS2007) (EPSG:4759) — the
        realization difference reaches ~5-20 cm in deforming regions.
        Take it from the per-survey WESM record, not an assumption.

    Returns
    -------
    CRS
        Projected CRS with the declared datum and pseudo-Mercator conversion.
    """
    base = CRS.from_epsg(base_epsg)
    mercator = CRS.from_epsg(3857).to_json_dict()
    mercator["base_crs"] = base.to_json_dict()
    mercator["name"] = f"Pseudo-Mercator ({base.name} based)"
    # no longer EPSG:3857 once the base datum is replaced
    mercator.pop("id", None)
    crs = CRS.from_json_dict(mercator)
    return crs.to_3d() if three_d else crs


def navd88_offset(lon: float, lat: float) -> float:
    """
    Local NAVD88-to-NAD83(2011)-ellipsoidal offset N (meters, ~-18..-35 in
    CONUS): the ellipsoidal height of the zero-orthometric surface. Used as
    the expected signature of an already-ellipsoidal source in the vertical
    datum check. Requires the geoid grid (verify with the preflight first).
    """
    t = Transformer.from_crs("EPSG:6318+5703", "EPSG:6319", always_xy=True)
    return float(t.transform(lon, lat, 0.0)[2])


def write_crs_file(crs: CRS, outfn: str | Path) -> str:
    """
    Write a CRS definition as pretty WKT2:2019.

    The file is kept in the run directory as provenance for the exact CRS
    used (cleanup preserves *.wkt).

    Parameters
    ----------
    crs
        CRS to write.
    outfn
        Output filename.

    Returns
    -------
    str
        The output filename, usable as a gdal/PDAL SRS argument.
    """
    outfn = Path(outfn)
    print(f"Writing CRS definition to {outfn}")
    outfn.write_text(crs.to_wkt(version="WKT2_2019", pretty=True))
    return str(outfn)


def library_versions() -> dict:
    """Versions of the geodesy-relevant libraries, for provenance metadata."""
    return {
        "proj": pyproj.proj_version_str,
        "pyproj": pyproj.__version__,
        "gdal": gdal.__version__,
        "proj_data_dir": pyproj.datadir.get_data_dir(),
    }


def preflight_vertical_transform(
    src_crs: CRS | str,
    dst_crs: CRS | str,
    download: bool = True,
    aoi_bounds: tuple = None,
    require_grids: list = None,
    allow_geoid_fallback: bool = False,
) -> dict:
    """
    Verify PROJ can rigorously transform src_crs -> dst_crs before compute.

    With required datum-shift grids missing (e.g. GEOID18 us_noaa_g2018u0.tif)
    and PROJ networking off, gdal.Warp silently falls back to a null vertical
    transformation, leaving heights wrong by the geoid undulation (~31 m in
    CONUS) with no error. Run this before tile processing: if the best
    transformation is unavailable, it first tries to download the missing
    grids (when networking is enabled), then raises rather than letting the
    run continue toward a silently wrong product.

    Parameters
    ----------
    src_crs
        Source CRS (anything pyproj accepts).
    dst_crs
        Target CRS.
    download
        Attempt to download missing grids to the PROJ user data directory
        when pyproj networking is enabled, by default True.
    aoi_bounds
        (west, south, east, north) in degrees. Scopes transformation
        selection to the area of interest and asserts the selected
        operation's area-of-use contains it — without this, the
        accuracy-ranked best operation can belong to another region (e.g.
        a CONUS geoid grid selected for a Puerto Rico/Alaska AOI), which is
        then applied out-of-area when the pipeline is enforced via
        gdalwarp -ct.
    require_grids
        Exact PROJ grid filename(s) (e.g. from `resolve_declared_geoid`)
        that the vertical step MUST use — the survey's production geoid is
        part of the data definition. If no candidate transformation uses
        them (the EPSG registry marks legacy geoid realizations as
        superseded, so PROJ never lists them), the pipeline is CONSTRUCTED
        explicitly by swapping the grid into the ranked pipeline, the
        grids are materialized locally, and the result is sanity-checked
        numerically against the ranked pipeline. There is no silent
        fallback: failure raises unless ``allow_geoid_fallback``.
    allow_geoid_fallback
        Operator escape hatch (--geoid-override best-available): when the
        required grids cannot be used, warn LOUDLY and continue with the
        ranked-best pipeline instead of raising. Default False. This is an
        unblocking permission, NOT a model selector: when the required
        grids ARE usable they are still used — there is deliberately no
        mode that forces the ranked model over usable declared grids (a
        historical-comparison run wanting that is a separate feature).

    Returns
    -------
    dict
        Provenance record: the selected transformation description, PROJ
        pipeline, grids used, and stated accuracy, for
        processing_metadata.yaml.

    Raises
    ------
    RuntimeError
        If the most accurate transformation cannot be instantiated.
    """
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_user_input(dst_crs)
    aoi = (
        AreaOfInterest(
            west_lon_degree=aoi_bounds[0],
            south_lat_degree=aoi_bounds[1],
            east_lon_degree=aoi_bounds[2],
            north_lat_degree=aoi_bounds[3],
        )
        if aoi_bounds is not None
        else None
    )

    def make_group():
        # ballpark operations are the silent-fallback failure mode this
        # preflight exists to prevent: never consider them
        return TransformerGroup(
            src_crs,
            dst_crs,
            always_xy=True,
            area_of_interest=aoi,
            allow_ballpark=False,
        )

    group = make_group()
    if not group.best_available and download and pyproj.network.is_network_enabled():
        print(
            "Best available transformation requires datum-shift grids; "
            "downloading to the PROJ user-writable data directory"
        )
        group.download_grids(verbose=True)
        group = make_group()
    if not group.best_available or not group.transformers:
        missing = sorted(
            {
                grid.short_name or grid.full_name
                for op in group.unavailable_operations
                for grid in op.grids
                if not grid.available
            }
        )
        raise RuntimeError(
            f"PROJ cannot rigorously transform '{src_crs.name}' -> '{dst_crs.name}'"
            f"{f' within the AOI {aoi_bounds} (no non-ballpark operation covers its area of use)' if aoi_bounds is not None else ''}: "
            f"missing datum-shift grids {missing}. If grids are the problem, install "
            "them (e.g. 'pyproj sync --file <grid>' or the conda-forge proj-data "
            "package) or set PROJ_NETWORK=ON to allow on-demand grid download. "
            "Refusing to continue: a silent fallback would leave output heights "
            "wrong by the geoid undulation (~31 m in CONUS)."
        )
    best = group.transformers[0]
    pipeline_source = "ranked"
    substitution_note = None
    explicit_transformer = None
    if require_grids:
        required = list(require_grids)
        matching = [
            t
            for t in group.transformers
            if all(g in (t.definition or "") for g in required)
        ]
        if matching:
            best = matching[0]
            pipeline_source = "ranked-declared-geoid"
        else:
            # The EPSG registry marks legacy NAVD88 geoid realizations as
            # superseded by the newest one, and PROJ's operation factory
            # discards superseded operations entirely (root cause of the
            # 2026-07-18 GEOID12B->g2018 substitution: the g2012b grid was
            # installed locally, yet the g2012b operation was never even a
            # candidate). Ranking cannot cooperate — construct the pipeline
            # explicitly by swapping the declared grid into the ranked
            # pipeline, whose structure is otherwise identical.
            try:
                base_def = group.transformers[0].definition or ""
                new_def = _swap_vgridshift_grids(base_def, required)
                _ensure_grids_local(required)
                explicit_transformer = Transformer.from_pipeline(new_def)
                if aoi_bounds is not None:
                    # numeric guard against a structurally wrong swap
                    # (sign/units/coverage): the declared geoid must differ
                    # from the ranked one by model differences (<~2 m),
                    # never by the full undulation (~30 m) or worse
                    cx = (aoi_bounds[0] + aoi_bounds[2]) / 2.0
                    cy = (aoi_bounds[1] + aoi_bounds[3]) / 2.0
                    to_src = Transformer.from_crs(
                        "EPSG:4979", src_crs, always_xy=True, allow_ballpark=True
                    )
                    src_pt = to_src.transform(cx, cy, 0.0)
                    z_ranked = best.transform(*src_pt)[2]
                    z_declared = explicit_transformer.transform(*src_pt)[2]
                    dz = abs(z_declared - z_ranked)
                    if not dz < 2.0:
                        raise RuntimeError(
                            f"Swapped-grid pipeline disagrees with the ranked "
                            f"pipeline by {dz:.3f} m at the AOI center — the "
                            f"declared grids {required} do not behave like a "
                            "geoid-model substitution; refusing."
                        )
                pipeline_source = "explicit-declared-geoid"
            except Exception as e:
                if allow_geoid_fallback:
                    print(
                        f"WARNING: declared geoid grids {required} cannot be "
                        f"used ({e}); geoid-override accepted — continuing "
                        "with the best-available model "
                        f"({group.transformers[0].description}). Heights will "
                        "differ from the survey's production geoid by the "
                        "model difference.",
                        file=sys.stderr,
                    )
                    best = group.transformers[0]
                    explicit_transformer = None
                    pipeline_source = "fallback-best-available"
                    substitution_note = (
                        f"declared grids {required} unusable ({e}); operator "
                        "accepted best-available substitution"
                    )
                else:
                    raise RuntimeError(
                        f"The survey's declared geoid grids {required} cannot "
                        f"be used for '{src_crs.name}' -> '{dst_crs.name}': "
                        f"{e}. Remedies: install the grid "
                        f"('pyproj sync --file {required[0].split('.')[0]}'), "
                        "set PROJ_NETWORK=ON, or consciously accept "
                        "substitution with --geoid-override best-available. "
                        "Refusing to silently substitute another geoid model "
                        "(cm-level vertical bias)."
                    ) from e
    # never enforce a pipeline outside its stated validity area
    if aoi_bounds is not None and best.area_of_use is not None:
        a = best.area_of_use
        west, south, east, north = aoi_bounds

        def lon_in(lon):
            # areas of use spanning the antimeridian (e.g. CONUS+Alaska)
            # have west > east
            if a.west <= a.east:
                return a.west <= lon <= a.east
            return lon >= a.west or lon <= a.east

        if not (
            a.south <= south and north <= a.north and lon_in(west) and lon_in(east)
        ):
            raise RuntimeError(
                f"Selected transformation '{best.description}' has area of use "
                f"'{a.name}' ({a.bounds}), which does not contain the AOI "
                f"{aoi_bounds}. Refusing to enforce an out-of-area pipeline."
            )
    if explicit_transformer is not None:
        definition = explicit_transformer.definition or ""
        description = f"{best.description} [declared geoid grids {list(require_grids)}]"
    else:
        definition = best.definition or ""
        description = best.description
    grids = sorted(
        {name for match in re.findall(r"grids=(\S+)", definition) for name in match.split(",")}
    )
    # every grid the selected pipeline touches must be a LOCAL file NOW —
    # never defer grid access to the warp stage (CDN blips fail hours in)
    _ensure_grids_local(grids)
    print(
        f"Transform preflight OK: '{src_crs.name}' -> '{dst_crs.name}' via "
        f"{description} (accuracy {best.accuracy} m, grids {grids or 'none'}, "
        f"pipeline {pipeline_source})"
    )
    record = {
        "source_crs": src_crs.name,
        "target_crs": dst_crs.name,
        "description": description,
        "proj_pipeline": definition,
        "grids": grids,
        "accuracy_m": best.accuracy,
        "area_of_use": best.area_of_use.name if best.area_of_use else None,
        "pipeline_source": pipeline_source,
    }
    if substitution_note:
        record["substitution"] = substitution_note
    return record


def set_coordinate_epoch(
    raster_fn: str | Path,
    epoch: float = DEFAULT_COORDINATE_EPOCH,
    crs: CRS = None,
) -> bool:
    """
    Stamp a coordinate epoch on a raster whose CRS is dynamic.

    3DEP sources are NAD83(2011) epoch-reduced to 2010.0, and the
    NAD83(2011)<->ITRF Helmert is evaluated at its 2010.0 reference epoch, so
    outputs in a dynamic frame (default WGS 84 (G2139) UTM) are coordinates
    at epoch 2010.0. Without the stamp they are ambiguous by ~1.65 cm/yr of
    plate motion when compared against ITRF/GNSS-era data. Static target CRSs
    (e.g. NAD83(2011)) carry no epoch and are left unstamped.

    Stamp before building overviews: the COG translate in gdal_add_overview
    carries the epoch through.

    Parameters
    ----------
    raster_fn
        Path to the raster file (modified in place).
    epoch
        Decimal-year coordinate epoch, by default 2010.0.
    crs
        Authoritative CRS of the raster. The GeoTIFF round-trip can drop
        the DYNAMIC property from custom-datum CRSs (observed for 2D
        demotions), making the file SRS look static; pass the intended CRS
        to decide dynamic-ness from it and rewrite the full definition
        along with the epoch.

    Returns
    -------
    bool
        True if the epoch was stamped, False if the CRS is static or missing.
    """
    from osgeo import osr

    with gdal.OpenEx(
        str(raster_fn),
        gdal.OF_RASTER | gdal.OF_UPDATE,
        open_options=["IGNORE_COG_LAYOUT_BREAK=YES"],
    ) as ds:
        if crs is not None:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.to_wkt())
        else:
            srs = ds.GetSpatialRef()
            srs = srs.Clone() if srs is not None else None
        if srs is None or not srs.IsDynamic():
            return False
        srs.SetCoordinateEpoch(epoch)
        ds.SetSpatialRef(srs)
    print(f"Stamped coordinate epoch {epoch} on {raster_fn}")
    return True
