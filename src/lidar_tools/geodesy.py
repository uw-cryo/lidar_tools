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
from pathlib import Path

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


# Selectable output-datum realizations for the auto-built local-UTM target.
# key -> (3D UTM builder, filename datum label). Arbitrary output CRSs beyond
# these are still supported by passing an explicit dst_crs WKT file.
OUTPUT_DATUM_BUILDERS = {
    "wgs84_g2139": (build_utm_g2139_3d, "WGS84_G2139"),
    "nad83_2011": (build_utm_nad83_2011_3d, "NAD83_2011"),
    "wgs84_g1674": (build_utm_g1674_3d, "WGS84_G1674"),
    "itrf2020": (build_utm_itrf2020_3d, "ITRF2020"),
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
    prefer_grids: str = None,
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
    prefer_grids
        Substring (e.g. 'g2012b') selecting the first available
        transformation whose pipeline uses a matching grid — for honoring a
        survey's production geoid model instead of PROJ's default ranking.

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
    if prefer_grids is not None:
        matching = [
            t for t in group.transformers if prefer_grids in (t.definition or "")
        ]
        if not matching:
            raise RuntimeError(
                f"No available transformation '{src_crs.name}' -> '{dst_crs.name}' "
                f"uses a grid matching '{prefer_grids}' (candidates: "
                f"{[t.description for t in group.transformers[:5]]})"
            )
        best = matching[0]
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
    definition = best.definition or ""
    grids = sorted(
        {name for match in re.findall(r"grids=(\S+)", definition) for name in match.split(",")}
    )
    print(
        f"Transform preflight OK: '{src_crs.name}' -> '{dst_crs.name}' via "
        f"{best.description} (accuracy {best.accuracy} m, grids {grids or 'none'})"
    )
    return {
        "source_crs": src_crs.name,
        "target_crs": dst_crs.name,
        "description": best.description,
        "proj_pipeline": definition,
        "grids": grids,
        "accuracy_m": best.accuracy,
        "area_of_use": best.area_of_use.name if best.area_of_use else None,
    }


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
