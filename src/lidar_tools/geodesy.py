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
from pyproj import CRS
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


def build_ept_3857_nad83_2011(three_d: bool = True) -> CRS:
    """
    Build Pseudo-Mercator on a NAD83(2011) base, describing 3DEP EPT
    coordinates by their true datum.

    EPT tiles were projected to EPSG:3857 with a null datum tie, so the
    numbers are NAD83(2011) values relabeled WGS84. Declaring the actual
    datum makes PROJ apply the ITRF<->NAD83(2011) time-dependent Helmert
    when transforming to an ITRF-based output CRS, instead of relabeling
    the coordinates (~1.3 m horizontal / ~0.9 m vertical error in CONUS).

    Parameters
    ----------
    three_d
        If True, add an ellipsoidal-height axis so heights are transformed
        too (for surveys whose EPT Z is already NAD83(2011) ellipsoidal).
        Use False for rasters whose band values are not heights (intensity).

    Returns
    -------
    CRS
        Projected CRS with NAD83(2011) datum and pseudo-Mercator conversion.
    """
    mercator = CRS.from_epsg(3857).to_json_dict()
    mercator["base_crs"] = CRS.from_epsg(NAD83_2011_EPSG).to_json_dict()
    mercator["name"] = "Pseudo-Mercator (NAD83(2011) based)"
    # no longer EPSG:3857 once the base datum is replaced
    mercator.pop("id", None)
    crs = CRS.from_json_dict(mercator)
    return crs.to_3d() if three_d else crs


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
    src_crs: CRS | str, dst_crs: CRS | str, download: bool = True
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
    group = TransformerGroup(src_crs, dst_crs, always_xy=True)
    if not group.best_available and download and pyproj.network.is_network_enabled():
        print(
            "Best available transformation requires datum-shift grids; "
            "downloading to the PROJ user-writable data directory"
        )
        group.download_grids(verbose=True)
        group = TransformerGroup(src_crs, dst_crs, always_xy=True)
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
            f"PROJ cannot rigorously transform '{src_crs.name}' -> '{dst_crs.name}': "
            f"missing datum-shift grids {missing}. Install them (e.g. "
            "'pyproj sync --file <grid>' or the conda-forge proj-data package) "
            "or set PROJ_NETWORK=ON to allow on-demand grid download. "
            "Refusing to continue: without these grids, output heights would "
            "be silently wrong by the geoid undulation (~31 m in CONUS)."
        )
    best = group.transformers[0]
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
