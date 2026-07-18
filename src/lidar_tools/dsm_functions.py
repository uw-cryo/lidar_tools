"""
Function library for lidar_tools
"""

from rasterio.warp import transform_bounds
from pyproj import CRS
import shapely
import geopandas as gpd
import requests
import rasterio
import xarray as xr
import rioxarray
import pystac_client
import numpy as np
import json
import sys
import time
from pathlib import Path

# import planetary_computer
from osgeo import gdal, gdalconst
import pdal
import odc.stac
import os

gdal.UseExceptions()

odc.stac.configure_rio(cloud_defaults=True)

def nearest_floor(x: int | float, a: int | float) -> int | float:
    """
    Round down to the nearest smaller multiple of a.
    """
    return np.floor(x / a) * a


def nearest_ceil(x: int | float, a: int | float) -> int | float:
    """
    Round down to the nearest larger multiple of a.
    """
    return np.ceil(x / a) * a


def tap_bounds(site_bounds: tuple | list | np.ndarray, res: int | float) -> list[float]:
    """
    calculate target aligned pixel bounds for a given site bounds and resolution.

    Parameters
    ----------
    site_bounds
        array of bounds with the following order [minx, miny, maxx, maxy]
    res
        resolution in same units of site_bounds

    Returns
    -------
    target_aligned_bounds
        Adjusted bounds such that extent is a multiple of resolution

    Notes
    -----
    - From https://github.com/uw-cryo/EarthLab_AirQuality_UAV/blob/main/notebooks/EarthLab_AQ_lidar_download_processing_function.ipynb
    - See also https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-tap
    """
    return [
        nearest_floor(site_bounds[0], res),
        nearest_floor(site_bounds[1], res),
        nearest_ceil(site_bounds[2], res),
        nearest_ceil(site_bounds[3], res),
    ]


#: per-dataset ept.json SRS cache (the metadata is immutable per dataset)
_EPT_SRS_CACHE: dict = {}


def _ept_srs_wkt(url: str, attempts: int = 4, backoff_s: float = 2.0) -> str:
    """Fetch (once) and cache the SRS WKT from an EPT dataset's ept.json.

    S3 occasionally returns a transient non-JSON error body (e.g. 503
    SlowDown) — retry with backoff instead of crashing a 1000-tile run.
    """
    if url in _EPT_SRS_CACHE:
        return _EPT_SRS_CACHE[url]
    last_exc = None
    for i in range(attempts):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            wkt = response.json()["srs"]["wkt"]
            _EPT_SRS_CACHE[url] = wkt
            return wkt
        except Exception as exc:  # transient S3 / network errors
            last_exc = exc
            print(f"ept.json fetch attempt {i + 1}/{attempts} failed for "
                  f"{url}: {exc}", file=sys.stderr)
            time.sleep(backoff_s * (i + 1))
    raise RuntimeError(f"could not fetch {url} after {attempts} attempts") \
        from last_exc


def return_readers(
    input_aoi: gpd.GeoDataFrame,
    pointcloud_resolution: float = 1.0,
    tile_size_km: float = 1.0,
    buffer_value: int = 5,
    return_specific_3dep_survey: str = None,
    return_all_intersecting_surveys: bool = False,
    ept_index_gdf: gpd.GeoDataFrame = None,
) -> tuple[list, list, list, list]:
    """
    This method takes an input aoi and finds overlapping 3DEP EPT data from https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json
    It then returns a series of readers corresponding to non-overlapping areas for PDAL processing pipelines

    Parameters
    ----------
    input_aoi
        The area of interest as a polygon.
    pointcloud_resolution
        The resolution of the point cloud data, by default 1.
    tile_size_km
        The size of the EPT processing tiles in kilometers, by default 1.0.
    buffer_value
        The buffer value in meters to apply to each tile for querying sorrounding tiles, by default 5.
    return_specific_3dep_survey
        A specific 3DEP survey to return, by default first intersecting survey is returned.
        Must be the EPT resource name (callers resolve WESM workunit aliases first;
        see survey.resolve_ept_resource).
    return_all_intersecting_surveys
        If True, return all intersecting surveys, by default False.
    ept_index_gdf
        Preloaded EPT resource boundary index (any CRS). When None, the hobu
        index is fetched and spatially filtered to the AOI here.

    Returns
    -------
    (readers, pointcloud_input_crs, extents, original_extents) :
        A tuple of lists consisting of the following:
         - PDAL reader dictionaries for each non-overlapping area.
         - coordinate reference systems from EPT metadata.
         - buffered extents
         - original extents
    """

    #Load EPT polygon boundary index for user AOI
    #Reproject to EPSG:3857 for subsequent intersection operations
    if ept_index_gdf is None:
        ept_index_gdf = gpd.read_file(
            "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson",
            mask=input_aoi
        ).to_crs(CRS.from_epsg(3857))
    else:
        # preloaded full index (avoids a second fetch): apply the same
        # AOI spatial filter the masked read would have
        ept_index_gdf = ept_index_gdf.to_crs(CRS.from_epsg(3857))
        aoi_union_3857 = input_aoi.to_crs(CRS.from_epsg(3857)).union_all()
        ept_index_gdf = ept_index_gdf[ept_index_gdf.intersects(aoi_union_3857)]
    # Can read from copy stored in the github repo if necessary
    # ept_index_gdf = gpd.read_file('../data/shapefiles/resources.geojson')

    print(f"Identified {len(ept_index_gdf)} 3DEP projects intersecting user AOI:")
    print(ept_index_gdf['name'], end="\n\n")

    # Reproject input AOI to EPSG:3857 (units of meters)
    input_aoi_3857 = input_aoi.to_crs(CRS.from_epsg(3857))

    # Prepare grid of tiles for the AOI bbox
    xmin, ymin, xmax, ymax = input_aoi_3857.total_bounds
    x_step = tile_size_km * 1000  # convert km to m
    y_step = tile_size_km * 1000  # convert km to m
    n_cols = int(np.ceil((xmax - xmin) / x_step))
    n_rows = int(np.ceil((ymax - ymin) / y_step))
    n_tiles = n_cols * n_rows

    readers = []
    pointcloud_input_crs = []
    original_extents = []
    extents = []

    print(f"Preparing PDAL pipelines for each {tile_size_km} x {tile_size_km} km tile: {n_cols} cols x {n_rows} rows, {n_tiles} total tiles\n")

    for i in range(n_cols):
        for j in range(n_rows):
            tilenum = (i*n_rows) + (j+1)
            aoi = shapely.geometry.Polygon.from_bounds(
                xmin + i * x_step,
                ymin + j * y_step,
                min(
                    xmin + (i + 1) * x_step, xmax
                ),  # Ensure the tile does not exceed AOI bounds
                min(
                    ymin + (j + 1) * y_step, ymax
                ),  # Ensure the tile does not exceed AOI bounds
            )

            # create tap bounds for the tile
            src_bounds_transformed_3857 = tap_bounds(aoi.bounds, pointcloud_resolution)
            aoi_3857 = shapely.geometry.Polygon.from_bounds(*src_bounds_transformed_3857)
            #print(aoi.bounds, src_bounds_transformed_3857)

            #Check to make sure the tile intersects the original user AOI, not just the bbox envelope
            if (input_aoi_3857.geometry.intersects(aoi_3857)).any():
                print(f"Column {i+1} of {n_cols}, Row {j+1} of {n_rows}, Tile {tilenum} of {n_tiles}")
                if buffer_value:
                    #print(f"The tile polygon will be buffered by {buffer_value:.2f} m")
                    # buffer the tile polygon by the buffer value
                    aoi_3857 = aoi_3857.buffer(buffer_value)
                    # now create tap bounds for the buffered tile
                    aoi_3857_bounds = tap_bounds(aoi_3857.bounds, pointcloud_resolution)
                    # now convert the buffered tile to a polygon
                    aoi_3857 = shapely.geometry.Polygon.from_bounds(*aoi_3857_bounds)
                    #print("The buffered tile bound is: ", aoi_3857.bounds)

                if return_specific_3dep_survey is not None:
                    return_all_intersecting_surveys = True
                #Better to do intersection with the geodataframe first, rather than looping through each polygon
                for _, row in (ept_index_gdf[ept_index_gdf.intersects(aoi)]).iterrows():
                    usgs_dataset_name = row["name"]
                    if return_specific_3dep_survey is not None:
                        if usgs_dataset_name == return_specific_3dep_survey:
                            add_survey = True
                        else:
                            add_survey = False
                    else:
                        add_survey = True
                    if add_survey:
                        print(f"3DEP Dataset(s): {usgs_dataset_name}")
                        url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json"
                        reader = {
                            "type": "readers.ept",
                            "filename": url,
                            "requests": 15,
                            "resolution": pointcloud_resolution,
                            "polygon": str(aoi_3857.wkt),
                        }

                        # SRS associated with the 3DEP dataset — cached per
                        # dataset + retried: this used to issue one S3 GET per
                        # TILE (1000+ for large AOIs) and a single transient
                        # non-JSON response (e.g. 503 SlowDown) killed the run
                        srs_wkt = _ept_srs_wkt(url)

                        pointcloud_input_crs.append(CRS.from_wkt(srs_wkt))
                        readers.append(reader)
                        extents.append(aoi_3857.bounds)
                        original_extents.append(src_bounds_transformed_3857)
                    if not return_all_intersecting_surveys:
                        break

    return readers, pointcloud_input_crs, extents, original_extents


def return_crs_local_lpc(lpc: str) -> CRS:
    """
    Given a local laz file, return the coordinate reference system (CRS) of the point cloud.

    Parameters
    ----------
    lpc
        Path to the local laz file.

    Returns
    -------
    crs
        The coordinate reference system of the point cloud.
    """
    pipeline = pdal.Reader(lpc).pipeline()
    pipeline.execute()
    srs_wkt2 = pipeline.srswkt2
    crs = CRS.from_wkt(srs_wkt2)
    pipeline = None
    return crs


def return_lpc_bounds(lpc: str, output_crs: CRS = None) -> list:
    """
     Given a local laz file, return the bounds of the point cloud.

     Parameters
     ----------
     lpc
         Path to the local laz file.
     output_crs
         The coordinate reference system to transform the bounds to, by default None.

     Returns
    --------
     bounds
         The bounds of the point cloud in the format [xmin, ymin, xmax, ymax].
    """
    pipeline = pdal.Reader(lpc).pipeline()
    # quickinfo reads header metadata only; do not execute the pipeline here
    # (a full execute loads every point into memory just to read bounds)
    quickinfo = pipeline.quickinfo["readers.las"]
    pdal_bounds = quickinfo["bounds"]
    minx, miny, maxx, maxy = (
        pdal_bounds["minx"],
        pdal_bounds["miny"],
        pdal_bounds["maxx"],
        pdal_bounds["maxy"],
    )
    output_bounds = [minx, miny, maxx, maxy]
    if output_crs is not None:
        src_crs = CRS.from_wkt(quickinfo["srs"]["compoundwkt"])
        if not isinstance(output_crs, CRS):
            output_crs = CRS.from_user_input(output_crs)
        if src_crs != output_crs:
            output_bounds = transform_bounds(
                src_crs, output_crs, minx, miny, maxx, maxy
            )
    pipeline = None
    return output_bounds


def return_local_lpc_reader(
    lpc: str,
    input_crs: CRS = None,
    output_crs: CRS = None,
    pointcloud_resolution: float = 1.0,
    aoi_bounds: gpd.GeoDataFrame = None,
    buffer_value: int = 5,  # this should be multiple of input resolution
) -> tuple[dict, CRS, list]:
    """
    Given a local laz file, return the PDAL reader for the point cloud.

    Parameters
    ----------
    lpc
        Path to the local laz file.
    input_crs
        Override CRS of the input point cloud.
    output_crs
        The CRS to transform the bounds to
    aoi_bounds
        The area of interest bounds to intersect with the point cloud bounds
    buffer_value
        The buffer value in meters to apply to the bounds

    Returns
    -------
    reader
        The PDAL reader for the point cloud.
    in_crs
        The coordinate reference system of the point cloud.
    output_bounds
        The bounds of the point cloud in the format [xmin, ymin, xmax, ymax] for DEM gridding.
    """
    # first attempt is that we just use the bounds of the laz file and grid everything within it
    # after initial testing, we will perform intersection with the aoi_bounds, and crop the laz file to that bounds with some buffer, and then grid to that bounds without the buffer

    # get the bounds of the laz file
    bounds = return_lpc_bounds(lpc)
    if input_crs is None:
        input_crs = return_crs_local_lpc(lpc)

    # adding function to utilize
    # if the bounds are not in the output crs, transform them

    lpc_polygon = shapely.geometry.Polygon.from_bounds(*bounds)
    lpc_gdf = gpd.GeoDataFrame(geometry=[lpc_polygon], crs=input_crs, index=[0])
    aoi_bounds_in_crs = aoi_bounds.to_crs(input_crs)

    reader = {"type": "readers.las", "filename": lpc}
    pipeline = {"pipeline": [reader]}

    intersection = lpc_gdf.intersection(aoi_bounds_in_crs.unary_union)

    if not intersection.is_empty.any():
        return_reader = True

        if intersection.area.values[0] < lpc_gdf.area.values[0]:
            output_bounds = intersection.total_bounds
            # crop to extent of intersection area
            if buffer_value is not None:
                intersection = intersection.buffer(buffer_value)
            intersection_bounds = intersection.total_bounds

            crop_filter = {
                "type": "filters.crop",
                "bounds": f"([{intersection_bounds[0]},{intersection_bounds[2]}],"
                f"[{intersection_bounds[1]},{intersection_bounds[3]}])",
            }
            pipeline["pipeline"] += [crop_filter]

        else:
            output_bounds = bounds

    else:
        return_reader = False
    if return_reader:
        if output_crs is not None:
            if input_crs != output_crs:
                output_bounds = transform_bounds(input_crs, output_crs, *output_bounds)

        tapped_bounds = tap_bounds(output_bounds, pointcloud_resolution)

        return pipeline, input_crs, tapped_bounds
    else:
        return None, None, None


# need to revisit this, a lot of the functionality is not used


def create_pdal_pipeline(
    filter_low_noise: bool = False,
    filter_high_noise: bool = False,
    hag_nn: float = None,
    filter_road: bool = False,
    reset_classes: bool = False,
    reclassify_ground: bool = False,
    return_only_ground: bool = False,
    percentile_filter: bool = False,
    percentile_threshold: float = 0.98,
    group_filter: str = "first,only",
    reproject: bool = True,
    proj_pipeline: str = None,
    save_pointcloud: bool = False,
    pointcloud_file: str = "pointcloud",
    input_crs: CRS = None,
    output_crs: CRS = None,
    output_type: str = "laz",
) -> dict:
    """
    Create a PDAL pipeline for processing point clouds.

    Parameters
    ----------
    filter_low_noise
        Whether to filter low noise points, by default False.
    filter_high_noise
        Whether to filter high noise points, by default False.
    hag_nn
        If specified, the height above ground (HAG) will be calculated using all nearest ground classified points, and all points greater than this value will be classified as high noise, by default None.
    filter_road : bool, optional
        Whether to filter road points, by default False.
    reset_classes
        Whether to reset point classifications, by default False.
    reclassify_ground
        Whether to reclassify ground points, by default False.
    return_only_ground
        Whether to return only ground points, by default False.
    percentile_filter
        Whether to apply a percentile filter, by default False.
    percentile_threshold
        The percentile threshold for the filter
    group_filter
        The group filter to apply, by default "first,only" for generating DSM.
    reproject
        Whether to reproject the point cloud, by default True.
    proj_pipeline
        A PROJ pipeline string to be used for reprojection of the point cloud. If specified, this will be used in combination with the input_crs and output_crs options.
    save_pointcloud
        Whether to save the point cloud to a file, by default False.
    pointcloud_file
        The filename for the output point cloud, by default 'pointcloud'.
    input_crs
        The input coordinate reference system, by default None.
    output_crs
        The output coordinate reference system, by default None.
    output_type
        The output type, either 'las' or 'laz', by default 'laz' if save_pointcloud is True.

    Returns
    -------
    dict
        A PDAL pipeline for processing point clouds.
    """
    if percentile_filter:
        assert abs(percentile_threshold) <= 1, (
            "Percentile threshold must be in range [0, 1]"
        )

    stage_filter_low_noise = {"type": "filters.range", "limits": "Classification![7:7]"}
    stage_filter_high_noise = {
        "type": "filters.range",
        "limits": "Classification![18:18]",
    }
    stage_filter_road = {"type": "filters.range", "limits": "Classification![11:11]"}
    stage_reset_classes = {"type": "filters.assign", "value": "Classification = 0"}
    stage_reclassify_ground = {
        "type": "filters.smrf",
        # added from pdal smrf documentation, in turn from Pingel, 2013
        "scalar": 1.2,
        "slope": 0.2,
        "threshold": 0.45,
        "window": 8.0,
    }
    stage_group_filter = {"type": "filters.returns", "groups": group_filter}
    stage_percentile_filter = {
        "type": "filters.python",
        "script": str(Path(__file__).parent / "filter_percentile.py"),
        "pdalargs": {"percentile_threshold": percentile_threshold},
        "function": "filter_percentile",
        "module": "anything",
    }
    stage_return_ground = {"type": "filters.range", "limits": "Classification[2:2]"}

    stage_save_pointcloud_las = {
        "type": "writers.las",
        "filename": f"{pointcloud_file}.las",
    }

    stage_save_pointcloud_laz = {
        "type": "writers.las",
        "compression": "true",
        "minor_version": "2",
        "dataformat_id": "0",
        "filename": f"{pointcloud_file}.laz",
    }

    # Build pipeline
    pipeline = []

    # resetting the original classifications resets
    # all point classifications to 0 (Unclassified)
    if reset_classes:
        pipeline.append(stage_reset_classes)
        if reclassify_ground:
            pipeline.append(stage_reclassify_ground)
    else:
        # we apply the percentile filter first as it
        # classifies detected outliers as 'high noise'

        if percentile_filter:
            pipeline.append(stage_percentile_filter)
        if filter_low_noise:
            pipeline.append(stage_filter_low_noise)
        if hag_nn is not None:
            # if hag_nn is specified, we classify all points with HAG greater than hag_nn as high noise
            stage_hag_nn = {
                "type": "filters.hag_nn"}
            stage_hag_nn_filter = {
                "type": "filters.assign",
                "value": [f"Classification = 18 WHERE HeightAboveGround > {hag_nn}"]
            }
            pipeline.append(stage_hag_nn)
            pipeline.append(stage_hag_nn_filter)

            filter_high_noise = True # ensure that we filter high noise points if hag_nn is specified
        if percentile_filter or filter_high_noise:
            pipeline.append(stage_filter_high_noise)
        if filter_road:
            pipeline.append(stage_filter_road)
        if group_filter is not None:
            pipeline.append(stage_group_filter)
    # For creating DTMs, we want to process only ground returns
    if return_only_ground:
        pipeline.append(stage_return_ground)

    if (output_crs is not None) & (input_crs is not None) and (reproject is True):
        if proj_pipeline is not None:
            stage_reprojection = {"type": "filters.projpipeline", "out_srs": str(output_crs)}
            stage_reprojection["coord_op"] = proj_pipeline
        else:
            stage_reprojection = {"type": "filters.reprojection", "out_srs": str(output_crs)}
            stage_reprojection["in_srs"] = str(input_crs)
        pipeline.append(stage_reprojection)

    # the pipeline can save the pointclouds to a separate file if needed
    if save_pointcloud:
        if output_type == "laz":
            pipeline.append(stage_save_pointcloud_laz)
        else:
            pipeline.append(stage_save_pointcloud_las)

    return pipeline


def create_dem_stage(
    dem_filename: str,
    extent: list,
    pointcloud_resolution: float = 1.0,
    gridmethod: str = "idw",
    dimension: str = "Z",
    data_type: str = "float32",
    nodata_value: int = -9999,
) -> list:
    """
    Create a PDAL stage for generating a DEM from a point cloud.

    Parameters
    ----------
    dem_filename
        The filename for the output DEM
    extent
        The extent of the DEM in the format [xmin, ymin, xmax, ymax]
    pointcloud_resolution
        The resolution of the point cloud
    gridmethod
        The grid method to use for generating the DEM, by default 'idw'
    dimension
        The dimension to use for the DEM, by default 'Z'
    data_type
        The data type for the raster values, by default 'float32'
    nodata_value
        The nodata value for the raster, by default -9999
    Returns
    -------
    dem_stage
        A list of PDAL stages for generating the DEM
    """

    # compute raster width and height
    width = (extent[2] - extent[0]) / pointcloud_resolution
    height = (extent[3] - extent[1]) / pointcloud_resolution
    #fix origin extent precision with respect to input resolution
    #from https://www.reddit.com/r/pythontips/comments/zw5ana/how_to_count_decimal_places/
    import decimal
    d = decimal.Decimal(str(pointcloud_resolution))
    precision = abs(d.as_tuple().exponent)

    origin_x = np.round(extent[0],precision)
    origin_y = np.round(extent[1],precision)
    dem_stage = {
        "type": "writers.gdal",
        "filename": dem_filename,
        "gdaldriver": "GTiff",
        "nodata": nodata_value,
        "data_type": data_type,
        "output_type": gridmethod,
        "resolution": float(pointcloud_resolution),
        "origin_x": origin_x,
        "origin_y": origin_y,
        "width": int(width),
        "height": int(height),
        "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES",
    }

    dem_stage.update({"dimension": dimension})

    return [dem_stage]

 # Dictionary mapping common dtype strings to GDAL data types
DTYPE_TO_GDAL = {
    "Byte": gdal.GDT_Byte,
    "UInt16": gdal.GDT_UInt16,
    "Int16": gdal.GDT_Int16,
    "UInt32": gdal.GDT_UInt32,
    "Int32": gdal.GDT_Int32,
    "Float32": gdal.GDT_Float32,
    "Float64": gdal.GDT_Float64,
    "CInt16": gdal.GDT_CInt16,
    "CInt32": gdal.GDT_CInt32,
    "CFloat32": gdal.GDT_CFloat32,
    "CFloat64": gdal.GDT_CFloat64,
}
def raise_file_limit(max_soft: int = 65536) -> int:
    """
    Raise the process soft open-file limit (RLIMIT_NOFILE) toward the hard limit.

    gdal.BuildVRT opens every tile simultaneously during mosaicking, which
    exceeds default soft limits (often 256-1024) for large AOIs (issue #43).

    Parameters
    ----------
    max_soft
        Upper bound on the requested soft limit, by default 65536.

    Returns
    -------
    int
        The resulting soft limit, or -1 if it could not be determined/raised.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max_soft if hard == resource.RLIM_INFINITY else min(hard, max_soft)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"Raised open-file limit (RLIMIT_NOFILE) from {soft} to {target}")
            soft = target
        return soft
    except Exception as e:
        print(f"Could not raise open-file limit: {e}", file=sys.stderr)
        return -1


def raster_mosaic(
    img_list: list,
    outfn: str,
    cog: bool = False,
    out_extent: list = None,

) -> None:
    """
    Given a list of input images, mosaic them into a COG raster by using vrt and gdal_translate

    Parameters
    ----------
    img_list
        List of input images to be mosaiced
    outfn
        Path to output mosaiced image
    cog
        Whether to create a COG-complaint raster (compressed and tiled)
    out_extent
        The extent of the output raster in the format [minx, miny, maxx, maxy]

    Returns
    -------
    None
    """
    # create vrt
    if type(img_list) is tuple:
        img_list = list(img_list)

    # Filter out None values from the image list
    img_list = [img for img in img_list if img is not None]
    if not img_list:
        # BuildVRT silently writes no file for an empty list, and the
        # downstream Translate then fails with a cryptic "vrt: No such file
        # or directory"; fail early with an actionable message instead
        raise ValueError(
            f"raster_mosaic: no input tiles to mosaic for {outfn} "
            "(all tiles empty or failed)"
        )
    vrt_fn = Path(outfn).with_suffix(".vrt")
    gdal.BuildVRT(vrt_fn, img_list, callback=gdal.TermProgress_nocb)
    if out_extent is not None:
        minx, miny, maxx, maxy = out_extent
        out_extent = [minx, maxy, maxx, miny]
    
    if cog:
        # translate to COG
        print(out_extent)
        gdal.Translate(
            outfn,
            vrt_fn,
            projWin=out_extent,
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            callback=gdal.TermProgress_nocb,
        )

    else:
        print(out_extent)
        # tiled + compressed intermediates enable multithreaded warping and
        # avoid very large uncompressed temp mosaics (issue #12 discussion)
        gdal.Translate(
            outfn,
            vrt_fn,
            projWin=out_extent,
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            callback=gdal.TermProgress_nocb,
        )
    # delete vrt
    os.remove(vrt_fn)


### Functions for datum checks


def get_esa_worldcover(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    version: str = "v200",
    mask_nodata: bool = False,
) -> xr.DataArray:
    """
    Adapted from easysnowdata.remote_sensing.get_esa_worldcover (MIT license)
    Author: Eric Gagliano https://github.com/egagli/easysnowdata/blob/main/easysnowdata/remote_sensing.py
    Fetches 10m ESA WorldCover global land cover data (2020 v100 or 2021 v200) for a given bounding box.

    Description:
    The discrete classification maps provide 11 classes defined using the Land Cover Classification System (LCCS)
    developed by the United Nations (UN) Food and Agriculture Organization (FAO).

    Parameters
    ----------
    bbox_input
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    version
        Version of the WorldCover data. The two versions are v100 (2020) and v200 (2021). Default is 'v200'.
    mask_nodata
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=0, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=0)

    Returns
    -------
    worldcover_da
        WorldCover DataArray with class information in attributes.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import easysnowdata
    >>>
    >>> # Define a bounding box for Mount Rainier
    >>> bbox = (-121.94, 46.72, -121.54, 46.99)
    >>>
    >>> # Fetch WorldCover data for the area
    >>> worldcover_da = easysnowdata.remote_sensing.get_esa_worldcover(bbox)
    >>>
    >>> # Plot the data using the example plot function
    >>> f, ax = worldcover_da.attrs['example_plot'](worldcover_da)

    Notes
    -----
    Data citation:
    Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A.,
    Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin,
    Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936.
    """

    def get_class_info():
        classes = {
            10: {"name": "Tree cover", "color": "#006400"},
            20: {"name": "Shrubland", "color": "#FFBB22"},
            30: {"name": "Grassland", "color": "#FFFF4C"},
            40: {"name": "Cropland", "color": "#F096FF"},
            50: {"name": "Built-up", "color": "#FA0000"},
            60: {"name": "Bare / sparse vegetation", "color": "#B4B4B4"},
            70: {"name": "Snow and ice", "color": "#F0F0F0"},
            80: {"name": "Permanent water bodies", "color": "#0064C8"},
            90: {"name": "Herbaceous wetland", "color": "#0096A0"},
            95: {"name": "Mangroves", "color": "#00CF75"},
            100: {"name": "Moss and lichen", "color": "#FAE6A0"},
        }
        return classes

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    if version == "v100":
        version_year = "2020"
    elif version == "v200":
        version_year = "2021"
    else:
        raise ValueError("Incorrect version number. Please provide 'v100' or 'v200'.")
    import planetary_computer

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox_gdf.total_bounds)
    worldcover_da = (
        odc.stac.load(
            search.items(), bbox=bbox_gdf.total_bounds, bands="map", chunks={}
        )["map"]
        .sel(time=version_year)
        .squeeze()
    )

    if mask_nodata:
        worldcover_da = worldcover_da.where(worldcover_da > 0)
        worldcover_da.rio.write_nodata(0, encoded=True, inplace=True)

    worldcover_da.attrs["class_info"] = get_class_info()
    # worldcover_da.attrs["cmap"] = get_class_cmap(worldcover_da.attrs["class_info"])
    worldcover_da.attrs["data_citation"] = (
        "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936."
    )

    # worldcover_da.attrs['example_plot'] = plot_classes

    return worldcover_da


def fetch_worldcover(
    raster_fn: str, match_grid_da: xr.DataArray = None
) -> xr.DataArray:
    """
    Fetches ESA WorldCover data for a given raster file extent.
    This function retrieves the ESA WorldCover data for the area defined by the raster file's extent.

    Parameters
    ----------
    raster_fn
        Path to the raster file
    match_grid_da
        Match the grid of the output data array to this data array. Default is None.

    Returns
    -------
    da_wc
        A DataArray containing the ESA WorldCover data for the specified area.
    """

    with rasterio.open(raster_fn) as dataset:
        bounds = dataset.bounds
        bounds = rasterio.warp.transform_bounds(dataset.crs, "EPSG:4326", *bounds)
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[shapely.box(*bounds)], crs="EPSG:4326", index=[0]
        )

    da_wc = get_esa_worldcover(bbox_gdf, mask_nodata=True)
    if match_grid_da is not None:
        da_wc = da_wc.rio.reproject_match(
            match_grid_da, resampling=rasterio.enums.Resampling.nearest
        )
    return da_wc


def common_mask(da_list: list, apply: bool = False) -> list | np.ndarray:
    """
    From a list of xarray dataarray objects sharing the same projection/extent/res, compute common mask where all input datasets have non-nan pixels

    Parameters
    ----------
    da_list
        List of xarray DataArray objects to compute the common mask from.
    apply
        If True, apply the common mask to the input DataArray objects. Default is False.

    Returns
    -------
    list or np.array
        If apply is True, returns a list of DataArray objects with the common mask applied.
        If apply is False, returns a numpy array representing the common mask.
    """
    # load nan layers as numpy array
    nan_arrays = np.array([np.isnan(da.values) for da in da_list])
    common_mask = 1 - np.any(nan_arrays, axis=0)

    if apply:
        common_mask_da_list = [da.where(common_mask, np.nan) for da in da_list]
        return common_mask_da_list
    else:
        return common_mask


def convert_bbox_to_geodataframe(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry,
) -> gpd.GeoDataFrame:
    """
    Adapted from easysnowdata.remote_sensing.get_esa_worldcover (MIT license)
    Author: Eric Gagliano https://github.com/egagli/easysnowdata/blob/main/easysnowdata/utils.py
    Converts the input to a GeoDataFrame.

    This function takes various input formats representing a bounding box and converts them
    to a standardized GeoDataFrame format.

    Parameters
    ----------
    bbox_input : GeoDataFrame or tuple or Shapely geometry or None
        The input bounding box in various formats.

    Returns
    -------
    GeoDataFrame
        The converted bounding box as a GeoDataFrame.

    Notes
    -----
    If no bounding box is provided (None), it returns a GeoDataFrame representing the entire world.
    """
    if bbox_input is None:
        # If no bounding box is provided, use the entire world
        print("No spatial subsetting because bbox_input was not provided.")
        bbox_input = gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(-180, -90, 180, 90)], crs="EPSG:4326"
        )
    if isinstance(bbox_input, gpd.GeoDataFrame):
        # If it's already a GeoDataFrame, return it
        return bbox_input
    if isinstance(bbox_input, tuple) and len(bbox_input) == 4:
        # If it's a tuple of four elements, treat it as (xmin, ymin, xmax, ymax)
        bbox_input = gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(*bbox_input)], crs="EPSG:4326"
        )
    elif isinstance(bbox_input, shapely.geometry.base.BaseGeometry):
        # If it's a Shapely geometry, convert it to a GeoDataFrame
        bbox_input = gpd.GeoDataFrame(geometry=[bbox_input], crs="EPSG:4326")

    return bbox_input


def get_copernicus_dem(
    bbox_input: gpd.GeoDataFrame
    | tuple
    | shapely.geometry.base.BaseGeometry
    | None = None,
    resolution: int = 30,
) -> xr.DataArray:
    """
    Adapted from easysnowdata.remote_sensing.get_esa_worldcover (MIT license)
    Author: Eric Gagliano https://github.com/egagli/easysnowdata/blob/main/easysnowdata/topography.py

    Fetches 30m or 90m Copernicus DEM from Microsoft Planetary Computer.

    This function retrieves the Copernicus Digital Elevation Model (DEM) data for a specified
    bounding box and resolution. The DEM represents the surface of the Earth including buildings,
    infrastructure, and vegetation.

    Parameters
    ----------
    bbox_input
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    resolution
        The resolution of the DEM, either 30 or 90 meters. Default is 30.

    Returns
    -------
    cop_dem_da
        A DataArray containing the Copernicus DEM data for the specified area.

    Raises
    ------
    ValueError
        If the resolution is not 30 or 90 meters.

    Notes
    -----
    The Copernicus DEM is a Digital Surface Model (DSM) derived from the WorldDEM, with additional
    editing applied to water bodies, coastlines, and other special features.

    Data citation:
    European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model.
    Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB. Accessed: 2024-03-18
    """
    import planetary_computer

    if resolution != 30 and resolution != 90:
        raise ValueError(
            "Copernicus DEM resolution is available in 30m and 90m. Please select either 30 or 90."
        )

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf = convert_bbox_to_geodataframe(bbox_input)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[f"cop-dem-glo-{resolution}"], bbox=bbox_gdf.total_bounds
    )
    cop_dem_da = odc.stac.load(search.items(), bbox=bbox_gdf.total_bounds, chunks={})[
        "data"
    ].squeeze()
    cop_dem_da = cop_dem_da.rio.write_nodata(-32767, encoded=True)

    return cop_dem_da


def fetch_cop30(raster_fn: str, match_grid_da: xr.DataArray = None) -> xr.DataArray:
    """
    Fetches Copernicus DEM data for a given raster file extent.
    This function retrieves the Copernicus DEM data for the area defined by the raster file's extent.

    Parameters
    ----------
    raster_fn
        Path to the raster file.
    match_grid_da
        Match the grid of the output data array to this data array. Default is None.

    Returns
    -------
    cop_da
        A DataArray containing the Copernicus DEM EGM2008 data for the specified area.
    """
    with rasterio.open(raster_fn) as dataset:
        bounds = dataset.bounds
        bounds = rasterio.warp.transform_bounds(dataset.crs, "EPSG:4326", *bounds)
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[shapely.box(*bounds)], crs="EPSG:4326", index=[0]
        )
    cop_da = get_copernicus_dem(bbox_gdf, resolution=30)
    if match_grid_da is not None:
        cop_da = cop_da.rio.reproject_match(
            match_grid_da, resampling=rasterio.enums.Resampling.bilinear
        )
    return cop_da


def confirm_3dep_vertical(raster_fn: str, bare_diff_tolerance: float = 3.0) -> bool:
    """
    Check if the 3DEP LiDAR DSM is with respect to geoid or ellipsoid by comparing it with COP30 EGM2008 DEM

    Parameters
    ----------
    raster_fn : str
        Path to the raster file.
    bare_diff_tolerance : float, optional
        Tolerance for the difference between COP30 EGM2008 and 3DEP LiDAR DSM over bareground and sparse vegetation surfaces, by default 3.0.

    Returns
    -------
    bool
        True if the 3DEP LiDAR DSM is with respect to geoid, False otherwise.
    """

    from lidar_tools import geodesy
    from pyproj import Transformer as _Transformer

    # decimated read: matching reference layers to a full production mosaic
    # grid needed ~174 GB at Las Vegas scale; a bounded ~1024^2 grid is
    # ample for a robust median datum decision
    lidar_da = _open_decimated_dataarray(raster_fn)
    worldcover_da = fetch_worldcover(raster_fn, lidar_da)
    cop30_da = fetch_cop30(raster_fn, lidar_da)
    lidar_da_masked, worldcover_da_masked, cop30_da_masked = common_mask(
        [lidar_da, worldcover_da, cop30_da], apply=True
    )
    dem_diff = lidar_da_masked - cop30_da_masked
    ## Mask out bare and sparse vegetation class
    bare_sparse_mask = worldcover_da_masked == 60
    dem_diff_bare = dem_diff.where(bare_sparse_mask, np.nan)
    valid_count = int(np.isfinite(dem_diff_bare.values).sum())
    median_diff = (
        float(np.nanmedian(dem_diff_bare.values)) if valid_count else float("nan")
    )
    # expected signature of an already-ellipsoidal source at this location
    cx = float(lidar_da.x.mean())
    cy = float(lidar_da.y.mean())
    lon, lat = _Transformer.from_crs(
        lidar_da.rio.crs, "EPSG:4326", always_xy=True
    ).transform(cx, cy)
    expected_undulation = geodesy.navd88_offset(lon, lat)
    print(
        f"Observed difference between COP30 EGM2008 and 3DEP LiDAR DSM over bareground and sparse vegetation surfaces is {median_diff:.2f} m ({valid_count} valid pixels); "
        f"local NAVD88->ellipsoid offset is {expected_undulation:+.2f} m"
    )
    out = datum_shift_required(
        median_diff,
        valid_count,
        expected_undulation,
        tolerance=bare_diff_tolerance,
    )
    if out:
        # this means that both COP30 and 3DEP LiDAR DSM are with respect to geoid
        print(
            "Looks like the 3DEP height estimates are with respect to geoid, will apply vertical datum shift to return heights with respect to ellipsoid"
        )
    else:
        # this means that 3DEP LiDAR DSM is with respect to ellipsoid
        print(
            "Looks like the 3DEP height estimates are already with respect to ellipsoid, geoid to ellipsoid transformation will not be attempted"
        )
    return out


def datum_shift_required(
    median_diff: float,
    valid_count: int,
    expected_undulation: float,
    tolerance: float = 3.0,
    min_valid_pixels: int = 100,
) -> bool:
    """
    Decide whether heights are geoid-referenced from a DSM-minus-COP30 sample.

    Three-state decision: the sample must match one of the two physically
    expected signatures, otherwise it is an error. A simple two-state test
    (anything beyond +/-tolerance means "ellipsoidal") silently mislabels
    AOIs where terrain steepness, snow/surface change, or reference-DEM
    error push the median past the tolerance — re-arming the silent ~30 m
    error class this check exists to prevent.

    Parameters
    ----------
    median_diff
        Median of (lidar DSM - COP30/EGM2008) over bare/sparse-vegetation pixels.
    valid_count
        Number of valid pixels contributing to the median.
    expected_undulation
        Local NAVD88-to-ellipsoid offset N in meters (negative in CONUS,
        ~-18..-35; see geodesy.navd88_offset). An already-ellipsoidal source
        must show median_diff within tolerance of this value.
    tolerance
        Half-width of the acceptance window for both signatures, by default 3.0.
    min_valid_pixels
        Minimum sample size for a reliable decision, by default 100.

    Returns
    -------
    bool
        True if heights are geoid-referenced (shift required), False if they
        match the already-ellipsoidal signature (no shift).

    Raises
    ------
    ValueError
        If the sample is too small, or the median matches NEITHER signature
        (unexplained offset) — never guess in that state.
    """
    if valid_count < min_valid_pixels or np.isnan(median_diff):
        raise ValueError(
            f"Vertical datum check failed: only {valid_count} valid bare/sparse-vegetation "
            "(ESA WorldCover class 60) pixels overlap the DSM and COP30 - cannot reliably "
            "determine whether heights are geoid- or ellipsoid-referenced for this AOI. "
            "Override the source vertical interpretation explicitly (ept_vertical) or use "
            "per-survey WESM metadata instead of this empirical check."
        )
    if abs(median_diff) <= tolerance:
        return True
    if abs(median_diff - expected_undulation) <= tolerance:
        return False
    raise ValueError(
        f"Vertical datum check failed: median offset {median_diff:+.2f} m matches "
        f"neither the geoid-referenced signature (~0 m) nor the ellipsoidal "
        f"signature (~{expected_undulation:+.2f} m) within +/-{tolerance} m. "
        "Possible causes: steep terrain/snow/surface change biasing the sample, "
        "a reference-DEM problem, or an unexpected source datum. Inspect the "
        "products, or override with ept_vertical='geoid'/'ellipsoid' if the "
        "source vertical datum is known."
    )


def check_raster_validity(raster_fn: str, deep: bool = False) -> bool:
    """
    Check if a raster file is valid and can be opened using rioxarray and CRS check

    Parameters
    ----------
    raster_fn
        Path to the raster file.
    deep
        Also read the last pixel row to catch files truncated by an
        interrupted run (used by resume before trusting an existing tile).

    Returns
    -------
    bool
        True if the raster file is valid, False otherwise.
    """
    try:
        da = rioxarray.open_rasterio(raster_fn, masked=True).squeeze()
        out = da.rio.crs is not None
        da = None
    except Exception:
        return False
    if out and deep:
        try:
            with gdal.OpenEx(str(raster_fn)) as ds:
                ds.GetRasterBand(1).ReadAsArray(
                    0, ds.RasterYSize - 1, ds.RasterXSize, 1
                )
        except Exception:
            return False
    return out


def _open_decimated_dataarray(raster_fn: str, max_dim: int = 1024):
    """
    Open a raster decimated to at most max_dim x max_dim (average resampling)
    as a georeferenced DataArray.

    Bounds the memory of whole-mosaic comparisons: reproject_match against a
    full production mosaic grid needed ~174 GB at Las Vegas scale, while a
    ~1024^2 grid is a few MB and is ample for a robust median datum decision.

    Parameters
    ----------
    raster_fn
        Path to the raster file.
    max_dim
        Maximum output dimension in pixels, by default 1024.

    Returns
    -------
    xr.DataArray
        Decimated raster with CRS and coordinates set.
    """
    with rasterio.open(raster_fn) as src:
        scale = max(src.width, src.height) / max_dim
        if scale <= 1:
            return rioxarray.open_rasterio(raster_fn, masked=True).squeeze()
        w = max(1, int(round(src.width / scale)))
        h = max(1, int(round(src.height / scale)))
        data = src.read(
            1,
            out_shape=(h, w),
            resampling=rasterio.enums.Resampling.average,
            masked=True,
        )
        transform = src.transform * rasterio.Affine.scale(
            src.width / w, src.height / h
        )
        xs = transform.c + transform.a * (np.arange(w) + 0.5)
        ys = transform.f + transform.e * (np.arange(h) + 0.5)
        da = xr.DataArray(
            np.ma.filled(data.astype("float32"), np.nan),
            dims=("y", "x"),
            coords={"y": ys, "x": xs},
        )
        return da.rio.write_crs(src.crs)


def gdal_warp(
    src_fn: str,
    dst_fn: str,
    src_srs: str,
    dst_srs: str,
    res: float = 1.0,
    resampling_alogrithm: str = "bilinear",
    out_extent: list = None,
    dtype: str = 'Float32',
    target_aligned_pixels: bool = True,
    coordinate_operation: str = None,
    coord_epoch: float = None,
) -> None:
    """
    Warp a raster file to a new coordinate reference system and resolution using GDAL.

    Parameters
    ----------
    src_fn
        Path to the source raster file.
    dst_fn
        Path to the destination raster file.
    src_srs
        Source coordinate reference system in WKT format.
    dst_srs
        Destination coordinate reference system in WKT format
    res
        Resolution for the output raster, by default 1.0.
    resampling_alogrithm
        Resampling algorithm to use, by default 'cubic'.
    out_extent
        The extent of the output raster in the format [minx, miny, maxx, maxy], by default None.
    dtype
        Data type for the output raster, by default 'Float32'.
        Common options include 'Byte', 'UInt16', 'Int16', 'UInt32', 'Int32', 'Float32', 'Float64'.
    target_aligned_pixels
        Align output grid to multiples of res (gdalwarp -tap), by default True.
        Set False to reproduce an existing grid via out_extent whose origin is not a multiple of res.
    coordinate_operation
        Explicit PROJ pipeline string (gdalwarp -ct) overriding GDAL's
        coordinate-operation selection. GDAL will not route horizontal-only
        transformations through an intermediate frame (e.g. the
        NAD83(2011)<->ITRF Helmert via ITRF2014), silently picking a null
        operation instead; pass the pipeline selected by
        geodesy.preflight_vertical_transform to enforce the rigorous path.
    coord_epoch
        Target coordinate epoch (decimal year, gdalwarp -t_coord_epoch) for
        dynamic-frame targets: the time-dependent Helmert is evaluated at
        this epoch, so the output coordinates ARE at this epoch. Mutually
        exclusive with coordinate_operation (a forced -ct pipeline has no
        free epoch parameter — GDAL silently ignores -t_coord_epoch when
        -ct is set). Requires GDAL's CLI-argument-string code path; the
        WarpOptions kwargs, transformerOptions, and an osr SRS with
        SetCoordinateEpoch() all drop the epoch (verified GDAL 3.12).
    Returns
    -------
    None
    This function does not return anything, it writes the output raster to the specified file.
    """

   

    tolerance = 0
    resampling_mapping = {
        "nearest": gdalconst.GRA_NearestNeighbour,
        "bilinear": gdalconst.GRA_Bilinear,
        "cubic": gdalconst.GRA_Cubic,
        "cubic_spline": gdalconst.GRA_CubicSpline,
    }
    resampling_alg = resampling_mapping[resampling_alogrithm]

    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    if coord_epoch is not None:
        if coordinate_operation is not None:
            raise ValueError(
                "coord_epoch and coordinate_operation are mutually exclusive: "
                "GDAL ignores -t_coord_epoch when a -ct pipeline is forced. "
                "With coord_epoch set, GDAL selects the operation itself "
                "(verified to pick the rigorous time-dependent Helmert for "
                "3D NAD83(2011)-based sources)."
            )
        # -t_coord_epoch only survives the CLI-argument string form
        cli_resampling = {"nearest": "near", "cubic_spline": "cubicspline"}.get(
            resampling_alogrithm, resampling_alogrithm
        )
        opts = (
            f"-overwrite -r {cli_resampling} -tr {res} {res} -et {tolerance} "
            f"-ot {dtype} -multi -t_coord_epoch {coord_epoch} "
            f'-s_srs "{src_srs}" -t_srs "{dst_srs}" '
            "-co COMPRESS=LZW -co TILED=YES -co COPY_SRC_OVERVIEWS=YES "
            "-co BIGTIFF=IF_SAFER"
        )
        if target_aligned_pixels:
            opts += " -tap"
        if out_extent is not None:
            opts += (f" -te {out_extent[0]} {out_extent[1]}"
                     f" {out_extent[2]} {out_extent[3]}")
        print(f"gdal.Warp CLI options: {opts}")
        ds = gdal.Warp(dst_fn, src_fn, options=opts,
                       callback=gdal.TermProgress_nocb)
    else:
        ds = gdal.Warp(
            dst_fn,
            src_fn,
            resampleAlg=resampling_alg,
            srcSRS=src_srs,
            xRes=res,
            yRes=res,
            dstSRS=dst_srs,
            errorThreshold=tolerance,
            # disable when matching an existing raster grid whose origin is not
            # a multiple of res (e.g. recovering interrupted-run intermediates)
            targetAlignedPixels=target_aligned_pixels,
            coordinateOperation=coordinate_operation,
            # use directly output format as COG when gaussian overview resampling is implemented upstream in GDAL
            outputBounds=out_extent,
            outputType=DTYPE_TO_GDAL.get(dtype),
            creationOptions=["COMPRESS=LZW", "TILED=YES", "COPY_SRC_OVERVIEWS=YES","BIGTIFF=IF_SAFER"],
            callback=gdal.TermProgress_nocb,
            multithread=True,
        )
    gdal.SetConfigOption("GDAL_NUM_THREADS", None)
    ds.Close()



def gdal_add_overview(raster_fn: str, ensure_cog=True, resampling: str = "AVERAGE") -> None:
    """
    Add overviews to a raster file using GDAL.
    Converts the raster to a COG,
        as adding overviews to tiled and compressed rasters does not automatically ensure COG compliance

    Parameters
    ----------
    raster_fn
        Path to the raster file.
    ensure_cog
        Whether to ensure the output raster is a COG, by default True.
    resampling
        Overview resampling kernel passed to BuildOverviews, by default "AVERAGE".
        "GAUSS" introduces a sub-pixel horizontal offset in the overview levels
        relative to the full-resolution grid — do not use for georeferenced delivery.

    Returns
    -------
    None
    This function does not return anything, it writes the output raster to the specified file.
    """
    print(f"Adding {resampling} overviews to {raster_fn}")
    with gdal.OpenEx(raster_fn, 1, open_options=["IGNORE_COG_LAYOUT_BREAK=YES"]) as ds:
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        ds.BuildOverviews(
            resampling, [2, 4, 8, 16], callback=gdal.TermProgress_nocb
        )

    if ensure_cog:
        temp_fn =Path(raster_fn).parent / f"{Path(raster_fn).stem}-cop-temp.tif"
        gdal.Translate(
            str(temp_fn),
            raster_fn,
            format="COG",
            creationOptions=["OVERVIEWS=FORCE_USE_EXISTING","BIGTIFF=IF_SAFER"],
            callback=gdal.TermProgress_nocb,
        )
        rename_rasters(str(temp_fn), raster_fn)


def raster_footprint(
    raster_fn: str, out_fn: str, simplify_px: float = 32.0
) -> str:
    """
    Write the valid-data footprint of a raster as a polygon GeoPackage in
    the raster CRS. Ships with every per-project mosaic so downstream QA
    (cross-project dz maps, seam attribution) can draw each project's
    actual coverage without re-deriving it from the ~GB rasters.

    Parameters
    ----------
    raster_fn
        Path to the raster; its nodata value defines "valid".
    out_fn
        Output GeoPackage path (overwritten if present).
    simplify_px
        Douglas-Peucker tolerance in full-resolution pixels, by default 32
        (~32 m at 1 m posting — plenty for map overlays, and it keeps the
        vertex count of a hole-riddled mosaic edge sane).

    Returns
    -------
    str
        The written GeoPackage path.
    """
    with gdal.OpenEx(raster_fn) as ds:
        gt = ds.GetGeoTransform()
        # trace a decimated overview when available: footprint fidelity is
        # bounded by the simplify tolerance anyway, and full-resolution
        # polygonization of a production mosaic takes minutes vs seconds
        ovr = min(3, ds.GetRasterBand(1).GetOverviewCount() - 1)
        Path(out_fn).unlink(missing_ok=True)
        gdal.Footprint(
            str(out_fn),
            ds,
            format="GPKG",
            layerName="footprint",
            ovr=ovr if ovr >= 0 else None,
            simplify=abs(gt[1]) * simplify_px,
        )
    return out_fn


def create_lpc_pipeline(
    local_laz_dir: str,
    target_wkt: str,
    output_prefix: str,
    extent_polygon: str,
    input_crs: str = None,
    dsm_gridding_choice: str = "first_idw",
    proj_pipeline: str = None,
    raster_resolution: float = 1.0,
    filter_high_noise: bool = True,
    filter_low_noise: bool = True,
    hag_nn: float = None,
    buffer_value: float = 5.0,
    products: list[str] | None = None) -> list[dict]:
    """
    Create single-read PDAL tile jobs for processing local LiDAR point clouds (LPC) to generate DEM products

    Parameters
    ----------
    local_laz_dir
        Directory containing local LiDAR point cloud files in LAZ or LAS format.
    input_crs
        Override input file CRS.
    target_wkt
        Path to the target WKT file defining the output coordinate reference system.
    output_prefix
        Prefix for the output files, which will be used to create output file names.
    extent_polygon
        Path to a polygon file defining the area of interest (AOI) for processing.
    dsm_gridding_choice : str
        The gridding method to use for DSM generation. 'first_idw' uses the first and only returns which are gridded using IDW, 'n-pct' computes points matching the nth percentile in a pointview (e.g., 98-pct), which are gridded using the max binning operator.
    proj_pipeline : str, optional
        PROJ pipeline string for reprojection, by default None.
        If None, the reprojection will be handled by GDAL using the input and output CRS.
    raster_resolution : float, optional
        Resolution for the output raster files, by default 1.0.
    filter_high_noise : bool, optional
        Remove high noise points (classification==18) from the point cloud before DSM and surface intensity processing. Default is True.
    hag_nn : float, optional
        If specified, the height above ground (HAG) will be calculated using all nearest ground classied points, and all points greater than this value will be classified as high noise, by default None.
    buffer_value : float, optional
        Buffer value to apply to the AOI bounds when reading points for rasterization, by default 5.0.
    products
        Canonical product names to build (see parse_products); by default
        all products.

    Returns
    -------
    tile_jobs
        One tile job dict per intersecting LAZ/LAS file (see
        create_tile_pipelines): per-filter-chain product executions with
        in-pipeline reprojection, executable with execute_tile_job. Local
        files are cheap to re-read, so no cache step is emitted; the
        filtered/reprojected points are still written alongside the DSM
        when it is requested (legacy save_pointcloud behavior).
    """
    lpc_files = sorted(Path(local_laz_dir).glob("*.laz"))
    lpc_files += sorted(Path(local_laz_dir).glob("*.las"))
    print(f"Number of local laz files: {len(lpc_files)}")
    readers = []
    original_extents = []
    input_crs_list = []
    aoi_bounds = gpd.read_file(extent_polygon)
    if isinstance(target_wkt, Path):
        target_wkt = str(target_wkt)
    for idx, lpc in enumerate(lpc_files):
        reader, in_crs, out_extent = return_local_lpc_reader(
            str(lpc),
            input_crs=input_crs,
            output_crs=target_wkt,
            pointcloud_resolution=raster_resolution,
            aoi_bounds=aoi_bounds,
            buffer_value=buffer_value,
        )
        # print(reader)
        if reader is not None:
            readers.append(reader)
            original_extents.append(out_extent)
            input_crs_list.append(in_crs)
    output_path = Path(output_prefix).parent
    prefix = Path(output_prefix).name
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    print(f"Number of readers: {len(readers)}")

    #Determine number of digits for unique pipeline id with zero padding
    ndigits = len(str(len(readers)))

    with open(target_wkt, "r") as f:
        contents = f.read()
    out_crs = CRS.from_string(contents)

    if products is None:
        products = list(PRODUCT_ORDER)

    tile_jobs = []
    for i, reader in enumerate(readers):
        # reader is a pipeline dict: readers.las + optional filters.crop
        tile_jobs.append(
            create_tile_pipelines(
                reader["pipeline"],
                tile_id=str(i).zfill(ndigits),
                output_path=output_path,
                prefix=prefix,
                extent=original_extents[i],
                raster_resolution=raster_resolution,
                products=products,
                dsm_gridding_choice=dsm_gridding_choice,
                filter_low_noise=filter_low_noise,
                filter_high_noise=filter_high_noise,
                hag_nn=hag_nn,
                # local files are cheap to re-read: no cache step
                use_cache=False,
                # reprojection happens in-pipeline, shared by every chain
                extra_pipeline_kwargs={
                    "reproject": True,
                    "proj_pipeline": proj_pipeline,
                    "input_crs": input_crs_list[i],
                    "output_crs": out_crs,
                },
                save_pointcloud="dsm" in products,
            )
        )

    return tile_jobs

def _set_dsm_gridding_params(dsm_gridding_choice: str):
    if dsm_gridding_choice == "first_idw":
        dsm_group_filter = "first,only"
        dsm_gridding_method = "idw"
        percentile_filter = False
        percentile_threshold = None
    else:
        dsm_group_filter = None
        dsm_gridding_method = "max"
        percentile_threshold = int(dsm_gridding_choice.split("-pct")[0])/100.0
        percentile_filter = True

    return dsm_group_filter, dsm_gridding_method, percentile_filter, percentile_threshold


# Canonical gridded products, in construction/reporting order. User-facing
# aliases expand to canonical names ("dtm" keeps its historical meaning of
# both DTM variants).
PRODUCT_ORDER = ("dsm", "dtm_no_fill", "dtm_fill", "intensity")
PRODUCT_ALIASES = {
    "all": PRODUCT_ORDER,
    "dtm": ("dtm_no_fill", "dtm_fill"),
}


def parse_products(products: str) -> list[str]:
    """
    Expand a comma-separated product selection into canonical product names.

    Parameters
    ----------
    products
        Comma-separated product names, e.g. "all", "dsm", "dsm,intensity",
        "dtm_fill". Aliases: "all" = every product, "dtm" = both DTM
        variants (dtm_no_fill, dtm_fill).

    Returns
    -------
    list[str]
        Canonical product names in PRODUCT_ORDER, deduplicated.
    """
    requested: set[str] = set()
    for token in products.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in PRODUCT_ALIASES:
            requested.update(PRODUCT_ALIASES[token])
        elif token in PRODUCT_ORDER:
            requested.add(token)
        else:
            valid = ", ".join(list(PRODUCT_ORDER) + list(PRODUCT_ALIASES))
            raise ValueError(f"Unknown product '{token}'. Valid: {valid}")
    if not requested:
        raise ValueError(f"No products requested (got '{products}')")
    return [name for name in PRODUCT_ORDER if name in requested]


def _product_specs(
    dsm_gridding_choice: str,
    filter_low_noise: bool,
    filter_high_noise: bool,
    hag_nn: float | None,
) -> dict[str, dict]:
    """
    Registry of gridded products: canonical name -> the create_pdal_pipeline
    kwargs building its filter chain, the create_dem_stage kwargs building
    its writers.gdal stage, writer-only extras, and the legacy per-tile
    filename template (resume compatibility and the downstream mosaic/cleanup
    globs depend on these exact names).

    Products whose *built* filter chains are identical share one PDAL
    execution with chained writers (writers pass points through unchanged),
    so a writer-only product variant (e.g. a filled DTM with a different
    window) merges onto an existing execution at zero extra point reads.
    """
    (
        dsm_group_filter,
        dsm_gridding_method,
        percentile_filter,
        percentile_threshold,
    ) = _set_dsm_gridding_params(dsm_gridding_choice)
    noise = {
        "filter_low_noise": filter_low_noise,
        "filter_high_noise": filter_high_noise,
    }
    return {
        "dsm": {
            "pipeline_kwargs": {
                **noise,
                "group_filter": dsm_group_filter,
                "percentile_filter": percentile_filter,
                "percentile_threshold": percentile_threshold,
                "hag_nn": hag_nn,
                "reproject": False,
            },
            "dem_kwargs": {"gridmethod": dsm_gridding_method, "dimension": "Z"},
            "writer_extra": {},
            "filename": "{prefix}_dsm_tile_aoi_{tile}.tif",
        },
        "dtm_no_fill": {
            "pipeline_kwargs": {
                **noise,
                "return_only_ground": True,
                "group_filter": None,
                "reproject": False,
            },
            "dem_kwargs": {"gridmethod": "idw", "dimension": "Z"},
            "writer_extra": {},
            "filename": "{prefix}_dtm_tile_aoi_no_fill{tile}.tif",
        },
        "dtm_fill": {
            "pipeline_kwargs": {
                **noise,
                "return_only_ground": True,
                "group_filter": None,
                "reproject": False,
            },
            "dem_kwargs": {"gridmethod": "idw", "dimension": "Z"},
            # gaps interpolated with a window of 4 cells; writer-only
            # difference from dtm_no_fill, so both share one execution
            "writer_extra": {"window_size": 4},
            "filename": "{prefix}_dtm_tile_aoi_fill4_{tile}.tif",
        },
        "intensity": {
            "pipeline_kwargs": {
                **noise,
                "return_only_ground": False,
                "group_filter": "first,only",
                "hag_nn": hag_nn,
                "reproject": False,
            },
            "dem_kwargs": {
                "gridmethod": "idw",
                "dimension": "Intensity",
                "nodata_value": 0,
                "data_type": "UInt16",
            },
            "writer_extra": {},
            "filename": "{prefix}_intensity_tile_aoi_{tile}.tif",
        },
    }


def _pdal_srs_string(crs: "CRS | str | None") -> str | None:
    """
    Render a CRS as a compact PDAL-friendly SRS string: an EPSG authority
    code when available (unambiguous and short), else full WKT. Passes str
    through unchanged; None -> None.
    """
    if crs is None or isinstance(crs, str):
        return crs
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()


def create_tile_pipelines(
    reader_stages: list[dict],
    tile_id: str,
    output_path: Path,
    prefix: str,
    extent: list,
    raster_resolution: float,
    products: list[str],
    dsm_gridding_choice: str = "first_idw",
    filter_low_noise: bool = True,
    filter_high_noise: bool = True,
    hag_nn: float | None = None,
    use_cache: bool = True,
    extra_pipeline_kwargs: dict | None = None,
    save_pointcloud: bool = False,
    source_crs: "CRS | str | None" = None,
) -> dict:
    """
    Build the single-read tile job for one tile: read points once, emit
    every requested product from that read (architecture review F3).

    PDAL cannot branch one reader into multiple writer leaves (only the
    first leaf of a multi-leaf pipeline executes, and a shared reader
    feeding N consumers re-executes N times), so the job is a short ordered
    sequence of standalone linear pipelines:

    1. fetch: reader -> writers.las cache LAZ (the only network read;
       LAS 1.4 / point format 6 retains PointSourceId and GpsTime for
       future per-lift segmentation).
    2. one execution per distinct filter chain among the requested
       products, reading the local cache, with the products' writers.gdal
       stages chained sequentially (writers pass points through).

    With a single distinct chain (or a single product) the cache is pure
    overhead: the reader stages are inlined into that one execution and
    no fetch step is emitted.

    Parameters
    ----------
    reader_stages
        Verbatim PDAL stage dicts producing the tile's points (e.g. one
        readers.ept dict, or readers.las + filters.crop).
    tile_id
        Zero-padded tile index used in filenames.
    output_path
        Run directory; per-tile pipeline JSONs are written to
        output_path/pipelines/, per-tile product rasters (and any saved
        pointclouds) to output_path/tiles/<product>/, and the cache LAZ to
        output_path/tiles/cache/.
    prefix
        Filename prefix (AOI stem).
    extent
        Output raster extent [xmin, ymin, xmax, ymax] for create_dem_stage.
    raster_resolution
        Output grid cell size.
    products
        Canonical product names (see parse_products), any subset of
        PRODUCT_ORDER.
    dsm_gridding_choice
        "first_idw" or "n-pct" (see _set_dsm_gridding_params).
    filter_low_noise, filter_high_noise, hag_nn
        Passed through to create_pdal_pipeline per product.
    use_cache
        Emit the fetch/cache step when more than one execution is needed.
        Set False for local file input where re-reading is cheap.
    extra_pipeline_kwargs
        Merged into every product's create_pdal_pipeline kwargs (the local
        input path injects reproject/proj_pipeline/input_crs/output_crs
        here so reprojection happens in-pipeline, shared by every product
        chain).
    save_pointcloud
        Write the filtered/reprojected points to a LAZ alongside the DSM
        (chained writers.las before the DSM's writers.gdal, matching the
        legacy local-input behavior including its historical .laz.laz
        double-extension filename). Only applies when "dsm" is requested.
    source_crs
        Authoritative CRS of the input points (e.g. the EPT's declared SRS).
        When caching, it is stamped on the cache writers.las (a_srs) AND
        forced onto the cache readers.las (override_srs) so the product
        rasters carry the correct CRS regardless of whether the LAS-header
        SRS survives the write/read round-trip in a given PDAL/GDAL build
        (a CRS-less cache read otherwise yields writers.gdal output with no
        CRS, which fails validity — observed on another environment).

    Returns
    -------
    dict
        Tile job: {"tile_id", "fetch": {"pipeline_json", "cache_file"} | None,
        "executions": [{"pipeline_json", "outputs": {product: tile_tif}}]}.
        All values are plain strings (cheap dask task payload).
    """
    specs = _product_specs(
        dsm_gridding_choice, filter_low_noise, filter_high_noise, hag_nn
    )
    ordered = [name for name in PRODUCT_ORDER if name in products]

    # Per-tile intermediates live in subdirectories so the run directory
    # holds only final products + provenance (tiles/ and pipelines/ are
    # removed by the post-run cleanup; ~1000-tile runs otherwise leave
    # thousands of files next to the mosaics). Tile rasters are further
    # split per product (tiles/dsm/, tiles/intensity/, ...; cache LAZ in
    # tiles/cache/) so recovery/mosaic rebuilds are one unambiguous glob.
    tiles_dir = output_path / "tiles"
    pipelines_dir = output_path / "pipelines"
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    # Group products by identical built filter chain: matching chains share
    # one execution with chained writers, distinct chains each re-read the
    # (local) cache. dict preserves insertion order -> deterministic naming.
    groups: dict[str, dict] = {}
    for name in ordered:
        chain = create_pdal_pipeline(
            **{**specs[name]["pipeline_kwargs"], **(extra_pipeline_kwargs or {})}
        )
        signature = json.dumps(chain, sort_keys=True)
        product_dir = tiles_dir / name
        product_dir.mkdir(parents=True, exist_ok=True)
        outfile = product_dir / specs[name]["filename"].format(
            prefix=prefix, tile=tile_id
        )
        writer = create_dem_stage(
            dem_filename=str(outfile),
            extent=extent,
            pointcloud_resolution=raster_resolution,
            **specs[name]["dem_kwargs"],
        )[0]
        writer.update(specs[name]["writer_extra"])
        group = groups.setdefault(signature, {"chain": chain, "members": []})
        group["members"].append({"name": name, "writer": writer, "outfile": outfile})

    crs_str = _pdal_srs_string(source_crs)

    fetch = None
    source_stages = reader_stages
    if use_cache and len(groups) > 1:
        cache_dir = tiles_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{prefix}_cache_tile_aoi_{tile_id}.laz"
        cache_writer = {
            "type": "writers.las",
            "filename": str(cache_file),
            "compression": True,
            "minor_version": 4,
            "dataformat_id": 6,
            # 3DEP EPT native scale is 0.01 m, so this round-trip is
            # lossless; offsets are data-driven and land on the same grid
            "scale_x": 0.01,
            "scale_y": 0.01,
            "scale_z": 0.01,
            "offset_x": "auto",
            "offset_y": "auto",
            "offset_z": "auto",
        }
        cache_reader = {"type": "readers.las", "filename": str(cache_file)}
        if crs_str:
            # pin the SRS explicitly on both ends of the cache round-trip:
            # a_srs guarantees the file is tagged, override_srs guarantees
            # the product rasters get the CRS even if a PDAL/GDAL build
            # fails to reconstruct it from the LAS header on read
            cache_writer["a_srs"] = crs_str
            cache_reader["override_srs"] = crs_str
        fetch_pipeline = {"pipeline": list(reader_stages) + [cache_writer]}
        fetch_fn = pipelines_dir / f"pipeline_fetch_{tile_id}.json"
        with open(fetch_fn, "w") as f:
            f.write(json.dumps(fetch_pipeline))
        fetch = {"pipeline_json": str(fetch_fn), "cache_file": str(cache_file)}
        source_stages = [cache_reader]

    executions = []
    for group in groups.values():
        members = group["members"]
        writers = [member["writer"] for member in members]
        member_names = [member["name"] for member in members]
        if save_pointcloud and "dsm" in member_names:
            dsm_outfile = members[member_names.index("dsm")]["outfile"]
            # legacy filename preserved verbatim, including the historical
            # .laz.laz double extension (downstream consumers glob for it)
            pointcloud_fn = os.path.splitext(str(dsm_outfile))[0] + ".laz"
            writers = [
                {
                    "type": "writers.las",
                    "compression": "true",
                    "minor_version": "2",
                    "dataformat_id": "0",
                    "filename": f"{pointcloud_fn}.laz",
                }
            ] + writers
        stages = list(source_stages) + group["chain"] + writers
        joined = "_".join(member["name"] for member in members)
        pipeline_fn = pipelines_dir / f"pipeline_{joined}_{tile_id}.json"
        with open(pipeline_fn, "w") as f:
            f.write(json.dumps({"pipeline": stages}))
        executions.append(
            {
                "pipeline_json": str(pipeline_fn),
                "outputs": {
                    member["name"]: str(member["outfile"]) for member in members
                },
            }
        )

    return {"tile_id": tile_id, "fetch": fetch, "executions": executions}


def create_ept_3dep_pipeline(
    extent_polygon: str,
    target_wkt: str,
    output_prefix: str,
    raster_resolution: float = 1.0,
    tile_size_km: float = 1.0,
    buffer_value: float = 5.0,
    dsm_gridding_choice: str = "first_idw",
    filter_high_noise: bool = True,
    filter_low_noise: bool = True,
    hag_nn: float = None,
    process_specific_3dep_survey: str = None,
    process_all_intersecting_surveys: bool = False,
    products: list[str] | None = None,
    ept_index_gdf: gpd.GeoDataFrame = None) -> list[dict]:

    """
    Create single-read PDAL tile jobs for processing 3DEP EPT point clouds to generate DEM products.

    Parameters
    ----------
    extent_polygon : str
        Path to the vector dataset containing a polygon defining the processing extent.
    output_prefix : str
        Path for output files, containing directory path and filename prefix (e.g., /tmp/CO_3DEP_ALS).
    target_wkt : str or None
        Path to a text file containing WKT2 definition for the output coordinate reference system (CRS). If unspecified, a local UTM CRS will be used.
    raster_resolution : float, optional
        Output grid cell size, default 1.0 m.
    tile_size_km : float, optional
        Processing tile dimension (square), default 1.0 km.
    buffer_value : float, optional
        Buffer distance to expand bounds when reading points ensuring sufficient tile collar for window operations, default 5.0 m.
    process_specific_3dep_survey: str
        Only process the specified 3DEP project name. This should be a string that matches the workunit name in the 3DEP metadata.
    process_all_intersecting_surveys: bool
        If true, process all available 3DEP EPT point clouds which intersect with the input polygon. If false, and process_specific_3dep_survey is not specified, first 3DEP project encountered will be processed.
    filter_high_noise
        Remove high noise points (classification==18) from the point cloud before DSM and surface intensity processing. Default is True.
    filter_low_noise
        Remove low points (classification==7) from the point cloud before DSM, DTM and surface intensity processing. Default is True.    
    hag_nn
        If specified, the height above ground (HAG) will be calculated using all nearest ground classified points, and all points greater than this value will be classified as high noise, by default None.
    products
        Canonical product names to build (see parse_products); by default
        all products.
    ept_index_gdf
        Preloaded EPT resource boundary index passed through to
        return_readers (avoids re-fetching the hobu index when the caller
        already loaded it for name resolution).

    Returns
    -------
    tile_jobs
        One tile job dict per (tile, survey) reader (see
        create_tile_pipelines): a single EPT fetch into a local cache plus
        per-filter-chain product executions, executable with
        execute_tile_job.
    """
    
    #Load the user-specified polygon dataset
    #Should check that this is EPSG:4326 (default for geojson)
    gdf = gpd.read_file(extent_polygon)

    # fetch the readers for the pointclouds
    readers, POINTCLOUD_CRS, extents, original_extents = return_readers(
        gdf,
        pointcloud_resolution=raster_resolution,
        tile_size_km=tile_size_km,
        buffer_value=buffer_value,
        return_specific_3dep_survey=process_specific_3dep_survey,
        return_all_intersecting_surveys=process_all_intersecting_surveys,
        ept_index_gdf=ept_index_gdf,
    )

    output_path = Path(output_prefix).parent
    prefix = Path(output_prefix).name
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    print(f"Number of readers: {len(readers)}")

    #Determine number of digits for unique pipeline id with zero padding
    ndigits = len(str(len(readers)))

    if products is None:
        products = list(PRODUCT_ORDER)

    tile_jobs = []
    for i, reader in enumerate(readers):
        tile_jobs.append(
            create_tile_pipelines(
                [reader],
                tile_id=str(i).zfill(ndigits),
                output_path=output_path,
                prefix=prefix,
                extent=original_extents[i],
                raster_resolution=raster_resolution,
                products=products,
                dsm_gridding_choice=dsm_gridding_choice,
                filter_low_noise=filter_low_noise,
                filter_high_noise=filter_high_noise,
                hag_nn=hag_nn,
                use_cache=True,
                source_crs=POINTCLOUD_CRS[i],
            )
        )

    return tile_jobs


def execute_pdal_pipeline(
    pdal_pipeline_path: str, skip_existing: bool = False, attempts: int = 3
) -> str:
    """
    Execute a PDAL pipeline
    #modified by Scott Henderson

    Parameters
    ----------
    pdal_pipeline_path : str
        The path to the PDAL pipeline json
    skip_existing
        If the pipeline's output already exists and passes a deep validity
        check, return it without re-executing (tile-level resume for
        interrupted runs). Truncated files from a killed run fail the check
        and are recomputed.
    attempts
        Execute up to this many times with backoff before giving up —
        EPT reads over the network fail transiently, and a multi-day run
        must not lose a tile to a single hiccup. By default 3.
    Returns
    -------
    output_fn
        The filename of the output raster is successfully saved, otherwise None is returned
    """
    # Policy: never silently drop tiles. Failures return None (the caller
    # accounts for and reports them) with a loud stderr message here.
    try:
        with open(pdal_pipeline_path) as f:
            pipelineDict = json.load(f)
        # maybe more robust to check for {'type': 'writers.gdal'}...
        outfile = pipelineDict["pipeline"][-1]["filename"]

        if (
            skip_existing
            and Path(outfile).exists()
            and check_raster_validity(outfile, deep=True)
        ):
            print(f"Resume: skipping existing valid tile {outfile}")
            return outfile

        for attempt in range(1, attempts + 1):
            try:
                pipeline = pdal.Pipeline(json.dumps(pipelineDict))
                suffix = f" (attempt {attempt}/{attempts})" if attempt > 1 else ""
                print(f"Executing {pdal_pipeline_path}{suffix}")
                pipeline.execute()
                pipeline = None
                break
            except Exception as e:
                if attempt == attempts:
                    raise
                wait = 5 * attempt
                print(
                    f"WARNING: pipeline {pdal_pipeline_path} failed "
                    f"(attempt {attempt}/{attempts}): {e}; retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)

        if check_raster_validity(outfile):
            return outfile
        print(
            f"ERROR: pipeline {pdal_pipeline_path} produced an invalid raster "
            f"(missing CRS or unreadable): {outfile}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"ERROR: PDAL pipeline {pdal_pipeline_path} failed: {e}",
            file=sys.stderr,
        )
        return None


def _execute_pipeline_with_retries(pipeline_json_path: str, attempts: int) -> int:
    """
    Execute one PDAL pipeline JSON with retry + backoff (EPT reads over the
    network fail transiently; a multi-day run must not lose a tile to a
    single hiccup). Raises the last error after `attempts` tries.

    Returns
    -------
    int
        The number of points the pipeline processed (used to detect
        legitimately empty tiles where a survey has no coverage).
    """
    with open(pipeline_json_path) as f:
        pipeline_dict = json.load(f)
    for attempt in range(1, attempts + 1):
        try:
            pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
            suffix = f" (attempt {attempt}/{attempts})" if attempt > 1 else ""
            print(f"Executing {pipeline_json_path}{suffix}")
            count = pipeline.execute()
            pipeline = None
            return count
        except Exception as e:
            if attempt == attempts:
                raise
            wait = 5 * attempt
            print(
                f"WARNING: pipeline {pipeline_json_path} failed "
                f"(attempt {attempt}/{attempts}): {e}; retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)


def execute_tile_job(
    job: dict, skip_existing: bool = False, attempts: int = 3
) -> dict:
    """
    Execute one single-read tile job (see create_tile_pipelines): fetch the
    tile's points into a local cache once, then run each pending product
    execution against the cache.

    Parameters
    ----------
    job
        Tile job dict from create_tile_pipelines.
    skip_existing
        Tile-level resume: an execution is skipped iff ALL of its outputs
        exist and pass a deep validity check (truncated tiles from a killed
        run fail and are recomputed). A partially-valid execution is re-run
        whole, overwriting its outputs. The cache is never trusted from
        disk: it is (re)fetched whenever any execution must run.
    attempts
        Retry budget per pipeline execution (network hiccups), by default 3.

    Returns
    -------
    dict
        ``{"empty": bool, "outputs": {product: path | None}}``. ``empty`` is
        True when the fetch returned zero points (the survey does not cover
        this tile): a legitimate no-data outcome, not a failure, so no
        rasters are written and no ERROR is logged. Otherwise ``outputs``
        maps each product to its tile raster path, or None where that
        product failed. Never raises (policy: failures/empties are accounted
        for and reported by the caller; dask retries re-run the job
        idempotently).
    """
    outputs: dict[str, str | None] = {}
    pending = []
    for execution in job["executions"]:
        exec_outputs = execution["outputs"]
        if skip_existing and all(
            Path(fn).exists() and check_raster_validity(fn, deep=True)
            for fn in exec_outputs.values()
        ):
            for name, fn in exec_outputs.items():
                print(f"Resume: skipping existing valid tile {fn}")
                outputs[name] = fn
        else:
            pending.append(execution)
            for name in exec_outputs:
                outputs[name] = None
    if not pending:
        return {"empty": False, "outputs": outputs}

    fetch = job.get("fetch")
    cache_file = fetch["cache_file"] if fetch else None
    try:
        if fetch:
            try:
                n_points = _execute_pipeline_with_retries(
                    fetch["pipeline_json"], attempts
                )
            except Exception as e:
                print(
                    f"ERROR: point fetch {fetch['pipeline_json']} failed: {e}",
                    file=sys.stderr,
                )
                return {"empty": False, "outputs": outputs}
            if n_points == 0:
                # legitimately empty tile: the survey has no points here, so
                # there is nothing to grid. Skip the product executions (which
                # would otherwise each write a full-size CRS-less nodata raster
                # and log a spurious "invalid raster" ERROR) and report the
                # tile as empty, not failed.
                print(f"Empty tile {job.get('tile_id')}: 0 points, skipped")
                return {"empty": True, "outputs": {name: None for name in outputs}}
        for execution in pending:
            try:
                _execute_pipeline_with_retries(execution["pipeline_json"], attempts)
            except Exception as e:
                print(
                    f"ERROR: PDAL pipeline {execution['pipeline_json']} failed: {e}",
                    file=sys.stderr,
                )
                continue
            for name, fn in execution["outputs"].items():
                if check_raster_validity(fn):
                    outputs[name] = fn
                else:
                    print(
                        f"ERROR: pipeline {execution['pipeline_json']} produced an "
                        f"invalid raster (missing CRS or unreadable): {fn}",
                        file=sys.stderr,
                    )
    finally:
        if cache_file:
            Path(cache_file).unlink(missing_ok=True)
    return {"empty": False, "outputs": outputs}


def rename_rasters(raster_fn, out_fn) -> None:
    """
    Rename the raster file to the final output name and the associated XML file if it exists.

    Parameters
    ----------
    raster_fn
        Path to the raster file to be renamed.
    out_fn
        Path to the output raster file name.

    Returns
    -------
    None
    This function does not return anything, it renames the raster file and its associated XML file
    """
    xml_fn = raster_fn + ".aux.xml"
    Path(raster_fn).rename(out_fn)
    if Path(xml_fn).exists():
        out_fn_xml = out_fn + ".aux.xml"
        Path(xml_fn).rename(out_fn_xml)
