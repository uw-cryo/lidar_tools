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
from pathlib import Path

# import planetary_computer
from osgeo import gdal, gdalconst
import pdal
import odc.stac
import os
import copy
import warnings
import glob

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


def return_readers(
    input_aoi: gpd.GeoDataFrame,
    pointcloud_resolution: float = 1.0,
    tile_size_km: float = 1.0,
    buffer_value: int = 5,
    return_specific_3dep_survey: str = None,
    return_all_intersecting_surveys: bool = False,
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
        A specific 3DEP survey to return, by default first intersecting survey is returned
    return_all_intersecting_surveys
        If True, return all intersecting surveys, by default False.

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
    ept_index_gdf = gpd.read_file(
        "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson",
        mask=input_aoi
    ).to_crs(CRS.from_epsg(3857))
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

                        # SRS associated with the 3DEP dataset
                        response = requests.get(url)
                        data = response.json()
                        srs_wkt = data["srs"]["wkt"]

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
    pipeline.execute()

    pdal_bounds = pipeline.quickinfo["readers.las"]["bounds"]
    # print(pdal_bounds)
    minx, miny, maxx, maxy = (
        pdal_bounds["minx"],
        pdal_bounds["miny"],
        pdal_bounds["maxx"],
        pdal_bounds["maxy"],
    )
    # print(f"Bounds of the point cloud: {minx}, {miny}, {maxx}, {maxy}")
    if output_crs is not None:
        if CRS.from_wkt(pipeline.srswkt2) != output_crs:
            output_bounds = transform_bounds(
                CRS.from_wkt(pipeline.srswkt2), output_crs, minx, miny, maxx, maxy
            )

    else:
        output_bounds = [minx, miny, maxx, maxy]
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
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            callback=gdal.TermProgress_nocb,    
        )

    else:
        print(out_extent)
        gdal.Translate(
            outfn, vrt_fn, projWin=out_extent, callback=gdal.TermProgress_nocb,
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

    lidar_da = rioxarray.open_rasterio(raster_fn, masked=True).squeeze()
    worldcover_da = fetch_worldcover(raster_fn, lidar_da)
    cop30_da = fetch_cop30(raster_fn, lidar_da)
    lidar_da_masked, worldcover_da_masked, cop30_da_masked = common_mask(
        [lidar_da, worldcover_da, cop30_da], apply=True
    )
    dem_diff = lidar_da_masked - cop30_da_masked
    ## Mask out bare and sparse vegetation class
    bare_sparse_mask = worldcover_da_masked == 60
    dem_diff_bare = dem_diff.where(bare_sparse_mask, np.nan)
    median_diff = np.nanmedian(dem_diff_bare.values)
    print(
        f"Observed difference between COP30 EGM2008 and 3DEP LiDAR DSM over bareground and sparse vegetation surfaces is {median_diff:.2f} m"
    )
    if np.abs(median_diff) <= bare_diff_tolerance:
        # this means that both COP30 and 3DEP LiDAR DSM are with respect to geoid
        print(
            "Looks like the 3DEP height estimates are with respect to geoid, will apply vertical datum shift to return heights with respect to ellipsoid"
        )
        out = True
    else:
        # this means that 3DEP LiDAR DSM is with respect to ellipsoid
        print(
            "Looks like the 3DEP height estimates are already with respect to ellipsoid, geoid to ellipsoid transformation will not be attempted"
        )
        out = False
    return out


def check_raster_validity(raster_fn: str) -> bool:
    """
    Check if a raster file is valid and can be opened using rioxarray and CRS check

    Parameters
    ----------
    raster_fn
        Path to the raster file.

    Returns
    -------
    bool
        True if the raster file is valid, False otherwise.
    """
    da = rioxarray.open_rasterio(raster_fn, masked=True).squeeze()
    if da.rio.crs is None:
        # print(f"Raster {raster_fn} does not have a valid CRS.")
        out = False
    else:
        # print(f"Raster {raster_fn} has a valid CRS.")
        out = True
    da = None
    return out


def gdal_warp(
    src_fn: str,
    dst_fn: str,
    src_srs: str,
    dst_srs: str,
    res: float = 1.0,
    resampling_alogrithm: str = "bilinear",
    out_extent: list = None,
    dtype: str = 'Float32',
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
    ds = gdal.Warp(
        dst_fn,
        src_fn,
        resampleAlg=resampling_alg,
        srcSRS=src_srs,
        xRes=res,
        yRes=res,
        dstSRS=dst_srs,
        errorThreshold=tolerance,
        targetAlignedPixels=True,
        # use directly output format as COG when gaussian overview resampling is implemented upstream in GDAL
        outputBounds=out_extent,
        outputType=DTYPE_TO_GDAL.get(dtype),
        creationOptions=["COMPRESS=LZW", "TILED=YES", "COPY_SRC_OVERVIEWS=YES","BIGTIFF=IF_SAFER"],
        callback=gdal.TermProgress_nocb,
        multithread=True,
    )
    gdal.SetConfigOption("GDAL_NUM_THREADS", None)
    ds.Close()



def gdal_add_overview(raster_fn: str,ensure_cog=True) -> None:
    """
    Add Gaussian overviews to a raster file using GDAL.
    Converts the raster to a COG,
        as adding Gaussian overviews added to tiled and compressed rasters does not automatically ensure COG compliance

    Parameters
    ----------
    raster_fn
        Path to the raster file.
    ensure_cog
        Whether to ensure the output raster is a COG, by default True.

    Returns
    -------
    None
    This function does not return anything, it writes the output raster to the specified file.
    """
    print(f"Adding Gaussian overviews to {raster_fn}")
    with gdal.OpenEx(raster_fn, 1, open_options=["IGNORE_COG_LAYOUT_BREAK=YES"]) as ds:
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        ds.BuildOverviews(
            "GAUSS", [2, 4, 8, 16], callback=gdal.TermProgress_nocb
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
    buffer_value: float = 5.0) -> tuple[list, list, list, list]:
    """
    Create PDAL pipelines for processing local LiDAR point clouds (LPC) to generate DEM products

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

    Returns
    -------
    dsm_pipeline_list
        List of paths to PDAL pipeline configuration files for generating DSMs.
    dtm_pipeline_no_fill_list
        List of paths to PDAL pipeline configuration files for generating DTM without interpolation.
    dtm_pipeline_fill_list
        List of paths to PDAL pipeline configuration files for generating DTM with interpolation.
    intensity_pipeline_list
        List of paths to PDAL pipeline configuration files for generating intensity rasters.
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
            pointcloud_resolution=1.0,
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

    dsm_pipeline_list = []
    dtm_pipeline_no_fill_list = []
    dtm_pipeline_fill_list = []
    intensity_pipeline_list = []

    dsm_group_filter, dsm_gridding_method, percentile_filter, percentile_threshold = _set_dsm_gridding_params(dsm_gridding_choice)

    for i, reader in enumerate(readers):
        #print(f"Processing reader #{i}")
        dsm_file = output_path / f"{prefix}_dsm_tile_aoi_{str(i).zfill(ndigits)}.tif"
        dtm_file_no_z_fill = output_path / f"{prefix}_dtm_tile_aoi_no_fill{str(i).zfill(ndigits)}.tif"
        dtm_file_z_fill = output_path / f"{prefix}_dtm_tile_aoi_fill4_{str(i).zfill(ndigits)}.tif"
        intensity_file = (
            output_path / f"{prefix}_intensity_tile_aoi_{str(i).zfill(ndigits)}.tif"
        )

        pipeline_dsm = copy.deepcopy(reader)
        pipeline_dtm_no_z_fill = copy.deepcopy(reader)
        pipeline_dtm_z_fill = copy.deepcopy(reader)
        pipeline_intensity = copy.deepcopy(reader)
        ## DSM creation block
        
        pipeline_dsm = reader
        pdal_pipeline_dsm = create_pdal_pipeline(
            group_filter=dsm_group_filter,
            percentile_filter=percentile_filter,
            percentile_threshold=percentile_threshold,
            reproject=True, # reproject to the output CRS
            proj_pipeline=proj_pipeline,
            input_crs=input_crs_list[i],
            output_crs=out_crs,
            save_pointcloud=True,
            pointcloud_file=os.path.splitext(dsm_file)[0]+".laz",
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            hag_nn=hag_nn)
        dsm_stage = create_dem_stage(
            dem_filename=str(dsm_file),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod=dsm_gridding_method,
            dimension="Z",
        )
        pipeline_dsm["pipeline"] += pdal_pipeline_dsm
        pipeline_dsm["pipeline"] += dsm_stage
        # Save a copy of each pipeline
        dsm_pipeline_config_fn = output_path / f"pipeline_dsm_{str(i).zfill(ndigits)}.json"
        with open(dsm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dsm))
        dsm_pipeline_list.append(dsm_pipeline_config_fn)
        # remove the pipeline from memory
        pipeline_dsm = None
        pdal_pipeline_dsm = None

        ## DTM creation block

        pdal_pipeline_dtm_no_z_fill = create_pdal_pipeline(
                return_only_ground=True,
                group_filter=None,
                reproject=True, # reproject to the output CRS
                proj_pipeline=proj_pipeline,
                input_crs=input_crs_list[i],
                filter_high_noise=filter_high_noise,
                filter_low_noise=filter_low_noise,
                output_crs=out_crs)

        pdal_pipeline_dtm_z_fill = pdal_pipeline_dtm_no_z_fill.copy() #for later

        dtm_stage = create_dem_stage(
            dem_filename=str(dtm_file_no_z_fill),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Z",
        )
        pipeline_dtm_no_z_fill["pipeline"] += pdal_pipeline_dtm_no_z_fill
        pipeline_dtm_no_z_fill["pipeline"] += dtm_stage

        #Save a copy of each pipeline
        dtm_pipeline_config_fn = output_path / f"pipeline_dtm_no_fill_{str(i).zfill(ndigits)}.json"
        with open(dtm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dtm_no_z_fill))

        dtm_pipeline_no_fill_list.append(dtm_pipeline_config_fn)
        pipeline_dtm_no_z_fill = None
        pdal_pipeline_dtm_no_z_fill = None

        dtm_stage = create_dem_stage(
            dem_filename=str(dtm_file_z_fill),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Z",
        )
        # add the z-fill stage to the pipeline
        dtm_stage[0]["window_size"] = 4

        pipeline_dtm_z_fill["pipeline"] += pdal_pipeline_dtm_z_fill
        pipeline_dtm_z_fill["pipeline"] += dtm_stage

        #Save a copy of each pipeline
        dtm_pipeline_config_fn = output_path / f"pipeline_dtm_fill_{str(i).zfill(ndigits)}.json"
        with open(dtm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dtm_z_fill))

        dtm_pipeline_fill_list.append(dtm_pipeline_config_fn)
        pipeline_dtm_z_fill = None
        pdal_pipeline_dtm_z_fill = None

        ## Intensity creation block

        pdal_pipeline_surface_intensity = create_pdal_pipeline(
                group_filter="first,only",
                reproject=True, # reproject to the output CRS
                input_crs=input_crs_list[i],
                output_crs=out_crs,
                proj_pipeline=proj_pipeline,
                filter_high_noise=filter_high_noise,
                filter_low_noise=filter_low_noise,
                hag_nn=hag_nn)

        intensity_stage = create_dem_stage(
            dem_filename=str(intensity_file),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Intensity",
            data_type="UInt16",
            nodata_value=0,
        )
        pipeline_intensity["pipeline"] += pdal_pipeline_surface_intensity
        pipeline_intensity["pipeline"] += intensity_stage

        # Save a copy of each pipeline
        intensity_pipeline_config_fn = (
            output_path / f"pipeline_intensity_{str(i).zfill(ndigits)}.json"
        )
        with open(intensity_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_intensity))

        intensity_pipeline_list.append(intensity_pipeline_config_fn)
        pipeline_intensity = None
        pdal_pipeline_surface_intensity = None

    # return the pipelines and filenames
    return (
        dsm_pipeline_list,  # list of PDAL pipelines for DSM creation
        dtm_pipeline_no_fill_list,  # list of PDAL pipelines for no-fill DTM creation
        dtm_pipeline_fill_list,  # list of PDAL pipelines for filled DTM creation
        intensity_pipeline_list,  # list of PDAL pipelines for Intensity creation
    )

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
    process_all_intersecting_surveys: bool = False) -> tuple[list, list, list, list]:

    """
    Create PDAL pipelines for processing 3DEP EPT point clouds to generate DEM products.

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

    Returns
    -------
    dsm_pipeline_list
        List of paths to PDAL pipeline configuration files for generating DSMs.
    dtm_pipeline_no_fill_list
        List of paths to PDAL pipeline configuration files for generating DTM without interpolation.
    dtm_pipeline_fill_list
        List of paths to PDAL pipeline configuration files for generating DTM with interpolation.
    intensity_pipeline_list
        List of paths to PDAL pipeline configuration files for generating intensity rasters.
    """ 
    
    #Load the user-specified polygon dataset
    #Should check that this is EPSG:4326 (default for geojson)
    gdf = gpd.read_file(extent_polygon)

    # specify the output CRS of DEMs
    with open(target_wkt, "r") as f:
        OUTPUT_CRS = " ".join(f.read().replace("\n", "").split())
    # fetch the readers for the pointclouds
    readers, POINTCLOUD_CRS, extents, original_extents = return_readers(
        gdf,
        pointcloud_resolution=raster_resolution,
        tile_size_km=tile_size_km,
        buffer_value=buffer_value,
        return_specific_3dep_survey=process_specific_3dep_survey,
        return_all_intersecting_surveys=process_all_intersecting_surveys,
    )

    output_path = Path(output_prefix).parent
    prefix = Path(output_prefix).name
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    print(f"Number of readers: {len(readers)}")

    #Determine number of digits for unique pipeline id with zero padding
    ndigits = len(str(len(readers)))

    dsm_pipeline_list = []
    dtm_pipeline_no_fill_list = []
    dtm_pipeline_fill_list = []
    intensity_pipeline_list = []
    
    dsm_group_filter, dsm_gridding_method, percentile_filter, percentile_threshold = _set_dsm_gridding_params(dsm_gridding_choice)

    for i, reader in enumerate(readers):
        #print(f"Processing reader #{i}")
        dsm_file = output_path / f"{prefix}_dsm_tile_aoi_{str(i).zfill(ndigits)}.tif"
        dtm_file_no_z_fill = output_path / f"{prefix}_dtm_tile_aoi_no_fill{str(i).zfill(ndigits)}.tif"
        dtm_file_z_fill = output_path / f"{prefix}_dtm_tile_aoi_fill4_{str(i).zfill(ndigits)}.tif"
        intensity_file = (
            output_path / f"{prefix}_intensity_tile_aoi_{str(i).zfill(ndigits)}.tif"
        )
        ## DSM creation block
        pipeline_dsm = {"pipeline": [reader]}
        pdal_pipeline_dsm = create_pdal_pipeline(
                group_filter=dsm_group_filter,
                percentile_filter=percentile_filter,
                percentile_threshold=percentile_threshold,
                reproject=False,
                input_crs=POINTCLOUD_CRS[i],
                filter_high_noise=filter_high_noise,
                filter_low_noise=filter_low_noise,
                hag_nn=hag_nn)

        dsm_stage = create_dem_stage(
            dem_filename=str(dsm_file),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod=dsm_gridding_method,
            dimension="Z",
        )
        pipeline_dsm["pipeline"] += pdal_pipeline_dsm
        pipeline_dsm["pipeline"] += dsm_stage

        # Save a copy of each pipeline
        dsm_pipeline_config_fn = output_path / f"pipeline_dsm_{str(i).zfill(ndigits)}.json"
        with open(dsm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dsm))
        dsm_pipeline_list.append(dsm_pipeline_config_fn)
        # remove the pipeline from memory
        pipeline_dsm = None
        pdal_pipeline_dsm = None

        ## DTM creation block
        ## DTM creation block without z-fill
        pipeline_dtm_no_z_fill = {"pipeline": [reader]}
        pdal_pipeline_dtm_no_z_fill = create_pdal_pipeline(
            return_only_ground=True,
            group_filter=None,
            reproject=False,
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            input_crs=POINTCLOUD_CRS[i],
        )
        pdal_pipeline_dtm_z_fill = pdal_pipeline_dtm_no_z_fill.copy()  # for later

        dtm_stage = create_dem_stage(
            dem_filename=str(dtm_file_no_z_fill),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Z",
        )
        pipeline_dtm_no_z_fill["pipeline"] += pdal_pipeline_dtm_no_z_fill
        pipeline_dtm_no_z_fill["pipeline"] += dtm_stage
        
        #Save a copy of each pipeline
        dtm_pipeline_config_fn = output_path / f"pipeline_dtm_no_fill_{str(i).zfill(ndigits)}.json"
        with open(dtm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dtm_no_z_fill))
        dtm_pipeline_no_fill_list.append(dtm_pipeline_config_fn)
        pipeline_dtm_no_z_fill = None
        pdal_pipeline_dtm_no_z_fill = None

        # add this to make a DTM which has gaps filled by an intepolation window size of 4
        pipeline_dtm_z_fill = {"pipeline": [reader]}
        dtm_stage = create_dem_stage(
            dem_filename=str(dtm_file_z_fill),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Z",
        )
        dtm_stage[0]["window_size"] = 4
        pipeline_dtm_z_fill["pipeline"] += pdal_pipeline_dtm_z_fill
        pipeline_dtm_z_fill["pipeline"] += dtm_stage

        #Save a copy of each pipeline
        dtm_pipeline_config_fn = output_path / f"pipeline_dtm_fill_{str(i).zfill(ndigits)}.json"
        with open(dtm_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_dtm_z_fill))
        dtm_pipeline_fill_list.append(dtm_pipeline_config_fn)
        pipeline_dtm_z_fill = None
        pdal_pipeline_dtm_z_fill = None

        ## Intensity pipeline
        pipeline_intensity = {"pipeline": [reader]}
        pdal_pipeline_surface_intensity = create_pdal_pipeline(
                return_only_ground=False,
                group_filter="first,only",
                reproject=False,
                input_crs=POINTCLOUD_CRS[i],
                filter_high_noise=filter_high_noise,
                filter_low_noise=filter_low_noise,
                hag_nn=hag_nn)

        intensity_stage = create_dem_stage(
            dem_filename=str(intensity_file),
            extent=original_extents[i],
            pointcloud_resolution=raster_resolution,
            gridmethod="idw",
            dimension="Intensity",
            nodata_value=0,
            data_type="UInt16",
        )

        pipeline_intensity["pipeline"] += pdal_pipeline_surface_intensity
        pipeline_intensity["pipeline"] += intensity_stage

        # Save a copy of each pipeline
        intensity_pipeline_config_fn = (
            output_path / f"pipeline_intensity_{str(i).zfill(ndigits)}.json"
        )
        with open(intensity_pipeline_config_fn, "w") as f:
            f.write(json.dumps(pipeline_intensity))
        intensity_pipeline_list.append(intensity_pipeline_config_fn)
        pipeline_intensity = None
        pdal_pipeline_surface_intensity = None

    return (
        dsm_pipeline_list,
        dtm_pipeline_no_fill_list,
        dtm_pipeline_fill_list,
        intensity_pipeline_list,
    )


def find_longitude_of_origin_from_utm(epsg_code: int) -> float:
    """
    Find the longitude of origin for a given UTM EPSG code.

    Parameters
    ----------
    epsg_code
        EPSG code for the UTM zone (e.g., 32617 for UTM zone 17N).

    Returns
    -------
    float
        Longitude of origin for the UTM zone in degrees.
    """

    crs = CRS.from_epsg(epsg_code)
    return crs.to_json_dict()["conversion"]["parameters"][1]["value"]


def write_local_utm_3DCRS_G2139(
    path_to_base_utm10_def: str, zone: str, outfn: str = None
) -> str:
    """
    Write a local UTM 3D CRS definition with respect to G2139 realization based on a base UTM 10N definition file.

    Parameters
    ----------
    path_to_base_utm10_def : str
        Path to the base UTM 10N file.
    zone : str
        UTM zone to create the CRS for
    outfn : str, optional
        Output filename for the modified CRS definition. If None, defaults to 'UTM_{zone}_WGS84_G2139_3D.wkt'.

    Returns
    -------
    str
        The filename of the output CRS definition file.
    """
    with open(path_to_base_utm10_def, "r") as f:  # open the file
        input_crs = f.read()
    if "N" in zone:
        zone_num = zone.split("N")[0]
        epsg_code = int(f"326{zone_num}")
    else:
        zone_num = zone.split("S")[0]
        epsg_code = int(f"327{zone_num}")
    # find center longitude
    center_long = find_longitude_of_origin_from_utm(epsg_code)
    mod_crs = input_crs.replace("UTM 10N", f"UTM {zone}")
    mod_crs = mod_crs.replace("UTM zone 10N", f"UTM zone {zone}")
    mod_crs = mod_crs.replace(
        '"Longitude of natural origin",-123',
        f'"Longitude of natural origin",{center_long}',
    )
    if outfn is None:
        outfn = os.path.join(
            os.path.split(path_to_base_utm10_def)[0], f"UTM_{zone}_WGS84_G2139_3D.wkt"
        )
    print(f"Writing 3D CRS at {outfn}")
    with open(outfn, "w") as f:
        f.write(mod_crs)
    return outfn


def execute_pdal_pipeline(pdal_pipeline_path: str) -> str:
    """
    Execute a PDAL pipeline
    #modified by Scott Henderson

    Parameters
    ----------
    pdal_pipeline_path : str
        The path to the PDAL pipeline json
    Returns
    -------
    output_fn
        The filename of the output raster is successfully saved, otherwise None is returned
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
        with open(pdal_pipeline_path) as f:
            pipelineDict = json.load(f)
            # maybe more robust to check for {'type': 'writers.gdal'}...
            outfile = pipelineDict["pipeline"][-1]["filename"]

            pipeline = pdal.Pipeline(json.dumps(pipelineDict))
            print(f"Executing {pdal_pipeline_path}")
            pipeline.execute()
            pipeline = None
        if check_raster_validity(outfile):
            return outfile
        else:
            return None
    except Exception as e:
        print(f"An error occurred while executing the PDAL pipeline: {e}")
        return None
    except RuntimeWarning as rw:
        print(f"PDAL RuntimeWarning: {rw}")
        return None
    


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
