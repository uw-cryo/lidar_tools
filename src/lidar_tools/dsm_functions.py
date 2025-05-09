"""
Generate DSMs from 3DEP EPT data
"""
from rasterio.warp import transform_bounds
from pyproj import CRS
import shapely
import geopandas as gpd
import requests
import subprocess
from shutil import which
import rasterio
import xarray as xr 
import rioxarray
import pystac_client
import numpy as np
#import planetary_computer
from osgeo import gdal, gdalconst 
import odc.stac
import os 
odc.stac.configure_rio(cloud_defaults=True)

def nearest_floor(x, a):
    """
    Round down to the nearest smaller multiple of a.
    From https://github.com/uw-cryo/EarthLab_AirQuality_UAV/blob/main/notebooks/EarthLab_AQ_lidar_download_processing_function.ipynb
    """
    return np.floor(x / a) * a
def nearest_ceil(x, a):
    """
    Round down to the nearest larger multiple of a.
    From https://github.com/uw-cryo/EarthLab_AirQuality_UAV/blob/main/notebooks/EarthLab_AQ_lidar_download_processing_function.ipynb
    """
    return np.ceil(x / a) * a
def tap_bounds(site_bounds, res):
    """
    Return target aligned pixel bounds for a given site bounds and resolution.
    From https://github.com/uw-cryo/EarthLab_AirQuality_UAV/blob/main/notebooks/EarthLab_AQ_lidar_download_processing_function.ipynb
    """
    return ([nearest_floor(site_bounds[0], res), nearest_floor(site_bounds[1], res), \
            nearest_ceil(site_bounds[2], res), nearest_ceil(site_bounds[3], res)])


def return_readers(input_aoi,
                   src_crs,
                   pointcloud_resolution=1,
                   n_rows = 5,
                   n_cols=5,
                   buffer_value=5,
                   return_all_intersecting_surveys=False):
    """
    This method takes an input aoi and finds overlapping 3DEP EPT data from https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json
    It then returns a series of readers corresponding to non-overlapping areas for PDAL processing pipelines

    Parameters
    ----------
    input_aoi : shapely.geometry.Polygon
        The area of interest as a polygon.
    src_crs : pyproj.CRS
        The coordinate reference system of the input AOI.
    pointcloud_resolution : int, optional
        The resolution of the point cloud data, by default 1.
    n_rows : int, optional
        The number of rows to divide the AOI into, by default 5.
    n_cols : int, optional
        The number of columns to divide the AOI into, by default 5.
    buffer_value : int, optional
        The buffer value in meters to apply to each tile, by default 5.
    return_all_interescting_surveys : bool, optional
        If True, return all intersecting surveys, by default False.
    Returns
    -------
    list of dict
        A list of PDAL readers for each non-overlapping area.
    list of pyproj.CRS
        A list of coordinate reference systems from EPT metadata.
    """
    xmin, ymin, xmax, ymax = input_aoi.bounds
    x_step = (xmax - xmin) / n_cols
    y_step = (ymax - ymin) / n_rows

    dst_crs = CRS.from_epsg(4326)

    readers = []
    pointcloud_input_crs = []
    original_extents = []
    extents = []

    for i in range(int(n_cols)):
        for j in range(int(n_rows)):
            aoi = shapely.geometry.Polygon.from_bounds(xmin+i*x_step, ymin+j*y_step, xmin+(i+1)*x_step, ymin+(j+1)*y_step)

            src_bounds_transformed = transform_bounds(src_crs, dst_crs, *aoi.bounds)
            aoi_4326 = shapely.geometry.Polygon.from_bounds(*src_bounds_transformed)

            src_bounds_transformed_3857 = transform_bounds(src_crs, CRS.from_epsg(3857), *aoi.bounds)
            #create tap bounds for the tile
            src_bounds_transformed_3857 = tap_bounds(src_bounds_transformed_3857, pointcloud_resolution)
            aoi_3857 = shapely.geometry.Polygon.from_bounds(*src_bounds_transformed_3857)
            print(aoi.bounds, src_bounds_transformed_3857)
            if buffer_value:
                print(f"The tile polygon will be buffered by {buffer_value:.2f} m")
                #buffer the tile polygon by the buffer value
                aoi_3857 = aoi_3857.buffer(buffer_value)
                # now create tap bounds for the buffered tile
                aoi_3857_bounds = tap_bounds(aoi_3857.bounds, pointcloud_resolution)
                # now convert the buffered tile to a polygon
                aoi_3857 = shapely.geometry.Polygon.from_bounds(*aoi_3857_bounds)
                print(f"The buffered tile bound is: ", aoi_3857.bounds)
                


            gdf = gpd.read_file('https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson').set_crs(4326)
            # in the eventuality that the above URL breaks, we store a local copy
            # gdf = gpd.read_file('../data/shapefiles/resources.geojson').set_crs(4326)

            for _, row in gdf.iterrows():
                if row.geometry.intersects(aoi_4326):
                    usgs_dataset_name = row['name']
                    print("Dataset being used: ", usgs_dataset_name)
                    url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json"
                    reader = {
                    "type": "readers.ept",
                    "filename": url,
                    "resolution": pointcloud_resolution,
                    "polygon": str(aoi_3857.wkt),
                    }

                    # SRS associated with the 3DEP dataset
                    response = requests.get(url)
                    data = response.json()
                    srs_wkt = data['srs']['wkt']

                    pointcloud_input_crs.append(CRS.from_wkt(srs_wkt))
                    readers.append(reader)
                    extents.append(aoi_3857.bounds)
                    original_extents.append(src_bounds_transformed_3857)
                    if not return_all_intersecting_surveys:
                        break

            

    return readers, pointcloud_input_crs, extents, original_extents


def return_reader_inclusive(input_aoi,
                   src_crs,
                   pointcloud_resolution=1):
    """
     This method takes an input aoi and finds overlapping 3DEP EPT data from https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json
    It then returns a series of readers corresponding to non-overlapping areas for PDAL processing pipelines

    Parameters
    ----------
    input_aoi : shapely.geometry.Polygon
        The area of interest as a polygon.
    src_crs : pyproj.CRS
        The coordinate reference system of the input AOI.
    pointcloud_resolution : int, optional
        The resolution of the point cloud data, by default 1.
    """
    xmin, ymin, xmax, ymax = input_aoi.bounds
    readers = []
    pointcloud_input_crs = []
    src_bounds_transformed = transform_bounds(src_crs, CRS.from_epsg(4326), *input_aoi.bounds)
    aoi_4326 = shapely.geometry.Polygon.from_bounds(*src_bounds_transformed)
    src_bounds_transformed_3857 = transform_bounds(src_crs, CRS.from_epsg(3857), *input_aoi.bounds)
    aoi_3857 = shapely.geometry.Polygon.from_bounds(*src_bounds_transformed_3857)
    gdf = gpd.read_file('https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson').set_crs(4326)
    for _, row in gdf.iterrows():
            if row.geometry.intersects(aoi_4326):
                usgs_dataset_name = row['name']
                print("Dataset being used: ", usgs_dataset_name)
                url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json"
                reader = {
                "type": "readers.ept",
                "filename": url,
                "resolution": pointcloud_resolution,
                "polygon": str(aoi_3857.wkt),
                }

                # SRS associated with the 3DEP dataset
                response = requests.get(url)
                data = response.json()
                srs_wkt = data['srs']['wkt']

                pointcloud_input_crs.append(CRS.from_wkt(srs_wkt))
                readers.append(reader)
    return readers, pointcloud_input_crs



def create_pdal_pipeline(filter_low_noise=False, filter_high_noise=False,
                         filter_road=False, reset_classes=False, reclassify_ground=False,
                         return_only_ground=False, percentile_filter=False, percentile_threshold=0.95,
                         group_filter="first,only", reproject=True, save_pointcloud=False,
                         pointcloud_file = 'pointcloud', input_crs=None,
                         output_crs=None, output_type='laz'):

    assert abs(percentile_threshold) <= 1, "Percentile threshold must be in range [0, 1]"
    assert output_type in ['las', 'laz'], "Output type must be either 'las' or 'laz'"
    assert output_crs is not None, "Argument 'output_crs' must be explicitly specified!"

    stage_filter_low_noise = {
        "type":"filters.range",
        "limits":"Classification![7:7]"
    }
    stage_filter_high_noise = {
        "type":"filters.range",
        "limits":"Classification![18:18]"
    }
    stage_filter_road = {
        "type":"filters.range",
        "limits":"Classification![11:11]"
    }
    stage_reset_classes = {
        "type":"filters.assign",
        "value":"Classification = 0"
    }
    stage_reclassify_ground = {
        "type":"filters.smrf",
        # added from pdal smrf documentation, in turn from Pingel, 2013
        "scalar":1.2,
        "slope":0.2,
        "threshold":0.45,
        "window":8.0
    }
    stage_group_filter = {
        "type":"filters.returns",
        "groups":group_filter
    }
    stage_percentile_filter =  {
        "type":"filters.python",
        "script":"filter_percentile.py",
        "pdalargs": {"percentile_threshold":percentile_threshold},
        "function":"filter_percentile",
        "module":"anything"
    }
    stage_return_ground = {
        "type":"filters.range",
        "limits":"Classification[2:2]"
    }

    stage_reprojection = {
        "type":"filters.reprojection",
        "out_srs":str(output_crs)
    }
    if input_crs is not None:
        stage_reprojection["in_srs"] = str(input_crs)

    stage_save_pointcloud_las = {
        "type": "writers.las",
        "filename": f"{pointcloud_file}.las"
    }
    
    stage_save_pointcloud_laz = {
        "type": "writers.las",
        "compression": "true",
        "minor_version": "2",
        "dataformat_id": "0",
        "filename": f"{pointcloud_file}.laz"
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
        if group_filter is not None:
            pipeline.append(stage_group_filter)
        if percentile_filter:
            pipeline.append(stage_percentile_filter)
        if filter_low_noise:
            pipeline.append(stage_filter_low_noise)
        if percentile_filter or filter_high_noise:
            pipeline.append(stage_filter_high_noise)
        if filter_road:
            pipeline.append(stage_filter_road)

    # For creating DTMs, we want to process only ground returns
    if return_only_ground:
        pipeline.append(stage_return_ground)

    if reproject:
        pipeline.append(stage_reprojection)

    # the pipeline can save the pointclouds to a separate file if needed
    if save_pointcloud:
        if output_type == 'laz':
            pipeline.append(stage_save_pointcloud_laz)
        else:
            pipeline.append(stage_save_pointcloud_las)

    return pipeline


def create_dem_stage(dem_filename, extent, pointcloud_resolution=1.,
                        gridmethod='idw', dimension='Z'):
    #compute raster width and height
    width = (extent[2] - extent[0]) / pointcloud_resolution 
    height = (extent[3] - extent[1]) / pointcloud_resolution
    origin_x = extent[0]
    origin_y = extent[1]
    dem_stage = {
            "type":"writers.gdal",
            "filename":dem_filename,
            "gdaldriver":'GTiff',
            "nodata":-9999,
            "output_type":gridmethod,
            "resolution":float(pointcloud_resolution),
            "origin_x":origin_x,
            "origin_y":origin_y,
            "width":width,
            "height":height,
            "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
    }

    dem_stage.update({
            'dimension':dimension
        })

    return [dem_stage]

def raster_mosaic(img_list,outfn):
    """
    Given a list of input images, mosaic them into a COG raster by using vrt and gdal_translate
    im_list: list
        List of input images to be mosaiced
    outfn: str
        Path to output mosaicked image
    """
    # create vrt
    vrt_fn = os.path.splitext(outfn)[0]+'.vrt'
    gdal.BuildVRT(vrt_fn,img_list,
                  callback=gdal.TermProgress_nocb)
    # translate to COG
    gdal.Translate(outfn,vrt_fn,options=gdal.TranslateOptions(creationOptions=['COMPRESS=LZW','TILED=YES','COPY_SRC_OVERVIEWS=YES']),
                   callback=gdal.TermProgress_nocb)
    # delete vrt
    os.remove(vrt_fn)


### Functions for datum checks

def get_esa_worldcover(
    bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
    version: str = "v200", mask_nodata: bool = False,
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
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    version : str, optional
        Version of the WorldCover data. The two versions are v100 (2020) and v200 (2021). Default is 'v200'.
    mask_nodata : bool, optional
        Whether to mask no data values. Default is False.
        If False: (dtype=uint8, rio.nodata=0, rio.encoded_nodata=None)
        If True: (dtype=float32, rio.nodata=nan, rio.encoded_nodata=0)

    Returns
    -------
    xarray.DataArray
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
        worldcover_da = worldcover_da.where(worldcover_da>0)
        worldcover_da.rio.write_nodata(0, encoded=True, inplace=True)

    worldcover_da.attrs["class_info"] = get_class_info()
    #worldcover_da.attrs["cmap"] = get_class_cmap(worldcover_da.attrs["class_info"])
    worldcover_da.attrs['data_citation'] = "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O. (2021). ESA WorldCover 10 m 2020 v100. doi:10.5281/zenodo.5571936."
    
    #worldcover_da.attrs['example_plot'] = plot_classes

    return worldcover_da


def fetch_worldcover(raster_fn: str,
                    match_grid_da: xr.DataArray =None):
    with rasterio.open(raster_fn) as dataset:
        bounds = dataset.bounds
        bounds = rasterio.warp.transform_bounds(dataset.crs, 'EPSG:4326', *bounds)
        bbox_gdf = gpd.GeoDataFrame(geometry=[shapely.box(*bounds)],crs='EPSG:4326',index=[0])
    
    da_wc = get_esa_worldcover(bbox_gdf,mask_nodata=True)
    if match_grid_da is not None:
        da_wc = da_wc.rio.reproject_match(match_grid_da,resampling=rasterio.enums.Resampling.nearest)
    return da_wc

def common_mask(da_list: list,
                apply: bool =False):
    """
    From a list of xarray dataarray objects sharing the same projection/extent/res, compute common mask where all input datasets have non-nan pixels
    """
    # load nan layers as numpy array
    nan_arrays = np.array([np.isnan(da.values) for da in da_list])
    common_mask = 1 - np.any(nan_arrays,axis=0)
    
    if apply:
        common_mask_da_list = [da.where(common_mask,np.nan) for da in da_list]
        return common_mask_da_list
    else:
        return common_mask
def convert_bbox_to_geodataframe(bbox_input):
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
               
def get_copernicus_dem(bbox_input: gpd.GeoDataFrame | tuple | shapely.geometry.base.BaseGeometry | None = None,
                       resolution: int = 30
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
    bbox_input : geopandas.GeoDataFrame or tuple or Shapely Geometry
        GeoDataFrame containing the bounding box, or a tuple of (xmin, ymin, xmax, ymax), or a Shapely geometry.
    resolution : int, optional
        The resolution of the DEM, either 30 or 90 meters. Default is 30.

    Returns
    -------
    xarray.DataArray
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
        raise ValueError("Copernicus DEM resolution is available in 30m and 90m. Please select either 30 or 90.")

    # Convert the input to a GeoDataFrame if it's not already one
    bbox_gdf =  convert_bbox_to_geodataframe(bbox_input)

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[f"cop-dem-glo-{resolution}"],bbox=bbox_gdf.total_bounds)
    cop_dem_da = odc.stac.load(search.items(),bbox=bbox_gdf.total_bounds,chunks={})['data'].squeeze()
    cop_dem_da = cop_dem_da.rio.write_nodata(-32767,encoded=True)

    return cop_dem_da


def fetch_cop30(raster_fn: str,
                match_grid_da: xr.DataArray = None) -> xr.DataArray:
    with rasterio.open(raster_fn) as dataset:
        bounds = dataset.bounds
        bounds = rasterio.warp.transform_bounds(dataset.crs, 'EPSG:4326', *bounds)
        bbox_gdf = gpd.GeoDataFrame(geometry=[shapely.box(*bounds)],crs='EPSG:4326',index=[0])
    cop_da = get_copernicus_dem(bbox_gdf,
                                        resolution=30)
    if match_grid_da is not None:
        cop_da = cop_da.rio.reproject_match(match_grid_da,resampling=rasterio.enums.Resampling.cubic)
    return cop_da



def confirm_3dep_vertical(raster_fn: str,
                bare_diff_tolerance: float = 3.0) -> bool:
    lidar_da = rioxarray.open_rasterio(raster_fn,masked=True).squeeze()
    worldcover_da = fetch_worldcover(raster_fn,lidar_da)
    cop30_da = fetch_cop30(raster_fn,lidar_da)
    lidar_da_masked,worldcover_da_masked,cop30_da_masked = common_mask([lidar_da,worldcover_da,cop30_da],apply=True)
    dem_diff = lidar_da_masked - cop30_da_masked
    ## Mask out bare and sparse vegetation class
    bare_sparse_mask = worldcover_da_masked == 60
    dem_diff_bare = dem_diff.where(bare_sparse_mask,np.nan)
    median_diff = np.nanmedian(dem_diff_bare.values)
    print(f"Observed difference between COP30 EGM2008 and 3DEP LiDAR DSM over bareground and sparse vegetation surfaces is {median_diff:.2f} m")
    if np.abs(median_diff) <= bare_diff_tolerance:
        #this means that both COP30 and 3DEP LiDAR DSM are with respect to geoid
        print("Looks like the 3DEP height estimates are with respect to geiod, will apply vertical datum shift to return heights with respect to ellipsoid")
        out = True
    else:
        #this means that 3DEP LiDAR DSM is with respect to ellipsoid
        print("Looks like the 3DEP height estimates are already with respect to ellipsoid, geoid to ellipsoid transformation will not be attempted")
        out = False
    return out

def check_raster_validity(raster_fn: str) -> bool:
    """
    Check if a raster file is valid and can be opened using rioxarray and CRS check

    Parameters
    ----------
    raster_fn : str
        Path to the raster file.

    Returns
    -------
    bool
        True if the raster file is valid, False otherwise.
    """
    da = rioxarray.open_rasterio(raster_fn,masked=True).squeeze()
    if da.rio.crs is None:
        #print(f"Raster {raster_fn} does not have a valid CRS.")
        out = False
    else:
        #print(f"Raster {raster_fn} has a valid CRS.")
        out = True
    return out   
def gdal_warp(src_fn: str,
                dst_fn: str, 
                src_srs: str, 
                dst_srs: str, 
                res: float = 1.0,
                resampling_alogrithm: str ='cubic') -> None:
    tolerance = 0
    resampling_mapping = {"nearest":  gdalconst.GRA_NearestNeighbour, "bilinear": gdalconst.GRA_Bilinear,
                  "cubic": gdalconst.GRA_Cubic, "cubic_spline": gdalconst.GRA_CubicSpline}
    resampling_alg = resampling_mapping[resampling_alogrithm]
    ds = gdal.Warp(dst_fn, src_fn,
                   resampleAlg=resampling_alg,
                   srcSRS=src_srs, xRes=res, yRes=res,
                   dstSRS=dst_srs, errorThreshold=tolerance,
                   targetAlignedPixels=True,
                   callback=gdal.TermProgress_nocb)
    ds = None

