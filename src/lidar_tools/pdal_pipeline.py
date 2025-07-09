"""
Generate a DSM from input polygon
"""

# Needs to happen before importing GDAL/PDAL
import os,sys
from dask.distributed import Client, LocalCluster


os.environ["PROJ_NETWORK"] = (
    "ON"  # Ensure this is 'ON' to get shift grids over the internet
)
print(f"PROJ_NETWORK is {os.environ['PROJ_NETWORK']}")

from lidar_tools import dsm_functions
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as _orient
import numpy as np
import json
from pathlib import Path
import warnings

import geopandas as gpd
from pathlib import Path


import requests


def create_dsm(
    extent_polygon: str,
    output_prefix: str,
    target_wkt: str = None,
    local_utm: bool = False,
    source_wkt: str = None,
    local_laz_dir: str = None,
    ept_tile_size_km: float = 1.0,
    process_specific_3dep_survey: str = None,
    process_all_intersecting_surveys: bool = False,
    num_process: int = 1,
    cleanup: bool = True,
    #output_resolution: float = 1.0, #to be added in a seperate PR
) -> None:
    """
    Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and intensity raster from a given extent and 3DEP point cloud data.

    Parameters
    ----------
    extent_polygon : str
        Path to the polygon file defining the processing extent.
    source_wkt : str or None
        Path to the WKT file defining the source coordinate reference system (CRS). If None, the CRS from the point cloud file is used.
    target_wkt : str
        Path to the WKT file defining the target coordinate reference system (CRS).
    output_prefix : str
        prefix with directory name and filename prefix for the project (e.g., CO_ALS_proc/CO_3DEP_ALS)
    local_utm: bool
        If true, compute the UTM zone from the extent polygon and use it to create the output rasters. If false, use the CRS defined in the target_wkt file.
    local_laz_dir: str
        If  the path to a local directory containing laz files is specified, the laz files are processed. If not specified, the function will process USGS 3DEP EPT tiles
    ept_tile_size_km: float
        The size of the EPT tiles to be processed. This is only used if local_laz_dir is not specified. The default is 1.0 km, which means that the function
        will process 1 km x 1 km tiles. If you want to process larger tiles, you can specify a larger value.
    process_specific_3dep_survey: str
        If specified, only process the given 3DEP survey. This should be a string that matches the workunit name in the 3DEP metadata
    process_all_intersecting_surveys: bool
        If true, process all available EPT surveys which intersect with the input polygon. If false, and process_specific_3dep_survey is not specified, only process the first available 3DEP EPT survey that intersects the input polygon.
    num_process: int, optional
        Number of processes to use for parallel processing. Default is 1, which means all pdal and gdal processing will be done serial
    cleanup: bool, optional
        If true, remove the intermediate tif files for the output tiles, leaving only the final mosaicked rasters. Default is True.
    Returns
    -------
    None

    """
    #figure out output projection
    #if user selectes local_utm, then compute the UTM zone from the extent polygon
    #this will supersed the target_wkt option
    
    if local_utm:
        gdf = gpd.read_file(extent_polygon)
        epsg_code = gdf.estimate_utm_crs().to_epsg()
        identifier_ns = str(epsg_code)[:3]
        identifier_zone = str(epsg_code)[3:]
        if identifier_ns == '326':
            zone = identifier_zone+'N'
        else:   
            zone = identifier_zone+'S'
        outdir = Path(output_prefix).parent
        if not outdir.exists():
            outdir.mkdir(parents=True, exist_ok=True)
        target_wkt =  outdir / f"UTM_{zone}_WGS84_G2139_3D.wkt"
        path_to_base_utm10_def = 'UTM_10.wkt' 
        url = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/UTM_10N_WGS84_G2139_3D.wkt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(path_to_base_utm10_def, "w") as f:
                f.write(response.text)
        target_wkt = dsm_functions.write_local_utm_3DCRS_G2139(path_to_base_utm10_def,zone=zone,outfn=target_wkt)

    # bounds for which pointcloud is created
    gdf = gpd.read_file(extent_polygon)
    _check_polygon_area(gdf)
    
    xmin, ymin, xmax, ymax = gdf.total_bounds
    input_aoi = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    input_crs = gdf.crs.to_wkt()

    # specify the output CRS of DEMs
    with open(target_wkt, "r") as f:
            contents = f.read()
    out_crs = CRS.from_string(contents)
    #print(out_crs)
    out_extent = gdf.to_crs(out_crs).total_bounds
    final_out_extent = dsm_functions.tap_bounds(out_extent,res=1) #this will change soon
    #print(f"Output extent in target CRS {out_crs} is {out_extent}")
    print(f"Output extent in target CRS is {final_out_extent}")
    gdf_out = gdf.to_crs(out_crs)
    gdf_out['geometry'] = gdf_out['geometry'].buffer(250) #buffer by 250m
    gdf_out = gdf_out.to_crs(input_crs) 
    extent_polygon = extent_polygon = outdir / "judicious_extent_polygon.geojson"
    gdf_out.to_file(extent_polygon, driver='GeoJSON')


    if local_laz_dir:
        print(f"This run will process laz files from {local_laz_dir}")
        ept_3dep = False
        (dsm_pipeline_list, dtm_no_fill_pipeline_list, dtm_fill_pipeline_list,
        intensity_pipeline_list) = dsm_functions.create_lpc_pipeline(
                                    local_laz_dir=local_laz_dir,
                                    target_wkt=target_wkt,output_prefix=output_prefix,
                                    aoi_bounds=extent_polygon)
        
    else:
        print("This run will process 3DEP EPT tiles")
        ept_3dep = True
        (dsm_pipeline_list, dtm_no_fill_pipeline_list, dtm_fill_pipeline_list,
        intensity_pipeline_list) = dsm_functions.create_ept_3dep_pipeline(
                extent_polygon, target_wkt, output_prefix,
                buffer_value=5,
                tile_size_km=ept_tile_size_km,
                process_specific_3dep_survey=process_specific_3dep_survey,
                process_all_intersecting_surveys=process_all_intersecting_surveys)
        
    if num_process == 1:
        print("Running DSM/DTM/intensity pipelines sequentially")
        final_dsm_fn_list = []
        final_dtm_no_fill_fn_list = []
        final_dtm_fill_fn_list = []
        final_intensity_fn_list = []
        for i, pipeline in enumerate(dsm_pipeline_list):
            outfn = dsm_functions.execute_pdal_pipeline(pipeline)
            if outfn is not None:
                final_dsm_fn_list.append(outfn)
        for i, pipeline in enumerate(dtm_no_fill_pipeline_list):
            outfn = dsm_functions.execute_pdal_pipeline(pipeline)
            if outfn is not None:
                final_dtm_no_fill_fn_list.append(outfn)
        for i, pipeline in enumerate(dtm_fill_pipeline_list):
            outfn = dsm_functions.execute_pdal_pipeline(pipeline)
            if outfn is not None:
                final_dtm_fill_fn_list.append(outfn)
        for i, pipeline in enumerate(intensity_pipeline_list):
            outfn = dsm_functions.execute_pdal_pipeline(pipeline)
            if outfn is not None:
                final_intensity_fn_list.append(outfn)
    else:
        print("Running DSM/DTM/intensity pipelines in parallel")
        num_pipelines = len(dsm_pipeline_list)
        if num_pipelines > num_process:
            n_jobs = num_process
        else:
            n_jobs = num_pipelines
        with Client(threads_per_worker=2, n_workers=n_jobs) as client:
            futures = client.map(dsm_functions.execute_pdal_pipeline,dsm_pipeline_list)
            final_dsm_fn_list = client.gather(futures)
            final_dsm_fn_list = [outfn for outfn in final_dsm_fn_list if outfn is not None]
        with Client(threads_per_worker=2, n_workers=n_jobs) as client:
            futures = client.map(dsm_functions.execute_pdal_pipeline,dtm_no_fill_pipeline_list)
            final_dtm_no_fill_fn_list = client.gather(futures)
            final_dtm_no_fill_fn_list = [outfn for outfn in final_dtm_no_fill_fn_list if outfn is not None]
        with Client(threads_per_worker=2, n_workers=n_jobs) as client:
            futures = client.map(dsm_functions.execute_pdal_pipeline,dtm_fill_pipeline_list)
            final_dtm_fill_fn_list = client.gather(futures)
            final_dtm_fill_fn_list = [outfn for outfn in final_dtm_fill_fn_list if outfn is not None]
        with Client(threads_per_worker=2, n_workers=n_jobs) as client:
            futures = client.map(dsm_functions.execute_pdal_pipeline,intensity_pipeline_list)
            final_intensity_fn_list = client.gather(futures)
            final_intensity_fn_list = [outfn for outfn in final_intensity_fn_list if outfn is not None]
        
    print("****Processing complete for all tiles****")
    

    #mosaicking step
    dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
    dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
    dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
    intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"
    if len(final_dsm_fn_list) > 1:
        print(
            f"Multiple DSM tiles created: {len(final_dsm_fn_list)}. Mosaicking required to create final DSM"
        )
        print("*** Now creating raster composites ***")
        if ept_3dep:
            cog = False
            out_extent = None
        else:
            out_extent = final_out_extent
            cog = True
        if num_process == 1:
            print("Running mosaicking sequentially")
            print(f"Creating DSM mosaic at {dsm_mos_fn}")
            dsm_functions.raster_mosaic(final_dsm_fn_list, dsm_mos_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating DTM mosaic at {dtm_mos_no_fill_fn}")
            dsm_functions.raster_mosaic(final_dtm_no_fill_fn_list, dtm_mos_no_fill_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating DTM mosaic with window size 4 at {dtm_mos_fill_fn}")
            dsm_functions.raster_mosaic(final_dtm_fill_fn_list, dtm_mos_fill_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
            dsm_functions.raster_mosaic(final_intensity_fn_list, intensity_mos_fn,
                cog=cog,out_extent=out_extent)
        else:
            #final_mos_list = []
            output_mos_list = [dsm_mos_fn, dtm_mos_no_fill_fn, dtm_mos_fill_fn, intensity_mos_fn]
            
            dems_list = [final_dsm_fn_list, final_dtm_no_fill_fn_list, final_dtm_fill_fn_list, final_intensity_fn_list]
            n_dems = len(dems_list)
            if n_dems > num_process:
                n_jobs = num_process
            else:
                n_jobs = n_dems
            with Client(n_workers=n_jobs) as client:
                futures = client.map(dsm_functions.raster_mosaic,
                                     dems_list,output_mos_list,[cog]*n_jobs,[out_extent]*n_jobs)
                output_mos_list = client.gather(futures)

    else:
        dsm_functions.rename_rasters(final_dsm_fn_list[0], dsm_mos_fn)
        dsm_functions.rename_rasters(final_dtm_no_fill_fn_list[0], dtm_mos_no_fill_fn)
        dsm_functions.rename_rasters(final_dtm_fill_fn_list[0], dtm_mos_fill_fn)
        dsm_functions.rename_rasters(final_intensity_fn_list[0], intensity_mos_fn)
        print("Only one tile created, no mosaicking required")

    # reprojection step
    dsm_reproj = dsm_mos_fn.split("-temp.tif")[0] + ".tif"
    dtm_no_fill_reproj = dtm_mos_no_fill_fn.split("-temp.tif")[0] + ".tif"
    dtm_fill_reproj = dtm_mos_fill_fn.split("-temp.tif")[0] + ".tif"
    intensity_reproj = intensity_mos_fn.split("-temp.tif")[0] + ".tif"
    if ept_3dep:
        if out_crs != CRS.from_epsg(3857):
            print("*********Reprojecting DSM, DTM and intensity rasters****")
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(dsm_mos_fn)
            if reproject_truth_val:
                # use input CRS which is EPSG:3857 with heights with respect to the NAVD88
                epsg_3857_navd88_fn = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/SRS_CRS.wkt"
                src_srs = epsg_3857_navd88_fn
            else:
                src_srs = "EPSG:3857"
            out_extent = final_out_extent
            print(src_srs)
            if num_process == 1:
                print("Running reprojection sequentially")
                print("Reprojecting DSM raster")
                dsm_functions.gdal_warp(dsm_mos_fn, dsm_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear",out_extent=out_extent)
                print("Reprojectiong DTM raster")
                dsm_functions.gdal_warp(dtm_mos_no_fill_fn, dtm_no_fill_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear", out_extent=out_extent)
                dsm_functions.gdal_warp(dtm_mos_fill_fn, dtm_fill_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear", out_extent=out_extent)
                print("Reprojecting intensity raster")
                dsm_functions.gdal_warp(intensity_mos_fn, intensity_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear" , out_extent=out_extent)
            else:
                print("Running reprojection in parallel")
                resolution = 1.0 #hardcoded for now, will change tomorrow
                reproj_fn_list = [dsm_reproj, dtm_no_fill_reproj, dtm_fill_reproj, intensity_reproj]
                dem_list = [dsm_mos_fn, dtm_mos_no_fill_fn, dtm_mos_fill_fn, intensity_mos_fn]
                n_dems = len(dem_list)
                if n_dems > num_process:
                    n_jobs = num_process
                else:
                    n_jobs = n_dems
                with Client(n_workers=n_jobs) as client:
                    futures = client.map(dsm_functions.gdal_warp,
                                        dem_list,reproj_fn_list,[src_srs]*n_jobs,
                                        [target_wkt]*n_jobs, [resolution]*n_jobs,
                                        ["bilinear"]*n_jobs,[out_extent]*n_jobs)
                    reproj_results = client.gather(futures)
    
    else:
        print("No reprojection required")
        # rename the temp files to the final output names
        dsm_functions.rename_rasters(dsm_mos_fn, dsm_reproj)
        dsm_functions.rename_rasters(dtm_mos_no_fill_fn, dtm_no_fill_reproj)
        dsm_functions.rename_rasters(dtm_mos_fill_fn, dtm_fill_reproj)
        dsm_functions.rename_rasters(intensity_mos_fn, intensity_reproj)
    
    print("****Building Gaussian overviews for all rasters****") 
    if num_process == 1:
        print("Running overview creation sequentially")
        dsm_functions.gdal_add_overview(dsm_reproj)
        dsm_functions.gdal_add_overview(dtm_no_fill_reproj)
        dsm_functions.gdal_add_overview(dtm_fill_reproj)
        dsm_functions.gdal_add_overview(intensity_reproj)
    else:
        
        print("Running overview creation in parallel")
        ovr_list = [dsm_reproj, dtm_no_fill_reproj, dtm_fill_reproj, intensity_reproj]
        ovr_results = []    
     
        n_dems = len(ovr_list)
        if n_dems > num_process:
            n_jobs = num_process
        else:
            n_jobs = n_dems
        with Client(n_workers=n_jobs) as client:
            futures = client.map(dsm_functions.gdal_add_overview, ovr_list)
            final_mos_list = client.gather(futures)
    if cleanup:
        print("User selected to remove intermediate tile outputs")
        tile_list = final_dsm_fn_list + final_dtm_no_fill_fn_list + final_dtm_fill_fn_list + final_intensity_fn_list
        tile_list = [tile for tile in tile_list if tile is not None]
        for fn in tile_list:
            try:
                Path(fn).unlink()
                aux_xml_fn = fn+".aux.xml"
                if Path(aux_xml_fn).exists():
                    Path(aux_xml_fn).unlink()
            except FileNotFoundError as e:
                print(f"Error {e} encountered for file {fn}")
                pass

        if ept_3dep:
            for fn in [dsm_mos_fn, dtm_mos_no_fill_fn, dtm_mos_fill_fn, intensity_mos_fn]:
            try:
                Path(fn).unlink()
                aux_xml_fn = fn+".aux.xml"
                if Path(aux_xml_fn).exists():
                    Path(aux_xml_fn).unlink()
            except FileNotFoundError as e:
                print(f"Error {e} encountered for file {fn}")
                pass
    print("****Processing complete****")


def geographic_area(gf: gpd.GeoDataFrame) -> gpd.pd.Series:
    """
    Estimate the geographic area of each polygon in a GeoDataFrame in m^2

    Parameters
    ----------
    gf : gpd.GeoDataFrame
        A GeoDataFrame containing the geometries for which the area needs to be calculated. The GeoDataFrame
        must have a geographic coordinate system (latitude and longitude).

    Returns
    -------
    pd.Series
        A Pandas Series containing the area of each polygon in the input GeoDataFrame in m^2.

    Raises
    ------
    TypeError
        If the GeoDataFrame does not have a geographic coordinate system.

    Notes
    ----------
    - Only works for areas up to 1/2 of globe (https://github.com/pyproj4/pyproj/issues/1401)
    """
    if gf.crs is None or not gf.crs.is_geographic:
        msg = "This function requires a GeoDataFrame with gf.crs.is_geographic==True"
        raise TypeError(msg)

    geod = gf.crs.get_geod()

    def area_calc(geom):
        if geom.geom_type not in ["MultiPolygon", "Polygon"]:
            return np.nan

        # For MultiPolygon do each separately
        if geom.geom_type == "MultiPolygon":
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal.
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(_orient(geom, 1))[0]

    return gf.geometry.apply(area_calc)


def _check_polygon_area(gf: gpd.GeoDataFrame) -> None:
    """
    Issue a warning if area is bigger than threshold
    """
    warn_if_larger_than = 100_000  # km^2

    # Fast track if projected and units are meters:
    if gf.crs.is_projected and gf.crs.axis_info[0].unit_name == "metre":
        area = gf.area * 1e-6
    else:
        area = geographic_area(gf.to_crs("EPSG:4326")) * 1e-6

    print(area.values[0])
    if area.to_numpy() >= warn_if_larger_than:
        msg = f"Very large AOI ({area.values[0]:e} km^2) requested, processing may be slow or crash. Recommended AOI size is <{warn_if_larger_than:e} km^2"
        warnings.warn(msg)
    else:
        print(f"Starting Processing of {area.values[0]:e} km^2 AOI")

