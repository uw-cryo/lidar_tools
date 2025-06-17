"""
Generate a DSM from input polygon
"""

# Needs to happen before importing GDAL/PDAL
import os,sys


os.environ["PROJ_NETWORK"] = (
    "ON"  # Ensure this is 'ON' to get shift grids over the internet
)
print(f"PROJ_NETWORK is {os.environ['PROJ_NETWORK']}")

from lidar_tools import dsm_functions
import pdal
from pyproj import CRS
import geopandas as gpd
from pathlib import Path
from dask.distributed import Client,progress

import dask





def create_dsm(
    extent_polygon: str,
    target_wkt: str,
    output_prefix: str,
    source_wkt: str = None,
    local_laz_dir: str = None,
    process_specific_3dep_survey: str = None,
    process_all_intersecting_surveys: bool = False,
    cleanup: bool = True,
    parallel: bool = False,
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
    cleanup: bool
        If true, remove the intermediate tif files for the output tiles
    reproject: bool
        If true, perform final reprojection from EPSG:3857 to user sepecified CRS
    local_laz_dir: str
        If specified, the path to a local directory containing laz files. If not specified, the function will process USGS 3DEP EPT tiles
    process_specific_3dep_survey: str
        If specified, only process the given 3DEP survey. This should be a string that matches the survey name in the 3DEP metadata
    process_all_intersecting_surveys: bool
        If true, process all intersecting surveys. If false, only process the first LiDAR survey that intersects the extent defined in the GeoJSON file.

    Returns
    -------
    None

    Notes
    -----
    After running this function, reproject the final DEM with the following commands:
    1. gdalwarp -s_srs SRS_CRS.wkt -t_srs UTM_13N_WGS84_G2139_3D.wkt -r cubic -tr 1.0 1.0 merged_dsm.tif merged_dsm_reprojected_UTM_13N_WGS84_G2139.tif

    """
    with open(target_wkt, "r") as f:
            contents = f.read()
    out_crs = CRS.from_string(contents)
    if local_laz_dir:
        print(f"This run will process laz files from {local_laz_dir}")
        ept_3dep = False
        (dsm_pipeline_list, dsm_fn_list, 
        dtm_pipeline_list, dtm_fn_list, 
        intensity_pipeline_list, intensity_fn_list) = dsm_functions.create_lpc_pipeline(
                                    local_laz_dir=local_laz_dir,
                                    target_wkt=target_wkt,output_prefix=output_prefix,
                                    aoi_bounds=extent_polygon)
    else:
        print("This run will process 3DEP EPT tiles")
        ept_3dep = True
        (dsm_pipeline_list, dsm_fn_list, 
        dtm_pipeline_list, dtm_fn_list, 
        intensity_pipeline_list, intensity_fn_list) = dsm_functions.create_ept_3dep_pipeline(
                                extent_polygon,target_wkt,output_prefix,
                                buffer_value=5,
                                tile_size_km=1.0,
                                process_specific_3dep_survey=process_specific_3dep_survey,
                                process_all_intersecting_surveys=process_all_intersecting_surveys
        )

    #execute the pipelines
    if not parallel:
        final_dsm_fn_list = []
        print("Running DSM/DTM/intensity pipelines sequentially")
        for i, pipeline in enumerate(dsm_pipeline_list):
            dsm = dsm_functions.execute_pdal_pipeline(pipeline,dsm_fn_list[i])
            if dsm is not None:
                final_dsm_fn_list.append(dsm)
        
        final_dtm_fn_list = []
        for i, pipeline in enumerate(dtm_pipeline_list):
            dtm = dsm_functions.execute_pdal_pipeline(pipeline,dtm_fn_list[i])
            if dtm is not None:
                final_dtm_fn_list.append(dtm)
        
        final_intensity_fn_list = []
        for i, pipeline in enumerate(intensity_pipeline_list):
            intensity = dsm_functions.execute_pdal_pipeline(pipeline,intensity_fn_list[i])
            if intensity is not None:
                final_intensity_fn_list.append(intensity)
    else:
        print("Running DSM/DTM/intensity pipelines in parallel")
        final_dsm_fn_list = []
        for idx, pipeline in enumerate(dsm_pipeline_list):
            #print(f"Executing DSM pipeline {idx+1} of {len(dsm_pipeline_list)}")
            final_dsm_list = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, dsm_fn_list[idx])
            final_dsm_fn_list.append(final_dsm_list)
        client = Client(threads_per_worker=2, n_workers=5)
        futures = dask.persist(*final_dsm_fn_list)
        final_dsm_fn_list = dask.compute(*futures)
        print(type(final_dsm_fn_list))
        print(final_dsm_fn_list)
        final_dtm_fn_list = []
        final_intensity_fn_list = []
        for idx,pipeline in enumerate(dtm_pipeline_list):
            #print(f"Executing DTM pipeline {idx+1} of {len(dtm_pipeline_list)}")
            final_dtm_list = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, dtm_fn_list[idx])
            final_dtm_fn_list.append(final_dtm_list)
        client = Client(threads_per_worker=2, n_workers=5)
        futures = dask.persist(*final_dtm_fn_list)
        final_dtm_fn_list = dask.compute(*futures)

        for idx,pipeline in enumerate(intensity_pipeline_list):
            #print(f"Executing intensity pipeline {idx+1} of {len(intensity_pipeline_list)}")
            final_intensity_list = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, intensity_fn_list[idx])
            final_intensity_fn_list.append(final_intensity_list)
        client = Client(threads_per_worker=2, n_workers=5)
        futures = dask.persist(*final_intensity_fn_list)
        final_intensity_fn_list = dask.compute(*futures)
    print("****Processing complete for all tiles****")
    
    if len(final_dsm_fn_list) > 1:
        print(
            f"Multiple DSM tiles created: {len(final_dsm_fn_list)}. Mosaicking required to create final DSM"
        )
        print("*** Now creating raster composites ***")
        if ept_3dep:
            dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
            dtm_mos_fn = f"{output_prefix}-DTM_mos-temp.tif"
            intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"
            cog = False
        else:
            dsm_mos_fn = f"{output_prefix}-DSM_mos.tif"
            dtm_mos_fn = f"{output_prefix}-DTM_mos.tif"
            intensity_mos_fn = f"{output_prefix}-intensity_mos.tif"
            cog = True
        if not parallel:
            print("Running mosaicking sequentially")
            print(f"Creating DSM mosaic at {dsm_mos_fn}")
            dsm_functions.raster_mosaic(final_dsm_fn_list, dsm_mos_fn,cog=cog)
            print(f"Creating DTM mosaic at {dtm_mos_fn}")
            dsm_functions.raster_mosaic(final_dtm_fn_list, dtm_mos_fn,cog=cog)
            print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
            dsm_functions.raster_mosaic(final_intensity_fn_list, intensity_mos_fn,cog=cog)
        else:
            final_mos_list = []
            output_mos_list = [dsm_mos_fn, dtm_mos_fn, intensity_mos_fn]
            for idx,lists in enumerate([final_dsm_fn_list, final_dtm_fn_list, final_intensity_fn_list]):
                
                final_mos = (dask.delayed(dsm_functions.raster_mosaic)(lists, output_mos_list[idx], cog))
                final_mos_list.append(final_mos)
            client = Client(threads_per_worker=2, n_workers=5)
            futures = dask.persist(*final_mos_list)
            final_mos_list = dask.compute(*futures)
    else:
        dsm_mos_fn = final_dsm_fn_list[0]
        dtm_mos_fn = final_dtm_fn_list[0]
        intensity_mos_fn = final_intensity_fn_list[0]

    
    if ept_3dep:
        if out_crs != CRS.from_epsg(3857):
            print("*********Reprojecting DSM, DTM and intensity rasters****")
            dsm_reproj = dsm_mos_fn.split("-temp.tif")[0] + ".tif"
            dtm_reproj = dtm_mos_fn.split("-temp.tif")[0] + ".tif"
            intensity_reproj = intensity_mos_fn.split("-temp.tif")[0] + ".tif"
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(dsm_mos_fn)
            if reproject_truth_val:
                # use input CRS which is EPSG:3857 with heights with respect to the NAVD88
                epsg_3857_navd88_fn = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/SRS_CRS.wkt"
                src_srs = epsg_3857_navd88_fn
            else:
                src_srs = "EPSG:3857"
            print(src_srs)
            if not parallel:
                print("Running reprojection sequentially")
                print("Reprojecting DSM raster")
                dsm_functions.gdal_warp(dsm_mos_fn, dsm_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear")
                print("Reprojectiong DTM raster")
                dsm_functions.gdal_warp(dtm_mos_fn, dtm_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear")
                print("Reprojecting intensity raster")
                dsm_functions.gdal_warp(intensity_mos_fn, intensity_reproj, src_srs, target_wkt,
                                        resampling_alogrithm="bilinear")
            else:
                print("Running reprojection in parallel")
                resolution = 1.0 #hardcoded for now, will change tomorrow
                reproj_fn_list = [dsm_reproj, dtm_reproj, intensity_reproj]
                dem_list = [dsm_mos_fn, dtm_mos_fn, intensity_mos_fn]
                out_list = []
                for idx, input_fn in enumerate(dem_list):
                    reproj = (dask.delayed(dsm_functions.gdal_warp)(input_fn, reproj_fn_list[idx], src_srs, target_wkt,
                                        resolution, "bilinear"))
                    out_list.append(reproj)
                client = Client(threads_per_worker=2, n_workers=3)
                futures = dask.persist(*out_list)
                out = dask.compute(*futures)
        else:
            print("No reprojection required")
            # rename the temp files to the final output names
            os.rename(dsm_mos_fn, dsm_reproj)
            os.rename(dtm_mos_fn, dtm_reproj)
            os.rename(intensity_mos_fn, intensity_reproj)
    else:
        dsm_reproj = dsm_mos_fn
        dtm_reproj = dtm_mos_fn
        intensity_reproj = intensity_mos_fn
    print("****Building Gaussian overviews for all rasters****") 
    if not parallel:
        print("Running overview creation sequentially")
        dsm_functions.gdal_add_overview(dsm_reproj)
        dsm_functions.gdal_add_overview(dtm_reproj)
        dsm_functions.gdal_add_overview(intensity_reproj)
    else:
        print("Running overview creation in parallel")
        ovr_list = [dsm_reproj, dtm_reproj, intensity_reproj]
        ovr_results = []    
        for ovr in ovr_list:
            ovr_results.append(dask.delayed(dsm_functions.gdal_add_overview)(ovr))
        client = Client(threads_per_worker=2, n_workers=3)
        futures = dask.persist(*ovr_results)
        out = dask.compute(*futures)

    if cleanup:
        print("User selected to remove intermediate tile outputs")
        tile_list = dsm_fn_list + dtm_fn_list + intensity_fn_list
        tile_list = [tile for tile in tile_list if tile is not None]
        for fn in tile_list:
            try:
                Path(fn).unlink()
            except FileNotFoundError as e:
                print(f"Error {e} encountered for file {fn}")
                pass
        if ept_3dep:
            if os.path.exists(dsm_mos_fn):
                os.remove(dsm_mos_fn)
            if os.path.exists(dtm_mos_fn):
                os.remove(dtm_mos_fn)
            if os.path.exists(intensity_mos_fn):
                os.remove(intensity_mos_fn)
    print("****Processing complete****")