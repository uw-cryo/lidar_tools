"""
Generate a DSM from input polygon
"""

# Needs to happen before importing GDAL/PDAL
import os,sys
from dask.distributed import Client, LocalCluster
import gc


os.environ["PROJ_NETWORK"] = (
    "ON"  # Ensure this is 'ON' to get shift grids over the internet
)
print(f"PROJ_NETWORK is {os.environ['PROJ_NETWORK']}")

from lidar_tools import dsm_functions
import pdal
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as _orient
import numpy as np
import json
from pathlib import Path
import warnings

import geopandas as gpd
from pathlib import Path
from dask.distributed import LocalCluster,Client,progress
from joblib import Parallel, delayed
import dask
import requests


def create_dsm(
    extent_polygon: str,
    output_prefix: str,
    target_wkt: str = None,
    local_utm: bool = False,
    source_wkt: str = None,
    local_laz_dir: str = None,
    process_specific_3dep_survey: str = None,
    process_all_intersecting_surveys: bool = False,
    cleanup: bool = True,
    parallel: bool = False,

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
        target_wkt = os.path.join(outdir, f"UTM_{zone}_WGS84_G2139_3D.wkt")
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
    extent_polygon = os.path.join(outdir, "judicious_extent_polygon.geojson")
    gdf_out.to_file(extent_polygon, driver='GeoJSON')

    temp_extent_polygon = gpd.read_file(extent_polygon).to_crs(out_crs).buffer(250)

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
        
        if not parallel:
            print("Running DSM/DTM/intensity pipelines sequentially")
            (final_dsm_fn_list, final_dtm_no_fill_fn_list, final_dtm_fill_fn_list, 
            final_intensity_fn_list) = dsm_functions.execute_ept_3dep_pipeline(
                extent_polygon, target_wkt, output_prefix,
                buffer_value=5,
                tile_size_km=5.0,
                process_specific_3dep_survey=process_specific_3dep_survey,
                process_all_intersecting_surveys=process_all_intersecting_surveys)

    #execute the pipelines
    
        
        # (final_dsm_fn_list,
        # final_dtm_fn_list, 
        # final_intensity_fn_list) = dsm_functions.execute_ept_3dep_pipeline(extent_polygon,target_wkt,output_prefix,
        #                         buffer_value=5,
        #                         tile_size_km=5.0,
        #                         process_specific_3dep_survey=process_specific_3dep_survey,
        #                         process_all_intersecting_surveys=process_all_intersecting_surveys)
        else:
            print("Running DSM/DTM/intensity pipelines in parallel")
            joblib_list = []
            final_dsm_task_list = []
            for idx, pipeline in enumerate(dsm_pipeline_list):
                #print(f"Executing DSM pipeline {idx+1} of {len(dsm_pipeline_list)}")
                delayed_task = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, dsm_fn_list[idx])
                final_dsm_task_list.append(delayed_task)
                joblib_list.append((pipeline,dsm_fn_list[idx]))
            #final_dsm_fn_list = Parallel(n_jobs=5)(delayed(dsm_functions.execute_pdal_pipeline)(pipeline, dsm_fn) for pipeline, dsm_fn in joblib_list)
        

        
            # Example usage
            #final_dsm_fn_list = _memory_safe_batch_processing(dsm_pipeline_list, dsm_fn_list, batch_size=50, restart_every=3)
            #final_dsm_fn_list = list(Parallel(n_jobs=6)(delayed(dsm_functions.execute_pdal_pipeline)(
            #pipeline, dsm_fn_list[idx]) for idx,pipeline  in enumerate(dsm_pipeline_list)))
            #with Client(threads_per_worker=2, n_workers=5 ) as client:
            #    futures = client.map(dsm_functions.execute_pdal_pipeline, dsm_pipeline_list, dsm_fn_list)
            #    final_dsm_fn_list = client.gather(futures)
            #cluster = LocalCluster(threads_per_worker=2, n_workers=5,processes=False)
            #dask.config.set(scheduler='multiprocessing')
            final_dsm_fn_list = [] 
            if len(final_dsm_task_list) > 20:
                print(f"Executing dask jobs in batches of 20")
                batch_size = 20
                for i in range(0, len(final_dsm_task_list), batch_size):
                    batch_tasks = final_dsm_task_list[i:i + batch_size]
                    cluster = LocalCluster(n_workers=5)
                    client = Client(cluster)
                    results = dask.compute(*batch_tasks)
                    final_dsm_fn_list.extend(results)
                    client.close()
            else:
                cluster = LocalCluster(n_workers=5)
                client = Client(cluster)
                final_dsm_fn_list = dask.compute(*final_dsm_task_list)
                client.close()
            final_dsm_task_list = None
            
            #futures = dask.persist(*final_dsm_fn_list)
            #final_dsm_fn_list = dask.compute(*futures)

            final_dtm_task_list = []
            #final_intensity_fn_list = []
            for idx,pipeline in enumerate(dtm_pipeline_list):
                #print(f"Executing DTM pipeline {idx+1} of {len(dtm_pipeline_list)}")
                delayed_task = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, dtm_fn_list[idx])
                final_dtm_task_list.append(delayed_task)
            final_dtm_fn_list = []
            if len(final_dtm_task_list) > 20:           
                print(f"Executing dask jobs in batches of 20")
                batch_size = 20
                for i in range(0, len(final_dtm_task_list), batch_size):
                    batch_tasks = final_dtm_task_list[i:i + batch_size]
                    cluster = LocalCluster(n_workers=5)
                    client = Client(cluster)
                    results = dask.compute(*batch_tasks)
                    final_dtm_fn_list.extend(results)
                    client.close()
            else:
                cluster = LocalCluster(n_workers=5)
                client = Client(cluster)
                final_dtm_fn_list = dask.compute(*final_dtm_task_list)
                client.close()
            final_dtm_task_list = None


            #final_dtm_fn_list = list(Parallel(n_jobs=6)(delayed(d
            #cluster = LocalCluster(threads_per_worker=2, n_workers=5,memory_limit='10GB',
            #                       memory_target_fraction=0.95,processes=True)
            #client = Client(cluster)
            #futures = dask.persist(*final_dtm_fn_list)
            #final_dtm_fn_list = dask.compute(*futures)
            #final_dtm_fn_list = dask.compute(*final_dtm_fn_list)
            #client.close()
            #final_dtm_fn_list = list(Parallel(n_jobs=6)(delayed(dsm_functions.execute_pdal_pipeline)(
            #    pipeline, dtm_fn_list[idx]) for idx,pipeline  in enumerate(dtm_pipeline_list)))

            #for idx,pipeline in enumerate(intensity_pipeline_list):
                #print(f"Executing intensity pipeline {idx+1} of {len(intensity_pipeline_list)}")
            #    final_intensity_list = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, intensity_fn_list[idx])
            #    final_intensity_fn_list.append(final_intensity_list)
            #cluster = LocalCluster(threads_per_worker=2, n_workers=5,memory_limit='10GB',
            #                       memory_target_fraction=0.95,processes=True)
            #client = Client(cluster)
            #futures = dask.persist(*final_intensity_fn_list)
            #final_intensity_fn_list = dask.compute(*futures)
            #final_intensity_fn_list = dask.compute(*final_intensity_fn_list)
            #client.close()
            #final_intensity_fn_list = list(Parallel(n_jobs=6)(delayed(dsm_functions.execute_pdal_pipeline)(
            #   pipeline, intensity_fn_list[idx]) for idx,pipeline  in enumerate(intensity_pipeline_list)))
            final_intensity_task_list = []
            final_intensity_fn_list = []
            for idx,pipeline in enumerate(intensity_pipeline_list):
                #print(f"Executing intensity pipeline {idx+1} of {len(intensity_pipeline_list)}")
                delayed_task = dask.delayed(dsm_functions.execute_pdal_pipeline)(pipeline, intensity_fn_list[idx])
                final_intensity_task_list.append(delayed_task)
            if len(final_intensity_task_list) > 20:
                print(f"Executing dask jobs in batches of 20")
                batch_size = 20
                for i in range(0, len(final_intensity_task_list), batch_size):
                    batch_tasks = final_intensity_task_list[i:i + batch_size]
                    cluster = LocalCluster(n_workers=5)
                    client = Client(cluster)
                    results = dask.compute(*batch_tasks)
                    final_intensity_fn_list.extend(results)
                    client.close()
            else:
                cluster = LocalCluster(n_workers=5)
                client = Client(cluster)
                final_intensity_fn_list = dask.compute(*final_intensity_task_list)
                client.close()
            final_intensity_task_list = None
    print("****Processing complete for all tiles****")
    
    if len(final_dsm_fn_list) > 1:
        print(
            f"Multiple DSM tiles created: {len(final_dsm_fn_list)}. Mosaicking required to create final DSM"
        )
        print("*** Now creating raster composites ***")
        if ept_3dep:
            dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
            dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
            dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
            intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"
            cog = False
            out_extent = None
        else:
            dsm_mos_fn = f"{output_prefix}-DSM_mos.tif"
            dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
            dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
            intensity_mos_fn = f"{output_prefix}-intensity_mos.tif"
            out_extent = final_out_extent
            cog = False
        if not parallel:
            print("Running mosaicking sequentially")
            print(f"Creating DSM mosaic at {dsm_mos_fn}")
            dsm_functions.raster_mosaic(final_dsm_fn_list, dsm_mos_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating DTM mosaic at {dtm_mos_no_fill_fn}")
            dsm_functions.raster_mosaic(dtm_no_fill_fn_list, dtm_mos_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating DTM mosaic with window size 4 at {dtm_mos_fill_fn}")
            dsm_functions.raster_mosaic(dtm_fill_fn_list, dtm_mos_fill_fn,
                cog=cog,out_extent=out_extent)
            print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
            dsm_functions.raster_mosaic(final_intensity_fn_list, intensity_mos_fn,
                cog=cog,out_extent=out_extent)
        else:
            #final_mos_list = []
            output_mos_list = [dsm_mos_fn, dtm_mos_fn, intensity_mos_fn]
            #for idx,lists in enumerate([final_dsm_fn_list, final_dtm_fn_list, final_intensity_fn_list]):
                
            #    final_mos = (dask.delayed(dsm_functions.raster_mosaic)(lists, output_mos_list[idx], cog))
             #   final_mos_list.append(final_mos)
            #client = Client(threads_per_worker=2, n_workers=5 )
            #futures = dask.persist(*final_mos_list)
            #final_mos_list = dask.compute(*futures)
            #final_mos_list = dask.compute(*final_mos_list)
            #client.close()
            dems_list = [final_dsm_fn_list, final_dtm_fn_list, final_intensity_fn_list]
            final_mos_list = Parallel(n_jobs=5)(delayed(dsm_functions.raster_mosaic)(
                lists,output_mos_list[idx], cog) for idx,lists  in enumerate(dems_list))
    else:
        dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
        dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
        dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
        
        
        intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"
        os.rename(final_dsm_fn_list[0], dsm_mos_fn)
        os.rename(final_dtm_no_fill_fn_list[0], dtm_mos_no_fill_fn)
        os.rename(final_dtm_fill_fn_list[0], dtm_mos_fill_fn)
        os.rename(final_intensity_fn_list[0], intensity_mos_fn)
        print("Only one tile created, no mosaicking required")

    
    if ept_3dep:
        if out_crs != CRS.from_epsg(3857):
            print("*********Reprojecting DSM, DTM and intensity rasters****")
            dsm_reproj = dsm_mos_fn.split("-temp.tif")[0] + ".tif"
            dtm_no_fill_reproj = dtm_mos_no_fill_fn.split("-temp.tif")[0] + ".tif"
            dtm_fill_reproj = dtm_mos_fill_fn.split("-temp.tif")[0] + ".tif"
            intensity_reproj = intensity_mos_fn.split("-temp.tif")[0] + ".tif"
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(dsm_mos_fn)
            if reproject_truth_val:
                # use input CRS which is EPSG:3857 with heights with respect to the NAVD88
                epsg_3857_navd88_fn = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/SRS_CRS.wkt"
                src_srs = epsg_3857_navd88_fn
            else:
                src_srs = "EPSG:3857"
            out_extent = final_out_extent
            print(src_srs)
            if not parallel:
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
                reproj_fn_list = [dsm_reproj, dtm_reproj, intensity_reproj]
                dem_list = [dsm_mos_fn, dtm_mos_fn, intensity_mos_fn]
                #out_list = []
                #for idx, input_fn in enumerate(dem_list):
                 #   reproj = (dask.delayed(dsm_functions.gdal_warp)(input_fn, reproj_fn_list[idx], src_srs, target_wkt,
                  #                      resolution, "bilinear"))
                   # out_list.append(reproj)
                #client = Client(threads_per_worker=2, n_workers=3 )
                #futures = dask.persist(*out_list)
                #out = dask.compute(*futures)
                #client.close()
                final_mos_list = Parallel(n_jobs=5)(delayed(dsm_functions.gdal_warp)(input_fn,reproj_fn_list[idx],
                    src_srs, target_wkt,resolution,"bilinear") for idx,input_fn  in enumerate(dem_list))
        else:
            print("No reprojection required")
            # rename the temp files to the final output names
            os.rename(dsm_mos_fn, dsm_reproj)
            os.rename(dtm_mos_no_fill_fn, dtm_no_fill_reproj)
            os.rename(dtm_mos_fill_fn, dtm_fill_reproj)
            os.rename(dtm_mos_fn, dtm_reproj)
            os.rename(intensity_mos_fn, intensity_reproj)
    else:
        dsm_reproj = dsm_mos_fn
        dtm_no_fill_reproj = dtm_mos_no_fill_fn
        dtm_fill_reproj = dtm_mos_fill_fn
        
        dtm_reproj = dtm_mos_fn
        intensity_reproj = intensity_mos_fn
    print("****Building Gaussian overviews for all rasters****") 
    if not parallel:
        print("Running overview creation sequentially")
        dsm_functions.gdal_add_overview(dsm_reproj)
        dsm_functions.gdal_add_overview(dtm_no_fill_reproj)
        dsm_functions.gdal_add_overview(dtm_fill_reproj)
        dsm_functions.gdal_add_overview(intensity_reproj)
    else:
        print("Running overview creation in parallel")
        ovr_list = [dsm_reproj, dtm_reproj, intensity_reproj]
        ovr_results = []    
        #for ovr in ovr_list:
        #    ovr_results.append(dask.delayed(dsm_functions.gdal_add_overview)(ovr))
        #client = Client(threads_per_worker=2, n_workers=3 )
        #futures = dask.persist(*ovr_results)
        #out = dask.compute(*futures)
        #client.close()
        final_mos_list = Parallel(n_jobs=5)(delayed(dsm_functions.gdal_add_overview)(
            ovr) for ovr in ovr_list)
    if cleanup:
        print("User selected to remove intermediate tile outputs")
        tile_list = final_dsm_fn_list + final_dtm_no_fill_fn_list + final_dtm_fill_fn_list + final_intensity_fn_list
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
            if os.path.exists(dtm_mos_no_fill_fn):
                os.remove(dtm_mos_no_fill_fn)
            if os.path.exists(dtm_mos_fill_fn):
                os.remove(dtm_mos_fill_fn)
            if os.path.exists(intensity_mos_fn):
                os.remove(intensity_mos_fn)
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

