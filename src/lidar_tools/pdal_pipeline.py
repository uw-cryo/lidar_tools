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
from typing import Literal, Annotated
import geopandas as gpd
import requests
import cyclopts

def rasterize(
    extent_polygon: str,
    output: str,
    input: str = None,
    src_crs: str = None,
    dst_crs: str = None,
    posting: float = 1.0,
    products: Literal["all","dsm","dtm","intensity"] = "all",
    threedep_projects: Literal["all","latest"] | str = "all",
    tile_size: float = 1.0,
    num_process: int = 1,
    use_bbox: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    cleanup: Annotated[bool, cyclopts.Parameter(negative="")] = False,
) -> None:
    """
    Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and/or Intensity raster from point cloud data.

    Parameters
    ----------
    extent_polygon :
        Path to the vector dataset containing a single polygon that defines the processing extent.
    output :
        Path to output directory (e.g., /tmp/CO_3DEP_ALS/).
    input:
        Path to directory containing laz point cloud files. If unspecified, the program will use USGS 3DEP EPT data on AWS.
    dst_crs :
        Path to file with PROJ-supported CRS definition for the output. If unspecified, a local UTM CRS will be used.
    src_crs :
        Path to file with PROJ-supported CRS definition for the coordinate reference system (CRS) of the input point cloud. If unspecified, the CRS defined in the source point cloud metadata will be used.
    posting :
        Output raster resolution in units of `dst_crs`.
    products :
        Which output products to generate: all products, digital surface model, digital terrain model, or intensity raster.
    threedep_projects :
        "all" processes all available 3DEP EPT point clouds which intersect with the input polygon. "first" 3DEP project encountered will be processed. "specific" should be a string that matches the "project" name in the 3DEP metadata.
    tile_size:
        The size of rasterized tiles processed from input EPT point clouds in units of `dst_crs`.
    num_process:
        Number of processes to use for parallel processing. Default is 1, which means all pdal and gdal processing will be done serially
    use_bbox :
        Use the bounding box of the input polygon for processing instead of the polygon itself.
    cleanup:
        Remove the intermediate tif files for the output tiles, leaving only the final mosaicked rasters.

    Returns
    -------
    None

    """
    #figure out output projection
    #if user selectes local_utm, then compute the UTM zone from the extent polygon
    #this will supersed the target_wkt option
    if target_wkt is None:
        local_utm = True

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
        path_to_base_utm10_def =  outdir / 'UTM_10.wkt'
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
                                    extent_polygon=extent_polygon,buffer_value=5)

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

