"""
Generate a DSM,DTM,Intensity rasters from input point clouds
"""

# Needs to happen before importing GDAL/PDAL
import os
from dask.distributed import Client, progress

os.environ["PROJ_NETWORK"] = (
    "ON"  # Ensure this is 'ON' to get shift grids over the internet
)
# print(f"PROJ_NETWORK is {os.environ['PROJ_NETWORK']}")

from lidar_tools import dsm_functions
from pyproj import CRS
from shapely.geometry.polygon import orient as _orient
import numpy as np
from pathlib import Path
import warnings
from typing import Literal, Annotated
import geopandas as gpd
import requests
import cyclopts
import shutil


def rasterize(
    geometry: str,
    input: str = "EPT_AWS",
    output: str = "/tmp/lidar-tools-output",
    src_crs: str = None,
    dst_crs: str = None,
    resolution: float = 1.0,
    dsm_gridding_choice: Literal["first_idw", "n-pct"] = "first_idw",
    products: Literal["all", "dsm", "dtm", "intensity"] = "all",
    threedep_project: Literal["all", "latest"] | str = "latest",
    tile_size: float = 1.0,
    num_process: int = 1,
    overwrite: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    cleanup: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    proj_pipeline: str = None,
    filter_noise: bool = True,
    height_above_ground_threshold: float = None,
) -> None:
    """
    Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and/or Intensity raster from point cloud data.

    Parameters
    ----------
    geometry
        Path to the vector dataset containing a single polygon that defines the processing extent.
    input
        Path to directory containing input LAS/LAZ files, otherwise uses USGS 3DEP EPT data on AWS.
    output
        Path to output directory.
    src_crs
        Path to file with PROJ-supported CRS definition to override CRS of input files.
    dst_crs
        Path to file with PROJ-supported CRS definition for the output. If unspecified, a local UTM CRS will be used.
    resolution
        Square output raster posting in units of `dst_crs`.
    dsm_gridding_choice
        The gridding method to use for DSM generation. 'first_idw' uses the first and only returns which are gridded using IDW, 'n-pct' computes points matching the nth percentile in a pointview (e.g., 98-pct), which are gridded using the max binning operator.
    products
        Which output products to generate: all products, digital surface model, digital terrain model, or intensity raster.
    threedep_project
        "all" processes all available 3DEP EPT point clouds which intersect with the input polygon.
        "first" 3DEP project encountered will be processed.
        "specific" should be a string that matches the "project" name in the 3DEP metadata.
    tile_size
        The size of rasterized tiles processed from input EPT point clouds in units of `dst_crs`.
    num_processes
        Number of processes to run PDAL pipelines in parallel.
    overwrite
        Overwrite output files if they already exist.
    cleanup
        Remove the intermediate tif files, keep only final mosaiced rasters.
    proj_pipeline
        A PROJ pipeline string to be used for reprojection of the point cloud. If specified, this will be used in combination with the target_wkt option.
    local_utm
        If true, automatically compute the local UTM zone from the extent polygon for final output products. If false, use the CRS defined in the target_wkt file.
    filter_noise
        Remove noise points (classification==18 and classification==7) from the point cloud before DSM, DTM and surface intensity processing. Default is True.
    height_above_ground_threshold
        If specified, the height above ground (HAG) will be calculated using all nearest ground classied points, and all points greater than this value will be classified as noise, by default None.

    Returns
    -------
    None
    """
    # Parse input polygon CRS and check that area isn't too large
    gdf = gpd.read_file(geometry)
    _check_polygon_area(gdf)
    input_crs = gdf.crs.to_wkt()

    outdir = Path(output)
    if outdir.exists():
        if overwrite:
            print(f"Overwriting existing output path: {outdir}")
            if outdir.is_file():
                outdir.unlink()
            elif outdir.is_dir():
                shutil.rmtree(outdir)
        else:
            raise FileExistsError(
                f"Output directory {outdir} already exists. Use --overwrite to allow overwriting."
            )

    # Set output filename prefix based on input polygon name
    outdir.mkdir(parents=True)
    output_prefix = outdir / Path(geometry).stem

    # Create custom 3D CRS UTM WKT2 with WGS84 G2139 datum realization
    if dst_crs is None:
        gdf = gpd.read_file(geometry)
        epsg_code = gdf.estimate_utm_crs().to_epsg()
        identifier_ns = str(epsg_code)[:3]
        identifier_zone = str(epsg_code)[3:]
        if identifier_ns == "326":
            zone = identifier_zone + "N"
        else:
            zone = identifier_zone + "S"
        target_wkt = outdir / f"UTM_{zone}_WGS84_G2139_3D.wkt"
        path_to_base_utm10_def = outdir / "UTM_10.wkt"
        # TODO: replace with local copy of file
        url = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/UTM_10N_WGS84_G2139_3D.wkt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(path_to_base_utm10_def, "w") as f:
                f.write(response.text)
        dst_crs = dsm_functions.write_local_utm_3DCRS_G2139(
            path_to_base_utm10_def, zone=zone, outfn=target_wkt
        )

    # Configure output raster extents and posting based on input polygon
    with open(dst_crs, "r") as f:
        contents = f.read()
        out_crs = CRS.from_string(contents)
    out_extent = gdf.to_crs(out_crs).total_bounds
    final_out_extent = dsm_functions.tap_bounds(out_extent, res=resolution)
    #fix extent precision with respect to input resolution
    #from https://www.reddit.com/r/pythontips/comments/zw5ana/how_to_count_decimal_places/
    import decimal
    d = decimal.Decimal(str(resolution))
    precision = abs(d.as_tuple().exponent)
    final_out_extent = [np.round(val,precision) for val in final_out_extent]
    
    # TODO: simplify and use tempfile (https://github.com/uw-cryo/lidar_tools/pull/25#discussion_r2177660328)
    # TODO: here and elsewhere use logging instead of prints
    print(f"Output extent in target CRS is {final_out_extent}")
    gdf_out = gdf.to_crs(out_crs)
    gdf_out["geometry"] = gdf_out["geometry"].buffer(250)  # NOTE: assumes meters
    gdf_out = gdf_out.to_crs(input_crs)
    extent_polygon = outdir / "judicious_extent_polygon.geojson"
    gdf_out.to_file(extent_polygon, driver="GeoJSON")

    # How to handle AOIs intersecting multiple 3DEP projects?
    if threedep_project == "all":
        process_all_intersecting_surveys = True
        process_specific_3dep_survey = None
    elif threedep_project == "latest":
        process_all_intersecting_surveys = False
        process_specific_3dep_survey = None
    else:
        process_all_intersecting_surveys = False
        process_specific_3dep_survey = threedep_project

    if filter_noise:
        filter_high_noise = True
        filter_low_noise = True
    else:
        filter_high_noise = False
        filter_low_noise = False
    # TODO: create EPT for local laz for common workflow? https://github.com/uw-cryo/lidar_tools/issues/14#issuecomment-3076045321
    # SB note: The main reason for seperate EPT and local laz pipelines is the difference in projection handling, not much due to difference in file formats.
    if input == "EPT_AWS":
        print("Processing 3DEP EPT tiles from AWS")
        # TODO: handle new positional args, skip products not requested
        
        (
            dsm_pipeline_list,
            dtm_no_fill_pipeline_list,
            dtm_fill_pipeline_list,
            intensity_pipeline_list,
        ) = dsm_functions.create_ept_3dep_pipeline(
            extent_polygon,
            dst_crs,
            output_prefix,
            buffer_value=10*resolution, # buffer is based on output resolution
            tile_size_km=tile_size,  # TODO: ensure we can do non-km units
            # TODO: handle new 3dep project keyword here
            dsm_gridding_choice=dsm_gridding_choice,
            process_specific_3dep_survey=process_specific_3dep_survey,
            process_all_intersecting_surveys=process_all_intersecting_surveys,
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            hag_nn=height_above_ground_threshold,
            raster_resolution=resolution
        )
    else:
        print(f"Processing local laz files from {input}")
        if src_crs:
            with open(src_crs, "r") as f:
                contents = f.read()
                src_projcrs = CRS.from_string(contents)
        else:
            src_projcrs = None
        print(src_projcrs)
        (
            dsm_pipeline_list,
            dtm_no_fill_pipeline_list,
            dtm_fill_pipeline_list,
            intensity_pipeline_list,
        ) = dsm_functions.create_lpc_pipeline(
            local_laz_dir=input,
            input_crs=src_projcrs,
            target_wkt=dst_crs,
            output_prefix=output_prefix,
            extent_polygon=extent_polygon,
            dsm_gridding_choice=dsm_gridding_choice,
            buffer_value=10*resolution, # buffer is based on output resolution
            proj_pipeline=proj_pipeline,
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            hag_nn=height_above_ground_threshold,
            raster_resolution=resolution
        )

    # TODO: refactor into function
    num_pipelines = len(dsm_pipeline_list)
    if num_process == 1:
        print(f"Executing PDAL in serial for products={products}")

        if products == "all" or products == "dsm":
            print("Generating DSM tiles")
            final_dsm_fn_list = []
            for pipeline in dsm_pipeline_list:
                outfn = dsm_functions.execute_pdal_pipeline(pipeline)
                if outfn is not None:
                    final_dsm_fn_list.append(outfn)

        if products == "all" or products == "dtm":
            print("Generating DTM tiles")
            final_dtm_no_fill_fn_list = []
            for pipeline in dtm_no_fill_pipeline_list:
                outfn = dsm_functions.execute_pdal_pipeline(pipeline)
                if outfn is not None:
                    final_dtm_no_fill_fn_list.append(outfn)

            final_dtm_fill_fn_list = []
            for pipeline in dtm_fill_pipeline_list:
                outfn = dsm_functions.execute_pdal_pipeline(pipeline)
                if outfn is not None:
                    final_dtm_fill_fn_list.append(outfn)

        if products == "all" or products == "intensity":
            print("Generating Intensity tiles")
            final_intensity_fn_list = []
            for pipeline in intensity_pipeline_list:
                outfn = dsm_functions.execute_pdal_pipeline(pipeline)
                if outfn is not None:
                    final_intensity_fn_list.append(outfn)

    else:
        n_jobs = num_process if num_pipelines > num_process else num_pipelines

        def run_parallel(pipeline_list):
            with Client(threads_per_worker=2, n_workers=n_jobs) as client:
                futures = client.map(dsm_functions.execute_pdal_pipeline, pipeline_list)
                progress(futures)
                results = client.gather(futures)
                return [outfn for outfn in results if outfn is not None]

        print(
            f"Executing PDAL in parallel with dask n_workers={n_jobs} for products={products}"
        )
        if products == "all" or products == "dsm":
            print("Generating DSM tiles")
            final_dsm_fn_list = run_parallel(dsm_pipeline_list)

        if products == "all" or products == "dtm":
            print("Generating DTM tiles")
            final_dtm_no_fill_fn_list = run_parallel(dtm_no_fill_pipeline_list)
            final_dtm_fill_fn_list = run_parallel(dtm_fill_pipeline_list)

        if products == "all" or products == "intensity":
            print("Generating Intensity tiles")
            final_intensity_fn_list = run_parallel(intensity_pipeline_list)

    print("****Processing complete for all tiles****")

    # Mosaicing
    # ===========
    dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
    dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
    dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
    intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"

    if num_pipelines > 1:
        print(
            f"Multiple tiles created: {num_pipelines}. Mosaicing required to create final rasters"
        )
        print("*** Now creating raster composites ***")
        if input == "EPT_AWS":
            cog = False
            out_extent = None
        else:
            out_extent = final_out_extent
            cog = True
            
        print("Running sequentially")
        if products == "all" or products == "dsm":
            print(f"Creating DSM mosaic at {dsm_mos_fn}")
            dsm_functions.raster_mosaic(
                final_dsm_fn_list, dsm_mos_fn, cog=cog, out_extent=out_extent
            )

        if products == "all" or products == "dtm":
            print(f"Creating DTM mosaic at {dtm_mos_no_fill_fn}")
            dsm_functions.raster_mosaic(
                final_dtm_no_fill_fn_list,
                dtm_mos_no_fill_fn,
                cog=cog,
                out_extent=out_extent,
            )

            print(f"Creating DTM mosaic with window size 4 at {dtm_mos_fill_fn}")
            dsm_functions.raster_mosaic(
                final_dtm_fill_fn_list, dtm_mos_fill_fn, cog=cog, out_extent=out_extent
            )

        if products == "all" or products == "intensity":
            print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
            dsm_functions.raster_mosaic(
                final_intensity_fn_list,
                intensity_mos_fn,
                cog=cog,
                out_extent=out_extent,
            )

    else:
        print("Only one tile created, no mosaicing required")
        if products == "all" or products == "dsm":
            dsm_functions.rename_rasters(final_dsm_fn_list[0], dsm_mos_fn)
        if products == "all" or products == "dtm":
            dsm_functions.rename_rasters(
                final_dtm_no_fill_fn_list[0], dtm_mos_no_fill_fn
            )
            dsm_functions.rename_rasters(final_dtm_fill_fn_list[0], dtm_mos_fill_fn)
        if products == "all" or products == "intensity":
            dsm_functions.rename_rasters(final_intensity_fn_list[0], intensity_mos_fn)

    # Reprojection
    # ============
    dsm_reproj = dsm_mos_fn.split("-temp.tif")[0] + ".tif"
    dtm_no_fill_reproj = dtm_mos_no_fill_fn.split("-temp.tif")[0] + ".tif"
    dtm_fill_reproj = dtm_mos_fill_fn.split("-temp.tif")[0] + ".tif"
    intensity_reproj = intensity_mos_fn.split("-temp.tif")[0] + ".tif"

    if input == "EPT_AWS":
        if out_crs != CRS.from_epsg(3857):
            print("*********Reprojecting rasters****")
            src_srs = "EPSG:3857"
            #This is hardcoded for dsm_mos_fn, but we could have dtm fn
            reproject_truth_val = False
            if products == "all" or products == "dsm":
                reproject_truth_val = dsm_functions.confirm_3dep_vertical(dsm_mos_fn)
            elif products == "dtm":
                reproject_truth_val = dtm_functions.confirm_3dep_vertical(dtm_mos_fill_fn)
            if reproject_truth_val:
                # use input CRS which is EPSG:3857 with heights with respect to the NAVD88
                epsg_3857_navd88_fn = "https://raw.githubusercontent.com/uw-cryo/lidar_tools/refs/heads/main/notebooks/SRS_CRS.wkt"
                src_srs = epsg_3857_navd88_fn
            out_extent = final_out_extent
            print(src_srs)
            print("Running reprojection sequentially")
            if products == "all" or products == "dsm":
                print("Reprojecting DSM raster")
                dsm_functions.gdal_warp(
                    dsm_mos_fn,
                    dsm_reproj,
                    src_srs,
                    dst_crs,
                    res=resolution,
                    resampling_alogrithm="bilinear",
                    out_extent=out_extent,
                )
            if products == "all" or products == "dtm":
                print("Reprojecting DTM raster")
                dsm_functions.gdal_warp(
                    dtm_mos_no_fill_fn,
                    dtm_no_fill_reproj,
                    src_srs,
                    dst_crs,
                    res=resolution,
                    resampling_alogrithm="bilinear",
                    out_extent=out_extent,
                )
                dsm_functions.gdal_warp(
                    dtm_mos_fill_fn,
                    dtm_fill_reproj,
                    src_srs,
                    dst_crs,
                    res=resolution,
                    resampling_alogrithm="bilinear",
                    out_extent=out_extent,
                )
            if products == "all" or products == "intensity":
                print("Reprojecting intensity raster")
                dsm_functions.gdal_warp(
                    intensity_mos_fn,
                    intensity_reproj,
                    src_srs,
                    dst_crs,
                    res=resolution,
                    resampling_alogrithm="bilinear",
                    out_extent=out_extent,
                )

    else:
        print("No reprojection required")
        # rename the temp files to the final output names
        if products == "all" or products == "dsm":
            dsm_functions.rename_rasters(dsm_mos_fn, dsm_reproj)
        if products == "all" or products == "dtm":
            dsm_functions.rename_rasters(dtm_mos_no_fill_fn, dtm_no_fill_reproj)
            dsm_functions.rename_rasters(dtm_mos_fill_fn, dtm_fill_reproj)
        if products == "all" or products == "intensity":
            dsm_functions.rename_rasters(intensity_mos_fn, intensity_reproj)

    print("****Building Gaussian overviews for all rasters****")
    print("Running overview creation sequentially")
    if products == "all" or products == "dsm":
        dsm_functions.gdal_add_overview(dsm_reproj)
    if products == "all" or products == "dtm":
        dsm_functions.gdal_add_overview(dtm_no_fill_reproj)
        dsm_functions.gdal_add_overview(dtm_fill_reproj)
    if products == "all" or products == "intensity":
        dsm_functions.gdal_add_overview(intensity_reproj)

    if cleanup:
        for tif_file in outdir.glob("*tile*.tif*"):
            tif_file.unlink()

    print("****Processing complete****")


def geographic_area(gf: gpd.GeoDataFrame) -> gpd.pd.Series:
    """
    Estimate the geographic area of each polygon in a GeoDataFrame in m^2

    Parameters
    ----------
    gf
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
    -----
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

    Parameters
    ----------
    gf
        A GeoDataFrame containing a polygon

    Returns
    -------
    None
        Just prints a warning if area is too large
    """
    warn_if_larger_than = 100_000  # km^2

    # Fast track if projected and units are meters:
    if gf.crs.is_projected and gf.crs.axis_info[0].unit_name == "metre":
        area = gf.area * 1e-6
    else:
        area = geographic_area(gf.to_crs("EPSG:4326")) * 1e-6

    # print(area.values[0])
    if area.to_numpy() >= warn_if_larger_than:
        msg = f"Very large AOI ({area.values[0]:e} km^2) requested, processing may be slow or crash. Recommended AOI size is <{warn_if_larger_than:e} km^2"
        warnings.warn(msg)
    else:
        print(f"Starting Processing of {area.values[0]:e} km^2 AOI")
