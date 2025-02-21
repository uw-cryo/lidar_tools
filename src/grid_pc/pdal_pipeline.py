"""
Generate a DSM from input polygon
"""
import os
from grid_pc import dsm_functions
import pdal
from shapely.geometry import Polygon
import geopandas as gpd
import json
from pathlib import Path

# NOTE: Hardcoding global settings for now, can expose as script arguments later
# -------------------------------------------------------
# Set pointcloud processing parameters
FILTER_LOW_NOISE = False
FILTER_HIGH_NOISE = False
FILTER_ROAD = False
RETURN_ONLY_GROUND = False # Set true for DTM
RESET_CLASSES = False
RECLASSIFY_GROUND = False
PERCENTILE_FILTER = False # Set to True to apply percentile based filtering of Z values
PERCENTILE_THRESHOLD = 0.95 # Percentile value to filter out noisy Z returns



REPROJECT = False
SAVE_POINTCLOUD=False
POINTCLOUD_RESOLUTION = 1
OUTPUT_TYPE='laz'
GRID_METHOD='idw'
DIMENSION='Z' # can be set to options accepted by writers.gdal. Set to 'intensity' to return intensity rasters

# -------------------------------------------------------


def create_dsm(extent_geojson: str, # processing_extent.geojson
               source_wkt: str | None, # SRS_CRS.wkt
               target_wkt: str, # UTM_13N_WGS84_G2139_3D.wkt
               output_prefix: str, # CO_ALS_proc/CO_3DEP_ALS
               mosaic: bool | True,
               cleanup: bool | True) -> None:
    """
    Create a Digital Surface Model (DSM) from a given extent and point cloud data.
    This function divides a region of interest into tiles and generates a DSM geotiff from EPT point clouds for each tile using PDAL

    Parameters
    ----------
    extent_geojson : str
        Path to the GeoJSON file defining the processing extent.
    source_wkt : str or None
        Path to the WKT file defining the source coordinate reference system (CRS). If None, the CRS from the point cloud file is used.
    target_wkt : str
        Path to the WKT file defining the target coordinate reference system (CRS).
    output_prefix : str
        prefix with directory name and filename prefix for the project (e.g., CO_ALS_proc/CO_3DEP_ALS)
    mosaic : bool
        Mosaic the output tiles using a weighted average algorithm
    cleanup: bool
        If true, remove the intermediate tif files for the output tiles
    Returns
    -------
    None

    Notes
    -----
    After running this function, reproject the final DEM with the following commands:
    1. gdalwarp -s_srs SRS_CRS.wkt -t_srs UTM_13N_WGS84_G2139_3D.wkt -r cubic -tr 1.0 1.0 merged_dsm.tif merged_dsm_reprojected_UTM_13N_WGS84_G2139.tif
    None
    """

    # bounds for which pointcloud is created
    gdf = gpd.read_file(extent_geojson)
    xmin, ymin, xmax, ymax = gdf.iloc[0].geometry.bounds

    input_aoi = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    input_crs = gdf.crs.to_wkt()

    # specify the output CRS of DEMs
    with open(target_wkt, 'r') as f:
        OUTPUT_CRS = ' '.join(f.read().replace('\n', '').split())

    # The method returns pointcloud readers, as well as the pointcloud file CRS as a WKT string
    # Specfying a buffer_value > 0 will generate overlapping DEM tiles, resulting in a seamless
    # final mosaicked DEM
    readers, POINTCLOUD_CRS = dsm_functions.return_readers(input_aoi, input_crs,
    pointcloud_resolution = 1, n_rows=5, n_cols=5, buffer_value=0)

    # NOTE: if source_wkt is passed, override POINTCLOUD_CRSs
    if source_wkt:
        with open(source_wkt, 'r') as f:
            src_wkt = f.read()
        POINTCLOUD_CRS = [src_wkt for _ in range(len(readers))]

    output_path = os.dirname(output_prefix)
    prefix = os.path.basename(output_prefix)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    print(f"Number of readers: {len(readers)}")
    dsm_fn_list = []
    dtm_fn_list = []
    intensity_fn_list = []
    for i, reader in enumerate(readers):
        print(f"Processing reader #{i}")
        dsm_file = output_path / f'{prefix}_dsm_tile_aoi_{str(i).zfill(4)}.tif'
        dtm_file = output_path / f'{prefix}_dtm_tile_aoi_{str(i).zfill(4)}.tif'
        intensity_file = output_path / f'{prefix}_intensity_tile_aoi_{str(i).zfill(4)}.tif'
        dsm_fn_list.append(dsm_file)
        dtm_fn_list.append(dtm_file)
        intensity_fn_list.append(intensity_file)
        
        ## DSM creation block
        pipeline_dsm = {'pipeline':[reader]}

        pdal_pipeline_dsm = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=RETURN_ONLY_GROUND,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter="first,only",
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )
        dsm_stage = dsm_functions.create_dem_stage(dem_filename=str(dsm_file),
                                        pointcloud_resolution=POINTCLOUD_RESOLUTION,
                                        gridmethod=GRID_METHOD, dimension='Z')
        pipeline_dsm['pipeline'] += pdal_pipeline_dsm
        pipeline_dsm['pipeline'] += dsm_stage
        
        # Save a copy of each pipeline
        dsm_pipeline_config_fn = os.path.join(output_path,f"pipeline_dsm_{str(i).zfill(4)}.json")
        with open(dsm_pipeline_config_fn, 'w') as f:
            f.write(json.dumps(pipeline_dsm))
        pipeline_dsm = pdal.Pipeline(json.dumps(pipeline_dsm))
        pipeline_dsm.execute()


        ## DTM creation block
        pipeline_dtm = {'pipeline':[reader]}
        pdal_pipeline_dtm = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=True,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter=None,
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )
        
        dtm_stage = dsm_functions.create_dem_stage(dem_filename=str(dtm_file),
                                        pointcloud_resolution=POINTCLOUD_RESOLUTION,
                                        gridmethod=GRID_METHOD, dimension='Z')
        # this is only required for the DTM
        dtm_stage[0]['window_size'] = 4
        
        pipeline_dtm['pipeline'] += pdal_pipeline_dtm
        pipeline_dtm['pipeline'] += dtm_stage

        # Save a copy of each pipeline
        dtm_pipeline_config_fn = os.path.join(output_path,f"pipeline_dtm_{str(i).zfill(4)}.json")
        with open(dtm_pipeline_config_fn, 'w') as f:
            f.write(json.dumps(pipeline_dtm))
        pipeline_dtm = pdal.Pipeline(json.dumps(pipeline_dtm))
        pipeline_dtm.execute()


        ## Intensity pipeline
        pipeline_intensity = {'pipeline':[reader]}
        pdal_pipeline_surface_intensity = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=False,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter="first,only",
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )

        intensity_stage = dsm_functions.create_dem_stage(dem_filename=str(intensity_file),
                                        pointcloud_resolution=POINTCLOUD_RESOLUTION,
                                        gridmethod=GRID_METHOD, dimension='Intensity')


        
        pipeline_intensity['pipeline'] += pdal_pipeline_surface_intensity
        pipeline_intensity['pipeline'] += intensity_stage

        
        # Save a copy of each pipeline
        intensity_pipeline_config_fn = os.path.join(output_path,f"pipeline_intensity_{str(i).zfill(4)}.json")
        with open(intensity_pipeline_config_fn, 'w') as f:
            f.write(json.dumps(pipeline_intensity))
        pipeline_intensity = pdal.Pipeline(json.dumps(pipeline_intensity))
        pipeline_intensity.execute()

    if mosaic:
        print("*** Now creating raster composites ***")
        dsm_mos_fn = f"{out_prefix}-DSM_mos.tif"
        print(f"Creating DSM mosaic at {dsm_mos_fn}")
        _ = dsm_functions.dem_mosaic(dsm_fn_list,dsm_mos_fn)
        dtm_mos_fn = f"{out_prefix}-DTM_mos.tif"
        print(f"Creating DTM mosaic at {dtm_mos_fn}")
        _ = dsm_functions.dem_mosaic(dtm_fn_list,dtm_mos_fn)
        intensity_mos_fn = f"{out_prefix}-Intensity_mos.tif"
        print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
        _ = dsm_functions.dem_mosaic(intensity_fn_list,intensity_mos_fn)
        if cleanup:
            print("User selected to remove intermediate tile outputs")
            for fn in dsm_fn_list + dtm_fn_list + intensity_fn_list:
                os.remove(fn)
        
                

