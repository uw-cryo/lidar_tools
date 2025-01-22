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

# "first,only" will return top of canopy, "last,only" will return lowest return
# set to None to use default values
GROUP_FILTER="first,only"

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
               output_path: str, # /tmp/dem/
               ) -> None:
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
    output_path : str
        Directory path where the output DSM files will be saved.

    Returns
    -------
    None

    Notes
    -----
    After running this function, create a final DEM with the following commands:
    1. dem_moaic -o merged_dsm.tif dem*.tif
    2. gdalwarp -s_srs SRS_CRS.wkt -t_srs UTM_13N_WGS84_G2139_3D.wkt -r cubic -tr 1.0 1.0 merged_dsm.tif merged_dsm_reprojected_UTM_13N_WGS84_G2139.tif
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


    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    print(f"Number of readers: {len(readers)}")
    for i, reader in enumerate(readers):
        print(f"Processing reader #{i}")
        dem_file = output_path / f'dem_tile_aoi_{str(i).zfill(4)}.tif'
        pipeline = {'pipeline':[reader]}

        pdal_pipeline_dsm = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=RETURN_ONLY_GROUND,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter=GROUP_FILTER,
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )

        pdal_pipeline_dtm = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=RETURN_ONLY_GROUND,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter=last,only,
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )
        pdal_pipeline_surface_intesity = dsm_functions.create_pdal_pipeline(
            filter_low_noise=FILTER_LOW_NOISE,
            filter_high_noise=FILTER_HIGH_NOISE,
            filter_road=FILTER_ROAD,
            reset_classes=RESET_CLASSES, reclassify_ground=RECLASSIFY_GROUND,
            return_only_ground=RETURN_ONLY_GROUND,
            percentile_filter=PERCENTILE_FILTER, percentile_threshold=PERCENTILE_THRESHOLD,
            group_filter=last,only,
            reproject=REPROJECT,
            save_pointcloud=SAVE_POINTCLOUD,
            pointcloud_file='pointcloud',
            input_crs = POINTCLOUD_CRS[i],
            output_crs=OUTPUT_CRS,
            output_type=OUTPUT_TYPE
        )
        dem_stage = dsm_functions.create_dem_stage(dem_filename=str(dem_file),
                                        pointcloud_resolution=POINTCLOUD_RESOLUTION,
                                        gridmethod=GRID_METHOD, dimension=DIMENSION)

        # this is only required for the DTM, since this function is for a DSM, I am commenting this out
        #dem_stage[0]['window_size'] = 4

        pipeline['pipeline'] += pdal_pipeline
        pipeline['pipeline'] += dem_stage

        # Save a copy of each pipeline
        pipeline_config_fn = os.path.join(output_path,f"pipeline_{i}.json")
        
        with open(pipeline_config_fn, 'w') as f:
            f.write(json.dumps(pipeline))

        # Skip execution for testing
        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()
