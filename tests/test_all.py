import lidar_tools
import pytest
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj

# Requires network
def test_return_readers():
    extent_geojson = "./notebooks/CO_LiDAR_test_area_final.geojson"
    gf = gpd.read_file(extent_geojson)
    xmin, ymin, xmax, ymax = gf.iloc[0].geometry.bounds
    input_aoi = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    input_crs = gf.crs.to_wkt()
    readers, crslist, buff_reader_extent_list, original_dem_tile_grid_extent_list  = lidar_tools.dsm_functions.return_readers(input_aoi,
                                                                input_crs,
                                                                pointcloud_resolution=10,
                                                                n_rows=2,
                                                                n_cols=2,
                                                                buffer_value=5)
    assert len(readers) == 4
    assert isinstance(crslist[0], pyproj.CRS, buff_reader_extent_list[0], original_dem_tile_grid_extent_list[0])[0]
