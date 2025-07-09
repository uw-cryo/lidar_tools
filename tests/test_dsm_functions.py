import lidar_tools
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
import numpy as np


def test_nearest_floor():
    result = lidar_tools.dsm_functions.nearest_floor(10, 4)
    expected = 8

    assert result == expected


def test_nearest_ceil():
    result = lidar_tools.dsm_functions.nearest_ceil(10, 4)
    expected = 12

    assert result == expected


def test_tap_bounds_floats():
    # Check that extent mins rounded down and extent max rounded up
    bounds = [47.08, -124.79, 49.05, -117.02]
    resolution = 0.1
    result = lidar_tools.dsm_functions.tap_bounds(bounds, resolution)
    expected = np.array([47.0, -124.8, 49.1, -117.0])

    np.testing.assert_almost_equal(result, expected)


def test_tap_bounds_integers():
    bounds = [363111, 4290903, 374892, 4298725]
    resolution = 2
    result = lidar_tools.dsm_functions.tap_bounds(bounds, resolution)
    expected = np.array([363110, 4290902, 374892, 4298726])

    np.testing.assert_almost_equal(result, expected)


# Requires network
def test_return_readers():
    extent_geojson = "./notebooks/uw-campus.geojson"
    gf = gpd.read_file(extent_geojson)
    readers, crslist, buff_reader_extent_list, original_dem_tile_grid_extent_list = (
        lidar_tools.dsm_functions.return_readers(
            gf,
            pointcloud_resolution=10,
            tile_size_km=1,
            buffer_value=5,
            return_specific_3dep_survey=None,
            return_all_intersecting_surveys=False)
    )

    assert len(readers) == 16
    assert {"type", "filename", "resolution", "polygon"} == set(readers[0].keys())
    assert isinstance(crslist[0], pyproj.CRS)
    assert buff_reader_extent_list[0] == (
        -13616292.0,
        6047853.0,
        -13615281.0,
        6048864.0,
    )
    assert original_dem_tile_grid_extent_list[0] == [
        -13616287.0,
        6047858.0,
        -13615286.0,
        6048859.0,
    ]
