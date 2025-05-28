import lidar_tools
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
import numpy as np
import pytest


@pytest.fixture
def small_aoi(scope="package"):
    # 5 vertices bounding box around UW Campus 3.5 km^2
    aoi_url = "./notebooks/uw-campus.geojson"
    return gpd.read_file(aoi_url)


def test_check_geographic_area(small_aoi):
    area = lidar_tools.pdal_pipeline.geographic_area(small_aoi)
    expected = 3488698
    actual = area.to_numpy().astype(np.int32)
    assert expected == actual


def test_check_large_aoi_warns_geographic():
    # NOTE: computation only works for a hemisphere
    bounds = (0, -90, 180.0, 90)
    polygon = Polygon.from_bounds(*bounds)
    gf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
    with pytest.warns(match="Very large AOI"):
        lidar_tools.pdal_pipeline._check_polygon_area(gf)


def test_check_large_aoi_warns_projected():
    # NOTE: computation only works for a hemisphere
    bounds = (0, 0, 100_001e3, 1e3)
    polygon = Polygon.from_bounds(*bounds)
    gf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:32610")
    with pytest.warns(match="Very large AOI"):
        lidar_tools.pdal_pipeline._check_polygon_area(gf)
