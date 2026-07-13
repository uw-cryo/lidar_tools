import lidar_tools
import geopandas as gpd
from shapely.geometry import Polygon
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


def test_cleanup_intermediates_nested_layout(tmp_path):
    """Cleanup must empty the per-product tiles/ tree and pipelines/, remove
    the emptied dirs deepest-first, keep saved pointclouds (and their dir),
    and never touch finals/WKTs/metadata."""
    d = tmp_path
    keep = [
        d / "aoi-DSM_mos.tif",
        d / "UTM_12N_NAD83_2011_3D.wkt",
        d / "aoi-processing_metadata.yaml",
        d / "tiles" / "dsm" / "aoi_dsm_tile_aoi_000.laz.laz",  # saved pointcloud
    ]
    remove = [
        d / "tiles" / "dsm" / "aoi_dsm_tile_aoi_000.tif",
        d / "tiles" / "intensity" / "aoi_intensity_tile_aoi_000.tif",
        d / "tiles" / "cache" / "aoi_cache_tile_aoi_000.laz",
        d / "pipelines" / "pipeline_fetch_000.json",
        d / "pipelines" / "pipeline_dsm_intensity_000.json",
        d / "aoi-DSM_mos-temp.tif",
        d / "judicious_extent_polygon.geojson",
    ]
    for fn in keep + remove:
        fn.parent.mkdir(parents=True, exist_ok=True)
        fn.write_bytes(b"x")

    lidar_tools.pdal_pipeline._cleanup_intermediates(d)

    assert all(fn.exists() for fn in keep)
    assert not any(fn.exists() for fn in remove)
    # emptied subdirs removed; the dir holding the kept pointcloud survives
    assert not (d / "pipelines").exists()
    assert not (d / "tiles" / "intensity").exists()
    assert not (d / "tiles" / "cache").exists()
    assert (d / "tiles" / "dsm").exists()
