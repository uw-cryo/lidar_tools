import lidar_tools
import geopandas as gpd
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
            pointcloud_resolution=1,
            tile_size_km=1,
            buffer_value=5,
            return_specific_3dep_survey=None,
            return_all_intersecting_surveys=False,
        )
    )

    assert len(readers) == 12
    assert {"type", "filename", "requests", "resolution", "polygon"} == set(
        readers[0].keys()
    )
    assert isinstance(crslist[0], pyproj.CRS)
    assert buff_reader_extent_list[0] == (
        -13615921.0,
        6048225.0,
        -13614910.0,
        6049236.0,
    )
    assert original_dem_tile_grid_extent_list[0] == [
        -13615916.0,
        6048230.0,
        -13614915.0,
        6049231.0,
    ]


def _make_const_uint16_raster(fn, value=120, origin=(-13615000, 6045000), size=100, res=0.5):
    """Constant UInt16 raster in EPSG:3857 (default origin near Seattle, UTM 10N)."""
    from osgeo import gdal, osr

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(fn), size, size, 1, gdal.GDT_UInt16)
    ds.SetGeoTransform((origin[0], res, 0, origin[1], 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.full((size, size), value, np.uint16))
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None


def test_gdal_warp_2d_source_preserves_band_values(tmp_path):
    """Non-height rasters (intensity) must never be vertically datum-shifted.

    Warping with a horizontal-only source SRS into a 3D target CRS must leave
    band values untouched (a compound source SRS with a vertical datum makes
    gdal.Warp shift single-band values by the geoid undulation).
    """
    from osgeo import gdal

    src = tmp_path / "const_3857.tif"
    _make_const_uint16_raster(src)
    dst = tmp_path / "warped.tif"
    lidar_tools.dsm_functions.gdal_warp(
        str(src),
        str(dst),
        "EPSG:3857",
        "./notebooks/UTM_10N_WGS84_G2139_3D.wkt",
        res=0.5,
        dtype="UInt16",
    )
    arr = gdal.Open(str(dst)).ReadAsArray()
    interior = arr[10:-10, 10:-10]
    vals = np.unique(interior[interior != 0])
    assert list(vals) == [120]


def test_gdal_warp_tap_false_reproduces_exact_extent(tmp_path):
    """target_aligned_pixels=False must honor an out_extent whose origin is not
    a multiple of res (needed to match grids of recovered interrupted runs)."""
    from osgeo import gdal

    src = tmp_path / "const_3857.tif"
    _make_const_uint16_raster(src)
    first = tmp_path / "warp1.tif"
    lidar_tools.dsm_functions.gdal_warp(
        str(src), str(first), "EPSG:3857", "EPSG:32610", res=0.5, dtype="UInt16"
    )
    gt = gdal.Open(str(first)).GetGeoTransform()
    # request a grid deliberately offset from res multiples
    minx, maxy = gt[0] + 5.13, gt[3] - 5.13
    out_extent = [minx, maxy - 20.0, minx + 20.0, maxy]
    second = tmp_path / "warp2.tif"
    lidar_tools.dsm_functions.gdal_warp(
        str(src),
        str(second),
        "EPSG:3857",
        "EPSG:32610",
        res=0.5,
        dtype="UInt16",
        out_extent=out_extent,
        target_aligned_pixels=False,
    )
    ds = gdal.Open(str(second))
    gt2 = ds.GetGeoTransform()
    np.testing.assert_almost_equal(gt2[0], minx)
    np.testing.assert_almost_equal(gt2[3], maxy)
    assert (ds.RasterXSize, ds.RasterYSize) == (40, 40)


def test_datum_shift_required():
    import pytest

    N = -31.0  # local NAVD88->ellipsoid offset (e.g. Casa Grande)
    # small median offset vs COP30/EGM2008 => geoid-referenced, shift required
    assert lidar_tools.dsm_functions.datum_shift_required(
        0.4, valid_count=5000, expected_undulation=N
    )
    # offset matching the local undulation => already ellipsoidal, no shift
    assert not lidar_tools.dsm_functions.datum_shift_required(
        -30.2, valid_count=5000, expected_undulation=N
    )
    # matches NEITHER signature (terrain/snow/reference error): never guess.
    # A two-state check would have silently declared this "ellipsoidal".
    with pytest.raises(ValueError, match="neither"):
        lidar_tools.dsm_functions.datum_shift_required(
            -10.0, valid_count=5000, expected_undulation=N
        )
    with pytest.raises(ValueError, match="neither"):
        lidar_tools.dsm_functions.datum_shift_required(
            8.0, valid_count=5000, expected_undulation=N
        )
    # empty or tiny samples must never silently decide (silent ~30 m error)
    with pytest.raises(ValueError, match="datum check failed"):
        lidar_tools.dsm_functions.datum_shift_required(
            float("nan"), valid_count=0, expected_undulation=N
        )
    with pytest.raises(ValueError, match="datum check failed"):
        lidar_tools.dsm_functions.datum_shift_required(
            0.1, valid_count=10, expected_undulation=N
        )


def test_raise_file_limit():
    import resource

    soft_before, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    result = lidar_tools.dsm_functions.raise_file_limit()
    assert result == -1 or result >= soft_before


def _make_test_laz(fn, epsg=32611):
    import pdal

    arr = np.zeros(
        10, dtype=[("X", np.float64), ("Y", np.float64), ("Z", np.float64)]
    )
    arr["X"] = np.linspace(500000, 500100, 10)
    arr["Y"] = np.linspace(4000000, 4000100, 10)
    arr["Z"] = 100.0
    pipeline = pdal.Writer.las(
        filename=str(fn), a_srs=f"EPSG:{epsg}", forward="all"
    ).pipeline(arr)
    pipeline.execute()


def test_return_lpc_bounds(tmp_path):
    laz = tmp_path / "test.laz"
    _make_test_laz(laz)
    # no output_crs: native bounds
    bounds = lidar_tools.dsm_functions.return_lpc_bounds(str(laz))
    np.testing.assert_allclose(bounds, [500000, 4000000, 500100, 4000100], atol=0.01)
    # output_crs equal to the point cloud CRS (previously UnboundLocalError)
    bounds = lidar_tools.dsm_functions.return_lpc_bounds(
        str(laz), output_crs=pyproj.CRS.from_epsg(32611)
    )
    np.testing.assert_allclose(bounds, [500000, 4000000, 500100, 4000100], atol=0.01)
    # output_crs differing: bounds transformed (UTM 11N -> lon/lat)
    bounds = lidar_tools.dsm_functions.return_lpc_bounds(
        str(laz), output_crs=pyproj.CRS.from_epsg(4326)
    )
    # easting 500000 in UTM 11N is exactly the -117 deg central meridian
    assert -117.01 < bounds[0] < -116.99 and 36 < bounds[1] < 37
