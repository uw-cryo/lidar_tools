from pathlib import Path

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


def test_check_raster_validity_deep_catches_truncation(tmp_path):
    fn = tmp_path / "tile.tif"
    _make_const_uint16_raster(fn, size=256)
    assert lidar_tools.dsm_functions.check_raster_validity(str(fn), deep=True)
    # truncate: header intact, pixel data missing (an interrupted write)
    data = fn.read_bytes()
    truncated = tmp_path / "truncated.tif"
    truncated.write_bytes(data[: len(data) // 2])
    assert not lidar_tools.dsm_functions.check_raster_validity(
        str(truncated), deep=True
    )


def test_execute_pdal_pipeline_skip_existing(tmp_path):
    import json

    # output already exists and is valid; the pipeline itself is garbage,
    # so returning the outfile proves execution was skipped
    outfn = tmp_path / "existing_tile.tif"
    _make_const_uint16_raster(outfn, size=64)
    pipeline_fn = tmp_path / "pipeline.json"
    pipeline_fn.write_text(
        json.dumps(
            {"pipeline": [{"type": "readers.las", "filename": "/nonexistent.laz"},
                          {"type": "writers.gdal", "filename": str(outfn),
                           "resolution": 1.0}]}
        )
    )
    result = lidar_tools.dsm_functions.execute_pdal_pipeline(
        str(pipeline_fn), skip_existing=True
    )
    assert result == str(outfn)
    # without skip_existing the garbage pipeline fails -> None (after retries)
    result = lidar_tools.dsm_functions.execute_pdal_pipeline(
        str(pipeline_fn), skip_existing=False, attempts=1
    )
    assert result is None


def test_open_decimated_dataarray(tmp_path):
    fn = tmp_path / "big.tif"
    _make_const_uint16_raster(fn, value=200, size=3000, res=0.5)
    da = lidar_tools.dsm_functions._open_decimated_dataarray(str(fn), max_dim=512)
    assert max(da.shape) <= 512
    assert da.rio.crs is not None
    np.testing.assert_allclose(float(np.nanmedian(da.values)), 200.0)
    # small rasters pass through at full resolution
    fn2 = tmp_path / "small.tif"
    _make_const_uint16_raster(fn2, size=100)
    da2 = lidar_tools.dsm_functions._open_decimated_dataarray(str(fn2), max_dim=512)
    assert da2.shape == (100, 100)


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


# ---------------------------------------------------------------------------
# Single-read / multi-product tile jobs (F3 consolidation)
# ---------------------------------------------------------------------------

_TILE_EXTENT = [500000.0, 4000000.0, 500020.0, 4000020.0]


def _make_multiclass_laz(fn, epsg=32611):
    """
    Fixture point cloud exercising every per-product filter: ground (2),
    unclassified (1), low noise (7), high noise (18), a tall HAG outlier,
    multi-return points (excluded from first,only products but kept for
    ground products), and varying Intensity.
    """
    import pdal

    n = 200
    rng = np.random.default_rng(42)
    arr = np.zeros(
        n,
        dtype=[
            ("X", np.float64),
            ("Y", np.float64),
            ("Z", np.float64),
            ("Classification", np.uint8),
            ("ReturnNumber", np.uint8),
            ("NumberOfReturns", np.uint8),
            ("Intensity", np.uint16),
            ("GpsTime", np.float64),
            ("PointSourceId", np.uint16),
        ],
    )
    # coordinates on the 0.01 m grid (matches 3DEP EPT native scale)
    arr["X"] = np.round(rng.uniform(_TILE_EXTENT[0], _TILE_EXTENT[2], n), 2)
    arr["Y"] = np.round(rng.uniform(_TILE_EXTENT[1], _TILE_EXTENT[3], n), 2)
    arr["Z"] = np.round(100.0 + rng.uniform(0, 3, n), 2)
    arr["Classification"] = rng.choice([1, 2], n, p=[0.5, 0.5])
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["Intensity"] = rng.integers(100, 2000, n)
    arr["GpsTime"] = 3.0e8 + np.arange(n)
    arr["PointSourceId"] = rng.choice([11, 12], n)
    # multi-return non-first points: kept for DTM (if ground), dropped by
    # the first,only group filter
    arr["ReturnNumber"][:20] = 2
    arr["NumberOfReturns"][:20] = 2
    arr["Classification"][:20] = 2
    arr["Z"][:20] = 99.5
    # low and high noise points (filtered everywhere by default flags)
    arr["Classification"][20:25] = 7
    arr["Z"][20:25] = 50.0
    arr["Classification"][25:30] = 18
    arr["Z"][25:30] = 400.0
    # tall unclassified outlier: survives noise filters, caught by hag_nn
    arr["Classification"][30:33] = 1
    arr["Z"][30:33] = 150.0
    pipeline = pdal.Writer.las(
        filename=str(fn),
        a_srs=f"EPSG:{epsg}",
        minor_version=4,
        dataformat_id=6,
        forward="all",
    ).pipeline(arr)
    pipeline.execute()


def _read_raster(fn):
    import rioxarray

    da = rioxarray.open_rasterio(fn, masked=True).squeeze()
    return da


def _legacy_product_pipelines(reader, outdir, prefix, hag_nn=None,
                              dsm_gridding_choice="first_idw"):
    """
    Replicate the pre-F3 per-product pipeline construction verbatim (one
    standalone pipeline per product, each embedding its own reader) so a
    registry regression breaks equivalence instead of matching it.
    """
    import json

    d = lidar_tools.dsm_functions
    (
        dsm_group_filter,
        dsm_gridding_method,
        percentile_filter,
        percentile_threshold,
    ) = d._set_dsm_gridding_params(dsm_gridding_choice)
    files = {
        "dsm": outdir / f"{prefix}_dsm_tile_aoi_000.tif",
        "dtm_no_fill": outdir / f"{prefix}_dtm_tile_aoi_no_fill000.tif",
        "dtm_fill": outdir / f"{prefix}_dtm_tile_aoi_fill4_000.tif",
        "intensity": outdir / f"{prefix}_intensity_tile_aoi_000.tif",
    }
    chains = {
        "dsm": d.create_pdal_pipeline(
            group_filter=dsm_group_filter,
            percentile_filter=percentile_filter,
            percentile_threshold=percentile_threshold,
            reproject=False,
            filter_high_noise=True,
            filter_low_noise=True,
            hag_nn=hag_nn,
        ),
        "dtm_no_fill": d.create_pdal_pipeline(
            return_only_ground=True,
            group_filter=None,
            reproject=False,
            filter_high_noise=True,
            filter_low_noise=True,
        ),
        "dtm_fill": d.create_pdal_pipeline(
            return_only_ground=True,
            group_filter=None,
            reproject=False,
            filter_high_noise=True,
            filter_low_noise=True,
        ),
        "intensity": d.create_pdal_pipeline(
            return_only_ground=False,
            group_filter="first,only",
            reproject=False,
            filter_high_noise=True,
            filter_low_noise=True,
            hag_nn=hag_nn,
        ),
    }
    writer_kwargs = {
        "dsm": dict(gridmethod=dsm_gridding_method, dimension="Z"),
        "dtm_no_fill": dict(gridmethod="idw", dimension="Z"),
        "dtm_fill": dict(gridmethod="idw", dimension="Z"),
        "intensity": dict(
            gridmethod="idw", dimension="Intensity", nodata_value=0,
            data_type="UInt16",
        ),
    }
    paths = {}
    for name, chain in chains.items():
        stage = d.create_dem_stage(
            dem_filename=str(files[name]),
            extent=_TILE_EXTENT,
            pointcloud_resolution=1.0,
            **writer_kwargs[name],
        )
        if name == "dtm_fill":
            stage[0]["window_size"] = 4
        pipeline_fn = outdir / f"legacy_pipeline_{name}.json"
        pipeline_fn.write_text(
            json.dumps({"pipeline": [reader] + chain + stage})
        )
        paths[name] = (pipeline_fn, files[name])
    return paths


def test_parse_products():
    parse = lidar_tools.dsm_functions.parse_products
    assert parse("all") == ["dsm", "dtm_no_fill", "dtm_fill", "intensity"]
    assert parse("dtm") == ["dtm_no_fill", "dtm_fill"]
    assert parse("intensity,dsm") == ["dsm", "intensity"]  # canonical order
    assert parse("dtm_fill") == ["dtm_fill"]
    import pytest

    with pytest.raises(ValueError):
        parse("bogus")
    with pytest.raises(ValueError):
        parse("")


def test_tile_job_structure_first_idw(tmp_path):
    import json

    d = lidar_tools.dsm_functions
    reader = {"type": "readers.ept", "filename": "https://example/ept.json"}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
        hag_nn=50.0,
    )
    # one network fetch into the cache, then two local executions
    fetch = json.loads(Path(job["fetch"]["pipeline_json"]).read_text())["pipeline"]
    assert [s["type"] for s in fetch] == ["readers.ept", "writers.las"]
    cache = fetch[-1]
    assert cache["minor_version"] == 4 and cache["dataformat_id"] == 6
    assert job["fetch"]["cache_file"].endswith("_cache_tile_aoi_000.laz")
    assert len(job["executions"]) == 2
    # dsm+intensity share one execution (identical chains): pinned sequence
    dsm_int = json.loads(
        Path(job["executions"][0]["pipeline_json"]).read_text()
    )["pipeline"]
    assert [s["type"] for s in dsm_int] == [
        "readers.las",
        "filters.range",     # low noise
        "filters.hag_nn",
        "filters.assign",
        "filters.range",     # high noise
        "filters.returns",   # first,only
        "writers.gdal",
        "writers.gdal",
    ]
    assert dsm_int[0]["filename"] == job["fetch"]["cache_file"]
    # legacy tile filenames preserved verbatim (resume + mosaic compat)
    outputs = job["executions"][0]["outputs"]
    assert outputs["dsm"].endswith("aoi_dsm_tile_aoi_000.tif")
    assert outputs["intensity"].endswith("aoi_intensity_tile_aoi_000.tif")
    # writer geometry matches create_dem_stage
    assert dsm_int[-2]["origin_x"] == _TILE_EXTENT[0]
    assert dsm_int[-2]["width"] == 20 and dsm_int[-2]["height"] == 20
    # the DTM pair shares the ground execution, fill differs only by writer
    dtm = json.loads(
        Path(job["executions"][1]["pipeline_json"]).read_text()
    )["pipeline"]
    assert [s["type"] for s in dtm] == [
        "readers.las",
        "filters.range",     # low noise
        "filters.range",     # high noise
        "filters.range",     # ground only
        "writers.gdal",
        "writers.gdal",
    ]
    dtm_outputs = job["executions"][1]["outputs"]
    assert dtm_outputs["dtm_no_fill"].endswith("aoi_dtm_tile_aoi_no_fill000.tif")
    assert dtm_outputs["dtm_fill"].endswith("aoi_dtm_tile_aoi_fill4_000.tif")
    writers = [s for s in dtm if s["type"] == "writers.gdal"]
    assert "window_size" not in writers[0] and writers[1]["window_size"] == 4


def test_tile_job_structure_npct(tmp_path):
    d = lidar_tools.dsm_functions
    reader = {"type": "readers.ept", "filename": "https://example/ept.json"}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
        dsm_gridding_choice="98-pct",
    )
    # percentile DSM diverges from intensity: three executions
    assert len(job["executions"]) == 3
    import json

    dsm = json.loads(
        Path(job["executions"][0]["pipeline_json"]).read_text()
    )["pipeline"]
    assert any(s["type"] == "filters.python" for s in dsm)
    assert [s for s in dsm if s["type"] == "writers.gdal"][0]["output_type"] == "max"


def test_tile_job_single_product_inlines_reader(tmp_path):
    import json

    d = lidar_tools.dsm_functions
    reader = {"type": "readers.ept", "filename": "https://example/ept.json"}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=["dsm"],
    )
    # cache would be pure overhead: reader inlined, no fetch step
    assert job["fetch"] is None
    assert len(job["executions"]) == 1
    stages = json.loads(
        Path(job["executions"][0]["pipeline_json"]).read_text()
    )["pipeline"]
    assert stages[0]["type"] == "readers.ept"
    # dtm pair alone still merges into one two-writer execution
    job2 = d.create_tile_pipelines(
        [reader],
        tile_id="001",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("dtm"),
    )
    assert job2["fetch"] is None
    assert len(job2["executions"]) == 1
    assert len(job2["executions"][0]["outputs"]) == 2


def _run_equivalence(tmp_path, hag_nn, dsm_gridding_choice):
    d = lidar_tools.dsm_functions
    laz = tmp_path / "points.laz"
    _make_multiclass_laz(laz)
    reader = {"type": "readers.las", "filename": str(laz)}

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = _legacy_product_pipelines(
        reader, legacy_dir, "aoi", hag_nn=hag_nn,
        dsm_gridding_choice=dsm_gridding_choice,
    )
    for name, (pipeline_fn, _) in legacy.items():
        assert d.execute_pdal_pipeline(str(pipeline_fn)) is not None, name

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=new_dir,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
        hag_nn=hag_nn,
        dsm_gridding_choice=dsm_gridding_choice,
    )
    results = d.execute_tile_job(job)
    assert all(fn is not None for fn in results.values()), results
    # cache removed in-task
    if job["fetch"]:
        assert not Path(job["fetch"]["cache_file"]).exists()

    for name, (_, legacy_fn) in legacy.items():
        old = _read_raster(str(legacy_fn))
        new = _read_raster(results[name])
        assert old.rio.crs == new.rio.crs, name
        np.testing.assert_array_equal(
            np.isnan(old.values), np.isnan(new.values), err_msg=name
        )
        # 0.011 m = one cache quantum; intensity DNs must be exact
        atol = 0.0 if name == "intensity" else 0.011
        np.testing.assert_allclose(
            old.values, new.values, atol=atol, err_msg=name
        )


def test_tile_job_equivalence_first_idw_hag(tmp_path):
    _run_equivalence(tmp_path, hag_nn=5.0, dsm_gridding_choice="first_idw")


def test_tile_job_equivalence_npct(tmp_path):
    _run_equivalence(tmp_path, hag_nn=None, dsm_gridding_choice="98-pct")


def test_execute_tile_job_resume(tmp_path):
    d = lidar_tools.dsm_functions
    laz = tmp_path / "points.laz"
    _make_multiclass_laz(laz)
    reader = {"type": "readers.las", "filename": str(laz)}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
    )
    first = d.execute_tile_job(job)
    assert all(fn is not None for fn in first.values())
    mtimes = {name: Path(fn).stat().st_mtime_ns for name, fn in first.items()}

    # full resume: every execution skipped, nothing rewritten
    second = d.execute_tile_job(job, skip_existing=True)
    assert second == first
    for name, fn in second.items():
        assert Path(fn).stat().st_mtime_ns == mtimes[name], name

    # partial-within-execution: losing intensity re-runs the dsm+intensity
    # execution whole (both rewritten); the DTM execution stays skipped
    Path(first["intensity"]).unlink()
    third = d.execute_tile_job(job, skip_existing=True)
    assert all(fn is not None for fn in third.values())
    assert Path(third["dsm"]).stat().st_mtime_ns > mtimes["dsm"]
    assert Path(third["intensity"]).exists()
    for name in ("dtm_no_fill", "dtm_fill"):
        assert Path(third[name]).stat().st_mtime_ns == mtimes[name], name


def test_execute_tile_job_fetch_failure(tmp_path):
    d = lidar_tools.dsm_functions
    reader = {"type": "readers.las", "filename": "/nonexistent.laz"}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
    )
    results = d.execute_tile_job(job, attempts=1)
    # never raises; all products reported failed; no cache left behind
    assert set(results) == {"dsm", "dtm_no_fill", "dtm_fill", "intensity"}
    assert all(fn is None for fn in results.values())
    assert not Path(job["fetch"]["cache_file"]).exists()


def test_execute_tile_job_empty_branch(tmp_path):
    # a tile whose points all fail one branch's filters (no ground) must
    # still produce the other products and report the empty ones as failed
    import pdal

    d = lidar_tools.dsm_functions
    laz = tmp_path / "noground.laz"
    n = 50
    arr = np.zeros(
        n,
        dtype=[
            ("X", np.float64),
            ("Y", np.float64),
            ("Z", np.float64),
            ("Classification", np.uint8),
            ("ReturnNumber", np.uint8),
            ("NumberOfReturns", np.uint8),
            ("Intensity", np.uint16),
        ],
    )
    rng = np.random.default_rng(7)
    arr["X"] = np.round(rng.uniform(_TILE_EXTENT[0], _TILE_EXTENT[2], n), 2)
    arr["Y"] = np.round(rng.uniform(_TILE_EXTENT[1], _TILE_EXTENT[3], n), 2)
    arr["Z"] = 100.0
    arr["Classification"] = 1  # no ground anywhere
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["Intensity"] = 500
    pipeline = pdal.Writer.las(
        filename=str(laz), a_srs="EPSG:32611", minor_version=4,
        dataformat_id=6, forward="all",
    ).pipeline(arr)
    pipeline.execute()

    reader = {"type": "readers.las", "filename": str(laz)}
    job = d.create_tile_pipelines(
        [reader],
        tile_id="000",
        output_path=tmp_path,
        prefix="aoi",
        extent=_TILE_EXTENT,
        raster_resolution=1.0,
        products=d.parse_products("all"),
    )
    results = d.execute_tile_job(job, attempts=1)
    # writers.gdal writes an all-nodata raster for an empty point view (same
    # as the legacy per-product path): every product "succeeds", the ground
    # products are fully masked
    assert all(fn is not None for fn in results.values())
    for name in ("dtm_no_fill", "dtm_fill"):
        da = _read_raster(results[name])
        assert np.isnan(da.values).all(), name
    da = _read_raster(results["dsm"])
    assert not np.isnan(da.values).all()


def test_lpc_tile_jobs(tmp_path):
    import json

    import pyproj as _pyproj

    d = lidar_tools.dsm_functions
    laz_dir = tmp_path / "laz"
    laz_dir.mkdir()
    _make_multiclass_laz(laz_dir / "points.laz")

    # AOI polygon covering the tile (geojson, EPSG:4326)
    aoi = gpd.GeoDataFrame(
        geometry=[
            gpd.GeoSeries.from_wkt(
                ["POLYGON((499990 3999990, 500030 3999990, 500030 4000030,"
                 " 499990 4000030, 499990 3999990))"]
            )
            .set_crs("EPSG:32611")
            .to_crs("EPSG:4326")
            .iloc[0]
        ],
        crs="EPSG:4326",
    )
    aoi_fn = tmp_path / "aoi.geojson"
    aoi.to_file(aoi_fn)

    target_wkt = tmp_path / "target.wkt"
    target_wkt.write_text(_pyproj.CRS.from_epsg(32611).to_wkt())

    outdir = tmp_path / "out"
    outdir.mkdir()
    jobs = d.create_lpc_pipeline(
        local_laz_dir=str(laz_dir),
        target_wkt=str(target_wkt),
        output_prefix=str(outdir / "aoi"),
        extent_polygon=str(aoi_fn),
        raster_resolution=1.0,
        products=d.parse_products("all"),
    )
    assert len(jobs) == 1
    job = jobs[0]
    # local input: no cache step, two executions (dsm+intensity, dtm pair)
    assert job["fetch"] is None
    assert len(job["executions"]) == 2

    dsm_int = json.loads(
        Path(job["executions"][0]["pipeline_json"]).read_text()
    )["pipeline"]
    types = [s["type"] for s in dsm_int]
    assert types[0] == "readers.las"
    # in-pipeline reprojection shared by the chain
    assert "filters.reprojection" in types
    # saved point cloud chained before the gdal writers, legacy .laz.laz name
    las_writers = [s for s in dsm_int if s["type"] == "writers.las"]
    assert len(las_writers) == 1
    assert las_writers[0]["filename"].endswith("_dsm_tile_aoi_0.laz.laz")
    assert types.index("writers.las") < types.index("writers.gdal")
    assert len([t for t in types if t == "writers.gdal"]) == 2

    dtm = json.loads(
        Path(job["executions"][1]["pipeline_json"]).read_text()
    )["pipeline"]
    dtm_types = [s["type"] for s in dtm]
    assert "filters.reprojection" in dtm_types
    assert "writers.las" not in dtm_types  # pointcloud saved only with dsm
    assert len(job["executions"][1]["outputs"]) == 2

    # executes end-to-end: all products valid, point cloud written
    results = d.execute_tile_job(job)
    assert all(fn is not None for fn in results.values()), results
    assert Path(las_writers[0]["filename"]).exists()
    da = _read_raster(results["dsm"])
    assert da.rio.crs == _pyproj.CRS.from_epsg(32611)
    assert not np.isnan(da.values).all()

    # subset without dsm: no point-cloud save writer anywhere
    jobs2 = d.create_lpc_pipeline(
        local_laz_dir=str(laz_dir),
        target_wkt=str(target_wkt),
        output_prefix=str(outdir / "aoi2"),
        extent_polygon=str(aoi_fn),
        raster_resolution=1.0,
        products=d.parse_products("dtm,intensity"),
    )
    for execution in jobs2[0]["executions"]:
        stages = json.loads(Path(execution["pipeline_json"]).read_text())["pipeline"]
        assert all(s["type"] != "writers.las" for s in stages)
