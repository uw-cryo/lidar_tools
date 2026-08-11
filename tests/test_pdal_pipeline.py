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


def _lv_aoi_file(tmp_path):
    import shapely

    aoi = gpd.GeoDataFrame(
        geometry=[shapely.box(-115.10, 36.00, -115.05, 36.05)], crs="EPSG:4326"
    )
    fn = tmp_path / "aoi.geojson"
    aoi.to_file(fn)
    return str(fn)


def _fake_ept_index(names, counts):
    import shapely

    return gpd.GeoDataFrame(
        {"name": names, "count": counts},
        geometry=[shapely.box(-115.2, 35.9, -114.9, 36.2)] * len(names),
        crs="EPSG:4326",
    )


def test_rasterize_pins_wesm_name_but_reads_resolved_ept(tmp_path, monkeypatch):
    """The WESM pin, output naming and metadata keep the workunit name;
    only the EPT reader join uses the resolved (FTP-era) resource name."""
    import glob

    import yaml

    from lidar_tools import catalog, dsm_functions, geodesy, pdal_pipeline

    wesm_rec = {
        "workunit": "NV_LasVegas_QL2_2016",
        "horiz_crs": "6521",
        "geoid": "GEOID12A",
        "ql": "QL 2",
    }
    monkeypatch.setattr(
        catalog, "workunit_record", lambda gdf, wu, **k: dict(wesm_rec, workunit=wu)
    )
    preflight_kwargs = []
    real_preflight_stub = lambda *a, **k: {"ok": True, "stub": True}  # noqa: E731

    def spy_preflight(*a, **k):
        preflight_kwargs.append(k)
        return real_preflight_stub(*a, **k)

    monkeypatch.setattr(
        catalog,
        "load_ept_resources",
        lambda *a, **k: _fake_ept_index(
            ["USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"], [9]
        ),
    )
    monkeypatch.setattr(geodesy, "preflight_vertical_transform", spy_preflight)
    captured = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(dsm_functions, "create_ept_3dep_pipeline", fake_create)

    outdir = tmp_path / "out"
    pdal_pipeline.rasterize_project(
        geometry=_lv_aoi_file(tmp_path),
        output=str(outdir),
        threedep_project="NV_LasVegas_QL2_2016",
        output_datum="nad83_2011",
        quiet=True,
    )

    # reader join got the RESOLVED EPT name
    assert captured["survey_name"] == "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"
    metas = glob.glob(str(outdir / "*processing_metadata.yaml"))
    assert len(metas) == 1
    # output naming keeps the WESM workunit name, never the EPT alias
    assert "NV_LasVegas_QL2_2016" in metas[0] and "USGS_LPC" not in metas[0]
    meta = yaml.safe_load(open(metas[0]))
    # the WESM pin rode through untouched (GEOID12A, workunit name)
    assert meta["survey_records"][0]["workunit"] == "NV_LasVegas_QL2_2016"
    assert meta["survey_records"][0]["geoid"] == "GEOID12A"
    # resolution provenance recorded: who resolved to what, at which tier
    assert meta["ept_resolution"]["workunit"] == "NV_LasVegas_QL2_2016"
    assert (
        meta["ept_resolution"]["ept_name"] == "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"
    )
    assert meta["ept_resolution"]["tier"] == 3
    assert meta["ept_resolution"]["boundary_intersects_aoi"] is True
    # 0 readers -> the no-data guard finished the run cleanly, loudly
    assert meta["run_status"]["state"] == "completed"
    assert "no data" in meta["run_status"]["note"]
    # declared GEOID12A resolved to the GEOID12B CONUS grid (NGS-identical
    # outside PR/USVI) and REQUIRED at preflight — never a silent fallback
    assert meta["declared_geoid"]["model"] == "GEOID12B"
    assert meta["declared_geoid"]["substituted_for"] == "GEOID12A"
    geoid_calls = [k for k in preflight_kwargs if k.get("require_grids")]
    assert geoid_calls, "no preflight call carried the declared-geoid grids"
    assert geoid_calls[0]["require_grids"] == ["us_noaa_g2012bu0.tif"]
    assert geoid_calls[0]["allow_geoid_fallback"] is False


def test_rasterize_unresolvable_ept_raises_lookuperror(tmp_path, monkeypatch):
    """No silent 0-reader runs: an unresolvable workunit fails loudly."""
    from lidar_tools import catalog, geodesy, pdal_pipeline

    monkeypatch.setattr(
        catalog,
        "workunit_record",
        lambda gdf, wu, **k: {"workunit": wu, "horiz_crs": "6340", "geoid": "GEOID18"},
    )
    monkeypatch.setattr(
        catalog,
        "load_ept_resources",
        lambda *a, **k: _fake_ept_index(["NV_Southern_5_D23"], [1]),
    )
    monkeypatch.setattr(
        geodesy,
        "preflight_vertical_transform",
        lambda *a, **k: {"ok": True, "stub": True},
    )
    with pytest.raises(LookupError, match="NV_Southern_4_D23"):
        pdal_pipeline.rasterize_project(
            geometry=_lv_aoi_file(tmp_path),
            output=str(tmp_path / "out"),
            threedep_project="NV_Southern_4_D23",
            output_datum="nad83_2011",
            quiet=True,
        )


def test_rasterize_wesm_failure_geoid_modes(tmp_path, monkeypatch):
    """A WESM fetch failure must not silently drop the declared-geoid
    requirement: hard error in 'declared' mode, loud proceed only when the
    operator already chose best-available."""
    from lidar_tools import catalog, dsm_functions, geodesy, pdal_pipeline

    def no_wesm(gdf, wu, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(catalog, "workunit_record", no_wesm)
    monkeypatch.setattr(
        catalog, "load_ept_resources", lambda *a, **k: _fake_ept_index(["WU_X"], [1])
    )
    monkeypatch.setattr(
        geodesy, "preflight_vertical_transform", lambda *a, **k: {"ok": True}
    )
    monkeypatch.setattr(dsm_functions, "create_ept_3dep_pipeline", lambda *a, **k: [])

    aoi = _lv_aoi_file(tmp_path)
    with pytest.raises(RuntimeError, match="geoid-override best-available"):
        pdal_pipeline.rasterize_project(
            geometry=aoi,
            output=str(tmp_path / "o1"),
            threedep_project="WU_X",
            output_datum="nad83_2011",
            quiet=True,
        )
    # conscious override: run proceeds on default datum handling
    pdal_pipeline.rasterize_project(
        geometry=aoi,
        output=str(tmp_path / "o2"),
        threedep_project="WU_X",
        output_datum="nad83_2011",
        geoid_override="best-available",
        quiet=True,
    )


def test_metadata_path_rejects_mixed_run_prefixes(tmp_path):
    """A directory holding metadata from two runs (e.g. --resume at a new
    resolution) must error, not silently update the alphabetically-first
    stale record."""
    from lidar_tools import pdal_pipeline

    only = tmp_path / "aoi_0.5m-processing_metadata.yaml"
    only.write_text("a: 1\n")
    assert pdal_pipeline._metadata_path(tmp_path) == only

    (tmp_path / "aoi_1m-processing_metadata.yaml").write_text("b: 2\n")
    with pytest.raises(RuntimeError, match="Multiple processing-metadata"):
        pdal_pipeline._metadata_path(tmp_path)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    assert pdal_pipeline._metadata_path(legacy).name == "processing_metadata.yaml"


def test_metadata_updates_target_the_running_prefix(tmp_path):
    """A run passes its own prefix, so its updates land in its own record
    even when the directory holds another run's metadata."""
    import yaml

    from lidar_tools import pdal_pipeline

    stale = tmp_path / "aoi_0.5m-processing_metadata.yaml"
    stale.write_text("run_status:\n  state: completed\n")
    mine = tmp_path / "aoi_1m-processing_metadata.yaml"
    mine.write_text("run_status:\n  state: started\n")

    pdal_pipeline._update_processing_metadata(
        tmp_path, "geodesy", {"x": 1}, filename_prefix="aoi_1m"
    )
    assert "geodesy" in yaml.safe_load(mine.read_text())
    # the other run's record is untouched
    assert yaml.safe_load(stale.read_text()) == {"run_status": {"state": "completed"}}


def test_engine_validates_before_overwrite_deletes(tmp_path):
    """Argument validation must precede the overwrite rmtree: a rejected
    call must not destroy the prior products it was pointed at."""
    import pytest

    from lidar_tools import pdal_pipeline

    outdir = tmp_path / "existing"
    outdir.mkdir()
    keep = outdir / "prior-DSM_mos.tif"
    keep.write_bytes(b"precious")
    with pytest.raises(ValueError, match="threedep_project is required"):
        pdal_pipeline.rasterize_project(
            geometry="unused.geojson",
            output=str(outdir),
            overwrite=True,
        )
    assert keep.read_bytes() == b"precious"  # nothing was deleted


def test_engine_rejects_selection_keywords():
    """Old-API literals ('all'/'latest') must error with guidance, not be
    treated as workunit names that fail later in WESM lookup."""
    import pytest

    from lidar_tools import pdal_pipeline

    for kw in ("all", "latest", "auto"):
        with pytest.raises(ValueError, match="selection keyword"):
            pdal_pipeline.rasterize_project(
                geometry="unused.geojson", output="/tmp/x", threedep_project=kw
            )


def test_engine_rejects_unnameable_local_input():
    import pytest

    from lidar_tools import pdal_pipeline

    with pytest.raises(ValueError, match="nameable input directory"):
        pdal_pipeline.rasterize_project(
            geometry="unused.geojson", output="/tmp/x", input="/"
        )


def _resume_params(**over):
    base = {
        "geometry_fingerprint": "abc123",
        "input": "EPT_AWS",
        "src_crs": None,
        "dst_crs": "/tmp/utm.wkt",
        "dsm_gridding_choice": "first_idw",
        "tile_size": 1.0,
        "filter_noise": True,
        "height_above_ground_threshold": None,
        "proj_pipeline": None,
        "ept_vertical": "auto",
        "geoid_override": "declared",
        "output_datum": "wgs84_g2139",
        "coord_epoch": None,
    }
    base.update(over)
    return base


def test_resume_accepts_identical_parameters():
    """The design center: a user re-running the identical command after a
    failure must resume, not be blocked."""
    from lidar_tools import pdal_pipeline

    p = _resume_params()
    assert pdal_pipeline.check_resume_compatible(dict(p), dict(p)) == []


def test_resume_rejects_changed_tile_parameters():
    import pytest

    from lidar_tools import pdal_pipeline

    with pytest.raises(ValueError, match="dsm_gridding_choice"):
        pdal_pipeline.check_resume_compatible(
            _resume_params(), _resume_params(dsm_gridding_choice="95-pct")
        )


def test_resume_rejects_edited_aoi_at_the_same_path():
    """Tiles are named by a bare index over a grid derived from the AOI
    bounds, so an AOI edited in place would silently re-use tiles covering
    different ground -- the path cannot detect that, the fingerprint can."""
    import pytest

    from lidar_tools import pdal_pipeline

    with pytest.raises(ValueError, match="geometry_fingerprint"):
        pdal_pipeline.check_resume_compatible(
            _resume_params(), _resume_params(geometry_fingerprint="deadbeef")
        )


def test_resume_reports_but_does_not_block_on_unverifiable():
    """Re-running the identical command after an interruption is the design
    centre, and EVERY directory written before a parameter existed lacks it
    (verified: real Casa Grande records carry no output_datum, and none
    predating this change carries geometry_fingerprint). Unverifiable must
    warn, never block -- only a CHANGED parameter blocks."""
    from lidar_tools import pdal_pipeline

    legacy = _resume_params()
    del legacy["output_datum"]
    del legacy["geometry_fingerprint"]
    unverified = pdal_pipeline.check_resume_compatible(legacy, _resume_params())
    assert sorted(unverified) == ["geometry_fingerprint", "output_datum"]

    # a missing/corrupt record verifies nothing, but still does not block
    assert set(pdal_pipeline.check_resume_compatible(None, _resume_params())) == set(
        pdal_pipeline.RESUME_TILE_PARAMS
    )


def test_resume_still_blocks_a_changed_parameter_in_a_legacy_record():
    """The dangerous case survives the relaxation: a key present in BOTH
    records and disagreeing still refuses."""
    import pytest

    from lidar_tools import pdal_pipeline

    legacy = _resume_params(dsm_gridding_choice="first_idw")
    del legacy["output_datum"]
    with pytest.raises(ValueError, match="dsm_gridding_choice"):
        pdal_pipeline.check_resume_compatible(
            legacy, _resume_params(dsm_gridding_choice="95-pct")
        )


def test_resume_path_spelling_does_not_block(tmp_path):
    """An identical resume spelled with a relative path or trailing slash
    must not throw away hours of valid tiles."""
    from lidar_tools import pdal_pipeline

    d = tmp_path / "wu_a"
    d.mkdir()
    prior = _resume_params(input=str(d))
    now = _resume_params(input=str(d) + "/")
    pdal_pipeline.check_resume_compatible(prior, now)  # no raise


def test_aoi_fingerprint_tracks_content_not_path(tmp_path):
    import geopandas as gpd
    import shapely

    from lidar_tools import pdal_pipeline

    a = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1)], crs="EPSG:4326")
    b = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 2, 2)], crs="EPSG:4326")
    assert pdal_pipeline._aoi_fingerprint(a) == pdal_pipeline._aoi_fingerprint(
        gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1, 1)], crs="EPSG:4326")
    )
    assert pdal_pipeline._aoi_fingerprint(a) != pdal_pipeline._aoi_fingerprint(b)


def test_engine_keyword_rejection_is_case_insensitive():
    """The driver casefolds these; the engine must agree or 'AUTO' reaches
    the misleading WESM lookup error."""
    import pytest

    from lidar_tools import pdal_pipeline

    for kw in ("AUTO", "Latest", "ALL"):
        with pytest.raises(ValueError, match="selection keyword"):
            pdal_pipeline.rasterize_project(
                geometry="unused.geojson", output="/tmp/x", threedep_project=kw
            )
