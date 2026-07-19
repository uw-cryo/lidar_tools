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

    from lidar_tools import dsm_functions, geodesy, pdal_pipeline, survey

    wesm_rec = {
        "workunit": "NV_LasVegas_QL2_2016",
        "horiz_crs": "6521",
        "geoid": "GEOID12A",
        "ql": "QL 2",
    }
    monkeypatch.setattr(
        survey, "workunit_record", lambda gdf, wu, **k: dict(wesm_rec, workunit=wu)
    )
    monkeypatch.setattr(
        survey,
        "load_ept_resources",
        lambda *a, **k: _fake_ept_index(
            ["USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"], [9]
        ),
    )
    monkeypatch.setattr(
        geodesy,
        "preflight_vertical_transform",
        lambda *a, **k: {"ok": True, "stub": True},
    )
    captured = {}

    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(dsm_functions, "create_ept_3dep_pipeline", fake_create)

    outdir = tmp_path / "out"
    pdal_pipeline.rasterize(
        geometry=_lv_aoi_file(tmp_path),
        output=str(outdir),
        threedep_project="NV_LasVegas_QL2_2016",
        output_datum="nad83_2011",
        quiet=True,
    )

    # reader join got the RESOLVED EPT name
    assert (
        captured["process_specific_3dep_survey"]
        == "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"
    )
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
        meta["ept_resolution"]["ept_name"]
        == "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018"
    )
    assert meta["ept_resolution"]["tier"] == 3
    assert meta["ept_resolution"]["boundary_intersects_aoi"] is True
    # 0 readers -> the no-data guard finished the run cleanly, loudly
    assert meta["run_status"]["state"] == "completed"
    assert "no data" in meta["run_status"]["note"]


def test_rasterize_unresolvable_ept_raises_lookuperror(tmp_path, monkeypatch):
    """No silent 0-reader runs: an unresolvable workunit fails loudly."""
    from lidar_tools import geodesy, pdal_pipeline, survey

    monkeypatch.setattr(
        survey,
        "workunit_record",
        lambda gdf, wu, **k: {"workunit": wu, "horiz_crs": "6340", "geoid": "GEOID18"},
    )
    monkeypatch.setattr(
        survey,
        "load_ept_resources",
        lambda *a, **k: _fake_ept_index(["NV_Southern_5_D23"], [1]),
    )
    monkeypatch.setattr(
        geodesy,
        "preflight_vertical_transform",
        lambda *a, **k: {"ok": True, "stub": True},
    )
    with pytest.raises(LookupError, match="NV_Southern_4_D23"):
        pdal_pipeline.rasterize(
            geometry=_lv_aoi_file(tmp_path),
            output=str(tmp_path / "out"),
            threedep_project="NV_Southern_4_D23",
            output_datum="nad83_2011",
            quiet=True,
        )
