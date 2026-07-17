import geopandas as gpd
import pytest
import shapely
import yaml

from lidar_tools import driver


@pytest.fixture
def aoi_file(tmp_path):
    fn = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(
        geometry=[shapely.box(-122.32, 47.64, -122.30, 47.66)], crs="EPSG:4326"
    ).to_file(fn, driver="GeoJSON")
    return str(fn)


def test_rasterize_projects_shared_grid_and_subdirs(tmp_path, aoi_file, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "rasterize", lambda **kw: calls.append(kw))
    outbase = tmp_path / "batch"
    driver.rasterize_projects(
        aoi_file, "WU_A,WU_B", str(outbase), resolution=0.5, num_process=3
    )
    assert len(calls) == 2
    # shared target CRS file created once in the base dir, passed to both
    wkts = list(outbase.glob("UTM_*_WGS84_G2139_3D.wkt"))
    assert len(wkts) == 1
    assert all(c["dst_crs"] == str(wkts[0]) for c in calls)
    assert [c["threedep_project"] for c in calls] == ["WU_A", "WU_B"]
    assert [c["output"] for c in calls] == [
        str(outbase / "WU_A"),
        str(outbase / "WU_B"),
    ]
    assert all(c["resolution"] == 0.5 and c["num_process"] == 3 for c in calls)
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    assert status["projects"] == {"WU_A": "completed", "WU_B": "completed"}


def test_rasterize_projects_one_failure_does_not_stop_batch(
    tmp_path, aoi_file, monkeypatch
):
    calls = []

    def fake_rasterize(**kw):
        calls.append(kw["threedep_project"])
        if kw["threedep_project"] == "WU_A":
            raise RuntimeError("boom")

    monkeypatch.setattr(driver, "rasterize", fake_rasterize)
    outbase = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="1/2 project runs failed"):
        driver.rasterize_projects(aoi_file, "WU_A, WU_B", str(outbase))
    assert calls == ["WU_A", "WU_B"]  # WU_B still ran
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    assert status["projects"]["WU_A"].startswith("failed")
    assert status["projects"]["WU_B"] == "completed"
