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


def test_rasterize_projects_output_datum_nad83(tmp_path, aoi_file, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "rasterize", lambda **kw: calls.append(kw))
    outbase = tmp_path / "batch"
    driver.rasterize_projects(
        aoi_file, "WU_A,WU_B", str(outbase), output_datum="nad83_2011"
    )
    # the shared target is the NAD83(2011) realization, built once, and the
    # datum choice is threaded through to every project
    wkts = list(outbase.glob("UTM_*_NAD83_2011_3D.wkt"))
    assert len(wkts) == 1
    assert not list(outbase.glob("*WGS84_G2139*"))
    assert all(c["dst_crs"] == str(wkts[0]) for c in calls)
    assert all(c["output_datum"] == "nad83_2011" for c in calls)


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


def test_rasterize_projects_flags_no_data_runs(tmp_path, aoi_file, monkeypatch, capsys):
    from pathlib import Path

    def fake_rasterize(**kw):
        outdir = Path(kw["output"])
        outdir.mkdir(parents=True, exist_ok=True)
        wu = kw["threedep_project"]
        run_status = {"state": "completed"}
        if wu == "WU_A":
            run_status["note"] = "no data (survey does not cover AOI)"
        (outdir / f"aoi_1m_{wu}-processing_metadata.yaml").write_text(
            yaml.dump({"run_status": run_status})
        )

    monkeypatch.setattr(driver, "rasterize", fake_rasterize)
    outbase = tmp_path / "batch"
    # a no-data project is a real outcome: the batch must NOT raise ...
    driver.rasterize_projects(aoi_file, "WU_A,WU_B", str(outbase))
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    # ... but it must never be recorded as a plain success
    assert status["projects"]["WU_A"].startswith("completed (no data)")
    assert "survey does not cover" in status["projects"]["WU_A"]
    assert status["projects"]["WU_B"] == "completed"
    err = capsys.readouterr().err
    assert "WU_A" in err and "WITHOUT products" in err
    assert "1/2" in err  # end-of-batch warning names the count


def test_rasterize_projects_warns_on_unreadable_metadata(
    tmp_path, aoi_file, monkeypatch, capsys
):
    from pathlib import Path

    def fake_rasterize(**kw):
        outdir = Path(kw["output"])
        outdir.mkdir(parents=True, exist_ok=True)
        # corrupt YAML: the run returned cleanly but its status is unreadable
        (outdir / "aoi_1m_WU_A-processing_metadata.yaml").write_text("{::not yaml")

    monkeypatch.setattr(driver, "rasterize", fake_rasterize)
    outbase = tmp_path / "batch"
    driver.rasterize_projects(aoi_file, "WU_A", str(outbase))
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    # still counted completed (the run itself succeeded) ...
    assert status["projects"]["WU_A"] == "completed"
    # ... but the operator is told the products could not be verified
    err = capsys.readouterr().err
    assert "unreadable processing metadata" in err
