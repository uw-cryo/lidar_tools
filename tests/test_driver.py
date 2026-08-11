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


def test_rasterize_shared_grid_and_subdirs(tmp_path, aoi_file, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    outbase = tmp_path / "batch"
    driver.rasterize(
        aoi_file, str(outbase), projects="WU_A,WU_B", resolution=0.5, num_process=3
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


def test_rasterize_output_datum_nad83(tmp_path, aoi_file, monkeypatch):
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    outbase = tmp_path / "batch"
    driver.rasterize(
        aoi_file, str(outbase), projects="WU_A,WU_B", output_datum="nad83_2011"
    )
    # the shared target is the NAD83(2011) realization, built once, and the
    # datum choice is threaded through to every project
    wkts = list(outbase.glob("UTM_*_NAD83_2011_3D.wkt"))
    assert len(wkts) == 1
    assert not list(outbase.glob("*WGS84_G2139*"))
    assert all(c["dst_crs"] == str(wkts[0]) for c in calls)
    assert all(c["output_datum"] == "nad83_2011" for c in calls)


def test_rasterize_one_failure_does_not_stop_batch(tmp_path, aoi_file, monkeypatch):
    calls = []

    def fake_rasterize(**kw):
        calls.append(kw["threedep_project"])
        if kw["threedep_project"] == "WU_A":
            raise RuntimeError("boom")

    monkeypatch.setattr(driver, "rasterize_project", fake_rasterize)
    outbase = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="1/2 project runs failed"):
        driver.rasterize(aoi_file, str(outbase), projects="WU_A, WU_B")
    assert calls == ["WU_A", "WU_B"]  # WU_B still ran
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    assert status["projects"]["WU_A"].startswith("failed")
    assert status["projects"]["WU_B"] == "completed"


def test_rasterize_flags_no_data_runs(tmp_path, aoi_file, monkeypatch, capsys):
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

    monkeypatch.setattr(driver, "rasterize_project", fake_rasterize)
    outbase = tmp_path / "batch"
    # a no-data project is a real outcome: the batch must NOT raise ...
    driver.rasterize(aoi_file, str(outbase), projects="WU_A,WU_B")
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    # ... but it must never be recorded as a plain success
    assert status["projects"]["WU_A"].startswith("completed (no data)")
    assert "survey does not cover" in status["projects"]["WU_A"]
    assert status["projects"]["WU_B"] == "completed"
    err = capsys.readouterr().err
    assert "WU_A" in err and "WITHOUT products" in err
    assert "1/2" in err  # end-of-batch warning names the count


def test_rasterize_warns_on_unreadable_metadata(
    tmp_path, aoi_file, monkeypatch, capsys
):
    from pathlib import Path

    def fake_rasterize(**kw):
        outdir = Path(kw["output"])
        outdir.mkdir(parents=True, exist_ok=True)
        # corrupt YAML: the run returned cleanly but its status is unreadable
        (outdir / "aoi_1m_WU_A-processing_metadata.yaml").write_text("{::not yaml")

    monkeypatch.setattr(driver, "rasterize_project", fake_rasterize)
    outbase = tmp_path / "batch"
    driver.rasterize(aoi_file, str(outbase), projects="WU_A")
    status = yaml.safe_load((outbase / "batch_status.yaml").read_text())
    # still counted completed (the run itself succeeded) ...
    assert status["projects"]["WU_A"] == "completed"
    # ... but the operator is told the products could not be verified
    err = capsys.readouterr().err
    assert "unreadable processing metadata" in err


def test_rasterize_passes_geoid_override(tmp_path, aoi_file, monkeypatch):
    from pathlib import Path

    seen = []

    def fake_rasterize(**kw):
        Path(kw["output"]).mkdir(parents=True, exist_ok=True)
        seen.append(kw)

    monkeypatch.setattr(driver, "rasterize_project", fake_rasterize)
    driver.rasterize(aoi_file, str(tmp_path / "b1"), projects="WU_A")
    assert seen[0]["geoid_override"] == "declared"  # hard-fail is the default
    driver.rasterize(
        aoi_file, str(tmp_path / "b2"), projects="WU_A", geoid_override="best-available"
    )
    assert seen[1]["geoid_override"] == "best-available"


def test_batch_status_accumulates_across_invocations(tmp_path, aoi_file, monkeypatch):
    """merge/preview/fetch-reports/report-metrics default to the workunits in
    batch_status.yaml, so a later run over one project must not drop the
    others from the batch."""
    out = tmp_path / "batch"
    out.mkdir()
    (out / "batch_status.yaml").write_text(
        yaml.dump(
            {
                "geometry": aoi_file,
                "dst_crs": "utm.wkt",
                "projects": {"WU_A": "completed", "WU_B": "completed"},
            }
        )
    )
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)

    driver.rasterize(
        geometry=aoi_file, output=str(out), projects="WU_C", dst_crs="utm.wkt"
    )
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_A", "WU_B", "WU_C"}


def test_batch_status_does_not_inherit_a_different_batch(
    tmp_path, aoi_file, monkeypatch
):
    """Downstream commands default to the workunits in batch_status.yaml, so a
    directory reused for a different AOI/grid must not carry the old projects
    forward — nor crash on a malformed file."""
    out = tmp_path / "batch"
    out.mkdir()
    (out / "batch_status.yaml").write_text(
        yaml.dump(
            {
                "geometry": "some_other_aoi.geojson",
                "dst_crs": "other.wkt",
                "projects": {"WU_ELSEWHERE": "completed"},
            }
        )
    )
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    driver.rasterize(
        geometry=aoi_file, output=str(out), projects="WU_C", dst_crs="utm.wkt"
    )
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_C"}  # the foreign batch is not inherited

    # a malformed status file degrades to a fresh batch instead of raising
    (out / "batch_status.yaml").write_text("- not\n- a mapping\n")
    driver.rasterize(
        geometry=aoi_file, output=str(out), projects="WU_D", dst_crs="utm.wkt"
    )
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_D"}


def test_projects_all_is_rejected_with_guidance(tmp_path, aoi_file, monkeypatch):
    """gh #68 postscript: the old 'all' mode blended surveys with overlap
    precedence set by EPT index file order. It is gone, and the error says
    what replaced it."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="auto.*merge"):
        driver.rasterize(aoi_file, str(tmp_path / "b"), projects="all")


def test_projects_auto_selects_and_orders(tmp_path, aoi_file, monkeypatch):
    """auto = catalog.select_workunits order, which is the default merge
    priority."""
    monkeypatch.setattr(driver.catalog, "load_wesm", lambda gdf: "WESM")
    monkeypatch.setattr(driver.catalog, "load_ept_resources", lambda: "EPT")
    monkeypatch.setattr(
        driver.catalog,
        "select_workunits",
        lambda gdf, wesm=None, ept=None: [
            {
                "workunit": "WU_NEW",
                "ql": "QL 1",
                "collect_start": "2023-01-01",
                "collect_end": "2023-02-01",
                "aoi_overlap_frac": 0.6,
                "priority": 1,
            },
            {
                "workunit": "WU_OLD",
                "ql": "QL 2",
                "collect_start": "2016-01-01",
                "collect_end": "2016-02-01",
                "aoi_overlap_frac": 1.0,
                "priority": 2,
            },
        ],
    )
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    driver.rasterize(aoi_file, str(tmp_path / "b"))  # auto is the default
    assert [c["threedep_project"] for c in calls] == ["WU_NEW", "WU_OLD"]
    status = yaml.safe_load((tmp_path / "b" / "batch_status.yaml").read_text())
    assert set(status["projects"]) == {"WU_NEW", "WU_OLD"}


def test_local_input_keys_on_directory_name(tmp_path, aoi_file, monkeypatch):
    """A local run lands in <output>/<input dir name>/ with no WESM
    selection; batch status records it like any project."""
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    laz_dir = tmp_path / "NEON_REDB_laz"
    laz_dir.mkdir()
    driver.rasterize(
        aoi_file, str(tmp_path / "b"), input=str(laz_dir), dst_crs="utm.wkt"
    )
    assert calls[0]["threedep_project"] is None  # engine derives/pins itself
    assert calls[0]["output"] == str(tmp_path / "b" / "NEON_REDB_laz")
    status = yaml.safe_load((tmp_path / "b" / "batch_status.yaml").read_text())
    assert set(status["projects"]) == {"NEON_REDB_laz"}


def test_local_input_rejects_project_lists(tmp_path, aoi_file, monkeypatch):
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    laz_dir = tmp_path / "laz"
    laz_dir.mkdir()
    with pytest.raises(ValueError, match="local input"):
        driver.rasterize(
            aoi_file,
            str(tmp_path / "b"),
            projects="WU_A,WU_B",
            input=str(laz_dir),
            dst_crs="utm.wkt",
        )


def test_single_survey_overrides_rejected_for_batches(tmp_path, aoi_file, monkeypatch):
    """One explicit source CRS or PROJ pipeline cannot be right across
    surveys with different source CRSs."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="exactly one survey"):
        driver.rasterize(
            aoi_file,
            str(tmp_path / "b"),
            projects="WU_A,WU_B",
            src_crs="some.wkt",
            dst_crs="utm.wkt",
        )
    with pytest.raises(ValueError, match="exactly one survey"):
        driver.rasterize(
            aoi_file,
            str(tmp_path / "b"),
            projects="WU_A,WU_B",
            proj_pipeline="+proj=pipeline",
            dst_crs="utm.wkt",
        )


def test_empty_projects_fails_fast(tmp_path, aoi_file, monkeypatch):
    """An unset shell variable must not silently launch a full auto batch."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="empty"):
        driver.rasterize(aoi_file, str(tmp_path / "b"), projects="")


def test_selector_keywords_rejected_inside_lists(tmp_path, aoi_file, monkeypatch):
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="mixes selector"):
        driver.rasterize(aoi_file, str(tmp_path / "b"), projects="auto,WU_A")
    with pytest.raises(ValueError, match="mixes selector"):
        driver.rasterize(aoi_file, str(tmp_path / "b"), projects="WU_A,Latest")


def test_projects_all_rejected_on_local_path_too(tmp_path, aoi_file, monkeypatch):
    """'all' must never become a literal survey name keyed as <output>/all/."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    laz = tmp_path / "laz"
    laz.mkdir()
    with pytest.raises(ValueError, match="removed"):
        driver.rasterize(aoi_file, str(tmp_path / "b"), projects="all", input=str(laz))


def test_output_that_looks_like_a_workunit_list_is_rejected(
    tmp_path, aoi_file, monkeypatch
):
    """Old rasterize-projects positional order (geometry, workunits, output)
    must error, not create a directory named 'WU_A,WU_B'."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="workunit list"):
        driver.rasterize(aoi_file, "WU_A,WU_B", projects=str(tmp_path / "out"))


def test_local_input_dot_resolves_to_real_directory_name(
    tmp_path, aoi_file, monkeypatch
):
    """'--input .' must key on the actual directory name, never '' (which
    collapsed the project dir onto the batch root)."""
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    laz = tmp_path / "my_laz_dir"
    laz.mkdir()
    monkeypatch.chdir(laz)
    driver.rasterize(aoi_file, str(tmp_path / "b"), input=".", dst_crs="utm.wkt")
    assert calls[0]["output"] == str(tmp_path / "b" / "my_laz_dir")
