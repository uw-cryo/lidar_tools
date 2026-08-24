import geopandas as gpd
import pytest
import shapely
import yaml

from lidar_tools import driver


def _rasterize(geometry, output, **kw):
    """Call the public rasterize API, funneling every option through
    RasterizeOpts — each call site exercises the dataclass contract."""
    driver.rasterize(geometry, output, opts=driver.RasterizeOpts(**kw))


def test_rasterize_without_opts_uses_defaults(tmp_path, aoi_file, monkeypatch):
    """The documented default usage is rasterize(geometry, output) with no
    opts at all — the `opts or RasterizeOpts()` fallback must construct
    the full default set (every helper call above passes opts explicitly,
    so only this test covers the bare-call path)."""
    calls = []
    monkeypatch.setattr(driver.catalog, "select_workunits", lambda *a: [])
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    driver.rasterize(aoi_file, str(tmp_path / "b"))
    # auto selection with the stub catalog selects nothing; the defaults
    # got as far as project resolution without a populated opts object
    assert calls == []


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    """The EPT path now loads WESM + the EPT index once per batch; tests
    must never fetch them. Tests that need real selection behavior
    override these stubs themselves."""
    monkeypatch.setattr(driver.catalog, "load_wesm", lambda gdf: "WESM_STUB")
    monkeypatch.setattr(driver.catalog, "load_ept_resources", lambda: "EPT_STUB")


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
    _rasterize(
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
    _rasterize(aoi_file, str(outbase), projects="WU_A,WU_B", output_datum="nad83_2011")
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
        _rasterize(aoi_file, str(outbase), projects="WU_A, WU_B")
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
    _rasterize(aoi_file, str(outbase), projects="WU_A,WU_B")
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
    _rasterize(aoi_file, str(outbase), projects="WU_A")
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
    _rasterize(aoi_file, str(tmp_path / "b1"), projects="WU_A")
    assert seen[0]["geoid_override"] == "declared"  # hard-fail is the default
    _rasterize(
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

    _rasterize(geometry=aoi_file, output=str(out), projects="WU_C", dst_crs="utm.wkt")
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
    _rasterize(geometry=aoi_file, output=str(out), projects="WU_C", dst_crs="utm.wkt")
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_C"}  # the foreign batch is not inherited

    # a malformed status file degrades to a fresh batch instead of raising
    (out / "batch_status.yaml").write_text("- not\n- a mapping\n")
    _rasterize(geometry=aoi_file, output=str(out), projects="WU_D", dst_crs="utm.wkt")
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_D"}


def test_projects_all_is_rejected_with_guidance(tmp_path, aoi_file, monkeypatch):
    """gh #68 postscript: the old 'all' mode blended surveys with overlap
    precedence set by EPT index file order. It is gone, and the error says
    what replaced it."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="auto.*merge"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="all")


def test_projects_auto_selects_and_orders(tmp_path, aoi_file, monkeypatch):
    """auto = catalog.select_workunits order, which is the default merge
    priority."""
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
    _rasterize(aoi_file, str(tmp_path / "b"))  # auto is the default
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
    _rasterize(aoi_file, str(tmp_path / "b"), input=str(laz_dir), dst_crs="utm.wkt")
    assert calls[0]["threedep_project"] is None  # engine derives/pins itself
    assert calls[0]["output"] == str(tmp_path / "b" / "NEON_REDB_laz")
    status = yaml.safe_load((tmp_path / "b" / "batch_status.yaml").read_text())
    assert set(status["projects"]) == {"NEON_REDB_laz"}


def test_local_input_rejects_project_lists(tmp_path, aoi_file, monkeypatch):
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    laz_dir = tmp_path / "laz"
    laz_dir.mkdir()
    with pytest.raises(ValueError, match="local input"):
        _rasterize(
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
        _rasterize(
            aoi_file,
            str(tmp_path / "b"),
            projects="WU_A,WU_B",
            src_crs="some.wkt",
            dst_crs="utm.wkt",
        )
    with pytest.raises(ValueError, match="exactly one survey"):
        _rasterize(
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
        _rasterize(aoi_file, str(tmp_path / "b"), projects="")


def test_selector_keywords_rejected_inside_lists(tmp_path, aoi_file, monkeypatch):
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="mixes selector"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="auto,WU_A")
    with pytest.raises(ValueError, match="mixes selector"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="WU_A,Latest")


def test_projects_all_rejected_on_local_path_too(tmp_path, aoi_file, monkeypatch):
    """'all' must never become a literal survey name keyed as <output>/all/."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    laz = tmp_path / "laz"
    laz.mkdir()
    with pytest.raises(ValueError, match="removed"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="all", input=str(laz))


def test_output_that_looks_like_a_workunit_list_is_rejected(
    tmp_path, aoi_file, monkeypatch
):
    """Old rasterize-projects positional order (geometry, workunits, output)
    must error, not create a directory named 'WU_A,WU_B'."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="workunit list"):
        _rasterize(aoi_file, "WU_A,WU_B", projects=str(tmp_path / "out"))


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
    _rasterize(aoi_file, str(tmp_path / "b"), input=".", dst_crs="utm.wkt")
    assert calls[0]["output"] == str(tmp_path / "b" / "my_laz_dir")


def test_explicit_lists_share_one_catalog_fetch(tmp_path, aoi_file, monkeypatch):
    """An explicit N-project batch loads WESM + the EPT index once and
    hands the frames to every engine run (no per-project refetch)."""
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    _rasterize(aoi_file, str(tmp_path / "b"), projects="WU_A,WU_B")
    assert len(calls) == 2
    assert all(c["wesm_gdf"] == "WESM_STUB" for c in calls)
    assert all(c["ept_index_gdf"] == "EPT_STUB" for c in calls)


def test_coord_epoch_static_datum_fails_fast_from_driver(
    tmp_path, aoi_file, monkeypatch
):
    """The engine's coord_epoch+nad83_2011 guard cannot fire from the CLI
    (the driver resolves dst_crs first), so the driver must run it."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="plate-fixed"):
        _rasterize(
            aoi_file,
            str(tmp_path / "b"),
            projects="WU_A",
            output_datum="nad83_2011",
            coord_epoch=2025.0,
        )


def test_path_shaped_projects_token_is_rejected(tmp_path, aoi_file, monkeypatch):
    """`rasterize aoi.geojson WU_A batch/` (removed positional order, single
    workunit) binds 'batch/' into projects: error with the new signature."""
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(ValueError, match="contains a path"):
        _rasterize(aoi_file, "WU_A", projects="batch/")


def test_comma_output_allowed_when_directory_exists(tmp_path, aoi_file, monkeypatch):
    """The old-signature guard must not permanently ban comma-named output
    directories: pre-creating one is the escape hatch."""
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    out = tmp_path / "a,b"
    out.mkdir()
    _rasterize(aoi_file, str(out), projects="WU_A", dst_crs="utm.wkt")
    assert len(calls) == 1


def test_catalog_prefetch_failure_degrades_for_explicit_lists(
    tmp_path, aoi_file, monkeypatch, capsys
):
    """A transient catalog read must not abort a whole batch before any
    project runs: explicit lists fall back to per-project fetching."""

    def boom(*a, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(driver.catalog, "load_wesm", boom)
    calls = []
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: calls.append(kw))
    _rasterize(aoi_file, str(tmp_path / "b"), projects="WU_A,WU_B")
    assert [c["threedep_project"] for c in calls] == ["WU_A", "WU_B"]
    assert all(c["wesm_gdf"] is None for c in calls)  # engine fetches its own
    assert "could not pre-load the catalog" in capsys.readouterr().err


def test_catalog_prefetch_failure_still_fatal_for_selectors(
    tmp_path, aoi_file, monkeypatch
):
    """auto/latest cannot proceed without the catalog, so the failure stays."""

    def boom(*a, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(driver.catalog, "load_wesm", boom)
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    with pytest.raises(OSError, match="connection reset"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="auto")


def test_resume_refusal_advice_is_not_rerun_the_same_command(
    tmp_path, aoi_file, monkeypatch
):
    """gh review: the batch advice told users to repeat a command that is
    deterministically going to fail the same way."""

    def refuse(**kw):
        raise ValueError("Cannot resume into this output directory: changed")

    monkeypatch.setattr(driver, "rasterize_project", refuse)
    with pytest.raises(RuntimeError, match="cannot resume into their existing"):
        _rasterize(aoi_file, str(tmp_path / "b"), projects="WU_A")


def test_batch_status_carries_forward_after_relocation(tmp_path, aoi_file, monkeypatch):
    """Moving a batch dir to another volume moves its WKT with it: the
    carry-forward check must not orphan the record over the stale absolute
    dst_crs path (the PCD-sweep clobber, 2026-08-23)."""
    import pyproj

    out = tmp_path / "batch"
    out.mkdir()
    (out / "UTM_12N_WGS84_G2139_3D.wkt").write_text(
        pyproj.CRS.from_epsg(32612).to_wkt()
    )
    (out / "batch_status.yaml").write_text(
        yaml.dump(
            {
                "geometry": aoi_file,
                # recorded before the move: absolute path that no longer exists
                "dst_crs": "/old/volume/batch/UTM_12N_WGS84_G2139_3D.wkt",
                "projects": {"WU_A": "completed"},
            }
        )
    )
    monkeypatch.setattr(driver, "rasterize_project", lambda **kw: None)
    _rasterize(
        aoi_file,
        str(out),
        projects="WU_B",
        dst_crs=str(out / "UTM_12N_WGS84_G2139_3D.wkt"),
    )
    projects = yaml.safe_load((out / "batch_status.yaml").read_text())["projects"]
    assert set(projects) == {"WU_A", "WU_B"}
