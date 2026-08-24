"""
Unit tests for the end-to-end `run` command: stage ordering, options
pass-through, and the partial/total-failure semantics. Every stage is
mocked at the lidar_tools.run module namespace — these tests exercise the
orchestration contract, not the stages themselves (each stage has its own
test module, and the Salton Sea e2e gate covers the real chain).
"""

from pathlib import Path

import pytest

from lidar_tools import run as run_mod
from lidar_tools.driver import RasterizeOpts
from lidar_tools.run import run


@pytest.fixture()
def stages(monkeypatch, tmp_path):
    """Mock all four stages, recording call order and arguments.

    The rasterize fake honors _rasterize_batch's return contract:
    (this invocation's status, cumulative batch projects) — seed
    `stages.batch.prior` to simulate completed projects carried forward
    from earlier invocations into the same batch directory."""
    calls = []

    def fake_search(geometry, output=None):
        calls.append(("search", geometry, output))

    def fake_batch(geometry, output, opts):
        calls.append(("rasterize", geometry, output, opts))
        st = dict(fake_batch.status)
        return st, {**fake_batch.prior, **st}

    fake_batch.prior = {}

    def fake_merge(batch_dir, normalize_intensity=True, **kw):
        calls.append(("merge", Path(batch_dir), normalize_intensity))
        return list(fake_merge.written)

    def fake_preview(path):
        calls.append(("preview", path))
        return [Path(path) / "composite-preview.png"]

    fake_batch.status = {"WU_A": "completed", "WU_B": "completed"}
    fake_merge.written = [Path("m/aoi_1m-DSM_mos.vrt")]
    monkeypatch.setattr(run_mod, "search", fake_search)
    monkeypatch.setattr(run_mod, "_rasterize_batch", fake_batch)
    monkeypatch.setattr(run_mod, "merge_projects", fake_merge)
    monkeypatch.setattr(run_mod, "preview", fake_preview)
    stages.calls = calls
    stages.batch = fake_batch
    stages.merge = fake_merge
    return stages


def test_stage_order_and_passthrough(stages, tmp_path):
    out = tmp_path / "batch"
    opts = RasterizeOpts(projects="WU_A,WU_B", resolution=0.5)
    run("aoi.geojson", str(out), opts=opts, normalize_intensity=False)
    names = [c[0] for c in stages.calls]
    assert names == ["search", "rasterize", "merge", "preview"]
    assert stages.calls[0][1] == "aoi.geojson"
    assert stages.calls[0][2] == str(out / "search")
    # the rasterize stage receives the SAME options object — divergence
    # between run and rasterize is impossible by construction
    assert stages.calls[1][3] is opts
    assert stages.calls[2] == ("merge", out, False)
    assert stages.calls[3][1] == str(out / "merge")


def test_default_opts_when_none(stages, tmp_path):
    run("aoi.geojson", str(tmp_path / "b"))
    assert stages.calls[1][3] == RasterizeOpts()


def test_summary_reports_previews_written_by_this_call(stages, tmp_path, capsys):
    """The preview path in the summary comes from preview()'s return,
    never from a directory glob that could name a stale figure."""
    out = tmp_path / "batch"
    (out / "merge").mkdir(parents=True)
    (out / "merge" / "zz_stale_older_run-preview.png").touch()
    run("aoi.geojson", str(out))
    text = capsys.readouterr().out
    assert "preview: " + str(out / "merge" / "composite-preview.png") in text
    assert "zz_stale" not in text


def test_partial_failure_merges_and_raises(stages, tmp_path, capsys):
    stages.batch.status = {"WU_A": "completed", "WU_B": "failed: boom"}
    # RuntimeError, never SystemExit: a Python caller's `except Exception`
    # must be able to contain a partial failure (same protocol as rasterize)
    with pytest.raises(RuntimeError) as exc:
        run("aoi.geojson", str(tmp_path / "b"))
    assert "re-invoke with the same arguments to resume" in str(exc.value)
    names = [c[0] for c in stages.calls]
    assert "merge" in names and "preview" in names
    assert "1 completed, 1 failed" in capsys.readouterr().out


def test_total_failure_with_no_prior_batch_stops_before_merge(stages, tmp_path):
    stages.batch.status = {"WU_A": "failed: boom", "WU_B": "failed: boom"}
    with pytest.raises(RuntimeError, match="2/2 project runs failed"):
        run("aoi.geojson", str(tmp_path / "b"))
    assert [c[0] for c in stages.calls] == ["search", "rasterize"]
    # the summary + durable record close the run BEFORE the raise (round 3)
    text = (tmp_path / "b" / "RUN_SUMMARY.md").read_text()
    assert "0 completed, 2 failed" in text and "exit nonzero:" in text


def test_total_failure_still_merges_prior_completed_projects(stages, tmp_path, capsys):
    """A later invocation whose own projects all fail must still merge the
    batch's previously completed products — a standalone `merge` would,
    and run must not diverge from the step-by-step chain."""
    stages.batch.prior = {"WU_A": "completed"}
    stages.batch.status = {"WU_B": "failed: boom"}
    with pytest.raises(RuntimeError, match="1 project\\(s\\) failed"):
        run("aoi.geojson", str(tmp_path / "b"))
    assert [c[0] for c in stages.calls] == ["search", "rasterize", "merge", "preview"]
    out = capsys.readouterr().out
    assert "run summary" in out
    # the durable record shows the cumulative batch the merge ran over
    assert "batch cumulative: 1 completed of 2 project(s) recorded" in out


def test_no_coverage_lookuperror_gets_summary_and_clean_error(
    stages, monkeypatch, tmp_path, capsys
):
    """An AOI outside 3DEP coverage (auto selection raises LookupError)
    must close with the run summary and a clean RuntimeError, not the
    selection internals' traceback (found by the McMurdo gate run)."""

    def no_coverage(geometry, output, opts):
        raise LookupError("No 3DEP collection intersects this AOI")

    monkeypatch.setattr(run_mod, "_rasterize_batch", no_coverage)
    with pytest.raises(RuntimeError, match="coverage inventory") as exc:
        run("aoi.geojson", str(tmp_path / "b"))
    assert isinstance(exc.value.__cause__, LookupError)
    out = capsys.readouterr().out
    assert "run summary" in out
    assert "No 3DEP collection intersects this AOI" in out


def test_empty_selection_still_prints_summary(stages, tmp_path, capsys):
    """An AOI intersecting nothing must not exit silently — the summary
    prints on every path, so a no-op is distinguishable from a crash."""
    stages.batch.status = {}
    run("aoi.geojson", str(tmp_path / "b"))  # must not raise
    out = capsys.readouterr().out
    assert "run summary" in out
    assert "0 completed, 0 failed" in out
    assert "skipped (no completed projects in the batch)" in out


def test_merge_failure_still_prints_summary_and_exits_nonzero(
    stages, monkeypatch, tmp_path, capsys
):
    def broken_merge(batch_dir, **kw):
        raise ValueError("not on one grid; refusing to merge")

    monkeypatch.setattr(run_mod, "merge_projects", broken_merge)
    with pytest.raises(RuntimeError, match="merge stage failed"):
        run("aoi.geojson", str(tmp_path / "b"))
    out = capsys.readouterr()
    assert "run summary" in out.out
    assert "merge: FAILED: not on one grid" in out.out
    assert "preview: skipped (merge failed)" in out.out


def test_nodata_completion_is_flagged_in_headline(stages, tmp_path, capsys):
    stages.batch.status = {
        "WU_A": "completed (no data): no data (survey does not cover AOI)",
        "WU_B": "completed",
    }
    run("aoi.geojson", str(tmp_path / "b"))
    out = capsys.readouterr().out
    assert "2 completed, 1 of those WITHOUT products (no data), 0 failed" in out


def test_no_merge_skips_merge_and_preview(stages, tmp_path, capsys):
    run("aoi.geojson", str(tmp_path / "b"), merge=False)
    assert [c[0] for c in stages.calls] == ["search", "rasterize"]
    out = capsys.readouterr().out
    assert "merge: skipped (--no-merge)" in out


def test_search_failure_is_advisory(stages, monkeypatch, tmp_path, capsys):
    def broken_search(geometry, output=None):
        raise OSError("WESM unreachable")

    monkeypatch.setattr(run_mod, "search", broken_search)
    run("aoi.geojson", str(tmp_path / "b"))
    assert [c[0] for c in stages.calls] == ["rasterize", "merge", "preview"]
    assert "search stage failed" in capsys.readouterr().err


def test_preview_failure_does_not_mask_success(stages, monkeypatch, tmp_path, capsys):
    def broken_preview(path):
        raise ValueError("no panels")

    monkeypatch.setattr(run_mod, "preview", broken_preview)
    run("aoi.geojson", str(tmp_path / "b"))  # must not raise
    assert "preview: FAILED: no panels" in capsys.readouterr().out


def test_empty_merge_skips_preview(stages, tmp_path, capsys):
    stages.merge.written = []
    run("aoi.geojson", str(tmp_path / "b"))
    assert [c[0] for c in stages.calls] == ["search", "rasterize", "merge"]
    assert "no product mosaics" in capsys.readouterr().out


def test_cli_registration_and_flattened_help(capsys):
    """`run` is registered and RasterizeOpts fields render as ordinary
    top-level options next to run's own flags."""
    from lidar_tools.cli import app

    with pytest.raises(SystemExit):
        app(["run", "--help"])
    out = capsys.readouterr().out
    for flag in [
        "--projects",
        "--resolution",
        "--output-datum",
        "--no-merge",
        "--normalize-intensity",
        "--coord-epoch",
    ]:
        assert flag in out, flag


def test_rasterizeopts_defaults_match_previous_cli():
    """The dataclass defaults are the pre-refactor rasterize defaults."""
    o = RasterizeOpts()
    assert (o.projects, o.input, o.resolution, o.products) == (
        "auto",
        "EPT_AWS",
        1.0,
        "all",
    )
    assert (o.num_process, o.resume, o.cleanup, o.quiet) == (1, True, True, False)
    assert (o.output_datum, o.dsm_gridding_choice, o.tile_size) == (
        "wgs84_g2139",
        "first_idw",
        1.0,
    )
    assert (o.filter_noise, o.ept_vertical, o.geoid_override) == (
        True,
        "auto",
        "declared",
    )
    assert (
        o.src_crs,
        o.dst_crs,
        o.proj_pipeline,
        o.height_above_ground_threshold,
        o.coord_epoch,
    ) == (None, None, None, None, None)


def test_run_summary_md_written_on_success(stages, tmp_path):
    out = tmp_path / "batch"
    run("aoi.geojson", str(out))
    text = (out / "RUN_SUMMARY.md").read_text()
    assert "lidar-tools run summary" in text
    assert "2 completed, 0 failed" in text
    assert "geometry: aoi.geojson" in text


def test_run_summary_md_written_on_partial_failure(stages, tmp_path):
    stages.batch.status = {"WU_A": "completed", "WU_B": "failed: boom"}
    out = tmp_path / "batch"
    with pytest.raises(RuntimeError):
        run("aoi.geojson", str(out))
    text = (out / "RUN_SUMMARY.md").read_text()
    assert "1 completed, 1 failed" in text
    assert "exit nonzero:" in text
    assert "re-invoke with the same arguments to resume" in text


def test_run_summary_md_written_on_no_coverage(stages, monkeypatch, tmp_path):
    def no_coverage(geometry, output, opts):
        raise LookupError("No 3DEP collection intersects this AOI")

    monkeypatch.setattr(run_mod, "_rasterize_batch", no_coverage)
    out = tmp_path / "batch"
    with pytest.raises(RuntimeError):
        run("aoi.geojson", str(out))
    text = (out / "RUN_SUMMARY.md").read_text()
    assert "No 3DEP collection intersects this AOI" in text
    assert "exit nonzero:" in text
