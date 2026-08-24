"""
The one-command end-to-end workflow: search -> rasterize -> merge ->
composite preview.

Thin orchestration only: every stage IS the public command it names,
called with the same defaults it has standalone, and the rasterize stage
receives the caller's options object unmodified — so `run` and the
step-by-step commands cannot diverge. The batch directory is the contract
between stages (search/, per-project subdirs + batch_status.yaml, merge/),
and re-invoking is safe: rasterize's resume guard skips valid tiles and
the merge/preview stages are idempotent rebuilds.
"""

import sys
from datetime import datetime
from pathlib import Path

from lidar_tools.catalog import search
from lidar_tools.driver import (
    RasterizeOpts,
    _completed_projects,
    _failed_projects,
    _failure_advice,
    _failure_message,
    _nodata_projects,
    _rasterize_batch,
)
from lidar_tools.merge import merge_projects
from lidar_tools.preview import preview


def _write_run_summary(
    outbase: Path, geometry: str, lines: list[str], failure: str | None
) -> None:
    """Persist the closing summary as <output>/RUN_SUMMARY.md: the batch
    directory's durable outcome record, written on every exit path so a
    directory reviewed later carries its own verdict."""
    body = [
        "# lidar-tools run summary",
        "",
        f"- generated: {datetime.now().astimezone().isoformat()}",
        f"- geometry: {geometry}",
        f"- batch dir: {outbase}",
        "",
        "```",
        *lines,
        "```",
    ]
    if failure:
        body += ["", f"exit nonzero: {failure}"]
    (outbase / "RUN_SUMMARY.md").write_text("\n".join(body) + "\n")


def run(
    geometry: str,
    output: str,
    *,
    opts: RasterizeOpts | None = None,
    normalize_intensity: bool = True,
    merge: bool = True,
) -> None:
    """
    End-to-end pipeline for an AOI: catalog search (provenance), one
    rasterize pass per covering survey on a shared grid, a priority merge
    of the per-project products, and a composite preview figure.

    A failed project does not abort the batch: the merge and preview run
    over the batch's completed projects, the closing summary lists both,
    and a RuntimeError (nonzero exit) closes the run. Re-invoke with the
    same arguments to resume.

    Parameters
    ----------
    geometry
        Path to the AOI polygon.
    output
        Batch output directory (same layout as ``rasterize``, plus
        ``search/`` and ``merge/``).
    opts
        Processing options, identical to ``rasterize``'s (they appear as
        ordinary top-level options; see RasterizeOpts).
    normalize_intensity
        Passed to the merge stage: map each project's intensity onto a
        common range in the merge VRT (see ``merge``). Default True.
    merge
        ``--no-merge`` skips the merge and composite-preview stages,
        leaving only the per-project products.

    Returns
    -------
    None
    """
    opts = opts or RasterizeOpts()
    outbase = Path(output)
    outbase.mkdir(parents=True, exist_ok=True)

    # 1. Search: provenance for the batch (surveys.gpkg records every
    # covering collection, EPT-backed or not; coverage_gaps.gpkg records
    # what nothing covers). Advisory: rasterize resolves the catalog
    # itself, so a transient search failure must not kill the run.
    # (The catalog — WESM bbox read and EPT index — is read twice per run,
    # here and in the rasterize stage; measured at ~3 s / ~0 s cached and
    # judged not worth shared-frame plumbing when this command was
    # designed. Revisit if large-AOI runs show otherwise.)
    print(f"===== search -> {outbase / 'search'} =====")
    try:
        search(geometry, output=str(outbase / "search"))
    except Exception as e:
        print(
            f"WARNING: search stage failed ({e}); continuing — the "
            "rasterize stage does its own catalog resolution",
            file=sys.stderr,
        )

    # 2. Rasterize: the shared batch loop; project selection stays inside
    # it (--projects auto by default) so run/rasterize cannot diverge.
    # batch_projects is the CUMULATIVE batch as recorded on disk —
    # completed projects from earlier runs into this directory included.
    try:
        status, batch_projects = _rasterize_batch(geometry, output, opts)
    except LookupError as e:
        # expected outcome when the selection has nothing processable —
        # either no 3DEP coverage at all, or coverage whose EPT builds do
        # not exist/resolve (two distinct catalog messages: carry the
        # actual reason rather than a canned line that may be wrong).
        # The search stage above already printed the full inventory, so
        # close with the summary and a clean error instead of the
        # selection internals' traceback.
        reason = str(e).split(". ")[0]
        lines = [
            "projects: 0 completed, 0 failed",
            f"merge: skipped ({reason})",
            f"preview: skipped ({reason})",
        ]
        print("\n===== run summary =====")
        print("\n".join(lines))
        msg = f"{e} The search stage above shows the coverage inventory."
        _write_run_summary(outbase, geometry, lines, msg)
        raise RuntimeError(msg) from e
    failed = _failed_projects(status)
    nodata = _nodata_projects(status)

    # 3 + 4. Merge over the cumulative batch (a standalone `merge` would
    # composite earlier runs' completed projects too — run must not
    # refuse where the step-by-step chain succeeds), then composite
    # preview. With nothing completed anywhere in the batch, fail with
    # rasterize's exact message — AFTER the summary and RUN_SUMMARY.md,
    # which close the run on every path; an empty selection is a no-op
    # summary with exit 0.
    written: list[Path] = []
    merge_error: Exception | None = None
    no_batch_data = not _completed_projects(batch_projects)
    if no_batch_data:
        merge_note = preview_note = "skipped (no completed projects in the batch)"
    elif not merge:
        merge_note = preview_note = "skipped (--no-merge)"
    else:
        print(f"\n===== merge -> {outbase / 'merge'} =====")
        try:
            written = merge_projects(outbase, normalize_intensity=normalize_intensity)
        except Exception as e:
            # summary + resume advice must still print (a stale
            # mismatched-grid mosaic raising here is a real failure, but
            # not one that should eat the batch report)
            merge_error = e
            merge_note = f"FAILED: {e}"
            preview_note = "skipped (merge failed)"
            print(f"WARNING: merge stage failed: {e}", file=sys.stderr)
        else:
            if written:
                merge_note = (
                    f"{len(written)} product composite(s) in {outbase / 'merge'}"
                )
                print(f"\n===== preview -> {outbase / 'merge'} =====")
                try:
                    pngs = preview(str(outbase / "merge"))
                    # report what THIS call wrote, never a directory glob
                    # that could name a stale figure
                    preview_note = (
                        "; ".join(str(p) for p in pngs) if pngs else "no PNG written"
                    )
                except Exception as e:
                    # a QA figure failure must not mask a successful batch
                    preview_note = f"FAILED: {e}"
                    print(f"WARNING: preview stage failed: {e}", file=sys.stderr)
            else:
                merge_note = "no product mosaics found"
                preview_note = "skipped (no product mosaics)"

    parts = [f"{len(_completed_projects(status))} completed"]
    if nodata:
        parts.append(f"{len(nodata)} of those WITHOUT products (no data)")
    parts.append(f"{len(failed)} failed")
    lines = [f"projects: {', '.join(parts)}"]
    lines += [f"  {w}: {s}" for w, s in status.items()]
    lines.append(f"merge: {merge_note}")
    lines += [f"  {p}" for p in written]
    lines.append(f"preview: {preview_note}")
    if set(batch_projects) != set(status):
        # re-invocation: the merge gate ran on the CUMULATIVE batch, so
        # the durable record must show it alongside this invocation
        lines.append(
            f"batch cumulative: {len(_completed_projects(batch_projects))} "
            f"completed of {len(batch_projects)} project(s) recorded"
        )
    print("\n===== run summary =====")
    print("\n".join(lines))
    failure: str | None = None
    if no_batch_data:
        # fail exactly like rasterize (same message), after the summary
        failure = _failure_message(status)
    elif failed or merge_error is not None:
        msgs = []
        if failed:
            msgs.append(f"{len(failed)} project(s) failed ({_failure_advice(status)})")
        if merge_error is not None:
            msgs.append(f"merge stage failed: {merge_error}")
        failure = "; ".join(msgs)
    _write_run_summary(outbase, geometry, lines, failure)
    if failure:
        # same failure protocol as rasterize: a RuntimeError the CLI turns
        # into a nonzero exit. Never SystemExit — that would tear through a
        # Python caller's `except Exception` and kill its batch loop.
        raise RuntimeError(failure)
