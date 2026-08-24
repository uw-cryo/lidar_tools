"""
The public rasterize command: select the surveys covering an AOI, then run
the per-project engine once per selection into per-project subdirectories
on one shared target grid.

Products stay per-project on disk so quality levels and acquisition dates
remain isolated (separate epochs = separate product sets); any combined
product is the explicit `merge` step, where overlap precedence is a
stated priority order — never an implicit side effect of processing.
The layout is uniform: a single-project run writes `<output>/<project>/`
exactly like a batch, so merge/preview/fetch-reports/report-metrics
consume every run the same way.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cyclopts
import geopandas as gpd
import yaml

from lidar_tools import catalog, geodesy
from lidar_tools.pdal_pipeline import rasterize_project


def _project_run_status(outdir: Path) -> dict:
    """
    Read the run_status block from a project's processing metadata
    (newest ``*processing_metadata.yaml``, covering both prefixed and
    legacy bare names). Empty dict when absent; unreadable metadata is
    WARNED about, never swallowed — a corrupt YAML must not let a
    no-data run masquerade as a plain success unnoticed.
    """
    metas = sorted(
        Path(outdir).glob("*processing_metadata.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for meta in metas:
        try:
            with open(meta) as f:
                content = yaml.safe_load(f) or {}
            return content.get("run_status") or {}
        except Exception as e:
            print(
                f"WARNING: unreadable processing metadata {meta} ({e}); "
                "cannot verify whether this run produced products",
                file=sys.stderr,
            )
            return {}
    return {}


def _resolve_projects(
    projects: str,
    input: str,
    gdf: gpd.GeoDataFrame,
) -> tuple[list[str | None], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Turn the `projects` selection into an ordered list of concrete keys.

    EPT path: "auto" = every EPT-backed intersecting survey in
    rank_collections priority order; "latest" = the newest EPT-backed
    survey; otherwise the given comma-separated workunit names, in
    priority order. Local path: at most one name (it labels the survey
    the files belong to, for WESM record pinning); default is the input
    directory's name, with `None` passed to the engine so pinning stays
    best-effort.

    Returns (keys, wesm_gdf, ept_gdf): on the EPT path the frames are
    loaded once here -- the selectors need them anyway, and explicit
    lists need them for per-project pinning/name resolution -- and handed
    back so every engine run reuses them instead of refetching. Local
    runs return (keys, None, None).
    """
    tokens = [t.strip() for t in str(projects).split(",") if t.strip()]
    if not tokens:
        # an empty value is a broken invocation (unset shell variable, a
        # failed upstream query), not a request for everything: auto must
        # be asked for by name — silently launching an N-project batch
        # here cost hours on the old path's equivalent mistakes
        raise ValueError(
            "--projects is empty: pass 'auto' explicitly, 'latest', or a "
            "comma-separated workunit list"
        )
    if any("/" in t for t in tokens):
        # workunit names never contain path separators; this is almost
        # always the removed rasterize-projects positional order
        # (geometry, workunits, output) binding a path into the projects
        # slot
        raise ValueError(
            f"--projects '{projects}' contains a path; the signature is: "
            "rasterize <geometry> <output> --projects A,B,C"
        )
    # selectors are recognized case-insensitively, and never as list
    # members: 'auto,WU_A' is a mistake, not a selection
    selectors = {t.casefold() for t in tokens} & {"auto", "latest", "all"}
    if selectors and len(tokens) > 1:
        raise ValueError(
            f"--projects '{projects}' mixes selector keyword(s) "
            f"{sorted(selectors)} into a workunit list; pass one selector "
            "alone, or only workunit names"
        )
    selector = tokens[0].casefold() if selectors else None
    if selector == "all":
        raise ValueError(
            "--projects all was removed: it blended every intersecting "
            "survey into one mosaic with overlap precedence set by EPT "
            "index file order. Use --projects auto (per-project products "
            "on one grid) and `lidar-tools merge` for an explicit, "
            "priority-ordered composite."
        )
    if input != "EPT_AWS":
        # local LAZ: one input directory is one project's data — a list
        # cannot map onto it, and auto-selection cannot enumerate a
        # directory into surveys
        if len(tokens) > 1 or selector == "latest":
            raise ValueError(
                f"--projects '{projects}' cannot apply to a local input "
                f"({input}): pass at most one name (the survey the files "
                "belong to), or leave the default"
            )
        if selector == "auto":
            # engine keys the run on the input dir name
            return [None], None, None
        return [tokens[0]], None, None
    # One catalog fetch per batch, shared by selection and by every
    # per-project run. A selector cannot proceed without it; an explicit
    # list can (each engine run fetches its own), so a transient read
    # failure degrades to the slower path instead of aborting the batch
    # before any project has run.
    try:
        wesm_gdf = catalog.load_wesm(gdf)
        ept_gdf = catalog.load_ept_resources()
    except Exception as e:
        if selector in ("auto", "latest"):
            raise
        print(
            f"WARNING: could not pre-load the catalog ({e}); each project "
            "will fetch it individually",
            file=sys.stderr,
        )
        wesm_gdf = ept_gdf = None
    if selector == "auto":
        selected = catalog.select_workunits(gdf, wesm_gdf, ept_gdf)
        print(f"Auto-selected {len(selected)} EPT-backed survey(s):")
        for rec in selected:
            print(
                f"  {rec['priority']}. {rec['workunit']} (ql {rec['ql']}, "
                f"collected {rec['collect_start']} - {rec['collect_end']}, "
                f"{rec['aoi_overlap_frac']:.1%} of AOI)"
            )
        return [rec["workunit"] for rec in selected], wesm_gdf, ept_gdf
    if selector == "latest":
        latest = catalog.select_latest_workunit(gdf, wesm_gdf, ept_gdf)
        if latest["undated"]:
            print(
                f"WARNING: no acquisition dates for any of the "
                f"{latest['n_candidates']} EPT-backed collection(s) here; "
                f"selected '{latest['workunit']}' (widest AOI coverage), "
                "which may not be the most recent",
                file=sys.stderr,
            )
        else:
            print(
                f"Latest survey selected: {latest['workunit']} "
                f"(collected {latest['collect_start']} - "
                f"{latest['collect_end']}, ql {latest['ql']}; "
                f"{latest['n_candidates']} EPT-backed candidate(s))"
            )
        if latest["aoi_overlap_frac"] < 0.95:
            # the most recent survey is frequently a sliver of the AOI, so
            # a single-survey run silently leaves most of it empty (gh #68)
            print(
                f"WARNING: {latest['workunit']} covers only "
                f"{latest['aoi_overlap_frac']:.1%} of the AOI, so this run "
                f"will have no data over the rest. {latest['n_candidates']} "
                "collection(s) cover this AOI — use `lidar-tools search` to "
                "inspect them, or --projects auto to process all of them.",
                file=sys.stderr,
            )
        return [latest["workunit"]], wesm_gdf, ept_gdf
    explicit: list[str | None] = list(tokens)
    return explicit, wesm_gdf, ept_gdf


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class RasterizeOpts:
    """
    The processing options of ``rasterize``, shared verbatim by ``run``
    (which hands the same populated object to the rasterize stage, so the
    two entry points cannot diverge). On the command line the fields
    appear as ordinary top-level options.

    Parameters
    ----------
    projects
        Which survey(s) to process, resolved against WESM/EPT:
        "auto" (default) = every AOI-intersecting survey with an EPT
        build, in catalog priority order (the order is the default merge
        priority; reordering later is a re-merge, not a re-run);
        "latest" = the most recently collected EPT-backed survey only;
        one or more comma-separated WESM workunit names = exactly those,
        in the given priority order. With a local ``--input``, at most
        one name (it labels the survey the files belong to, for WESM
        record pinning).
    input
        Directory of classified LAS/LAZ files to read instead of the
        3DEP EPT service on AWS (non-3DEP sources — NEON, NCALM, vendor
        deliveries — enter here until catalog adapters land).
    resolution
        Shared output posting in target CRS units.
    products
        Comma-separated product selection (e.g. "all", "dsm,intensity").
    num_process
        Worker count for the per-tile PDAL pipelines.
    resume
        Continue interrupted per-project runs (skip existing valid
        tiles), by default True — a failed batch re-invoked as-is picks
        up where it stopped. To rebuild a project from scratch, delete
        its subdirectory.
    cleanup
        Remove per-tile intermediates after each project run.
    quiet
        Suppress dask progress bars.
    src_crs
        Path to a CRS definition overriding the input files' CRS.
        Single-project/local runs only: one override cannot be right
        across surveys with different source CRSs.
    dst_crs
        Optional path to a target CRS definition shared by all projects.
        Default: a 3D UTM CRS built from the AOI (datum per
        `output_datum`) and written to the base directory once.
    output_datum
        Datum realization of the auto-built shared UTM target, used only
        when `dst_crs` is not given. Dynamic frames: 'wgs84_g2139'
        (default), 'wgs84_g1674', 'itrf2020', 'itrf2014', 'itrf2008'.
        Static: 'nad83_2011' (the source realization of 3DEP;
        ellipsoidal heights, no epoch stamp).
    dsm_gridding_choice
        DSM gridding strategy (see the engine docs), applied to every
        project.
    tile_size
        Processing tile size in km.
    proj_pipeline
        Explicit PROJ pipeline for the point transformations.
        Single-project/local runs only, for the same reason as src_crs.
    filter_noise
        Remove classified noise points before gridding.
    height_above_ground_threshold
        When set, points this far above the nearest ground points are
        reclassified as high noise.
    ept_vertical
        Vertical interpretation override, applied to every project (use
        separate runs when collections need different overrides).
    geoid_override
        'declared' (default) hard-fails when a survey's declared
        production geoid cannot be used; 'best-available' consciously
        accepts model substitution.
    coord_epoch
        Coordinate epoch (decimal year) stamped on the output CRS when
        the target datum is dynamic.
    """

    projects: str = "auto"
    input: str = "EPT_AWS"
    resolution: float = 1.0
    products: str = "all"
    num_process: int = 1
    resume: bool = True
    cleanup: bool = True
    quiet: bool = False
    src_crs: str | None = None
    dst_crs: str | None = None
    output_datum: Literal[
        "wgs84_g2139",
        "nad83_2011",
        "wgs84_g1674",
        "itrf2020",
        "itrf2008",
        "itrf2014",
    ] = "wgs84_g2139"
    dsm_gridding_choice: str = "first_idw"
    tile_size: float = 1.0
    proj_pipeline: str | None = None
    filter_noise: bool = True
    height_above_ground_threshold: float | None = None
    ept_vertical: Literal["auto", "geoid", "ellipsoid"] = "auto"
    geoid_override: Literal["declared", "best-available"] = "declared"
    coord_epoch: float | None = None


def rasterize(
    geometry: str,
    output: str,
    *,
    opts: RasterizeOpts | None = None,
) -> None:
    """
    Create DSM, DTM (with and without gap filling) and/or intensity
    rasters for an AOI: one subdirectory per selected survey, all on one
    shared target grid, so the per-project products are co-registered and
    `merge` can composite them without resampling.

    Parameters
    ----------
    geometry
        Path to the AOI polygon (same AOI for every project).
    output
        Base output directory; each project writes to
        ``<output>/<project>/`` and the batch status accumulates in
        ``<output>/batch_status.yaml`` — also for single-project runs, so
        every downstream command consumes every run the same way.
    opts
        Processing options (see RasterizeOpts); on the command line the
        fields are ordinary top-level options.

    Returns
    -------
    None
    """
    status, _ = _rasterize_batch(geometry, output, opts or RasterizeOpts())
    _raise_on_failed(status)


def _rasterize_batch(
    geometry: str, output: str, opts: RasterizeOpts
) -> tuple[dict[str, str], dict[str, str]]:
    """
    The batch loop shared by ``rasterize`` and ``run``: validate, resolve
    the shared target grid, run the engine once per selected project,
    record batch_status.yaml, and print the batch summary. Returns
    ``(status, batch_projects)``: this invocation's per-project status,
    and the cumulative batch mapping as written to batch_status.yaml
    (carried-forward earlier runs included). Raising on failures is the
    caller's responsibility (`rasterize` always raises, `run` merges the
    batch's completed projects first).
    """
    # dst_crs is the one option the body reassigns (resolved to a concrete
    # CRS file below); everything else is read via opts.<field> directly
    dst_crs = opts.dst_crs

    if (
        opts.coord_epoch is not None
        and dst_crs is None
        and opts.output_datum == "nad83_2011"
    ):
        # mirrors the engine's fail-fast, which cannot fire from here
        # because dst_crs is resolved to a concrete file below
        raise ValueError(
            "--coord-epoch applies to dynamic-frame targets only: "
            "NAD83(2011) is plate-fixed (epoch-invariant), so there is no "
            "epoch-dependent transformation to pin. Drop --coord-epoch or "
            "choose a dynamic --output-datum (e.g. itrf2020, wgs84_g2139)."
        )
    if not Path(output).exists() and "," in Path(output).name:
        # the removed rasterize-projects command took (geometry, workunits,
        # output); an old-style invocation binds the workunit list here and
        # would create a directory literally named after it
        raise ValueError(
            f"output '{output}' looks like a workunit list; the signature "
            "is: rasterize <geometry> <output> --projects A,B,C. (If the "
            "output directory really is named with a comma, create it "
            "first.)"
        )
    gdf = gpd.read_file(geometry)
    keys, wesm_gdf, ept_gdf = _resolve_projects(opts.projects, opts.input, gdf)

    if len(keys) > 1 and (opts.src_crs or opts.proj_pipeline):
        # one explicit source CRS or PROJ pipeline cannot be right across
        # projects with different source CRSs — silently applying it to
        # all of them would be a correctness bug, not a convenience
        raise ValueError(
            "--src-crs/--proj-pipeline apply to exactly one survey; "
            f"this selection has {len(keys)}. Run those projects "
            "individually, or drop the override."
        )

    outbase = Path(output)
    outbase.mkdir(parents=True, exist_ok=True)

    if dst_crs is None:
        utm_crs = gdf.estimate_utm_crs()
        epsg_code = utm_crs.to_epsg() if utm_crs is not None else None
        if epsg_code is None:
            raise ValueError(
                "Could not derive a UTM EPSG code for the AOI "
                "(estimate_utm_crs found no exact EPSG match — polar or "
                "zone-spanning AOI?). Pass --dst-crs explicitly."
            )
        out_crs_obj, wkt_name = geodesy.build_utm_target(epsg_code, opts.output_datum)
        target = outbase / wkt_name
        if not target.exists():
            geodesy.write_crs_file(out_crs_obj, target)
        dst_crs = str(target)
    print(f"Shared target grid: {dst_crs} at {opts.resolution} m")

    status = {}
    for key in keys:
        # local runs without a project label key on the input dir name,
        # mirroring the engine's own filename/pinning convention
        dirname = key if key is not None else Path(opts.input).resolve().name
        if not dirname:
            raise ValueError(
                f"cannot derive a project key from input '{opts.input}' "
                "(filesystem root?); pass --projects <name> to label the run"
            )
        outdir = outbase / dirname
        print(f"\n===== {dirname} -> {outdir} =====")
        try:
            rasterize_project(
                geometry=geometry,
                input=opts.input,
                output=str(outdir),
                src_crs=opts.src_crs,
                dst_crs=dst_crs,
                output_datum=opts.output_datum,
                resolution=opts.resolution,
                dsm_gridding_choice=opts.dsm_gridding_choice,
                products=opts.products,
                threedep_project=key,
                tile_size=opts.tile_size,
                num_process=opts.num_process,
                cleanup=opts.cleanup,
                proj_pipeline=opts.proj_pipeline,
                filter_noise=opts.filter_noise,
                height_above_ground_threshold=opts.height_above_ground_threshold,
                quiet=opts.quiet,
                ept_vertical=opts.ept_vertical,
                geoid_override=opts.geoid_override,
                resume=opts.resume and outdir.exists(),
                coord_epoch=opts.coord_epoch,
                wesm_gdf=wesm_gdf,
                ept_index_gdf=ept_gdf,
            )
            # a clean return is NOT proof of products: a 0-reader run
            # records "no data" in its run_status note and must never be
            # reported as a plain success in the batch. Match the specific
            # state+note the pipeline writes — an unrelated future note
            # must not flip a products-bearing run to "(no data)".
            run_status = _project_run_status(outdir)
            note = run_status.get("note") or ""
            if run_status.get("state") == "completed" and "no data" in note:
                status[dirname] = f"completed (no data): {note}"
                print(
                    f"WARNING: {dirname} completed WITHOUT products: {note}",
                    file=sys.stderr,
                )
            else:
                status[dirname] = "completed"
        except Exception as e:
            # one failed project must not take down the rest of the batch
            print(f"ERROR: {dirname} failed: {e}")
            status[dirname] = f"failed: {e}"

    # Carry forward projects from earlier invocations into the same batch
    # directory: merge / preview / fetch-reports / report-metrics all default
    # to the workunits recorded here, so overwriting with just this run's
    # list silently shrinks their input (Casa Grande: a later single-project
    # run left the 5-project batch listing one, and the re-merge had to name
    # all five by hand).
    status_fn = outbase / "batch_status.yaml"
    projects_rec: dict = {}
    if status_fn.exists():
        prior: object = {}
        try:
            with open(status_fn) as f:
                prior = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(
                f"WARNING: {status_fn} is not readable YAML ({e}); starting a "
                "fresh batch status",
                file=sys.stderr,
            )
        if not isinstance(prior, dict):
            print(
                f"WARNING: {status_fn} is not a mapping; starting a fresh batch status",
                file=sys.stderr,
            )
            prior = {}
        # Only carry projects forward within the SAME batch: the downstream
        # defaults (merge/preview/fetch-reports/report-metrics) act on this
        # list, so inheriting another AOI's or grid's projects would point
        # them at products that do not belong together.
        same_batch = str(prior.get("geometry")) == str(geometry) and _same_crs_file(
            prior.get("dst_crs"), dst_crs
        )
        if prior and not same_batch:
            print(
                f"WARNING: {status_fn} records a different AOI/target grid "
                f"({prior.get('geometry')}, {prior.get('dst_crs')}); its "
                "projects are NOT carried forward",
                file=sys.stderr,
            )
        elif same_batch:
            carried = prior.get("projects")
            if isinstance(carried, dict):
                projects_rec = dict(carried)
            elif carried is not None:
                print(
                    f"WARNING: 'projects' in {status_fn} is not a mapping; ignoring it",
                    file=sys.stderr,
                )
    projects_rec.update(status)
    with open(status_fn, "w") as f:
        yaml.dump(
            {
                "geometry": str(geometry),
                "dst_crs": str(dst_crs),
                "projects": projects_rec,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print("\nBatch summary:")
    for dirname, state in status.items():
        print(f"  {dirname}: {state}")
    nodata = _nodata_projects(status)
    if nodata:
        print(
            f"WARNING: {len(nodata)}/{len(status)} project runs produced NO "
            f"products: {nodata} — check EPT availability/name resolution or "
            "use the local point-cloud path (rasterize --input)",
            file=sys.stderr,
        )
    return status, projects_rec


def _same_crs_file(a, b) -> bool:
    """Whether two dst_crs values name the same target CRS. Path equality
    first; then CRS CONTENT (a relocated batch dir moves its WKT file with
    it, and the stale recorded path must not orphan the batch record — the
    PCD-sweep clobber, 2026-08-23); as a last resort the basename, which
    encodes zone+datum (e.g. UTM_12N_WGS84_G2139_3D.wkt), for records
    whose old absolute path no longer exists."""
    from lidar_tools.pdal_pipeline import _read_crs_param

    if str(a) == str(b):
        return True
    if a is None or b is None:
        return False
    crs_a, crs_b = _read_crs_param(a), _read_crs_param(b)
    if crs_a is not None and crs_b is not None:
        return crs_a == crs_b
    if Path(str(a)).name == Path(str(b)).name:
        # last-resort match on the zone+datum-encoding basename: right for
        # a relocated batch, guessable-wrong for a reused generic name —
        # never silent (round-3 finding)
        print(
            f"WARNING: dst_crs matched by basename only ({Path(str(a)).name}); "
            "the recorded CRS file is unreadable, so content could not be "
            "verified",
            file=sys.stderr,
        )
        return True
    return False


def _completed_projects(status: dict[str, str]) -> list[str]:
    """Projects recorded as completed — including "completed (no data)",
    which is a real (product-less) outcome, not a failure. The status
    vocabulary is written by _rasterize_batch; classify through these
    helpers, never by ad-hoc prefix checks."""
    return [w for w, s in status.items() if str(s).startswith("completed")]


def _failed_projects(status: dict[str, str]) -> list[str]:
    return [w for w, s in status.items() if str(s).startswith("failed")]


def _nodata_projects(status: dict[str, str]) -> list[str]:
    return [w for w, s in status.items() if str(s).startswith("completed (no data)")]


def _failure_advice(status: dict[str, str]) -> str:
    """Resume advice for the failed projects in a batch status.

    "re-run the same command" is the right advice for a transient failure
    and exactly the wrong advice for a resume-compatibility refusal,
    which is deterministic."""
    failed = _failed_projects(status)
    blocked = [w for w in failed if "Cannot resume" in status[w]]
    return (
        "re-invoke with the same arguments to resume"
        if not blocked
        else f"{blocked} cannot resume into their existing output "
        "directories (see above); use a fresh output directory or "
        "delete those project subdirectories to rebuild"
    )


def _failure_message(status: dict[str, str]) -> str | None:
    """The batch-failure message `rasterize` raises with, or None when
    nothing failed — shared with `run`, which prints/persists the summary
    before raising the identical message."""
    failed = _failed_projects(status)
    if not failed:
        return None
    return (
        f"{len(failed)}/{len(status)} project runs failed: {failed} "
        f"({_failure_advice(status)})"
    )


def _raise_on_failed(status: dict[str, str]) -> None:
    msg = _failure_message(status)
    if msg:
        raise RuntimeError(msg)
