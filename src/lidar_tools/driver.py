"""
Per-project batch driver: run the pipeline once per selected collection
(workunit) into per-project subdirectories on a shared target grid.

Products stay per-project on disk so quality levels and acquisition dates
remain isolated (separate epochs = separate product sets); any combined
"all projects" product is a later, explicit merge step informed by the
compare stage — never an implicit side effect of processing.
"""

import sys
from pathlib import Path
from typing import Literal

import geopandas as gpd
import yaml

from lidar_tools import geodesy
from lidar_tools.pdal_pipeline import rasterize


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


def rasterize_projects(
    geometry: str,
    workunits: str,
    output: str,
    resolution: float = 1.0,
    products: str = "all",
    num_process: int = 1,
    resume: bool = True,
    cleanup: bool = True,
    quiet: bool = False,
    dst_crs: str = None,
    output_datum: Literal["wgs84_g2139", "nad83_2011"] = "wgs84_g2139",
    ept_vertical: Literal["auto", "geoid", "ellipsoid"] = "auto",
    geoid_override: Literal["declared", "best-available"] = "declared",
) -> None:
    """
    Run rasterize once per workunit into per-project subdirectories that
    share one target grid (same CRS file, same resolution -> co-registered
    per-project products).

    Parameters
    ----------
    geometry
        Path to the AOI polygon (same AOI for every project).
    workunits
        Comma-separated WESM workunit names, in priority order (see
        `lidar-tools survey` for the per-AOI inventory and proposed
        priorities).
    output
        Base output directory; each workunit writes to `<output>/<workunit>/`.
    resolution
        Shared output posting in target CRS units.
    products
        Comma-separated product selection passed through to rasterize
        (e.g. "all", "dsm,intensity"; see rasterize for names/aliases).
    num_process
        Worker count passed through to rasterize.
    resume
        Continue interrupted per-project runs (skip existing valid tiles),
        by default True — a failed batch can be re-invoked as-is.
    cleanup
        Remove per-tile intermediates after each project run.
    quiet
        Suppress dask progress bars.
    dst_crs
        Optional path to a target CRS definition shared by all projects.
        Default: a 3D UTM CRS built from the AOI (datum per `output_datum`)
        and written to the base directory once.
    output_datum
        Datum realization of the auto-built shared UTM target, used only
        when `dst_crs` is not given: 'wgs84_g2139' (default) or 'nad83_2011'
        (static source realization; ellipsoidal heights, no epoch). Passed
        through to every project; ignored when an explicit `dst_crs` is set.
    ept_vertical
        Vertical interpretation override passed through to rasterize
        (applies to every project in the batch; use per-project runs when
        collections need different overrides).
    geoid_override
        Passed through to rasterize: 'declared' (default) hard-fails when
        a survey's declared production geoid cannot be used;
        'best-available' consciously accepts model substitution.

    Returns
    -------
    None
    """
    wu_list = [w.strip() for w in str(workunits).split(",") if w.strip()]
    if not wu_list:
        raise ValueError("No workunits given")

    outbase = Path(output)
    outbase.mkdir(parents=True, exist_ok=True)

    if dst_crs is None:
        gdf = gpd.read_file(geometry)
        epsg_code = gdf.estimate_utm_crs().to_epsg()
        out_crs_obj, wkt_name = geodesy.build_utm_target(epsg_code, output_datum)
        target = outbase / wkt_name
        if not target.exists():
            geodesy.write_crs_file(out_crs_obj, target)
        dst_crs = str(target)
    print(f"Shared target grid: {dst_crs} at {resolution} m")

    status = {}
    for workunit in wu_list:
        outdir = outbase / workunit
        print(f"\n===== {workunit} -> {outdir} =====")
        try:
            rasterize(
                geometry=geometry,
                input="EPT_AWS",
                output=str(outdir),
                dst_crs=dst_crs,
                output_datum=output_datum,
                resolution=resolution,
                products=products,
                threedep_project=workunit,
                num_process=num_process,
                cleanup=cleanup,
                quiet=quiet,
                ept_vertical=ept_vertical,
                geoid_override=geoid_override,
                resume=resume and outdir.exists(),
            )
            # a clean return is NOT proof of products: a 0-reader run
            # records "no data" in its run_status note and must never be
            # reported as a plain success in the batch. Match the specific
            # state+note the pipeline writes — an unrelated future note
            # must not flip a products-bearing run to "(no data)".
            run_status = _project_run_status(outdir)
            note = run_status.get("note") or ""
            if run_status.get("state") == "completed" and "no data" in note:
                status[workunit] = f"completed (no data): {note}"
                print(
                    f"WARNING: {workunit} completed WITHOUT products: {note}",
                    file=sys.stderr,
                )
            else:
                status[workunit] = "completed"
        except Exception as e:
            # one failed project must not take down the rest of the batch
            print(f"ERROR: {workunit} failed: {e}")
            status[workunit] = f"failed: {e}"

    with open(outbase / "batch_status.yaml", "w") as f:
        yaml.dump(
            {"geometry": str(geometry), "dst_crs": str(dst_crs), "projects": status},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print("\nBatch summary:")
    for workunit, state in status.items():
        print(f"  {workunit}: {state}")
    nodata = [w for w, s in status.items() if s.startswith("completed (no data)")]
    if nodata:
        print(
            f"WARNING: {len(nodata)}/{len(status)} project runs produced NO "
            f"products: {nodata} — check EPT availability/name resolution or "
            "use the local point-cloud path (rasterize --input)",
            file=sys.stderr,
        )
    failed = [w for w, s in status.items() if s.startswith("failed")]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(status)} project runs failed: {failed} "
            "(re-invoke with the same arguments to resume)"
        )
