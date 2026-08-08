"""
Per-project batch driver: run the pipeline once per selected collection
(workunit) into per-project subdirectories on a shared target grid.

Products stay per-project on disk so quality levels and acquisition dates
remain isolated (separate epochs = separate product sets); any combined
"all projects" product is a later, explicit merge step informed by the
compare stage — never an implicit side effect of processing.
"""

from pathlib import Path
from typing import Literal

import geopandas as gpd
import yaml

from lidar_tools import geodesy
from lidar_tools.pdal_pipeline import rasterize


def rasterize_projects(
    geometry: str,
    workunits: str,
    output: str,
    resolution: float = 1.0,
    products: Literal["all", "dsm", "dtm", "intensity"] = "all",
    num_process: int = 1,
    resume: bool = True,
    cleanup: bool = True,
    quiet: bool = False,
    dst_crs: str = None,
    ept_vertical: Literal["auto", "geoid", "ellipsoid"] = "auto",
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
        Product selection passed through to rasterize.
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
        Default: a WGS84 (G2139) 3D UTM CRS built from the AOI and written
        to the base directory once.
    ept_vertical
        Vertical interpretation override passed through to rasterize
        (applies to every project in the batch; use per-project runs when
        collections need different overrides).

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
        zone = geodesy.utm_zone_label(epsg_code)
        target = outbase / f"UTM_{zone}_WGS84_G2139_3D.wkt"
        if not target.exists():
            geodesy.write_crs_file(geodesy.build_utm_g2139_3d(epsg_code), target)
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
                resolution=resolution,
                products=products,
                threedep_project=workunit,
                num_process=num_process,
                cleanup=cleanup,
                quiet=quiet,
                ept_vertical=ept_vertical,
                resume=resume and outdir.exists(),
            )
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
    failed = [w for w, s in status.items() if s != "completed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(status)} project runs failed: {failed} "
            "(re-invoke with the same arguments to resume)"
        )
