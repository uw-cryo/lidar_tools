"""
Create and exectute PDAL pipelines
"""

import os
import re
import sys
from dask.distributed import Client, progress, fire_and_forget

# Needs to happen before importing GDAL/PDAL. PROJ fetches datum shift grids
# over the internet; respect an explicit user setting (e.g. PROJ_NETWORK=OFF).
os.environ.setdefault("PROJ_NETWORK", "ON")
# Error out instead of silently falling back to a less accurate transformation
# when the best one is unavailable (e.g. missing geoid grid with networking
# off => heights silently wrong by the ~31 m geoid undulation in CONUS)
os.environ.setdefault("PROJ_ONLY_BEST_DEFAULT", "YES")

from lidar_tools import dsm_functions, geodesy, survey
from pyproj import CRS
from shapely.geometry.polygon import orient as _orient
import numpy as np
from pathlib import Path
import warnings
from typing import Literal, Annotated
import geopandas as gpd
import cyclopts
import shutil
import subprocess
import yaml
from datetime import datetime
from importlib.metadata import version


def _write_processing_metadata(
    output_dir: Path,
    geometry: str,
    input: str,
    output: str,
    src_crs: str,
    dst_crs: str,
    resolution: float,
    dsm_gridding_choice: str,
    products: str,
    threedep_project: str,
    tile_size: float,
    num_process: int,
    overwrite: bool,
    cleanup: bool,
    proj_pipeline: str,
    filter_noise: bool,
    height_above_ground_threshold: float,
    quiet: bool,
    ept_vertical: str = "auto",
    resume: bool = False,
) -> None:
    """
    Write processing metadata to a YAML file in the output directory.

    Parameters
    ----------
    output_dir
        Path to the output directory where the metadata file will be written.
    geometry, input, output, src_crs, dst_crs, resolution, dsm_gridding_choice,
    products, threedep_project, tile_size, num_process, overwrite, cleanup,
    proj_pipeline, filter_noise, height_above_ground_threshold, quiet
        All input parameters from the rasterize function.

    Returns
    -------
    None
    """
    metadata: dict = {
        "lidar_tools_version": version("lidar_tools"),
        "processing_timestamp": datetime.now().isoformat(),
        "input_parameters": {
            "geometry": str(geometry),
            "input": str(input),
            "output": str(output),
            "src_crs": str(src_crs) if src_crs else None,
            "dst_crs": str(dst_crs) if dst_crs else None,
            "resolution": resolution,
            "dsm_gridding_choice": dsm_gridding_choice,
            "products": products,
            "threedep_project": threedep_project,
            "tile_size": tile_size,
            "num_process": num_process,
            "overwrite": overwrite,
            "cleanup": cleanup,
            "proj_pipeline": str(proj_pipeline) if proj_pipeline else None,
            "filter_noise": filter_noise,
            "height_above_ground_threshold": height_above_ground_threshold,
            "quiet": quiet,
            "ept_vertical": ept_vertical,
            "resume": resume,
        }
    }
    # exact code state for reproducing branch-based production runs
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        metadata["git_commit"] = None

    metadata_file = output_dir / "processing_metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"Processing metadata written to {metadata_file}")


def _update_processing_metadata(output_dir: Path, section: str, data) -> None:
    """
    Merge a top-level section into an existing processing_metadata.yaml.

    Parameters
    ----------
    output_dir
        Directory containing processing_metadata.yaml.
    section
        Top-level key to set or replace.
    data
        YAML-serializable content for the section.

    Returns
    -------
    None
    """
    metadata_file = Path(output_dir) / "processing_metadata.yaml"
    metadata: dict = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = yaml.safe_load(f) or {}
    metadata[section] = data
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def _cleanup_intermediates(outdir: Path) -> None:
    """
    Remove per-tile intermediates so the run directory keeps only final
    products, WKT CRS definitions and metadata: per-tile rasters + cache LAZ
    from tiles/<product>/ and tiles/cache/ (cache normally deleted in-task;
    this is the backstop for hard kills), pipeline JSONs from pipelines/,
    and top-level *temp.tif mosaics. Emptied subdirectories are removed —
    anything else in them (e.g. saved per-tile pointclouds) is kept.
    """
    for subdir_name, patterns in (
        ("tiles", ["*_tile_aoi_*.tif*", "*_cache_tile_aoi_*.laz"]),
        ("pipelines", ["pipeline*.json"]),
    ):
        subdir = outdir / subdir_name
        for pattern in patterns:
            for file in subdir.rglob(pattern):
                file.unlink()
        # deepest-first so emptied product subdirs go before tiles/ itself
        for child in sorted(subdir.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass  # still holds files we keep
        try:
            subdir.rmdir()
        except OSError:
            pass  # missing, or still holds files we keep
    for file in outdir.glob("*temp.tif"):
        file.unlink()


def rasterize(
    geometry: str,
    input: str = "EPT_AWS",
    output: str = "/tmp/lidar-tools-output",
    src_crs: str = None,
    dst_crs: str = None,
    output_datum: Literal[
        "wgs84_g2139", "nad83_2011", "wgs84_g1674", "itrf2020"
    ] = "wgs84_g2139",
    resolution: float = 1.0,
    dsm_gridding_choice: str = "first_idw",
    products: str = "all",
    threedep_project: Literal["all", "latest"] | str = "latest",
    tile_size: float = 1.0,
    num_process: int = 1,
    overwrite: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    cleanup: bool = True,
    proj_pipeline: str = None,
    filter_noise: bool = True,
    height_above_ground_threshold: float = None,
    quiet: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    ept_vertical: Literal["auto", "geoid", "ellipsoid"] = "auto",
    resume: Annotated[bool, cyclopts.Parameter(negative="")] = False,
    coord_epoch: float = None,
) -> None:
    """
    Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and/or Intensity raster from point cloud data.

    Parameters
    ----------
    geometry
        Path to the vector dataset containing a single polygon that defines the processing extent.
    input
        Path to directory containing input LAS/LAZ files, otherwise uses USGS 3DEP EPT data on AWS.
    output
        Path to output directory.
    src_crs
        Path to file with PROJ-supported CRS definition to override CRS of input files.
    dst_crs
        Path to file with PROJ-supported CRS definition for the output. If unspecified, a local UTM CRS is auto-built for the AOI (datum per `output_datum`).
    output_datum
        Datum realization of the auto-built local-UTM target, used only when
        `dst_crs` is not given: 'wgs84_g2139' (default; dynamic frame,
        outputs stamped at epoch 2010.0) or 'nad83_2011' (static source
        realization of 3DEP; ellipsoidal heights, no epoch, no ITRF Helmert).
        For any other target, pass an explicit `dst_crs` WKT file.
    resolution
        Square output raster posting in units of `dst_crs`.
    dsm_gridding_choice
        The gridding method to use for DSM generation. 'first_idw' uses the first and only returns which are gridded using IDW, 'n-pct' computes points matching the nth percentile in a pointview (e.g., 98-pct), which are gridded using the max binning operator.
    products
        Comma-separated output products to generate: any subset of dsm,
        dtm_no_fill, dtm_fill, intensity (e.g. "dsm,intensity"). Aliases:
        "all" (default) = every product; "dtm" = both DTM variants. All
        requested products for a tile are generated from a single point
        read.
    threedep_project
        "all" processes all available 3DEP EPT point clouds which intersect with the input polygon.
        "first" 3DEP project encountered will be processed.
        "specific" should be a string that matches the "project" name in the 3DEP metadata.
    tile_size
        The size of rasterized tiles processed from input EPT point clouds in units of `dst_crs`.
    num_processes
        Number of processes to run PDAL pipelines in parallel.
    overwrite
        Overwrite output files if they already exist.
    cleanup
        Remove the intermediate tif files, keep only final mosaiced rasters.
    proj_pipeline
        A PROJ pipeline string to be used for reprojection of the point cloud. If specified, this will be used in combination with the target_wkt option.
    local_utm
        If true, automatically compute the local UTM zone from the extent polygon for final output products. If false, use the CRS defined in the target_wkt file.
    filter_noise
        Remove noise points (classification==18 and classification==7) from the point cloud before DSM, DTM and surface intensity processing. Default is True.
    height_above_ground_threshold
        If specified, the height above ground (HAG) will be calculated using all nearest ground classied points, and all points greater than this value will be classified as noise, by default None.
    quiet
        Suppress dask progress bar (useful for CI logs)
    ept_vertical
        Vertical interpretation of the EPT source heights: 'auto' decides
        empirically per run (bare-ground sample vs COP30); 'geoid' (NAVD88
        orthometric) or 'ellipsoid' skip the empirical check when the
        source datum is known (e.g. from per-survey metadata) or when the
        check cannot sample reliably (steep terrain, snow, small AOIs).
    resume
        Continue an interrupted run in an existing output directory: tiles
        whose outputs already exist and pass a deep validity check are
        skipped; truncated/invalid tiles are recomputed. Mutually exclusive
        with overwrite.

    Returns
    -------
    None
    """
    if dsm_gridding_choice != "first_idw" and not re.match(r"^\d{1,2}-pct$", dsm_gridding_choice):
        raise ValueError(
            f"Invalid dsm_gridding_choice: {dsm_gridding_choice}. Must be 'first_idw' or match the format 'n-pct' (e.g., '98-pct')."
        )

    # canonical product names (validates the selection early)
    requested = dsm_functions.parse_products(products)

    # Parse input polygon CRS and check that area isn't too large
    gdf = gpd.read_file(geometry)
    _check_polygon_area(gdf)
    input_crs = gdf.crs.to_wkt()
    # lon/lat AOI bounds for scoping PROJ transformation selection
    aoi_lonlat = tuple(gdf.to_crs("EPSG:4326").total_bounds)

    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    outdir = Path(output)
    if outdir.exists():
        if resume:
            print(
                f"Resuming into existing output path: {outdir} "
                "(existing valid tiles will be skipped)"
            )
        elif overwrite:
            print(f"Overwriting existing output path: {outdir}")
            if outdir.is_file():
                outdir.unlink()
            elif outdir.is_dir():
                shutil.rmtree(outdir)
        else:
            raise FileExistsError(
                f"Output directory {outdir} already exists. Use --overwrite to "
                "replace it or --resume to continue an interrupted run."
            )

    # Set output filename prefix based on input polygon name
    outdir.mkdir(parents=True, exist_ok=True)
    output_prefix = outdir / Path(geometry).stem

    # Write processing metadata to YAML file
    _write_processing_metadata(
        output_dir=outdir,
        geometry=geometry,
        input=input,
        output=output,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resolution=resolution,
        dsm_gridding_choice=dsm_gridding_choice,
        products=products,
        threedep_project=threedep_project,
        tile_size=tile_size,
        num_process=num_process,
        overwrite=overwrite,
        cleanup=cleanup,
        proj_pipeline=proj_pipeline,
        filter_noise=filter_noise,
        height_above_ground_threshold=height_above_ground_threshold,
        quiet=quiet,
        ept_vertical=ept_vertical,
        resume=resume,
    )
    _update_processing_metadata(
        outdir,
        "run_status",
        {"state": "started", "timestamp": datetime.now().isoformat()},
    )

    # Create custom 3D CRS UTM WKT2 for the AOI's local zone on the selected
    # output datum realization, built programmatically with pyproj (correct
    # southern-hemisphere false northing; no runtime network fetch of a WKT
    # template). Default is the dynamic WGS 84 (G2139); output_datum can
    # select the static NAD83(2011) source realization instead.
    if dst_crs is None:
        epsg_code = gdf.estimate_utm_crs().to_epsg()
        out_crs_obj, wkt_name = geodesy.build_utm_target(epsg_code, output_datum)
        dst_crs = geodesy.write_crs_file(out_crs_obj, outdir / wkt_name)

    # Configure output raster extents and posting based on input polygon
    with open(dst_crs, "r") as f:
        contents = f.read()
        out_crs = CRS.from_string(contents)
    out_extent = gdf.to_crs(out_crs).total_bounds
    final_out_extent = dsm_functions.tap_bounds(out_extent, res=resolution)
    #fix extent precision with respect to input resolution
    #from https://www.reddit.com/r/pythontips/comments/zw5ana/how_to_count_decimal_places/
    import decimal
    d = decimal.Decimal(str(resolution))
    precision = abs(d.as_tuple().exponent)
    final_out_extent = [np.round(val,precision) for val in final_out_extent]
    
    # TODO: simplify and use tempfile (https://github.com/uw-cryo/lidar_tools/pull/25#discussion_r2177660328)
    # TODO: here and elsewhere use logging instead of prints
    print(f"Output extent in target CRS is {final_out_extent}")
    gdf_out = gdf.to_crs(out_crs)
    #This is problematic if output CRS is units of decimal degrees, instead of meters
    gdf_out["geometry"] = gdf_out["geometry"].buffer(250)  # NOTE: assumes meters
    gdf_out = gdf_out.to_crs(input_crs)
    extent_polygon = outdir / "judicious_extent_polygon.geojson"
    gdf_out.to_file(extent_polygon, driver="GeoJSON")

    # How to handle AOIs intersecting multiple 3DEP projects?
    if threedep_project == "all":
        process_all_intersecting_surveys = True
        process_specific_3dep_survey = None
    elif threedep_project == "latest":
        process_all_intersecting_surveys = False
        process_specific_3dep_survey = None
    else:
        process_all_intersecting_surveys = False
        process_specific_3dep_survey = threedep_project

    if filter_noise:
        filter_high_noise = True
        filter_low_noise = True
    else:
        filter_high_noise = False
        filter_low_noise = False

    # Per-survey record (WESM): pinned into processing metadata and used to
    # drive datum handling — declared horizontal realization (base datum for
    # the EPT null-tie interpretation) and production geoid model. Only
    # available when a specific workunit was requested; 'latest'/'all' runs
    # fall back to the NAD83(2011)+best-geoid defaults.
    survey_record = None
    ept_base_epsg = geodesy.NAD83_2011_EPSG
    geoid_hint = None
    if input == "EPT_AWS" and process_specific_3dep_survey is not None:
        try:
            survey_record = survey.workunit_record(gdf, process_specific_3dep_survey)
        except Exception as e:
            print(
                f"WARNING: could not fetch the WESM record for "
                f"'{process_specific_3dep_survey}' ({e}); using default datum "
                "handling (NAD83(2011), best available geoid)",
                file=sys.stderr,
            )
        if survey_record is not None:
            _update_processing_metadata(outdir, "survey_records", [survey_record])
            if survey_record.get("horiz_crs"):
                # hard-errors on non-NAD83-family (e.g. Pacific-plate PA11)
                ept_base_epsg = geodesy.geographic_base_epsg(
                    survey_record["horiz_crs"]
                )
            geoid_hint = geodesy.geoid_grid_hint(survey_record.get("geoid"))
            print(
                f"Survey record pinned: {process_specific_3dep_survey} "
                f"(base datum EPSG:{ept_base_epsg}, geoid "
                f"{survey_record.get('geoid')}, ql {survey_record.get('ql')})"
            )

    # TODO: create EPT for local laz for common workflow? https://github.com/uw-cryo/lidar_tools/issues/14#issuecomment-3076045321
    # SB note: The main reason for seperate EPT and local laz pipelines is the difference in projection handling, not much due to difference in file formats.
    # Fail fast, before hours of tile compute, if PROJ cannot rigorously
    # perform the datum transformations this run will need (e.g. missing
    # geoid grids would otherwise surface as a silent ~31 m vertical error
    # or a crash at the final warp stage)
    transform_checks = []
    ept_checks = {}

    if input == "EPT_AWS":
        print("Processing 3DEP EPT tiles from AWS")
        if out_crs != CRS.from_epsg(3857):
            # both candidate source interpretations of EPT data; which one
            # applies is decided empirically (or by ept_vertical) after
            # mosaicking. The selected PROJ pipelines are enforced at the
            # final warps (gdalwarp -ct).
            for key, branch, ept_src_crs, grids_hint in [
                (
                    "geoid",
                    f"geoid (EPSG:3857 as base EPSG:{ept_base_epsg} + NAVD88 heights)",
                    geodesy.build_ept_3857_navd88_compound(base_epsg=ept_base_epsg),
                    geoid_hint,
                ),
                (
                    "ellipsoid",
                    f"ellipsoid (EPSG:3857 as base EPSG:{ept_base_epsg} 3D)",
                    geodesy.build_ept_3857_nad83_2011(base_epsg=ept_base_epsg),
                    None,
                ),
            ]:
                try:
                    record = geodesy.preflight_vertical_transform(
                        ept_src_crs,
                        out_crs,
                        aoi_bounds=aoi_lonlat,
                        prefer_grids=grids_hint,
                    )
                except RuntimeError:
                    if grids_hint is None:
                        raise
                    print(
                        f"WARNING: no transformation using the survey's "
                        f"declared geoid ('{grids_hint}') is available; "
                        "falling back to the best available geoid model",
                        file=sys.stderr,
                    )
                    record = geodesy.preflight_vertical_transform(
                        ept_src_crs, out_crs, aoi_bounds=aoi_lonlat
                    )
                record["branch"] = branch
                transform_checks.append(record)
                ept_checks[key] = record
        tile_jobs = dsm_functions.create_ept_3dep_pipeline(
            extent_polygon,
            dst_crs,
            output_prefix,
            buffer_value=10*resolution, # buffer is based on output resolution
            tile_size_km=tile_size,  # TODO: ensure we can do non-km units
            # TODO: handle new 3dep project keyword here
            dsm_gridding_choice=dsm_gridding_choice,
            process_specific_3dep_survey=process_specific_3dep_survey,
            process_all_intersecting_surveys=process_all_intersecting_surveys,
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            hag_nn=height_above_ground_threshold,
            raster_resolution=resolution,
            products=requested,
        )
    else:
        print(f"Processing local laz files from {input}")
        if src_crs:
            with open(src_crs, "r") as f:
                contents = f.read()
                src_projcrs = CRS.from_string(contents)
        else:
            src_projcrs = None
        print(src_projcrs)
        if src_projcrs is not None:
            record = geodesy.preflight_vertical_transform(
                src_projcrs, out_crs, aoi_bounds=aoi_lonlat
            )
            record["branch"] = "local point cloud (in-pipeline reprojection)"
            transform_checks.append(record)
        tile_jobs = dsm_functions.create_lpc_pipeline(
            local_laz_dir=input,
            input_crs=src_projcrs,
            target_wkt=dst_crs,
            output_prefix=output_prefix,
            extent_polygon=extent_polygon,
            dsm_gridding_choice=dsm_gridding_choice,
            buffer_value=10*resolution, # buffer is based on output resolution
            proj_pipeline=proj_pipeline,
            filter_high_noise=filter_high_noise,
            filter_low_noise=filter_low_noise,
            hag_nn=height_above_ground_threshold,
            raster_resolution=resolution,
            products=requested,
        )

    # Record geodesy provenance now (before compute) so it survives an
    # interrupted run; the coordinate epoch is filled in after stamping
    geodesy_record: dict = {
        "versions": geodesy.library_versions(),
        "vertical_transform_preflight": transform_checks,
        "coordinate_epoch": None,
    }
    _update_processing_metadata(outdir, "geodesy", geodesy_record)

    # BuildVRT opens every tile at once during mosaicking; the default soft
    # open-file limit fails for large AOIs (issue #43)
    dsm_functions.raise_file_limit()

    # TODO: refactor into function
    num_pipelines = len(tile_jobs)

    # one task per tile: each tile job reads its points once and emits every
    # requested product from that read (single-read consolidation, F3)
    if num_process == 1:
        print(f"Executing PDAL in serial for products={requested}")
        tile_results = [
            dsm_functions.execute_tile_job(job, resume) for job in tile_jobs
        ]
    else:
        n_jobs = num_process if num_pipelines > num_process else num_pipelines
        print(
            f"Executing PDAL in parallel with dask n_workers={n_jobs} for products={requested}"
        )
        # one cluster for the whole run: the scheduler interleaves tiles
        # instead of paying a cluster spinup + idle drain per product
        # (~20% of wall time measured at UW-campus scale)
        with Client(
            n_workers=n_jobs,
            processes=True,  # run PDAL pipelines in isolated processes
            threads_per_worker=1,
        ) as client:
            print(f"Dask dashboard available at: {client.dashboard_link}")
            futures = [
                client.submit(
                    dsm_functions.execute_tile_job,
                    job,
                    resume,
                    retries=3,  # resubmit on worker death (KilledWorker)
                    pure=False,
                )
                for job in tile_jobs
            ]
            # fire_and_forget for better memory management
            for future in futures:
                fire_and_forget(future)
            if not quiet:
                progress(futures)
            tile_results = client.gather(futures)

    # reassemble tile-ordered per-product lists for mosaicking; empty tiles
    # (survey has no points there) are a legitimate no-data outcome, tracked
    # separately from real failures
    results: dict[str, list] = {name: [] for name in requested}
    n_empty = 0
    for tile_result in tile_results:
        if tile_result is None:
            continue
        if tile_result.get("empty"):
            n_empty += 1
            continue
        for name, outfn in tile_result["outputs"].items():
            if outfn is not None:
                results[name].append(outfn)

    final_dsm_fn_list = results.get("dsm", [])
    final_dtm_no_fill_fn_list = results.get("dtm_no_fill", [])
    final_dtm_fill_fn_list = results.get("dtm_fill", [])
    final_intensity_fn_list = results.get("intensity", [])

    print("****Processing complete for all tiles****")

    # Tile accounting: empty tiles (survey has no points there) are expected
    # and excluded, not failures; real failures are counted against the
    # data-bearing tiles and reported loudly (they leave holes in the mosaic)
    product_labels = {
        "dsm": "DSM",
        "dtm_no_fill": "DTM_no_fill",
        "dtm_fill": "DTM_fill",
        "intensity": "intensity",
    }
    data_total = num_pipelines - n_empty
    tile_counts = {
        product_labels[name]: (len(results[name]), data_total) for name in requested
    }
    summary = ", ".join(
        f"{name}: {ok}/{total}" for name, (ok, total) in tile_counts.items()
    )
    if n_empty:
        print(
            f"{n_empty}/{num_pipelines} tiles had no points (survey does not "
            "cover them); excluded from the mosaics as expected."
        )
    n_failed = sum(total - ok for ok, total in tile_counts.values())
    if n_failed:
        print(
            f"WARNING: {n_failed} tile pipelines failed or produced invalid "
            f"rasters (valid/data-tiles {summary}). Final mosaics will contain "
            "gaps; see ERROR messages above for the failing pipelines.",
            file=sys.stderr,
        )
    else:
        print(f"All data tiles produced valid rasters ({summary})")

    # No-data guard: a survey may not cover the AOI at all (0 readers) or
    # every intersecting tile may be empty. There is nothing to mosaic, warp
    # or stamp; finish cleanly with a clear status instead of crashing on an
    # empty tile list (final_*_fn_list[0]) or a cryptic empty-VRT error.
    if not any(results[name] for name in requested):
        print(
            f"No data produced for this AOI (readers: {num_pipelines}, "
            f"empty tiles: {n_empty}); the survey likely does not cover it. "
            "Skipping mosaic/reprojection; no products written."
        )
        _update_processing_metadata(
            outdir,
            "run_status",
            {
                "state": "completed",
                "note": "no data (survey does not cover AOI)",
                "timestamp": datetime.now().isoformat(),
            },
        )
        if cleanup:
            _cleanup_intermediates(outdir)
        print("****Processing complete (no data)****")
        return

    # Mosaicing
    # ===========
    dsm_mos_fn = f"{output_prefix}-DSM_mos-temp.tif"
    dtm_mos_no_fill_fn = f"{output_prefix}-DTM_no_fill_mos-temp.tif"
    dtm_mos_fill_fn = f"{output_prefix}-DTM_fill_window_size_4_mos-temp.tif"
    intensity_mos_fn = f"{output_prefix}-intensity_mos-temp.tif"

    if num_pipelines > 1:
        print(
            f"Multiple tiles created: {num_pipelines}. Mosaicing required to create final rasters"
        )
        print("*** Now creating raster composites ***")
        if input == "EPT_AWS":
            cog = False
            out_extent = None
        else:
            out_extent = final_out_extent
            cog = True
            
        print("Running sequentially")
        if "dsm" in requested:
            print(f"Creating DSM mosaic at {dsm_mos_fn}")
            dsm_functions.raster_mosaic(
                final_dsm_fn_list, dsm_mos_fn, cog=cog, out_extent=out_extent
            )

        if "dtm_no_fill" in requested:
            print(f"Creating DTM mosaic at {dtm_mos_no_fill_fn}")
            dsm_functions.raster_mosaic(
                final_dtm_no_fill_fn_list,
                dtm_mos_no_fill_fn,
                cog=cog,
                out_extent=out_extent,
            )

        if "dtm_fill" in requested:
            print(f"Creating DTM mosaic with window size 4 at {dtm_mos_fill_fn}")
            dsm_functions.raster_mosaic(
                final_dtm_fill_fn_list, dtm_mos_fill_fn, cog=cog, out_extent=out_extent
            )

        if "intensity" in requested:
            print(f"Creating intensity raster mosaic at {intensity_mos_fn}")
            dsm_functions.raster_mosaic(
                final_intensity_fn_list,
                intensity_mos_fn,
                cog=cog,
                out_extent=out_extent,
            )

    else:
        print("Only one tile created, no mosaicing required")
        if "dsm" in requested:
            dsm_functions.rename_rasters(final_dsm_fn_list[0], dsm_mos_fn)
        if "dtm_no_fill" in requested:
            dsm_functions.rename_rasters(
                final_dtm_no_fill_fn_list[0], dtm_mos_no_fill_fn
            )
        if "dtm_fill" in requested:
            dsm_functions.rename_rasters(final_dtm_fill_fn_list[0], dtm_mos_fill_fn)
        if "intensity" in requested:
            dsm_functions.rename_rasters(final_intensity_fn_list[0], intensity_mos_fn)

    # Reprojection
    # ============
    dsm_reproj = dsm_mos_fn.split("-temp.tif")[0] + ".tif"
    dtm_no_fill_reproj = dtm_mos_no_fill_fn.split("-temp.tif")[0] + ".tif"
    dtm_fill_reproj = dtm_mos_fill_fn.split("-temp.tif")[0] + ".tif"
    intensity_reproj = intensity_mos_fn.split("-temp.tif")[0] + ".tif"

    if input == "EPT_AWS" and out_crs != CRS.from_epsg(3857):
        print("*********Reprojecting rasters****")
        #This is hardcoded for dsm_mos_fn, but we could have dtm fn
        reproject_truth_val = False
        if ept_vertical != "auto":
            # source vertical datum supplied by the user / survey metadata
            reproject_truth_val = ept_vertical == "geoid"
            print(f"EPT vertical interpretation overridden: {ept_vertical}")
        elif "dsm" in requested:
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(dsm_mos_fn)
        elif "dtm_fill" in requested:
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(dtm_mos_fill_fn)
        elif "dtm_no_fill" in requested:
            reproject_truth_val = dsm_functions.confirm_3dep_vertical(
                dtm_mos_no_fill_fn
            )
        # enforce the preflight-selected pipeline: GDAL's own operation
        # selection is not guaranteed to match (it null-ties some pairs)
        height_ct = ept_checks["geoid" if reproject_truth_val else "ellipsoid"][
            "proj_pipeline"
        ]
        if reproject_truth_val:
            # EPT heights are NAVD88 orthometric: warp with the compound
            # source SRS (TRUE base datum declared — see
            # build_ept_3857_navd88_compound) so the geoid-to-ellipsoid
            # shift uses the survey-consistent geoid route
            src_srs = geodesy.write_crs_file(
                geodesy.build_ept_3857_navd88_compound(base_epsg=ept_base_epsg),
                outdir / f"EPT_3857_base{ept_base_epsg}_NAVD88_compound.wkt",
            )
        else:
            # EPT heights are already ellipsoidal: declare the survey's true
            # base datum (3D) so the ITRF Helmert applies to positions and
            # heights, instead of a null relabel that leaves outputs
            # ~1.3 m horizontal / ~0.9 m vertical off in CONUS
            src_srs = geodesy.write_crs_file(
                geodesy.build_ept_3857_nad83_2011(base_epsg=ept_base_epsg),
                outdir / f"EPT_3857_base{ept_base_epsg}_3D.wkt",
            )
        out_extent = final_out_extent
        print(src_srs)
        print("Running reprojection sequentially")
        # An explicit coord_epoch PINS the full operation with the epoch
        # baked into the -ct pipeline (projinfo --t_epoch). GDAL free
        # selection with -t_coord_epoch is deliberately NOT used: it proved
        # unstable across source datum declarations (LV 2026-07-10 — null
        # horizontal ties for WGS84-realization targets, and for ITRF
        # targets once the compound source declared its true NAD83 base).
        if coord_epoch is not None:
            need = ["+proj=helmert"]
            if reproject_truth_val:
                need.append("vgridshift")
            height_ct = geodesy.epoch_pinned_pipeline(
                src_srs,
                dst_crs,
                coord_epoch,
                aoi_bounds=aoi_lonlat,
                require_substrings=need,
            )
            print(f"epoch-pinned -ct pipeline: {height_ct}")
            transform_checks.append(
                {
                    "branch": f"epoch-pinned (coord_epoch={coord_epoch})",
                    "proj_pipeline": height_ct,
                }
            )
        if "dsm" in requested:
            print("Reprojecting DSM raster")
            dsm_functions.gdal_warp(
                dsm_mos_fn,
                dsm_reproj,
                src_srs,
                dst_crs,
                res=resolution,
                resampling_alogrithm="bilinear",
                out_extent=out_extent,
                coordinate_operation=height_ct,
            )
        if "dtm_no_fill" in requested:
            print("Reprojecting DTM raster")
            dsm_functions.gdal_warp(
                dtm_mos_no_fill_fn,
                dtm_no_fill_reproj,
                src_srs,
                dst_crs,
                res=resolution,
                resampling_alogrithm="bilinear",
                out_extent=out_extent,
                coordinate_operation=height_ct,
            )
        if "dtm_fill" in requested:
            dsm_functions.gdal_warp(
                dtm_mos_fill_fn,
                dtm_fill_reproj,
                src_srs,
                dst_crs,
                res=resolution,
                resampling_alogrithm="bilinear",
                out_extent=out_extent,
                coordinate_operation=height_ct,
            )
        if "intensity" in requested:
            print("Reprojecting intensity raster")
            # Intensity values are not heights: warp 2D -> 2D. A vertical
            # axis on EITHER side makes gdal.Warp treat the band values as
            # heights (compound source: ~-30 m geoid shift, values below
            # the undulation clamp to 0 = nodata; even a 2D source against
            # the 3D target gets promoted and picks up the ~-0.7 m Helmert
            # dz, truncating UInt16 DNs). Still declare the NAD83(2011)
            # datum so the horizontal Helmert matches the height products.
            intensity_src_srs = geodesy.write_crs_file(
                geodesy.build_ept_3857_nad83_2011(
                    three_d=False, base_epsg=ept_base_epsg
                ),
                outdir / f"EPT_3857_base{ept_base_epsg}_2D.wkt",
            )
            intensity_dst_srs = geodesy.write_crs_file(
                out_crs.to_2d(),
                outdir / (Path(str(dst_crs)).stem + "_2D.wkt"),
            )
            # GDAL will not route the horizontal-only NAD83-family->target
            # transform through the ITRF Helmert on its own (it silently
            # selects the null tie): enforce the pipeline pyproj selects
            intensity_check = geodesy.preflight_vertical_transform(
                geodesy.build_ept_3857_nad83_2011(
                    three_d=False, base_epsg=ept_base_epsg
                ),
                out_crs.to_2d(),
                aoi_bounds=aoi_lonlat,
            )
            intensity_check["branch"] = "intensity (2D horizontal-only)"
            transform_checks.append(intensity_check)
            dsm_functions.gdal_warp(
                intensity_mos_fn,
                intensity_reproj,
                intensity_src_srs,
                intensity_dst_srs,
                res=resolution,
                dtype="UInt16",
                resampling_alogrithm="bilinear",
                out_extent=out_extent,
                coordinate_operation=intensity_check["proj_pipeline"],
            )

    else:
        # local point clouds are reprojected in-pipeline; EPT runs with an
        # EPSG:3857 target need no warp (previously these were never renamed
        # and the run crashed at the overview stage)
        print("No reprojection required")
        # rename the temp files to the final output names
        if "dsm" in requested:
            dsm_functions.rename_rasters(dsm_mos_fn, dsm_reproj)
        if "dtm_no_fill" in requested:
            dsm_functions.rename_rasters(dtm_mos_no_fill_fn, dtm_no_fill_reproj)
        if "dtm_fill" in requested:
            dsm_functions.rename_rasters(dtm_mos_fill_fn, dtm_fill_reproj)
        if "intensity" in requested:
            dsm_functions.rename_rasters(intensity_mos_fn, intensity_reproj)

    product_reproj = {
        "dsm": dsm_reproj,
        "dtm_no_fill": dtm_no_fill_reproj,
        "dtm_fill": dtm_fill_reproj,
        "intensity": intensity_reproj,
    }
    final_products = [product_reproj[name] for name in requested]

    # 3DEP sources are NAD83(2011) epoch-reduced to 2010.0, so products in a
    # dynamic frame (default WGS84 G2139 target) are coordinates at epoch
    # 2010.0; unstamped they are ambiguous by ~1.65 cm/yr of plate motion.
    # Stamp before overview creation so the COG translate carries it over.
    # Static target CRSs are left unstamped (set_coordinate_epoch no-ops).
    # Pass the authoritative CRS: the GeoTIFF round-trip can drop the
    # DYNAMIC property (intensity is warped to the 2D demotion of the
    # target, whose file SRS reads back as static).
    intensity_warped_2d = input == "EPT_AWS" and out_crs != CRS.from_epsg(3857)
    epoch_to_stamp = (
        coord_epoch if coord_epoch is not None else geodesy.DEFAULT_COORDINATE_EPOCH
    )
    stamped = [
        fn
        for fn in final_products
        if geodesy.set_coordinate_epoch(
            fn,
            epoch_to_stamp,
            crs=out_crs.to_2d()
            if (fn == intensity_reproj and intensity_warped_2d)
            else out_crs,
        )
    ]
    geodesy_record["coordinate_epoch"] = epoch_to_stamp if stamped else None
    geodesy_record["epoch_stamped_products"] = [Path(fn).name for fn in stamped]
    _update_processing_metadata(outdir, "geodesy", geodesy_record)

    print("****Building Gaussian overviews for all rasters****")
    print("Running overview creation sequentially")
    for raster_fn in final_products:
        dsm_functions.gdal_add_overview(raster_fn)

    _update_processing_metadata(
        outdir,
        "run_status",
        {
            "state": "completed",
            "timestamp": datetime.now().isoformat(),
            "tiles_total": num_pipelines,
            "tiles_empty": n_empty,
            "tiles_data": data_total,
        },
    )

    if cleanup:
        print("Cleaning up intermediate outputs")
        _cleanup_intermediates(outdir)

    try:
        from lidar_tools.preview import product_preview

        preview_fn = product_preview(outdir)
        if preview_fn is not None:
            print(f"Preview figure: {preview_fn}")
    except Exception as e:  # a figure failure must never fail a finished run
        print(f"WARNING: preview figure generation failed: {e}", file=sys.stderr)

    print("****Processing complete****")


def geographic_area(gf: gpd.GeoDataFrame) -> gpd.pd.Series:
    """
    Estimate the geographic area of each polygon in a GeoDataFrame in m^2

    Parameters
    ----------
    gf
        A GeoDataFrame containing the geometries for which the area needs to be calculated. The GeoDataFrame
        must have a geographic coordinate system (latitude and longitude).

    Returns
    -------
    pd.Series
        A Pandas Series containing the area of each polygon in the input GeoDataFrame in m^2.

    Raises
    ------
    TypeError
        If the GeoDataFrame does not have a geographic coordinate system.

    Notes
    -----
    - Only works for areas up to 1/2 of globe (https://github.com/pyproj4/pyproj/issues/1401)
    """
    if gf.crs is None or not gf.crs.is_geographic:
        msg = "This function requires a GeoDataFrame with gf.crs.is_geographic==True"
        raise TypeError(msg)

    geod = gf.crs.get_geod()

    def area_calc(geom):
        if geom.geom_type not in ["MultiPolygon", "Polygon"]:
            return np.nan

        # For MultiPolygon do each separately
        if geom.geom_type == "MultiPolygon":
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal.
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(_orient(geom, 1))[0]

    return gf.geometry.apply(area_calc)


def _check_polygon_area(gf: gpd.GeoDataFrame) -> None:
    """
    Issue a warning if area is bigger than threshold

    Parameters
    ----------
    gf
        A GeoDataFrame containing a polygon

    Returns
    -------
    None
        Just prints a warning if area is too large
    """
    warn_if_larger_than = 10000  # km^2

    # Fast track if projected and units are meters:
    if gf.crs.is_projected and gf.crs.axis_info[0].unit_name == "metre":
        area = gf.area * 1e-6
    else:
        area = geographic_area(gf.to_crs("EPSG:4326")) * 1e-6

    if area.to_numpy() >= warn_if_larger_than:
        msg = f"Very large AOI area ({area.values[0]:.2f} km^2). Recommended using an area of less than {warn_if_larger_than} km^2"
        warnings.warn(msg)
    else:
        print(f"Starting Processing of {area.values[0]:.2f} km^2 AOI")

