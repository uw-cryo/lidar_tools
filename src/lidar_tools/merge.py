"""
Priority merge of per-project products from a rasterize-projects batch.

The per-project mosaics share one target grid (same CRS file, same
resolution, same tap-aligned origin), so the merge is a VRT composite with
zero resampling: sources are painted in order and the highest-priority
project is painted last (on top). GDAL exposes "virtual overviews" for the
VRT derived from the sources' own COG overviews (verified empirically), so
component overviews carry through without regeneration.
"""

import os
from pathlib import Path

import yaml

#: product mosaic filename suffixes eligible for merging
PRODUCT_SUFFIXES = (
    "DSM_mos",
    "DTM_no_fill_mos",
    "DTM_fill_window_size_4_mos",
    "intensity_mos",
)


def _raster_signature(fn: Path) -> tuple:
    """Geotransform + size + CRS name, used to enforce grid alignment."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.OpenEx(str(fn))
    srs = ds.GetSpatialRef()
    return (
        ds.GetGeoTransform(),
        ds.RasterXSize,
        ds.RasterYSize,
        srs.GetName() if srs is not None else None,
    )


def merge_projects(
    batch_dir: str | Path,
    workunits: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Build per-product priority-merge VRTs from a rasterize-projects batch.

    Parameters
    ----------
    batch_dir
        rasterize-projects base directory (per-project subdirectories +
        batch_status.yaml).
    workunits
        Project names in priority order, FIRST = highest priority (same
        semantics as the rasterize-projects workunits argument: put the
        higher-quality / more recent survey first). Default: the project
        order recorded in batch_status.yaml.
    output_dir
        Where the VRTs are written, by default <batch_dir>/merge.

    Returns
    -------
    list[Path]
        The written VRT paths (one per product found in >=1 project).

    Raises
    ------
    ValueError
        If the per-project mosaics for a product do not share an identical
        grid (geotransform/size/CRS) — merging then would need resampling,
        which this stage deliberately refuses to do.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    batch_dir = Path(batch_dir)
    if workunits is None:
        status_fn = batch_dir / "batch_status.yaml"
        if not status_fn.exists():
            raise FileNotFoundError(
                f"{status_fn} not found; pass workunits explicitly."
            )
        with open(status_fn) as f:
            workunits = list(yaml.safe_load(f)["projects"])
    output_dir = Path(output_dir) if output_dir else batch_dir / "merge"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    merge_meta: dict = {"priority_order": list(workunits), "products": {}}
    for suffix in PRODUCT_SUFFIXES:
        # sources in priority order (first = highest priority)
        sources: list[Path] = []
        for wu in workunits:
            hits = sorted((batch_dir / wu).glob(f"*-{suffix}.tif"))
            if hits:
                sources.append(hits[0])
        if not sources:
            continue

        signatures = {fn: _raster_signature(fn) for fn in sources}
        if len(set(signatures.values())) > 1:
            detail = "\n".join(f"  {fn}: {sig}" for fn, sig in signatures.items())
            raise ValueError(
                f"{suffix}: per-project mosaics are not on one grid; refusing "
                f"to merge without resampling:\n{detail}"
            )

        out_fn = output_dir / f"{sources[0].name.rsplit('.', 1)[0]}.vrt"
        # VRT sources paint in list order (last on top) -> reverse so the
        # highest-priority project wins in overlaps. Build from inside the
        # output dir with relative paths so the VRT stays portable when the
        # volume is mounted elsewhere.
        rel_sources = [
            os.path.relpath(fn, output_dir) for fn in reversed(sources)
        ]
        cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            ds = gdal.BuildVRT(out_fn.name, rel_sources)
            ds.FlushCache()
            band = ds.GetRasterBand(1)
            n_ovr = band.GetOverviewCount()
            ds = None
        finally:
            os.chdir(cwd)
        print(
            f"{suffix}: merged {len(sources)} project(s) -> {out_fn} "
            f"({n_ovr} virtual overview levels)"
        )
        written.append(out_fn)
        merge_meta["products"][suffix] = {
            "vrt": out_fn.name,
            "sources_priority_order": [str(fn) for fn in sources],
            "virtual_overviews": n_ovr,
        }

    if written:
        with open(output_dir / "merge_metadata.yaml", "w") as f:
            yaml.dump(merge_meta, f, default_flow_style=False, sort_keys=False)
    return written


def merge(
    batch_dir: str,
    workunits: str | None = None,
    output_dir: str | None = None,
) -> None:
    """
    Merge the per-project products of a rasterize-projects batch into
    per-product VRTs (highest-priority project on top, no resampling; see
    merge_projects).

    Parameters
    ----------
    batch_dir
        rasterize-projects base directory.
    workunits
        Comma-separated project names in priority order, first = highest
        priority. Default: the order recorded in batch_status.yaml (i.e.
        the order given to rasterize-projects).
    output_dir
        Output directory for the VRTs, by default <batch_dir>/merge.
    """
    written = merge_projects(
        batch_dir,
        workunits=[w.strip() for w in workunits.split(",")] if workunits else None,
        output_dir=output_dir,
    )
    if not written:
        print(f"No product mosaics found under {batch_dir}")
