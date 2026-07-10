"""
One-page preview figure of the product mosaics in a rasterize output
directory, for quick visual QA after a run (and iteration on issues
without opening a GIS).
"""

from pathlib import Path

import numpy as np

#: product mosaic filename suffixes in display order: (suffix, label, kind)
_PRODUCT_PANELS = (
    ("DSM_mos", "DSM", "elevation"),
    ("DTM_no_fill_mos", "DTM (no fill)", "elevation"),
    ("DTM_fill_window_size_4_mos", "DTM (window fill)", "elevation"),
    ("intensity_mos", "Intensity", "intensity"),
)


def _read_decimated(fn: Path, max_dim: int) -> dict:
    """
    Read one band decimated to <= max_dim pixels on the long edge (GDAL
    serves this from the COG overviews) as a masked array plus georeference.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.OpenEx(str(fn))
    band = ds.GetRasterBand(1)
    scale = max(1, int(np.ceil(max(ds.RasterXSize, ds.RasterYSize) / max_dim)))
    nx = max(1, ds.RasterXSize // scale)
    ny = max(1, ds.RasterYSize // scale)
    arr = band.ReadAsArray(buf_xsize=nx, buf_ysize=ny)
    nodata = band.GetNoDataValue()
    masked = (
        np.ma.masked_equal(arr, nodata)
        if nodata is not None
        else np.ma.masked_invalid(arr)
    )
    gt = ds.GetGeoTransform()
    extent = (
        gt[0],
        gt[0] + gt[1] * ds.RasterXSize,
        gt[3] + gt[5] * ds.RasterYSize,
        gt[3],
    )
    srs = ds.GetSpatialRef()
    return {
        "arr": masked,
        "extent": extent,
        "crs": srs.GetName() if srs is not None else "unknown CRS",
        "res": gt[1],
    }


def product_preview(
    project_dir: str | Path, out_fn: str | Path | None = None, max_dim: int = 2048
) -> Path | None:
    """
    Render every product mosaic found in project_dir onto one preview page.

    Elevation panels share one robust (2-98 %) color range so DSM/DTM
    differences read directly; intensity gets its own grayscale range.

    Parameters
    ----------
    project_dir
        A rasterize output directory (one project) containing *_mos.tif.
    out_fn
        Output PNG path, by default <project_dir>/preview.png.
    max_dim
        Decimated read size for the long edge, by default 2048 px.

    Returns
    -------
    Path | None
        The written PNG path, or None if no product mosaics were found.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_dir = Path(project_dir)
    panels = []
    for suffix, label, kind in _PRODUCT_PANELS:
        hits = sorted(project_dir.glob(f"*-{suffix}.tif"))
        if hits:
            panels.append({**_read_decimated(hits[0], max_dim), "label": label, "kind": kind})
    if not panels:
        return None
    if out_fn is None:
        out_fn = project_dir / "preview.png"

    elev = [
        np.ma.compressed(p["arr"]) for p in panels if p["kind"] == "elevation"
    ]
    elev = np.concatenate([e for e in elev if e.size]) if any(e.size for e in elev) else None
    clim = {
        "elevation": np.percentile(elev, [2, 98]) if elev is not None else (0, 1),
    }

    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 7 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    for ax, p in zip(axes, panels):
        arr = p["arr"]
        if p["kind"] == "elevation":
            vmin, vmax = clim["elevation"]
            cmap, unit = "viridis", "m"
        else:
            data = np.ma.compressed(arr)
            vmin, vmax = np.percentile(data, [2, 98]) if data.size else (0, 1)
            cmap, unit = "gray", "DN"
        im = ax.imshow(
            arr, extent=p["extent"], cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        valid = 1.0 - np.ma.getmaskarray(arr).mean()
        ax.set_title(f"{p['label']}  ({valid:.1%} valid)", fontsize=11)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.65, pad=0.02, label=unit)
    fig.suptitle(
        f"{project_dir.name} — {panels[0]['crs']} @ {panels[0]['res']:g} m",
        fontsize=13,
    )
    fig.savefig(out_fn, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(out_fn)


def preview(path: str, max_dim: int = 2048) -> None:
    """
    Write preview page(s) for a rasterize output directory, or for every
    project subdirectory of a rasterize-projects batch directory.

    Parameters
    ----------
    path
        A project output directory containing product mosaics, or a batch
        directory whose immediate subdirectories contain them.
    max_dim
        Decimated read size for the long edge, by default 2048 px.
    """
    p = Path(path)
    written = []
    fn = product_preview(p, max_dim=max_dim)
    if fn is not None:
        written.append(fn)
    else:
        for sub in sorted(d for d in p.iterdir() if d.is_dir()):
            fn = product_preview(sub, max_dim=max_dim)
            if fn is not None:
                written.append(fn)
    for fn in written:
        print(f"Wrote preview to {fn}")
    if not written:
        print(f"No product mosaics found under {p}")
