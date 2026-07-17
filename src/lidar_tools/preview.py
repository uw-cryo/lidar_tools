"""
One-page preview figure of the product mosaics in a rasterize output
directory, for quick visual QA after a run.

Style follows the group's DEM-figure conventions (README example /
groundcontrol plot.py / vantor figure scripts): color shaded relief
(elevation colormap at alpha 0.4 over a gray hillshade), per-panel
colorbars labeled with the datum, anchored scale bars, no coordinate
ticks, one row of panels, and a processing-metadata footer.

The hillshade/scalebar helpers mirror groundcontrol.plot — candidates for
the planned shared plotting library; keep the implementations in sync.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

#: product mosaic filename suffixes in display order: (suffix, label, kind)
_PRODUCT_PANELS = (
    ("DSM_mos", "DSM", "elevation"),
    ("DTM_no_fill_mos", "DTM (no fill)", "elevation"),
    ("DTM_fill_window_size_4_mos", "DTM (fill window 4 px)", "elevation"),
    ("intensity_mos", "Intensity", "intensity"),
)


def _hillshade(
    z: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    azdeg: float = 315.0,
    altdeg: float = 45.0,
) -> np.ndarray:
    """Gradient hillshade of a north-up elevation array, in [0, 1].

    Plain-numpy Horn/ESRI formulation, ported from groundcontrol.plot
    (verified there against matplotlib LightSource). NaNs propagate.
    """
    z = np.asarray(z, dtype="float64")
    g_south, g_east = np.gradient(z, dy, dx)
    slope = np.arctan(np.hypot(g_east, g_south))
    aspect = np.arctan2(g_south, -g_east)
    zenith = np.radians(90.0 - altdeg)
    az_math = np.radians((360.0 - azdeg + 90.0) % 360.0)
    hs = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        az_math - aspect
    )
    hs = np.clip(hs, 0.0, 1.0)
    hs[~np.isfinite(z)] = np.nan
    return hs


def _nice_scale_length(span: float) -> float:
    """A round 1/2/5 x 10^k length covering roughly 1/5 of ``span``."""
    target = span / 5.0
    k = np.floor(np.log10(target))
    base = target / 10.0**k
    nice = 1.0 if base < 1.5 else (2.0 if base < 3.5 else (5.0 if base < 7.5 else 10.0))
    return float(nice * 10.0**k)


def _add_scalebar(ax) -> None:
    """Anchored lower-right scale bar in map units (groundcontrol.plot style)."""
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    x0, x1 = ax.get_xlim()
    span = abs(x1 - x0)
    length = _nice_scale_length(span)
    label = f"{length / 1000.0:g} km" if length >= 1000 else f"{length:g} m"
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        "lower right",
        pad=0.3,
        sep=3,
        borderpad=0.5,
        frameon=True,
        size_vertical=span / 250.0,
        color="black",
        fontproperties=FontProperties(size=7),
    )
    bar.patch.set_alpha(0.85)
    ax.add_artist(bar)


def _decimal_year(dt: datetime) -> float:
    year_start = datetime(dt.year, 1, 1)
    year_end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - year_start) / (year_end - year_start)


def _elevation_cmap():
    """The group's rainbow-over-hillshade colormap (imview cpt_rainbow when
    available, matplotlib turbo otherwise — same fallback as vantor scripts)."""
    import matplotlib.pyplot as plt

    try:
        from imview.lib import gmtColormap

        return gmtColormap.get_rainbow()
    except Exception:
        return plt.cm.turbo


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
    epoch = srs.GetCoordinateEpoch() if srs is not None else 0.0
    return {
        "arr": masked,
        "extent": extent,
        "crs": srs.GetName() if srs is not None else "unknown CRS",
        "epoch": epoch,
        "res": gt[1],
        "size": (ds.RasterXSize, ds.RasterYSize),
        # decimated pixel size, for hillshading the decimated array
        "res_dec": (extent[1] - extent[0]) / nx,
    }


def _metadata_file(dirname: Path, kind: str) -> Path | None:
    """<prefix>-<kind>.yaml in dirname, falling back to the legacy bare
    <kind>.yaml of pre-2026-07-13 runs; None if neither exists."""
    hits = sorted(dirname.glob(f"*-{kind}.yaml"))
    if hits:
        return hits[0]
    legacy = dirname / f"{kind}.yaml"
    return legacy if legacy.exists() else None


def _project_metadata_files(project_dir: Path) -> list[Path]:
    """Processing-metadata files feeding this directory's products —
    the directory's own file, or every source project's for a merge dir."""
    own = _metadata_file(project_dir, "processing_metadata")
    if own is not None:
        return [own]
    merge_meta = _metadata_file(project_dir, "merge_metadata")
    found: list[Path] = []
    if merge_meta is not None:
        with open(merge_meta) as f:
            meta = yaml.safe_load(f)
        seen = set()
        for prod in meta.get("products", {}).values():
            for src in prod.get("sources_priority_order", []):
                fn = _metadata_file(Path(src).parent, "processing_metadata")
                if fn is not None and fn not in seen:
                    seen.add(fn)
                    found.append(fn)
    return found


def _footer_lines(project_dir: Path, panel: dict) -> list[str]:
    """Processing-details footer: 3D CRS + epoch, acquisition range
    (datetime + decimal year), grid, tile counts, projects, provenance."""
    epoch_txt = (
        f"{panel['epoch']:.2f}" if panel["epoch"] else "not stamped (static frame)"
    )
    lines = [
        f"CRS: {panel['crs']} (3D, ellipsoidal heights) | coord epoch: {epoch_txt}"
        f" | grid: {panel['res']:g} m, {panel['size'][0]} x {panel['size'][1]} px"
    ]
    metas = []
    for fn in _project_metadata_files(project_dir):
        with open(fn) as f:
            metas.append(yaml.safe_load(f))
    if not metas:
        return lines

    records = [r for m in metas for r in m.get("survey_records", [])]
    if records:
        projs = ", ".join(
            f"{r['workunit']} ({r.get('ql', '?')})" for r in records
        )
        starts = [datetime.fromisoformat(r["collect_start"]) for r in records]
        ends = [datetime.fromisoformat(r["collect_end"]) for r in records]
        t0, t1 = min(starts), max(ends)
        lines.append(f"projects: {projs}")
        lines.append(
            f"acquisition: {t0.date()} to {t1.date()} "
            f"({_decimal_year(t0):.2f} to {_decimal_year(t1):.2f})"
        )
    ip = metas[0].get("input_parameters", {})
    prov = []
    # per-project tile accounting recorded at run completion (runs after
    # 2026-07-10); summed across source projects for a merge directory
    counts = [m.get("run_status", {}) for m in metas]
    if all("tiles_total" in c for c in counts):
        total = sum(c["tiles_total"] for c in counts)
        empty = sum(c["tiles_empty"] for c in counts)
        tile_km = ip.get("tile_size")
        tile_txt = f" ({tile_km:g} km)" if tile_km else ""
        prov.append(f"tiles: {total} total, {empty} empty{tile_txt}")
    if ip.get("input"):
        prov.append(f"source: {ip['input']}")
    ver = metas[0].get("lidar_tools_version")
    commit = str(metas[0].get("git_commit", ""))[:12]
    stamp = str(metas[0].get("processing_timestamp", ""))[:10]
    prov.append(f"lidar_tools {ver} @ {commit}, processed {stamp}")
    lines.append(" | ".join(prov))
    return lines


def product_preview(
    project_dir: str | Path,
    out_fn: str | Path | None = None,
    max_dim: int = 1600,
    dpi: int = 300,
) -> Path | None:
    """
    Render every product mosaic found in project_dir onto one preview page:
    one row of panels (~10 x 4 in), color shaded relief for the elevation
    products (shared robust 2-98 % range), grayscale intensity, per-panel
    scale bars, and a processing-metadata footer.

    Parameters
    ----------
    project_dir
        A rasterize output directory (one project) or a merge directory
        containing *_mos.tif / *_mos.vrt products.
    out_fn
        Output PNG path, by default <project_dir>/<prefix>-preview.png
        where prefix is the product filename prefix (AOI name + grid
        posting, e.g. aoi_1m-preview.png).
    max_dim
        Decimated read size for the long edge, by default 1600 px.
    dpi
        Output resolution, by default 300 (panel detail at the 4 in height).

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
    prefix = None
    for suffix, label, kind in _PRODUCT_PANELS:
        # .tif = per-project mosaics, .vrt = merge-stage composites
        hits = sorted(project_dir.glob(f"*-{suffix}.tif")) or sorted(
            project_dir.glob(f"*-{suffix}.vrt")
        )
        if hits:
            if prefix is None:
                prefix = hits[0].name.rsplit(f"-{suffix}", 1)[0]
            panels.append(
                {**_read_decimated(hits[0], max_dim), "label": label, "kind": kind}
            )
    if not panels:
        return None
    if out_fn is None:
        # inherit the product prefix (AOI name + grid posting)
        out_fn = project_dir / f"{prefix}-preview.png"

    elev_parts = [
        np.ma.compressed(p["arr"]) for p in panels if p["kind"] == "elevation"
    ]
    elev_parts = [e for e in elev_parts if e.size]
    elev_clim = (
        np.percentile(np.concatenate(elev_parts), [2, 98]) if elev_parts else (0, 1)
    )
    rainbow = _elevation_cmap()
    datum = panels[0]["crs"].split(" / ")[0]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4.0))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        arr = np.ma.filled(p["arr"].astype("float64"), np.nan)
        if p["kind"] == "elevation":
            hs = _hillshade(arr, dx=p["res_dec"], dy=p["res_dec"])
            ax.imshow(
                hs, cmap="gray", vmin=0, vmax=1, extent=p["extent"],
                interpolation="bilinear",
            )
            im = ax.imshow(
                arr, cmap=rainbow, alpha=0.5, extent=p["extent"],
                vmin=elev_clim[0], vmax=elev_clim[1], interpolation="bilinear",
            )
            cb_label = f"Elevation (m, {datum})"
        else:
            data = arr[np.isfinite(arr)]
            vmin, vmax = np.percentile(data, [2, 98]) if data.size else (0, 1)
            im = ax.imshow(
                arr, cmap="gray", vmin=vmin, vmax=vmax, extent=p["extent"],
                interpolation="bilinear",
            )
            cb_label = "Intensity (DN)"
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, extend="both")
        cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(p["label"], fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scalebar(ax)

    footer = _footer_lines(project_dir, panels[0])
    title = project_dir.name
    if title in ("merge",) and project_dir.parent.name:
        title = f"{project_dir.parent.name} / merge"
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.text(
        0.01, 0.01, "\n".join(footer), fontsize=6, family="monospace",
        va="bottom", ha="left",
    )
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.9, bottom=0.05 + 0.032 * len(footer),
        wspace=0.12,
    )
    fig.savefig(out_fn, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(out_fn)


def preview(path: str, max_dim: int = 1600, dpi: int = 300) -> None:
    """
    Write preview page(s) for a rasterize output directory, or for every
    project subdirectory (and merge directory) of a rasterize-projects
    batch directory.

    Parameters
    ----------
    path
        A project/merge output directory containing product mosaics, or a
        batch directory whose immediate subdirectories contain them.
    max_dim
        Decimated read size for the long edge, by default 1600 px.
    dpi
        Output PNG resolution, by default 300.
    """
    p = Path(path)
    written = []
    fn = product_preview(p, max_dim=max_dim, dpi=dpi)
    if fn is not None:
        written.append(fn)
    else:
        for sub in sorted(d for d in p.iterdir() if d.is_dir()):
            fn = product_preview(sub, max_dim=max_dim, dpi=dpi)
            if fn is not None:
                written.append(fn)
    for fn in written:
        print(f"Wrote preview to {fn}")
    if not written:
        print(f"No product mosaics found under {p}")
