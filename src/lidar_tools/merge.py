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
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

#: product mosaic filename suffixes eligible for merging
PRODUCT_SUFFIXES = (
    "DSM_mos",
    "DTM_no_fill_mos",
    "DTM_fill_window_size_4_mos",
    "intensity_mos",
)

#: common target range for normalized intensity (DN-like, arbitrary units)
INTENSITY_TARGET = (1000.0, 60000.0)
#: minimum same-ground sample (at the decimated stats grid) to trust an
#: overlap-fitted refinement over the plain global stretch. Counted on the
#: max_dim-capped grid, so one decimated pixel covers more ground on a
#: larger AOI: the same physical overlap that cleared 100k px on the
#: cal-range batch fell to ~33-93k px on the full Casa Grande AOI and
#: silently degraded every source to global-stretch. 20k samples keep the
#: 81-point quantile fit stable while still refusing sliver overlaps.
MIN_OVERLAP_PX = 20_000
#: nodata for the Float32 normalized intensity band (0 stays a valid value)
NORM_NODATA = -9999.0


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


def _read_decimated_band(fn: Path, max_dim: int = 8000) -> tuple:
    """Decimated band read (served from the COG overviews) -> (float array,
    valid mask, source nodata, dtype max). Aligned grids give identical
    decimated shapes across sources."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.OpenEx(str(fn))
    band = ds.GetRasterBand(1)
    scale = max(1, int(np.ceil(max(ds.RasterXSize, ds.RasterYSize) / max_dim)))
    arr = band.ReadAsArray(
        buf_xsize=max(1, ds.RasterXSize // scale),
        buf_ysize=max(1, ds.RasterYSize // scale),
    ).astype(np.float64)
    nodata = band.GetNoDataValue()
    mask = np.isfinite(arr) if nodata is None else (arr != nodata)
    dtype_max = {
        "Byte": 255.0,
        "UInt16": 65535.0,
        "Int16": 32767.0,
        "UInt32": float(2**32 - 1),
    }.get(gdal.GetDataTypeName(band.DataType), np.inf)
    return arr, mask, nodata, dtype_max


def _intensity_normalization(sources: list[Path]) -> list[dict]:
    """
    One linear map (gain, offset) per source onto the common
    ``INTENSITY_TARGET`` range, in priority order (first = anchor).

    Each source starts from its own robust stretch (2-98 % of its valid
    pixels -> target range); sources overlapping the higher-priority
    composite by at least MIN_OVERLAP_PX decimated pixels get that stretch
    refined by a linear fit through the same-ground paired quantiles
    (10-90 %), which removes the land-cover-mix bias of the global stretch.
    The two maps compose into the single returned gain/offset.
    """
    params: list[dict] = []
    composite: np.ndarray | None = None
    comp_mask: np.ndarray | None = None
    for src in sources:
        arr, mask, _, dtype_max = _read_decimated_band(src)
        valid = arr[mask]
        p2, p98 = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        if p98 <= p2:
            p2, p98 = float(valid.min()), float(valid.min()) + 1.0
        gain = (INTENSITY_TARGET[1] - INTENSITY_TARGET[0]) / (p98 - p2)
        offset = INTENSITY_TARGET[0] - gain * p2
        method = "global-stretch"
        n_overlap = 0
        if composite is not None:
            assert comp_mask is not None
            overlap = comp_mask & mask
            n_overlap = int(overlap.sum())
            if n_overlap >= MIN_OVERLAP_PX:
                # same-ground refinement: paired quantiles of this source
                # (in target space) vs the higher-priority composite
                qq = np.arange(10, 91)
                qx = np.percentile(gain * arr[overlap] + offset, qq)
                qy = np.percentile(composite[overlap], qq)
                if qx[-1] > qx[0]:
                    g2, o2 = np.polyfit(qx, qy, 1)
                    # g2 <= 0 (anti-correlated overlap) would invert the map
                    # and break the increasing clamp LUT; keep the stretch
                    if g2 > 0:
                        gain, offset = g2 * gain, g2 * offset + o2
                        method = "overlap-refined"
        saturated = float((valid == dtype_max).mean()) if valid.size else 0.0
        if saturated > 0.01:
            print(
                f"WARNING: {src.name}: {saturated:.1%} of valid intensity "
                "pixels sit at the dtype ceiling (saturated) — the global "
                "stretch percentiles may be biased",
                file=sys.stderr,
            )
        if composite is None:
            composite = np.full(arr.shape, np.nan)
            comp_mask = np.zeros(arr.shape, dtype=bool)
        assert comp_mask is not None
        newly = mask & ~comp_mask
        composite[newly] = gain * arr[newly] + offset
        comp_mask |= mask
        params.append(
            {
                "source": str(src),
                "gain": float(gain),
                "offset": float(offset),
                "method": method,
                "overlap_px": n_overlap,
                "stretch_percentiles": [float(p2), float(p98)],
                "saturated_frac": saturated,
            }
        )
    return params


def _apply_vrt_normalization(vrt_fn: Path, params: list[dict]) -> None:
    """
    Rewrite a BuildVRT product as the normalized composite: Float32 band
    (nodata NORM_NODATA — a linear map with gain > 1 would clamp the dark
    tail to 0 = source nodata in a UInt16 band, silently punching holes)
    and a per-ComplexSource two-point LUT encoding each source's linear map
    CLAMPED to INTENSITY_TARGET (GDAL LUTs hold their endpoint value outside
    the table). Unclamped ScaleRatio/ScaleOffset let a source's un-fitted
    radiometric regimes land entire spans outside the target range (a
    bimodal lift in the full-AOI PimaCo_2 mapped 13% of its pixels to
    ~-140k, dragging every default display stretch of the composite), and a
    valid pixel could map to exactly NORM_NODATA and vanish.
    """
    # keyed by resolved path: basenames collide across projects (same AOI
    # prefix), so the filename alone would assign every source one map
    by_path = {Path(p["source"]).resolve(): p for p in params}
    tree = ET.parse(vrt_fn)
    band = tree.getroot().find("VRTRasterBand")
    assert band is not None, f"no VRTRasterBand in {vrt_fn}"
    band.set("dataType", "Float32")
    nd = band.find("NoDataValue")
    if nd is None:
        nd = ET.SubElement(band, "NoDataValue")
    nd.text = f"{NORM_NODATA:g}"
    for source in band.findall("ComplexSource"):
        el = source.find("SourceFilename")
        assert el is not None and el.text, f"sourceless ComplexSource in {vrt_fn}"
        key = (
            (vrt_fn.parent / el.text) if el.get("relativeToVRT") == "1"
            else Path(el.text)
        ).resolve()
        p = by_path[key]
        # raw DNs whose mapped values hit the target endpoints; the LUT
        # interpolates linearly between them and clamps beyond
        src_lo = (INTENSITY_TARGET[0] - p["offset"]) / p["gain"]
        src_hi = (INTENSITY_TARGET[1] - p["offset"]) / p["gain"]
        ET.SubElement(source, "LUT").text = (
            f"{src_lo:.10g}:{INTENSITY_TARGET[0]:g},"
            f"{src_hi:.10g}:{INTENSITY_TARGET[1]:g}"
        )
    tree.write(vrt_fn)


def merge_projects(
    batch_dir: str | Path,
    workunits: list[str] | None = None,
    output_dir: str | Path | None = None,
    normalize_intensity: bool = True,
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
    normalize_intensity
        When more than one project contributes intensity, map each source
        onto a common range with one linear map per source in the VRT (own
        robust stretch, refined by a same-ground overlap fit against the
        higher-priority composite; see _intensity_normalization), clamped
        to the target range (see _apply_vrt_normalization). Vendors
        deliver raw amplitude on incompatible scales (gh #34) — an
        unnormalized cross-project composite is unusable. The per-project
        mosaics stay raw; the normalized band is Float32 (nodata -9999).
        Elevation products are never touched. Default True.

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
        source_wus: list[str] = []
        for wu in workunits:
            hits = sorted((batch_dir / wu).glob(f"*-{suffix}.tif"))
            if hits:
                sources.append(hits[0])
                source_wus.append(wu)
        if not sources:
            continue

        signatures = {fn: _raster_signature(fn) for fn in sources}
        if len(set(signatures.values())) > 1:
            detail = "\n".join(f"  {fn}: {sig}" for fn, sig in signatures.items())
            raise ValueError(
                f"{suffix}: per-project mosaics are not on one grid; refusing "
                f"to merge without resampling:\n{detail}"
            )

        # the composite drops the per-project token from the source name
        # (aoi_1m_AZ_PimaCo_1_2021-DSM_mos.tif -> aoi_1m-DSM_mos.vrt);
        # sources without the token (older runs) pass through unchanged
        base = sources[0].name.rsplit(".", 1)[0]
        out_fn = output_dir / f"{base.replace(f'_{source_wus[0]}-', '-')}.vrt"
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
        norm_params = None
        if suffix == "intensity_mos" and normalize_intensity and len(sources) > 1:
            norm_params = _intensity_normalization(sources)
            _apply_vrt_normalization(out_fn, norm_params)
        print(
            f"{suffix}: merged {len(sources)} project(s) -> {out_fn} "
            f"({n_ovr} virtual overview levels)"
        )
        if norm_params:
            detail = ", ".join(
                f"{Path(p['source']).parent.name}: {p['method']}"
                for p in norm_params
            )
            print(
                f"  intensity normalized to common range "
                f"{INTENSITY_TARGET[0]:g}-{INTENSITY_TARGET[1]:g} ({detail})"
            )
        written.append(out_fn)
        merge_meta["products"][suffix] = {
            "vrt": out_fn.name,
            "sources_priority_order": [str(fn) for fn in sources],
            "virtual_overviews": n_ovr,
        }
        if norm_params:
            merge_meta["products"][suffix]["intensity_normalization"] = {
                "target_range": list(INTENSITY_TARGET),
                "clamped_to_target": True,
                "band_dtype": "Float32",
                "nodata": NORM_NODATA,
                "sources": norm_params,
            }

    if written:
        # inherit the composite prefix (AOI + posting; product suffixes
        # carry no '-', so rsplit is safe even for dashed AOI names)
        prefix = written[0].name.rsplit("-", 1)[0]
        with open(output_dir / f"{prefix}-merge_metadata.yaml", "w") as f:
            yaml.dump(merge_meta, f, default_flow_style=False, sort_keys=False)
    return written


def merge(
    batch_dir: str,
    workunits: str | None = None,
    output_dir: str | None = None,
    normalize_intensity: bool = True,
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
    normalize_intensity
        Map each project's intensity onto a common range with a per-source
        clamped linear map in the merge VRT (vendor amplitude scales are
        incompatible across projects, gh #34); per-project mosaics stay
        raw. Default True.
    """
    written = merge_projects(
        batch_dir,
        workunits=[w.strip() for w in workunits.split(",")] if workunits else None,
        output_dir=output_dir,
        normalize_intensity=normalize_intensity,
    )
    if not written:
        print(f"No product mosaics found under {batch_dir}")
