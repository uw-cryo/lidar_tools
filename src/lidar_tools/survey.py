"""
Survey discovery and metadata for an AOI: which lidar collections cover the
area, with quality level, acquisition dates, CRS/datum/geoid declarations,
and point-cloud-service (EPT) availability.

This is the metadata backbone for per-project processing: an explicit,
inspectable inventory comes first; selection and processing consume its
records. Sources are pluggable — USGS WESM is the primary provider for
3DEP/CONUS, but the summarize/coverage logic is provider-agnostic
(any polygon layer with per-collection attributes works).

Notes
-----
- WESM workunit polygons are read remotely with a bbox filter
  (GeoPackage spatial index over /vsicurl), so only the AOI's pages of the
  ~3.4 GB file are fetched. Pass a local path for offline/pinned use.
- EPT availability is determined SPATIALLY against the hobu boundary index:
  name joins between WESM workunits and EPT resources are unreliable
  (renames, misspellings, FTP-era naming).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

WESM_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WESM.gpkg"
)
EPT_RESOURCES_URL = (
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/"
    "resources.geojson"
)

# WESM attributes carried into survey records (kept if present)
WESM_FIELDS = [
    "workunit",
    "workunit_id",
    "project",
    "collect_start",
    "collect_end",
    "ql",
    "spec",
    "p_method",
    "horiz_crs",
    "vert_crs",
    "geoid",
    "lpc_pub_date",
    "lpc_category",
    "lpc_reason",
    "lpc_link",
    "metadata_link",
]


def load_wesm(aoi_gdf: gpd.GeoDataFrame, wesm_source: str = WESM_URL) -> gpd.GeoDataFrame:
    """
    Read WESM workunit polygons intersecting the AOI bounding box.

    Parameters
    ----------
    aoi_gdf
        AOI polygon(s), any CRS.
    wesm_source
        Path or URL of the WESM GeoPackage. Remote URLs are read through
        /vsicurl with a bbox filter (only the needed pages are fetched).

    Returns
    -------
    gpd.GeoDataFrame
        WESM rows whose polygons intersect the AOI bounds (EPSG:4326).
    """
    src = str(wesm_source)
    if src.startswith(("http://", "https://")):
        src = f"/vsicurl/{src}"
    bbox = tuple(aoi_gdf.to_crs("EPSG:4326").total_bounds)
    return gpd.read_file(src, bbox=bbox)


def _yaml_safe(value):
    """Convert record values to plain YAML-serializable types."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def record_from_wesm(wesm_gdf: gpd.GeoDataFrame, workunit: str) -> dict:
    """
    Extract one workunit's sanitized attribute record (WESM_FIELDS) from a
    WESM query result. Pure/offline; raises ValueError when absent.
    """
    rows = wesm_gdf[wesm_gdf["workunit"] == workunit]
    if rows.empty:
        available = sorted(wesm_gdf.get("workunit", pd.Series(dtype=str)).astype(str))
        raise ValueError(
            f"Workunit '{workunit}' not found in the WESM query result "
            f"(available here: {available})"
        )
    row = rows.iloc[0]
    return {k: _yaml_safe(row[k]) for k in WESM_FIELDS if k in rows.columns}


def workunit_record(
    aoi_gdf: gpd.GeoDataFrame, workunit: str, wesm_source: str = WESM_URL
) -> dict:
    """
    Fetch the live WESM record for one workunit intersecting the AOI —
    the per-survey source of truth for datum/geoid/QL/date handling,
    pinned into processing metadata by the pipeline.
    """
    return record_from_wesm(load_wesm(aoi_gdf, wesm_source), workunit)


def load_ept_resources(url: str = EPT_RESOURCES_URL) -> gpd.GeoDataFrame:
    """
    Load the EPT resource boundary index (hobu usgs-lidar mirror).

    Returns
    -------
    gpd.GeoDataFrame
        One row per EPT resource with `name`, `url` (when present), and a
        boundary polygon.
    """
    return gpd.read_file(url)


def summarize_surveys(
    wesm_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    ept_gdf: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Build per-collection survey records for an AOI.

    Pure geometry/attribute logic (no network): computes each collection's
    overlap with the AOI and, when an EPT boundary index is supplied, the
    spatially-matched EPT resources and the fraction of the collection's
    AOI footprint they cover.

    Parameters
    ----------
    wesm_gdf
        Collection polygons + attributes (WESM or provider-equivalent).
    aoi_gdf
        AOI polygon(s), any CRS.
    ept_gdf
        Optional EPT boundary index with `name` and geometry.

    Returns
    -------
    gpd.GeoDataFrame
        One row per collection intersecting the AOI, in EPSG:4326, with
        `aoi_overlap_frac`, `ept_names`, `ept_coverage_frac`, plus the
        provider attributes (WESM_FIELDS where present), sorted by
        quality level then recency.
    """
    aoi = aoi_gdf.to_crs("EPSG:4326").union_all()
    w = wesm_gdf.to_crs("EPSG:4326")
    w = w[w.intersects(aoi)].copy()
    if w.empty:
        return w

    w["aoi_overlap_frac"] = w.geometry.apply(
        lambda g: g.intersection(aoi).area / aoi.area
    )

    ept_names: list = []
    ept_cov: list = []
    if ept_gdf is not None:
        e = ept_gdf.to_crs("EPSG:4326")
        for g in w.geometry:
            footprint = g.intersection(aoi)
            hits = e[e.intersects(footprint)]
            ept_names.append(sorted(hits["name"].tolist()))
            if footprint.is_empty or footprint.area == 0:
                ept_cov.append(0.0)
            else:
                covered = hits.union_all().intersection(footprint)
                ept_cov.append(covered.area / footprint.area)
    else:
        ept_names = [None] * len(w)
        ept_cov = [None] * len(w)
    w["ept_names"] = ept_names
    w["ept_coverage_frac"] = ept_cov

    cols = [c for c in WESM_FIELDS if c in w.columns]
    out = w[
        cols + ["aoi_overlap_frac", "ept_names", "ept_coverage_frac", "geometry"]
    ]
    sort_cols = [c for c in ["ql", "collect_end"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(
            sort_cols, ascending=[True, False][: len(sort_cols)]
        )
    return out.reset_index(drop=True)


def _collect_midpoint(row) -> pd.Timestamp:
    """Midpoint of a collection's acquisition window (NaT when unknown)."""
    try:
        t0 = pd.Timestamp(row.get("collect_start"))
        t1 = pd.Timestamp(row.get("collect_end"))
        return t0 + (t1 - t0) / 2
    except (TypeError, ValueError):
        return pd.NaT


def assign_epochs(surveys_gdf: gpd.GeoDataFrame, gap_years: float = 1.5) -> gpd.GeoDataFrame:
    """
    Group collections into acquisition epochs: clusters of collection
    midpoints separated by more than gap_years. AOIs commonly hold a few
    distinct well-covered timesteps (e.g. Las Vegas 2022-2023 vs 2016;
    San Francisco 2023 vs 2010) that can be built as separate product sets
    for change analysis; the epoch label makes those groups explicit.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with an `epoch` column (e.g. '2023' or '2022-2023';
        'undated' when no dates are available).
    """
    s = surveys_gdf.copy()
    mids = {idx: _collect_midpoint(r) for idx, r in s.iterrows()}
    dated = sorted(
        (m, idx) for idx, m in mids.items() if pd.notna(m)
    )
    labels = {}
    cluster: list = []

    def flush():
        if not cluster:
            return
        years = sorted({m.year for m, _ in cluster})
        label = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
        for _, idx in cluster:
            labels[idx] = label

    prev = None
    for m, idx in dated:
        if prev is not None and (m - prev).days / 365.25 > gap_years:
            flush()
            cluster = []
        cluster.append((m, idx))
        prev = m
    flush()
    s["epoch"] = [labels.get(idx, "undated") for idx in s.index]
    return s


def pick_anchor(surveys_gdf: gpd.GeoDataFrame) -> int:
    """
    Choose the anchor collection: the reference against which additional
    collections are judged. Among spec-meeting collections (when any),
    the one covering the most of the AOI, ties broken by recency. There
    are no hard absolute rules for combining surveys — the anchor gives
    the relative frame ("how different is each addition from the best
    base coverage").

    Returns
    -------
    int
        Index (label) of the anchor row in surveys_gdf.
    """
    c = surveys_gdf
    if "lpc_category" in c.columns:
        # "Meets" and "Meets with variance" (common for Geiger/SPL
        # collections) both count as spec-meeting
        meets = c["lpc_category"].astype(str).str.startswith("Meets")
        if meets.any():
            c = c[meets]
    sort_cols = [x for x in ["aoi_overlap_frac", "collect_end"] if x in c.columns]
    return c.sort_values(sort_cols, ascending=False).index[0]


def relative_metrics(surveys_gdf: gpd.GeoDataFrame, anchor_idx: int = None) -> gpd.GeoDataFrame:
    """
    Annotate each collection with differences relative to the anchor:
    years between acquisitions, and whether the declared horizontal CRS
    and geoid model match. These are the decision inputs for whether
    merging is acceptable — judged per AOI, not by absolute rules.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with `anchor` (bool), `dt_years` (acquisition midpoint offset
        from the anchor, signed), `same_geoid`, `same_horiz_crs`.
    """
    out = surveys_gdf.copy()
    if anchor_idx is None:
        anchor_idx = pick_anchor(out)
    a = out.loc[anchor_idx]
    out["anchor"] = out.index == anchor_idx

    mid_a = _collect_midpoint(a)
    out["dt_years"] = [
        round((_collect_midpoint(r) - mid_a).days / 365.25, 2)
        if pd.notna(_collect_midpoint(r)) and pd.notna(mid_a)
        else None
        for _, r in out.iterrows()
    ]
    if "geoid" in out.columns:
        out["same_geoid"] = out["geoid"] == a.get("geoid")
    if "horiz_crs" in out.columns:
        out["same_horiz_crs"] = out["horiz_crs"] == a.get("horiz_crs")
    return out


def rank_collections(surveys_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Order collections by inclusion priority: the anchor first, then
    spec-meeting collections by quality level, temporal proximity to the
    anchor, and AOI overlap; legacy/'Other' collections last. Adds a
    1-based `priority` column and sorts by it. This is a default ordering
    for inspection, not a hard rule — combinability is assessed per AOI
    (see the compare stage).

    Returns
    -------
    gpd.GeoDataFrame
        Sorted copy with `priority` column.
    """
    s = surveys_gdf.copy()
    if "anchor" not in s.columns:
        s = relative_metrics(s)
    ql_order = {"QL 0": 0, "QL 1": 1, "QL 2": 2, "QL 3": 3}
    s["_ql"] = s.get("ql", pd.Series(index=s.index, dtype=object)).map(
        lambda q: ql_order.get(str(q), 9)
    )
    if "lpc_category" in s.columns:
        s["_meets"] = (~s["lpc_category"].astype(str).str.startswith("Meets")).astype(int)
    else:
        s["_meets"] = 0
    s["_absdt"] = s["dt_years"].abs().fillna(999.0)
    s = s.sort_values(
        ["anchor", "_meets", "_ql", "_absdt", "aoi_overlap_frac"],
        ascending=[False, True, True, True, False],
    ).drop(columns=["_ql", "_meets", "_absdt"])
    s["priority"] = range(1, len(s) + 1)
    return s


QL_COLORS = {
    "QL 0": "#1a7a1a",
    "QL 1": "#6abf40",
    "QL 2": "#e8a838",
    "QL 3": "#d46a6a",
}


def plot_coverage(
    surveys_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    gaps_gdf: gpd.GeoDataFrame = None,
    out_fn: str = None,
    title: str = None,
):
    """
    Map of collection footprints over the AOI: polygons colored by quality
    level (gray = legacy/'Other'), numbered and keyed to a legend carrying
    workunit, QL, dates, geoid, EPT coverage, and years-from-anchor; the
    anchor outlined in bold; no-coverage gaps hatched red.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surveys = surveys_gdf.to_crs("EPSG:4326")
    if "priority" not in surveys.columns:
        surveys = rank_collections(surveys)
    if "epoch" not in surveys.columns:
        surveys = assign_epochs(surveys)
    aoi = aoi_gdf.to_crs("EPSG:4326")
    aoi_geom = aoi.union_all()

    # numbers = proposed inclusion priority (1 = anchor); draw larger
    # footprints first so smaller ones stay visible on top
    draw_order = surveys.geometry.area.sort_values(ascending=False).index
    n_lines = len(surveys) + surveys["epoch"].nunique() + 3
    # centered map on top, single-line legend grouped by epoch below
    text_in = 0.17 * n_lines + 0.3
    fig_h = 6.6 + text_in
    fig = plt.figure(figsize=(12, fig_h))
    ax = fig.add_axes(
        [0.06, (text_in + 0.55) / fig_h, 0.88, 1 - (text_in + 0.55) / fig_h - 0.05]
    )
    ax.set_anchor("N")  # center the (aspect-constrained) map horizontally
    legend_entries = {}
    for idx in draw_order:
        row = surveys.loc[idx]
        n = int(row["priority"])
        clipped = row.geometry.intersection(aoi_geom)
        if clipped.is_empty:
            continue
        color = QL_COLORS.get(str(row.get("ql", "")), "#9a9a9a")
        is_anchor = bool(row.get("anchor", False))
        gpd.GeoSeries([clipped], crs="EPSG:4326").plot(
            ax=ax,
            facecolor=color,
            alpha=0.35,
            edgecolor="black" if is_anchor else color,
            linewidth=2.5 if is_anchor else 1.0,
        )
        pt = clipped.representative_point()
        ax.annotate(
            str(n),
            (pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", alpha=0.9),
        )
        dt = row.get("dt_years")
        dt_txt = f", {dt:+.1f} yr" if dt not in (None, 0.0) and pd.notna(dt) else ""
        ept_cov = row.get("ept_coverage_frac")
        ept_txt = (
            f", EPT {ept_cov:.0%}" if ept_cov is not None and ept_cov < 0.99 else ""
        )
        frac = row["aoi_overlap_frac"]
        frac_txt = f"{frac:.1%}" if frac < 0.01 else f"{frac:.0%}"
        legend_entries[n] = (
            f"  {n}. {row.get('workunit', '?')}  [{row.get('ql', '?')}] "
            f"{str(row.get('collect_start', '?'))[:10]}..{str(row.get('collect_end', '?'))[:10]} "
            f"{row.get('geoid', '?')}, {frac_txt} of AOI{dt_txt}{ept_txt}"
            f"{'  << ANCHOR' if is_anchor else ''}"
        )

    # group legend by epoch: anchor's epoch first, then by temporal
    # distance from it
    by_priority = surveys.sort_values("priority")
    anchor_epoch = by_priority[by_priority["anchor"]]["epoch"].iloc[0]
    epoch_dt = by_priority.groupby("epoch")["dt_years"].apply(
        lambda v: v.abs().min()
    )
    epoch_order = sorted(
        epoch_dt.index,
        key=lambda e: (e != anchor_epoch, epoch_dt[e] if pd.notna(epoch_dt[e]) else 999),
    )
    legend_lines = []
    for epoch in epoch_order:
        members = by_priority[by_priority["epoch"] == epoch]
        legend_lines.append(f"epoch {epoch}:")
        legend_lines.extend(
            legend_entries[int(p)] for p in members["priority"] if int(p) in legend_entries
        )
    if gaps_gdf is not None and not gaps_gdf.empty:
        gaps_gdf.to_crs("EPSG:4326").plot(
            ax=ax, facecolor="none", edgecolor="red", hatch="///", linewidth=1.0
        )
        legend_lines.append(
            f"red hatch: no lidar coverage "
            f"({gaps_gdf['gap_frac'].sum():.1%} of AOI)"
        )

    aoi.boundary.plot(ax=ax, color="black", linewidth=2, linestyle="--")
    minx, miny, maxx, maxy = aoi.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect(1 / max(0.1, abs(np.cos(np.deg2rad((miny + maxy) / 2)))))
    ax.set_title(title or "Lidar collection coverage")
    ax.locator_params(axis="x", nbins=5)
    legend_lines.insert(
        0,
        "collections by proposed epoch (numbers = proposed inclusion "
        "priority, 1 = anchor; user decides):",
    )
    # fixed canvas (no tight bbox): size the font so the longest line fits
    max_chars = max(len(line) for line in legend_lines)
    usable_pt = 0.90 * 12 * 72
    fontsize = min(8.5, usable_pt / (0.62 * max_chars))
    fig.text(
        0.05,
        (text_in + 0.15) / fig_h,
        "\n".join(legend_lines),
        fontsize=fontsize,
        family="monospace",
        va="top",
        ha="left",
    )
    if out_fn is not None:
        fig.savefig(out_fn, dpi=130)
        print(f"Wrote coverage map to {out_fn}")
    return fig


def plot_coverage_panels(
    surveys_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    out_fn: str = None,
):
    """
    Small-multiples companion to plot_coverage: one subplot per collection
    showing its footprint alone against the AOI outline, in priority order —
    disambiguates areas where the combined map overlaps heavily.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surveys = surveys_gdf.to_crs("EPSG:4326")
    if "priority" not in surveys.columns:
        surveys = rank_collections(surveys)
    surveys = surveys.sort_values("priority")
    aoi = aoi_gdf.to_crs("EPSG:4326")
    aoi_geom = aoi.union_all()
    minx, miny, maxx, maxy = aoi.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05

    n = len(surveys)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), squeeze=False
    )
    for ax, (_, row) in zip(axes.flat, surveys.iterrows()):
        clipped = row.geometry.intersection(aoi_geom)
        color = QL_COLORS.get(str(row.get("ql", "")), "#9a9a9a")
        if not clipped.is_empty:
            gpd.GeoSeries([clipped], crs="EPSG:4326").plot(
                ax=ax, facecolor=color, alpha=0.55, edgecolor=color
            )
        aoi.boundary.plot(ax=ax, color="black", linewidth=1.2, linestyle="--")
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect(1 / max(0.1, abs(np.cos(np.deg2rad((miny + maxy) / 2)))))
        ax.set_xticks([])
        ax.set_yticks([])
        frac = row["aoi_overlap_frac"]
        frac_txt = f"{frac:.1%}" if frac < 0.01 else f"{frac:.0%}"
        ax.set_title(
            f"{int(row['priority'])}. {row.get('workunit', '?')}\n"
            f"[{row.get('ql', '?')}] {str(row.get('collect_end', '?'))[:4]}, "
            f"{frac_txt}",
            fontsize=8.5,
        )
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    if out_fn is not None:
        fig.savefig(out_fn, dpi=130, bbox_inches="tight")
        print(f"Wrote per-collection panels to {out_fn}")
    return fig


def coverage_gaps(
    surveys_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    AOI area not covered by any of the given survey polygons.

    Permanent holes are an expected outcome (e.g. the Spring Mountains west
    of Las Vegas have no 3DEP lidar at all) and must be reported, not
    treated as a selection failure.

    Returns
    -------
    gpd.GeoDataFrame
        Zero or more gap polygons (EPSG:4326) with a `gap_frac` attribute
        giving each gap's share of the AOI area.
    """
    aoi = aoi_gdf.to_crs("EPSG:4326").union_all()
    if surveys_gdf.empty:
        gap = aoi
    else:
        gap = aoi.difference(surveys_gdf.to_crs("EPSG:4326").union_all())
    if gap.is_empty:
        return gpd.GeoDataFrame(
            {"gap_frac": []}, geometry=[], crs="EPSG:4326"
        )
    parts = list(gap.geoms) if hasattr(gap, "geoms") else [gap]
    return gpd.GeoDataFrame(
        {"gap_frac": [p.area / aoi.area for p in parts]},
        geometry=parts,
        crs="EPSG:4326",
    )


def survey(
    geometry: str,
    output: str = None,
    wesm_source: str = WESM_URL,
    ept_index: str = EPT_RESOURCES_URL,
    min_overlap: float = 0.0,
) -> None:
    """
    Report the lidar collections covering an AOI.

    Prints a per-collection table (quality level, acquisition dates,
    declared CRS/datum/geoid, EPT availability, AOI overlap) and the
    uncovered fraction of the AOI.

    Parameters
    ----------
    geometry
        Path to a vector dataset with the AOI polygon.
    output
        Optional directory: writes surveys.gpkg (records + geometry),
        surveys.yaml (records for pinning into processing metadata), and
        coverage_gaps.gpkg when gaps exist.
    wesm_source
        WESM GeoPackage URL or local path.
    ept_index
        EPT boundary index (GeoJSON) URL or local path.
    min_overlap
        Optional exploration filter: drop collections overlapping less than
        this fraction of the AOI. Default 0 — the inventory shows
        everything available; thinning is a selection decision, not a
        display rule. Dropped collections are listed when the filter is
        used.

    Returns
    -------
    None
    """
    from datetime import datetime, timezone

    import yaml

    aoi_gdf = gpd.read_file(geometry)
    # WESM is updated near-daily: read live (bbox-filtered) and record when
    wesm_accessed = datetime.now(timezone.utc).isoformat()
    print(f"Querying WESM for collections intersecting {geometry} ...")
    wesm = load_wesm(aoi_gdf, wesm_source)
    ept = load_ept_resources(ept_index)
    surveys = summarize_surveys(wesm, aoi_gdf, ept)
    if min_overlap > 0 and not surveys.empty:
        dropped = surveys[surveys["aoi_overlap_frac"] < min_overlap]
        if not dropped.empty:
            print(
                f"Dropped {len(dropped)} collections with < {min_overlap:.0%} "
                f"AOI overlap: {', '.join(dropped['workunit'].astype(str))}"
            )
        surveys = surveys[surveys["aoi_overlap_frac"] >= min_overlap].reset_index(
            drop=True
        )
    if surveys.empty:
        print("No collections intersect the AOI.")
        return
    surveys = assign_epochs(rank_collections(relative_metrics(surveys)))

    show = [
        c
        for c in [
            "priority",
            "epoch",
            "workunit",
            "ql",
            "collect_start",
            "collect_end",
            "horiz_crs",
            "vert_crs",
            "geoid",
            "lpc_category",
            "aoi_overlap_frac",
            "ept_coverage_frac",
            "anchor",
            "dt_years",
            "same_geoid",
            "same_horiz_crs",
        ]
        if c in surveys.columns
    ]
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(surveys[show].round(3).to_string(index=False))

    gaps = coverage_gaps(surveys, aoi_gdf)
    uncovered = float(gaps["gap_frac"].sum()) if not gaps.empty else 0.0
    print(
        f"\nAOI coverage: {1 - uncovered:.1%} covered by the listed collections; "
        f"{uncovered:.1%} has no coverage"
    )
    for _, row in surveys.iterrows():
        if row.get("ept_coverage_frac") is not None and row["ept_coverage_frac"] < 0.99:
            print(
                f"NOTE: {row.get('workunit', '?')} is only "
                f"{row['ept_coverage_frac']:.0%} covered by EPT resources "
                "(may require the local point-cloud path)"
            )

    if output is not None:
        outdir = Path(output)
        outdir.mkdir(parents=True, exist_ok=True)
        surveys_out = surveys.copy()
        surveys_out["ept_names"] = surveys_out["ept_names"].apply(
            lambda v: ",".join(v) if isinstance(v, list) else v
        )
        surveys_out.to_file(outdir / "surveys.gpkg", driver="GPKG")
        records = [
            {k: _yaml_safe(v) for k, v in rec.items()}
            for rec in surveys_out.drop(columns="geometry").to_dict(orient="records")
        ]
        with open(outdir / "surveys.yaml", "w") as f:
            yaml.dump(
                {
                    "aoi": str(geometry),
                    "wesm_source": str(wesm_source),
                    "wesm_accessed": wesm_accessed,
                    "collections": records,
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        if not gaps.empty:
            gaps.to_file(outdir / "coverage_gaps.gpkg", driver="GPKG")
        plot_coverage(
            surveys,
            aoi_gdf,
            gaps,
            out_fn=str(outdir / "coverage_map.png"),
            title=f"Lidar collection coverage: {Path(geometry).stem}",
        )
        plot_coverage_panels(
            surveys, aoi_gdf, out_fn=str(outdir / "coverage_map_panels.png")
        )
        print(f"Wrote survey records to {outdir}")


def _parse_index_url(index_url: str) -> tuple[str, str]:
    """
    Split a TNM "index.html?prefix=" browser link (the form WESM publishes
    as metadata_link) into the bucket endpoint and the object prefix
    (trailing slash included).
    """
    from urllib.parse import urlsplit

    parts = urlsplit(index_url)
    prefix = dict(
        kv.split("=", 1) for kv in parts.query.split("&") if "=" in kv
    ).get("prefix", "").rstrip("/") + "/"
    return f"{parts.scheme}://{parts.netloc}", prefix


def _s3_list_prefix(index_url: str, recursive: bool = True) -> list[dict]:
    """
    List objects under the S3 prefix of a TNM index link (only the
    prefix's own level when recursive=False). Follows ListObjectsV2
    pagination; returns [{"key", "size"}] with keys relative to the
    prefix.
    """
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    import requests

    endpoint, prefix = _parse_index_url(index_url)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    objects: list[dict] = []
    token = None
    while True:
        url = f"{endpoint}/?list-type=2&prefix={quote(prefix)}"
        if not recursive:
            url += "&delimiter=%2F"
        if token:
            url += f"&continuation-token={quote(token)}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for obj in root.findall(f"{ns}Contents"):
            key = obj.findtext(f"{ns}Key")
            size = obj.findtext(f"{ns}Size")
            assert key is not None and size is not None, "malformed S3 listing"
            objects.append({"key": key[len(prefix):], "size": int(size)})
        if (root.findtext(f"{ns}IsTruncated") or "").lower() != "true":
            return objects
        token = root.findtext(f"{ns}NextContinuationToken")


def fetch_reports(
    batch_dir: str,
    workunits: str | None = None,
    include: str = ".pdf",
) -> None:
    """
    Stage each project's vendor reports (QA/QC, survey/control, mapping)
    next to its products.

    3DEP vendor deliverables live under the workunit's staged-metadata S3
    prefix (the metadata_link recorded in survey_records of the project's
    processing metadata). Downstream accuracy assessment needs the vendor
    QC numbers on disk next to the rasters, not behind a browser link, so
    this fetches the report files into <project>/vendor_reports/
    (preserving subpaths), writes the full remote listing to
    remote_inventory.txt in the same directory (nothing is dropped
    silently), and records the staging in the processing metadata.
    Already-downloaded files (matching size) are skipped, so re-runs only
    fill gaps.

    Parameters
    ----------
    batch_dir
        rasterize-projects base directory (per-project subdirectories +
        batch_status.yaml).
    workunits
        Comma-separated project names, default: all projects recorded in
        batch_status.yaml. Projects without a survey record carrying a
        metadata_link (e.g. manually ingested vendor rasters) are reported
        and skipped.
    include
        Comma-separated filename extensions to download, by default
        ".pdf" — the vendor report documents. The staged-metadata prefix
        also holds bulky non-report payloads (ground-control monument
        photos, breakline geodatabases); widen deliberately, e.g.
        ".pdf,.jpg" to add the monument photos.

    Returns
    -------
    None
    """
    import sys
    import time

    import yaml

    import requests

    batch = Path(batch_dir)
    if workunits is None:
        status_fn = batch / "batch_status.yaml"
        if not status_fn.exists():
            raise FileNotFoundError(
                f"{status_fn} not found; pass workunits explicitly."
            )
        with open(status_fn) as f:
            names = list(yaml.safe_load(f)["projects"])
    else:
        names = [w.strip() for w in workunits.split(",")]
    exts = tuple(e.strip().lower() for e in include.split(",") if e.strip())

    for wu in names:
        pdir = batch / wu
        meta_hits = sorted(pdir.glob("*-processing_metadata.yaml")) or [
            pdir / "processing_metadata.yaml"
        ]
        meta_fn = meta_hits[0]
        if not meta_fn.exists():
            print(f"{wu}: no processing metadata, skipping")
            continue
        meta = yaml.safe_load(meta_fn.read_text()) or {}
        records = meta.get("survey_records") or []
        link = next(
            (
                rec["metadata_link"]
                for rec in records
                if rec.get("metadata_link")
                and rec.get("workunit", wu) == wu
            ),
            None,
        )
        if link is None:
            print(f"{wu}: no metadata_link in survey records, skipping")
            continue

        endpoint, wu_prefix = _parse_index_url(link)
        # workunit reports + the project level one up: the project-wide
        # USGS report and the vertical_accuracy/ checkpoint data (measured
        # per-point errors, shared across the project's workunits) live
        # there, not under the workunit
        proj_prefix = wu_prefix.rstrip("/").rsplit("/", 1)[0] + "/"
        proj_link = f"{endpoint}/index.html?prefix={proj_prefix}"
        va_link = f"{endpoint}/index.html?prefix={proj_prefix}vertical_accuracy/"
        # (prefix, dest subdir, extension filter — None = everything; the
        # vertical_accuracy tree is curated point data, take all of it)
        layers = [
            (link, "", exts),
            (proj_link, "project_level/", exts),
            (va_link, "project_level/vertical_accuracy/", None),
        ]
        outdir = pdir / "vendor_reports"
        outdir.mkdir(parents=True, exist_ok=True)
        fetched: list[str] = []
        failed: list[str] = []
        n_remote = 0
        with open(outdir / "remote_inventory.txt", "w") as inv:
            for layer_link, sub, layer_exts in layers:
                objects = _s3_list_prefix(layer_link, recursive=sub != "project_level/")
                n_remote += len(objects)
                _, layer_prefix = _parse_index_url(layer_link)
                for obj in objects:
                    inv.write(f"{obj['size']:>12} {sub}{obj['key']}\n")
                for obj in objects:
                    if layer_exts and not obj["key"].lower().endswith(layer_exts):
                        continue
                    dest = outdir / sub / obj["key"]
                    if dest.exists() and dest.stat().st_size == obj["size"]:
                        fetched.append(f"{sub}{obj['key']}")
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    # long multi-GB staging runs hit transient S3 resets;
                    # retry each file, and give up on a persistently
                    # failing object (recorded in `failed`, never fatal)
                    ok = False
                    for attempt in range(4):
                        try:
                            resp = requests.get(
                                f"{endpoint}/{layer_prefix}{obj['key']}",
                                timeout=600,
                                stream=True,
                            )
                            resp.raise_for_status()
                            with open(tmp, "wb") as f:
                                for chunk in resp.iter_content(1 << 20):
                                    f.write(chunk)
                            ok = True
                            break
                        except requests.exceptions.RequestException as e:
                            if attempt == 3:
                                print(
                                    f"WARNING: giving up on "
                                    f"{sub}{obj['key']} after 4 attempts "
                                    f"({e})",
                                    file=sys.stderr,
                                )
                            else:
                                time.sleep(2 ** (attempt + 1))
                    if not ok:
                        failed.append(f"{sub}{obj['key']}")
                        continue
                    tmp.rename(dest)
                    fetched.append(f"{sub}{obj['key']}")
        meta["vendor_reports"] = {
            "source_prefix": link,
            "project_prefix": proj_link,
            "directory": outdir.name,
            "include": list(exts),
            "files": fetched,
            "failed": failed,
            "remote_objects_total": n_remote,
        }
        with open(meta_fn, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
        print(
            f"{wu}: staged {len(fetched)}/{n_remote} remote objects"
            + (f" ({len(failed)} FAILED)" if failed else "")
            + " "
            f"({', '.join(exts)}) -> {outdir}"
        )
