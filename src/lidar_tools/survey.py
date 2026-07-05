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

    def midpoint(row):
        try:
            t0 = pd.Timestamp(row.get("collect_start"))
            t1 = pd.Timestamp(row.get("collect_end"))
            return t0 + (t1 - t0) / 2
        except (TypeError, ValueError):
            return pd.NaT

    mid_a = midpoint(a)
    out["dt_years"] = [
        round((midpoint(r) - mid_a).days / 365.25, 2)
        if pd.notna(midpoint(r)) and pd.notna(mid_a)
        else None
        for _, r in out.iterrows()
    ]
    if "geoid" in out.columns:
        out["same_geoid"] = out["geoid"] == a.get("geoid")
    if "horiz_crs" in out.columns:
        out["same_horiz_crs"] = out["horiz_crs"] == a.get("horiz_crs")
    return out


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
    aoi = aoi_gdf.to_crs("EPSG:4326")
    aoi_geom = aoi.union_all()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    # draw larger footprints first so smaller ones stay clickable/visible
    order = surveys.geometry.area.sort_values(ascending=False).index
    legend_lines = []
    for n, idx in enumerate(order, start=1):
        row = surveys.loc[idx]
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
        legend_lines.append(
            f"{n}. {row.get('workunit', '?')}  [{row.get('ql', '?')}] "
            f"{str(row.get('collect_start', '?'))[:10]}..{str(row.get('collect_end', '?'))[:10]} "
            f"{row.get('geoid', '?')}, {frac_txt} of AOI"
            f"{dt_txt}{ept_txt}"
            f"{'  << ANCHOR' if is_anchor else ''}"
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
    fig.text(
        0.01,
        0.01,
        "\n".join(legend_lines),
        fontsize=8.5,
        family="monospace",
        va="bottom",
    )
    fig.subplots_adjust(bottom=0.06 + 0.025 * len(legend_lines))
    if out_fn is not None:
        fig.savefig(out_fn, dpi=130, bbox_inches="tight")
        print(f"Wrote coverage map to {out_fn}")
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
        Drop collections overlapping less than this fraction of the AOI.

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
        surveys = surveys[surveys["aoi_overlap_frac"] >= min_overlap].reset_index(
            drop=True
        )
    if surveys.empty:
        print("No collections intersect the AOI.")
        return
    surveys = relative_metrics(surveys)

    show = [
        c
        for c in [
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
        def _yaml_safe(value):
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            if isinstance(value, np.generic):
                return value.item()
            if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
                return None
            return value

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
        print(f"Wrote survey records to {outdir}")
