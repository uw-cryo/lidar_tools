"""
Archive-wide conformance checks: does the whole 3DEP catalog satisfy the
assumptions this code encodes?

Every site we process exercises a handful of workunits, so an assumption
that holds in Arizona and Nevada can still be wrong for 14% of the
archive. WESM publishes its full attribute table (~3.3k workunits) as a
few-MB CSV, so the assumptions can be tested against every survey offline,
before a run fails somewhere new. This is the same method that produced the
EPT name resolver's tiers (measured over 2,277 EPT x 3,260 WESM names).

The checks are name/attribute-level and need no geometry, so they run from
WESM.csv alone. Region-specific resolution (which geoid grid applies where)
is reported as the model's available regions rather than resolved per
workunit; a geometry-aware pass would need the 3.4 GB GeoPackage.

Used by scripts/conformance_audit.py; kept importable so the checks are
unit-testable and can later back per-AOI preflight warnings.
"""

from dataclasses import dataclass, field, asdict

from lidar_tools import geodesy

#: p_method values that are lidar. Anything else (IFSAR radar, bathymetric
#: sonar-adjacent products) reaches our pipeline as a point cloud without
#: ASPRS return/classification semantics.
LIDAR_METHODS = {
    "linear-mode lidar",
    "geiger-mode lidar",
    "single photon lidar",
    "flash lidar",
    "topobathymetric lidar",
    "bathymetric lidar",
}

#: methods our gridding assumptions do not cover at all
NON_LIDAR_METHODS = {"ifsar"}


@dataclass
class Finding:
    """One conformance violation class, with its share of the archive."""

    check: str
    severity: str  # "blocker" | "degraded" | "unsupported" | "info"
    count: int
    total: int
    detail: str
    examples: list = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.count / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["fraction"] = round(self.fraction, 4)
        return d


def _counts(series) -> list:
    """(value, count) pairs for a pandas Series, nulls folded to ''."""
    return list(series.fillna("").astype(str).value_counts().items())


def _resolves_as_esri(code: str) -> bool:
    """True when `code` is a real ESRI authority code. The definitive test —
    ask PROJ — not a numeric-range heuristic, so an arbitrary invalid number
    is never misreported as a fixable ESRI code."""
    from pyproj import CRS

    try:
        CRS.from_authority("ESRI", code)
    except Exception:
        return False
    return True


def check_horizontal_crs(series) -> list:
    """
    Can `geodesy.geographic_base_epsg` derive a base datum from every
    declared `horiz_crs`?

    Failures split into two very different cases: ESRI authority codes
    stored in an EPSG-shaped field (resolvable — we simply never try the
    ESRI authority), and datums genuinely outside the NAD83 family
    (Mariana/Pacific plates), where a hard failure is correct.
    """
    total = int(series.shape[0])
    esri, unsupported, invalid = [], [], []
    for value, count in _counts(series):
        try:
            geodesy.geographic_base_epsg(value)
            continue
        except Exception as exc:
            msg = str(exc)
        if value.isdigit() and _resolves_as_esri(value):
            # ESRI authority codes live in this EPSG-shaped field; resolve
            # against the ESRI authority rather than trusting a numeric
            # range, so an arbitrary bad number cannot masquerade as a
            # fixable ESRI code
            esri.append((value, count))
        elif "has base" in msg or "non-NAD83" in msg.lower():
            unsupported.append((value, count))
        else:
            invalid.append((value, count))

    findings = []
    if esri:
        n = sum(c for _, c in esri)
        findings.append(
            Finding(
                check="horiz_crs.esri_authority_code",
                severity="blocker",
                count=n,
                total=total,
                detail=(
                    "declared horiz_crs is an ESRI authority code in an "
                    "EPSG-shaped field; these resolve as ESRI:<code> and are "
                    "NAD83-family, so the hard failure is ours, not the data's"
                ),
                examples=[v for v, _ in esri[:8]],
            )
        )
    if unsupported:
        findings.append(
            Finding(
                check="horiz_crs.non_nad83_datum",
                severity="unsupported",
                count=sum(c for _, c in unsupported),
                total=total,
                detail=(
                    "declared datum is outside the NAD83 family (e.g. "
                    "NAD83(MA11) on the Mariana plate); hard-failing is "
                    "correct, but the message should name the plate"
                ),
                examples=[v for v, _ in unsupported[:8]],
            )
        )
    if invalid:
        findings.append(
            Finding(
                check="horiz_crs.unparseable",
                severity="blocker",
                count=sum(c for _, c in invalid),
                total=total,
                detail="declared horiz_crs could not be parsed at all",
                examples=[v for v, _ in invalid[:8]],
            )
        )
    return findings


def check_geoid(series) -> list:
    """
    Is every declared geoid model recognized by the enforcement table?
    An unrecognized model hard-fails under the default
    `--geoid-override declared`; an absent/unknown declaration silently
    disables enforcement, which is the failure mode that work removed.
    """
    total = int(series.shape[0])
    unknown_decl, unrecognized = [], []
    for value, count in _counts(series):
        # mirror resolve_declared_geoid's normalization exactly, so this
        # reports what enforcement will actually do (geoid_grid_hint uses a
        # different, narrower table and would flag the bare '12B' form that
        # enforcement resolves fine)
        key = str(value).upper().replace(" ", "")
        if key in geodesy._GEOID_NAME_UNDECLARED:
            unknown_decl.append((value or "(null)", count))
            continue
        key = geodesy._GEOID_NAME_ALIASES.get(key, key)
        # GEOID12A is remapped to the GEOID12B grids outside PR/USVI
        if key == "GEOID12A":
            continue
        if key not in geodesy.GEOID_GRID_FILES:
            unrecognized.append((value, count))

    findings = []
    if unrecognized:
        findings.append(
            Finding(
                check="geoid.unrecognized_model",
                severity="blocker",
                count=sum(c for _, c in unrecognized),
                total=total,
                detail=(
                    "declared geoid model maps to no PROJ grid; the run "
                    "hard-fails under the default --geoid-override declared"
                ),
                examples=[v for v, _ in unrecognized[:8]],
            )
        )
    if unknown_decl:
        findings.append(
            Finding(
                check="geoid.undeclared",
                severity="degraded",
                count=sum(c for _, c in unknown_decl),
                total=total,
                detail=(
                    "no declared geoid, so enforcement silently switches off "
                    "and best-available substitution returns without a warning"
                ),
                examples=[v for v, _ in unknown_decl[:8]],
            )
        )
    return findings


def check_acquisition_method(series) -> list:
    """Flag surveys whose acquisition method our gridding does not model."""
    total = int(series.shape[0])
    non_lidar, unknown = [], []
    for value, count in _counts(series):
        key = value.strip().lower()
        if key in NON_LIDAR_METHODS:
            non_lidar.append((value, count))
        elif key and key not in LIDAR_METHODS:
            unknown.append((value, count))

    findings = []
    if non_lidar:
        findings.append(
            Finding(
                check="p_method.not_lidar",
                severity="unsupported",
                count=sum(c for _, c in non_lidar),
                total=total,
                detail=(
                    "acquisition method is not lidar (IFSAR radar), so ASPRS "
                    "return/classification filtering is meaningless; the "
                    "pipeline should refuse these rather than grid them"
                ),
                examples=[v for v, _ in non_lidar[:8]],
            )
        )
    if unknown:
        findings.append(
            Finding(
                check="p_method.unrecognized",
                severity="info",
                count=sum(c for _, c in unknown),
                total=total,
                detail="acquisition method not in the known-method list",
                examples=[v for v, _ in unknown[:8]],
            )
        )
    return findings


def check_metadata_completeness(df) -> list:
    """Fields our commands depend on, and what breaks when they are absent."""
    total = int(df.shape[0])
    spec = [
        (
            "lpc_link",
            "staging/prepare cannot reconcile TESM against the "
            "links file, so tile truth is unavailable",
            "degraded",
        ),
        (
            "metadata_link",
            "fetch-reports has no S3 prefix to stage vendor "
            "reports from and skips the project",
            "degraded",
        ),
        (
            "collect_start",
            "preview footers and epoch selection have no acquisition date",
            "degraded",
        ),
        ("collect_end", "'latest' selection cannot date-order this survey", "degraded"),
        ("ql", "quality level unknown, so merge priority ordering is blind", "info"),
    ]
    findings = []
    for column, detail, severity in spec:
        if column not in df.columns:
            continue
        missing = int(df[column].isna().sum())
        if missing:
            findings.append(
                Finding(
                    check=f"metadata.missing_{column}",
                    severity=severity,
                    count=missing,
                    total=total,
                    detail=detail,
                    examples=sorted(df.loc[df[column].isna(), "workunit"].astype(str))[
                        :8
                    ]
                    if "workunit" in df.columns
                    else [],
                )
            )
    return findings


def audit_wesm(df) -> dict:
    """
    Run every check against a WESM attribute table.

    Parameters
    ----------
    df
        WESM rows (``pandas.DataFrame``), e.g. from WESM.csv.

    Returns
    -------
    dict
        ``{"n_workunits", "findings": [...]}`` sorted most-severe first.
    """
    findings = []
    if "horiz_crs" in df.columns:
        findings += check_horizontal_crs(df["horiz_crs"])
    if "geoid" in df.columns:
        findings += check_geoid(df["geoid"])
    if "p_method" in df.columns:
        findings += check_acquisition_method(df["p_method"])
    findings += check_metadata_completeness(df)

    rank = {"blocker": 0, "unsupported": 1, "degraded": 2, "info": 3}
    findings.sort(key=lambda f: (rank.get(f.severity, 9), -f.count))
    return {"n_workunits": int(df.shape[0]), "findings": findings}


def format_report(result: dict) -> str:
    """Human-readable conformance report."""
    n = result["n_workunits"]
    lines = [f"3DEP archive conformance: {n} workunits", ""]
    if not result["findings"]:
        lines.append("  no violations — every assumption holds archive-wide")
        return "\n".join(lines)
    for f in result["findings"]:
        lines.append(f"  [{f.severity:11s}] {f.check}: {f.count} ({f.fraction:.1%})")
        lines.append(f"      {f.detail}")
        if f.examples:
            lines.append(f"      e.g. {', '.join(map(str, f.examples[:6]))}")
    return "\n".join(lines)
