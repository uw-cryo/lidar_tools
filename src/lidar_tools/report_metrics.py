"""
Standardized metric extraction from staged vendor/QC reports.

3DEP vendor deliverables bury the numbers that matter for processing and
QA — acquisition period, tested vertical/horizontal accuracy, point
density — in per-vendor PDF report formats and FGDC XML templates. This
module reduces a project's staged vendor_reports/ directory (see
survey.fetch_reports) to one machine-readable record per project so
cross-project comparison and downstream QA don't require reading PDFs.

Two extraction layers, most-structured first:

1. FGDC CSDGM XML (vendor_provided_xml/*.xml): acquisition begin/end
   dates and the ASPRS compliance statements. Vendors sometimes ship the
   template with unfilled blanks ("___") — that is surfaced, not hidden.
2. PDF text (pdftotext -layout): vendor-varying report tables matched by
   labeled patterns (NVA/VVA tables, aggregate nominal pulse density,
   RMSEr). Every extracted number carries the source file and matched
   line in an `evidence` list — heuristic extractions must be auditable.

WESM survey records (already pinned in the project's processing
metadata) provide the reference acquisition window and QL; disagreement
between WESM and the vendor XML is flagged, not resolved (observed at
Casa Grande: a vendor XML dated a full year after the WESM window).

Note ground-return density is generally NOT reported by vendors (reports
carry first-return / aggregate pulse density only); a ground-density
estimate has to come from the point cloud or the DTM_no_fill cell
occupancy.
"""

import re
import shutil
import subprocess
from pathlib import Path

import yaml

#: PDFs larger than this are still parsed (they are usually THE vendor
#: report); the limit only guards against pathological inputs.
MAX_PDF_BYTES = 500 * 1024 * 1024


def _pdf_text(fn: Path) -> str:
    """Extract text with layout preserved; empty string on failure."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext not found: report-metrics needs poppler "
            "(shipped in the pixi environment)."
        )
    if fn.stat().st_size > MAX_PDF_BYTES:
        return ""
    proc = subprocess.run(
        ["pdftotext", "-layout", str(fn), "-"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return proc.stdout if proc.returncode == 0 else ""


def _iso(d: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD (FGDC caldate form); passthrough otherwise."""
    d = d.strip()
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if re.fullmatch(r"\d{8}", d) else d


def parse_fgdc_xml(text: str) -> dict:
    """
    Pull acquisition dates and compliance statements from an FGDC CSDGM
    metadata document (vendor-provided workunit-level XML).
    """
    out: dict = {}

    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
        return " ".join(m.group(1).split()) if m else None

    beg, end = tag("begdate"), tag("enddate")
    if beg:
        out["acquisition_start"] = _iso(beg)
    if end:
        out["acquisition_end"] = _iso(end)
    for key, name in [
        ("vertical_accuracy_statement", "vertaccr"),
        ("horizontal_accuracy_statement", "horizpar"),
    ]:
        val = tag(name)
        if val:
            out[key] = val
    # vendors sometimes deliver the boilerplate with the numbers left blank
    out["unfilled_template_fields"] = any(
        "___" in v for v in out.values() if isinstance(v, str)
    )
    return out


def parse_usgs_validation_text(text: str) -> dict:
    """
    Parse the standardized USGS-NGTOC Data Validation Report (post-2020
    workunits): QL, base-spec version, P-method, geoid, and the verdict.
    """
    out = {}
    patterns = {
        "quality_level": r"Quality Level:\s*(\S+)",
        "lidar_base_spec": r"Lidar Base Spec:\s*([\d.]+)",
        "p_method": r"P-Method:\s*([^\n]+?)\s*$",
        "geoid": r"Geoid Model:\s*([^\n]+?)\s*$",
        "horizontal_epsg": r"Horizontal EPSG Code:\s*(\d+)",
        "vertical_epsg": r"Vertical EPSG Code:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.M)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"delivered data is\s+([A-Z\s]+?)\s+3D Elevation", text)
    if m:
        out["verdict"] = " ".join(m.group(1).split())
    return out


#: labeled numeric patterns tried against every PDF's text; group names
#: become metric keys. Values in meters / points-per-square-meter.
_METRIC_PATTERNS: list[tuple[str, str]] = [
    # VeriDaaS/Sanborn Geiger-style vertical accuracy table rows
    ("nva_rmsez_m/nva_95pct_m",
     r"NVA of (?:Point Cloud|Bare Earth)\s+(?P<n_nva>\d+)\s+"
     r"(?P<nva_rmsez_m>[\d.]+)\s+(?P<nva_95pct_m>[\d.]+)"),
    ("vva_95th_m",
     r"VVA of (?:Bare Earth|DEM)\s+(?P<n_vva>\d+)\s+[\d.]+\s+"
     r"(?P<vva_95th_m>[\d.]+)"),
    # target/measured/count table style (PimaCo reports)
    ("nva_95pct_m",
     r"^\s*NVA\s+[\d.]+\s*m\s+(?P<nva_95pct_m>[\d.]+)\s+(?P<n_nva>\d+)\s*$"),
    ("vva_95th_m",
     r"^\s*VVA\s+[\d.]+\s*m\s+(?P<vva_95th_m>[\d.]+)\s+(?P<n_vva>\d+)\s*$"),
    # Merrick MARS QC style (value pairs are metric/US-foot)
    ("nva_rmsez_m",
     r"Non-vegetated Vertical Accuracy \(NVA\) RMSE\(z\)\s+"
     r"(?P<nva_rmsez_m>[\d.]+)/"),
    ("nva_95pct_m",
     r"NVA\) at the 95% Confidence Level \+/-\s+(?P<nva_95pct_m>[\d.]+)/"),
    ("vva_95th_m",
     r"VVA\) at the 95(?:th)? [Pp]ercentile\s+(?P<vva_95th_m>[\d.]+)/"),
    # densities
    ("anpd_ppsm",
     r"Aggregate Nominal Pulse Density \(pls/(?:m..?|m2)\)\s+"
     r"(?P<anpd_ppsm>[\d.]+)"),
    ("anps_m",
     r"Aggregate Nominal Pulse Spacing \(m\)\s+(?P<anps_m>[\d.]+)"),
    ("first_return_density_ppsm",  # MARS QC C-4 aggregate row
     r"^\s*Aggregate\s+[\d,]+\s+[\d,]+\s+"
     r"(?P<first_return_density_ppsm>[\d.]+)/[\d.]+"),
    ("measured_density_ppsm",  # "Density    9.66 pts / m2" (single value)
     r"^\s*Density\s+(?P<measured_density_ppsm>[\d.]+)\s*pts?\s*/\s*m2?\s*$"),
    ("measured_density_ppsm",  # required / planned / achieved 3-column row
     r"Density\s+[\d.]+\s*pts?\s*/\s*m2\s+[\d.]+\s*pts?\s*/\s*m2\s+"
     r"(?P<measured_density_ppsm>[\d.]+)\s*pts?\s*/\s*m2"),
    # swath-to-swath consistency (precision, not absolute accuracy)
    ("swath_relative_dz_mean_m",
     r"relative vertical accuracy[\s\S]{0,120}?"
     r"\(\s*(?P<swath_relative_dz_mean_m>[\d.]+)\s*meters?\)"),
    # horizontal accuracy
    ("horizontal_acc95_m",
     r"compiled to meet (?P<horizontal_acc95_m>[\d.]+)\s*m(?:eter)? "
     r"horizontal accuracy"),
]


def parse_report_text(text: str, source: str) -> tuple[dict, list[dict]]:
    """
    Match the labeled metric patterns against one document's text.

    Returns (metrics, evidence): metric key -> float, and one evidence
    entry per match recording the source file and the matched line so a
    misfire is visible in the output rather than laundered into a number.
    """
    metrics: dict = {}
    evidence = []
    for _, pat in _METRIC_PATTERNS:
        for m in re.finditer(pat, text, re.M):
            line = " ".join(
                text[text.rfind("\n", 0, m.start()) + 1:
                     max(text.find("\n", m.end()), m.end())].split()
            )
            for key, val in m.groupdict().items():
                if val is None:
                    continue
                num = float(val)
                # first match wins: reports repeat their summary tables
                # (duplicated appendices), and the primary table comes first
                if key not in metrics:
                    metrics[key] = num
                    evidence.append({"metric": key, "file": source, "line": line})
    # RMSEr / ACCr are laid out as a label line followed by value lines
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        label = raw.strip()
        if label in ("RMSEr", "ACCr"):
            key = "horizontal_rmser_m" if label == "RMSEr" else "horizontal_acc95_m"
            for nxt in lines[i + 1: i + 4]:
                vm = re.search(r"([\d.]+)\s*m\s*$", nxt.strip())
                if vm and key not in metrics:
                    metrics[key] = float(vm.group(1))
                    evidence.append(
                        {"metric": key, "file": source,
                         "line": f"{label} ... {nxt.strip()}"}
                    )
                    break
    return metrics, evidence


def _wesm_reference(pdir: Path, workunit: str) -> dict:
    """Acquisition window / QL / spec from the pinned WESM survey record."""
    hits = sorted(pdir.glob("*-processing_metadata.yaml")) or [
        pdir / "processing_metadata.yaml"
    ]
    if not hits or not hits[0].exists():
        return {}
    meta = yaml.safe_load(hits[0].read_text()) or {}
    for rec in meta.get("survey_records") or []:
        if rec.get("workunit", workunit) == workunit:
            return {
                k: rec.get(k)
                for k in ("collect_start", "collect_end", "ql", "spec")
                if rec.get(k) is not None
            }
    return {}


def extract_project_metrics(pdir: Path, workunit: str) -> dict:
    """
    Reduce one project's vendor_reports/ directory to a standardized
    metric record (see module docstring for the layers).
    """
    vdir = pdir / "vendor_reports"
    record: dict = {"workunit": workunit}
    if not vdir.is_dir():
        record["note"] = "no vendor_reports directory (run fetch-reports)"
        return record

    fgdc = {}
    xmls = sorted(vdir.rglob("vendor_provided_xml/*.xml"))
    # the point-cloud document is the authoritative one for acquisition
    xmls.sort(key=lambda p: ("ClassifiedPointCloud" not in p.name
                             and "Classified" not in p.name))
    if xmls:
        fgdc = parse_fgdc_xml(xmls[0].read_text(errors="replace"))
        fgdc["source"] = str(xmls[0].relative_to(vdir))
    record["fgdc"] = fgdc or None

    metrics: dict = {}
    evidence: list[dict] = []
    usgs_validation = None
    pdfs = sorted(vdir.rglob("*.pdf"))
    for fn in pdfs:
        text = _pdf_text(fn)
        if not text:
            continue
        rel = str(fn.relative_to(vdir))
        if "Data Validation Report" in text[:400]:
            usgs_validation = parse_usgs_validation_text(text)
            usgs_validation["source"] = rel
            continue
        got, ev = parse_report_text(text, rel)
        for key, val in got.items():
            metrics.setdefault(key, val)
        evidence.extend(ev)
    record["usgs_validation"] = usgs_validation
    record["metrics"] = metrics
    record["pdfs_parsed"] = len(pdfs)

    wesm = _wesm_reference(pdir, workunit)
    record["wesm"] = wesm or None
    # flag vendor-XML acquisition dates that fall outside the WESM window
    # (± 45 d slack for WESM day-precision and mobilization edges)
    if fgdc.get("acquisition_start") and wesm.get("collect_start"):
        import datetime as dt

        slack = dt.timedelta(days=45)
        f0 = dt.datetime.fromisoformat(fgdc["acquisition_start"])
        f1 = dt.datetime.fromisoformat(fgdc["acquisition_end"])
        w0 = dt.datetime.fromisoformat(str(wesm["collect_start"]))
        w1 = dt.datetime.fromisoformat(str(wesm["collect_end"]))
        record["acquisition_dates_consistent"] = bool(
            w0 - slack <= f0 and f1 <= w1 + slack
        )

    if metrics.get("first_return_density_ppsm") or metrics.get("anpd_ppsm"):
        dens = metrics.get("first_return_density_ppsm") or metrics["anpd_ppsm"]
        # mean point spacing bounds the finest grid posting worth producing
        record["derived"] = {
            "first_return_mean_spacing_m": round(dens ** -0.5, 3)
        }
    record["evidence"] = evidence
    return record


# comparison-table rows: (label, function of record -> value)
_TABLE_ROWS = [
    ("acquisition (vendor XML)", lambda r: _fmt_span(r.get("fgdc") or {})),
    ("acquisition (WESM)", lambda r: _fmt_span_wesm(r.get("wesm") or {})),
    ("dates consistent", lambda r: r.get("acquisition_dates_consistent")),
    ("QL (WESM)", lambda r: (r.get("wesm") or {}).get("ql")),
    ("USGS validation", lambda r: (r.get("usgs_validation") or {}).get("verdict")),
    ("geoid", lambda r: (r.get("usgs_validation") or {}).get("geoid")),
    ("NVA RMSEz [m]", lambda r: r["metrics"].get("nva_rmsez_m")),
    ("NVA 95% [m]", lambda r: r["metrics"].get("nva_95pct_m")),
    ("VVA 95th [m]", lambda r: r["metrics"].get("vva_95th_m")),
    ("checkpoints NVA/VVA", lambda r: _fmt_ckpts(r["metrics"])),
    ("swath relative dz [m]",
     lambda r: r["metrics"].get("swath_relative_dz_mean_m")),
    ("horiz RMSEr [m]", lambda r: r["metrics"].get("horizontal_rmser_m")),
    ("horiz 95% [m]", lambda r: r["metrics"].get("horizontal_acc95_m")),
    ("ANPD [pls/m2]", lambda r: r["metrics"].get("anpd_ppsm")),
    ("first-return density [p/m2]",
     lambda r: r["metrics"].get("first_return_density_ppsm")
     or r["metrics"].get("measured_density_ppsm")),
    ("mean spacing [m]",
     lambda r: (r.get("derived") or {}).get("first_return_mean_spacing_m")),
]


def _fmt_span(d):
    a, b = d.get("acquisition_start"), d.get("acquisition_end")
    return f"{a} - {b}" if a and b else None


def _fmt_span_wesm(d):
    a, b = d.get("collect_start"), d.get("collect_end")
    return f"{str(a)[:10]} - {str(b)[:10]}" if a and b else None


def _fmt_ckpts(m):
    a, b = m.get("n_nva"), m.get("n_vva")
    return f"{a:g}/{b:g}" if a and b else (f"{a:g}/-" if a else None)


def print_comparison(records: list[dict]) -> None:
    """Print the cross-project metric table (rows = metrics)."""
    names = [r["workunit"] for r in records]
    width = max(len(label) for label, _ in _TABLE_ROWS) + 2
    cwidth = max(max((len(n) for n in names), default=10) + 2, 25)
    print("".ljust(width) + "".join(n.ljust(cwidth) for n in names))
    for label, fn in _TABLE_ROWS:
        vals = []
        for r in records:
            try:
                v = fn(r)
            except (KeyError, TypeError):
                v = None
            vals.append("-" if v is None else str(v))
        if any(v != "-" for v in vals):
            print(label.ljust(width) + "".join(v.ljust(cwidth) for v in vals))


def report_metrics(
    batch_dir: str,
    workunits: str | None = None,
) -> None:
    """
    Extract standardized metrics from each project's staged vendor
    reports (see fetch-reports) and print a cross-project comparison.

    Writes <prefix>-report_metrics.yaml next to each project's products:
    acquisition period (vendor FGDC XML vs the pinned WESM record, with a
    consistency flag), USGS validation summary (QL, base spec, P-method,
    geoid, verdict), tested vertical accuracy (NVA RMSEz / NVA 95% / VVA
    95th, checkpoint counts), horizontal accuracy (RMSEr / 95%), pulse
    density (ANPD / first-return density) and the derived mean point
    spacing (a floor for useful grid posting). Every number extracted
    from a PDF carries the source file and matched line in `evidence`.

    Parameters
    ----------
    batch_dir
        rasterize-projects base directory with staged vendor_reports/
        subdirectories.
    workunits
        Comma-separated project names, default: all projects in
        batch_status.yaml.
    """
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

    records = []
    for wu in names:
        pdir = batch / wu
        print(f"Extracting report metrics: {wu}")
        rec = extract_project_metrics(pdir, wu)
        records.append(rec)
        hits = sorted(pdir.glob("*-processing_metadata.yaml"))
        prefix = (
            hits[0].name.rsplit("-", 1)[0] if hits else wu
        )
        out_fn = pdir / f"{prefix}-report_metrics.yaml"
        with open(out_fn, "w") as f:
            yaml.dump(rec, f, default_flow_style=False, sort_keys=False)
    print()
    print_comparison(records)
