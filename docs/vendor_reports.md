# Vendor reports: staging and standardized metric extraction

Two commands turn the per-vendor report sprawl of a 3DEP delivery into
machine-readable per-project records that sit next to the products:

```bash
lidar-tools fetch-reports <batch_dir>     # stage the documents
lidar-tools report-metrics <batch_dir>    # extract + compare
```

Both default to the projects in `batch_status.yaml`; pass
`--workunits a,b,c` to override.

## What fetch-reports stages

For each project, using the `metadata_link` already pinned in the
project's processing metadata (WESM survey record):

- **Workunit prefix** (`.pdf,.xml` by default, `--include` to widen):
  vendor mapping/project reports, USGS data-validation report, QC
  report suites, POS/GNSS trajectory reports, plus the small FGDC
  metadata XMLs (`vendor_provided_xml/`) that report-metrics parses
  for acquisition dates and compliance statements.
- **Project level** (one directory up, shared across workunits): the
  standardized `USGS_<project>_Project_Report.pdf` — the document the
  per-workunit validation reports defer to for measured accuracy.
- **`vertical_accuracy/` tree** (whole, minus monument photos): the
  USGS standardized checkpoint GPKG (`USGS_standard_*_VA.gpkg`,
  per-point surveyed coordinates with NVA/VVA typing — directly usable
  as a control source) and the contractor-provided control points and
  accuracy reports. Checkpoint monument photos are excluded (one
  project carried 3,227 of them, 25.3 GB against 0.06 GB of data);
  they remain listed in the inventory.

Everything lands in `<project>/vendor_reports/`, the complete remote
listing goes to `remote_inventory.txt` (nothing is dropped silently),
the staging is recorded in the processing metadata, and re-runs skip
size-matched files. Transient S3 resets are retried per file;
persistently failing objects are recorded under `failed` and never
kill the run.

## What report-metrics extracts

Three layers, most-structured first, into
`<prefix>-report_metrics.yaml` per project plus a printed
cross-project table:

1. **Vendor FGDC XML** (workunit-level `vendor_provided_xml/`):
   acquisition begin/end dates and ASPRS compliance statements.
   Unfilled template blanks (`___`) are surfaced. Dates are
   cross-checked against the pinned WESM collection window and flagged
   when inconsistent — Casa Grande's PimaCo workunits ship XMLs dated
   a full year outside their WESM windows, with identical copy-pasted
   dates across two workunits.
2. **USGS project report** (both the LBS-2020 and LBS-2022 template
   generations): contractor, P-method, base spec, authoritative
   collection dates, and the **tested** NVA/VVA (point cloud and DEM).
   This is the one cross-vendor-comparable accuracy source.
3. **Vendor/QC PDF text** (pdftotext, poppler ships in the pixi env):
   per-vendor-format labeled patterns for NVA/VVA tables, aggregate
   nominal pulse density and spacing, first-return density,
   swath-to-swath relative vertical accuracy, and horizontal
   RMSEr/95%. Every extracted number carries its source file and
   matched line in an `evidence` list so a pattern misfire is visible
   rather than laundered into a number.

USGS-tested and vendor-reported accuracy are kept side by side and
never merged: at Casa Grande the vendor reported NVA 0.166 m where
USGS tested 0.107 m for the same workunit. The disagreement is
information — downstream accuracy reporting (e.g. groundcontrol) should
carry both.

Vendor formats vary (three distinct report families across the four
Casa Grande workunits, all post-2020); new formats are added as
patterns in `_METRIC_PATTERNS` with a pinned fixture per family in
`tests/test_report_metrics.py`.

## What is deliberately absent

**Ground-return density is not in any vendor report** (first-return /
pulse density only). Estimate it from the point cloud, or from the
products: the fraction of valid `DTM_no_fill` cells within the project
footprint is the ground-cell occupancy at the product posting. At Casa
Grande this saturates (98–100 % at 1 m and 0.5 m) — occupancy at a
candidate posting, not a density point estimate, is the meaningful
answer once penetration is good.
