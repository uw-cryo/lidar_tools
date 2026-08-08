# Front-end / staging implementation plan

2026-07-18. Branch `frontend-staging` (from `f3-single-read-multiproduct` @ 52ad174).
Companion working artifacts — session `README.md` (§5–§8 designs + decisions),
`threedep_source_map.md`, `laz_vs_ept_study.md`, `ept_wesm_mapping.csv`
(archive-wide name resolution, tier-labeled) — live UNTRACKED in
`sandbox/20260718_lv_nad83_regen/` on working machines.

Goal: everything that happens BEFORE `rasterize-projects` — discovery, naming
resolution, staging, probes, project selection — becomes explicit, inspectable,
and offline-consumable, so batch runs never silently miss or misread a source.

Phases are ordered by risk-adjusted value; P1 is implemented on this branch,
P2 partially (skeleton + tests), P3–P4 are designs with implementation notes.

---

## P1 — Easy wins (implemented here)

### P1a. EPT-resource name resolver (fixes the silent 0-reader join)

Problem (measured): exact `==` join resolves only 45.4% of the 2,277-resource
EPT archive against canonical WESM workunit names; 4-tier normalization reaches
98.2% with zero ambiguity (`ept_wesm_mapping.csv`; the review's F1 fix — hyphen
folding instead of a one-sided `ARRA-` strip — added the 46 ARRA pairs over the
originally reported 96.2%). Live blocker:
`NV_LasVegas_QL2_2016` → EPT `USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018`.

Design:
- `survey.normalize_ept_name(name, tier)` + `survey.resolve_ept_resource(
  workunit, ept_gdf)` → `{"ept_name", "tier", "candidates"}`.
  Tiers: 1 exact → 2 casefold → 3 strip `^USGS_LPC_` / `_LAS_\d{4}$` →
  4 hyphen↔underscore folding (subsumes `ARRA-`/`ARRA_` drift; the prefix is
  never stripped — review F1). Multiple same-tier candidates
  (re-released builds): larger `count` wins (an exact-name candidate always
  short-circuits at tier 1 — review F7).
  Unresolvable → `LookupError` with the spatially-nearby names in the message
  (hard-fail; never silent 0 readers). Spatial-gate tier for the ~3.8% rename
  tail is a P2 hook (`resolve_ept_resource(..., wesm_geom=...)`) — omitted from
  the P1 join deliberately: it needs the WESM polygon, which `rasterize` has
  only when the WESM fetch succeeded, and a wrong spatial pick is worse than a
  loud failure (nested collections: Southern_4's polygon overlaps ClarkCo_2's
  and QL2_2016's EPT boundaries).
- Wiring in `pdal_pipeline.rasterize` (NOT deep in `return_readers`): resolve
  right after the WESM-record pin; echo `workunit X -> EPT resource Y (tier N)`;
  record `ept_resolution` into processing metadata; pass the RESOLVED name +
  the already-loaded full EPT index down through
  `create_ept_3dep_pipeline(ept_index_gdf=...)` → `return_readers(
  ept_index_gdf=...)` (also removes a duplicate 8.7 MB index fetch).
  The WESM lookup keeps the WESM workunit name → the GEOID12A pin for
  `NV_LasVegas_QL2_2016` stays intact; only the reader join uses the alias.
- Resolved-but-disjoint-from-AOI (legitimate "survey doesn't cover AOI"):
  proceed to the existing no-data completed path, now with an accurate note —
  distinguishable from "no EPT resource exists" (hard LookupError).

Files: `survey.py` (+~90), `pdal_pipeline.py` (~25 in the EPT_AWS block),
`dsm_functions.py` (optional `ept_index_gdf` passthrough, ~15).
Tests: `tests/test_survey.py` (resolver tiers, tie-break, LookupError),
`tests/test_pdal_pipeline.py` or `test_dsm_functions.py` (injected index gdf;
no network).

### P1b. No-data runs must not report as plain `completed`

Problem: `driver.rasterize_projects` collapses every non-exception to
`completed` (`driver.py:119`); a 0-reader run writes products=nothing but the
batch reads as success (Southern_4, 2026-07-18 am).

Design: after each project run, read the child's `run_status` from
`<outdir>/*processing_metadata.yaml` (glob newest, legacy-name tolerant):
- `state: completed` + no note → `completed`
- `state: completed` + `note: no data...` → `completed (no data): <note>` in
  `batch_status.yaml`, plus a stderr `WARNING:` line per such project and a
  final-summary count. Batch still exits 0 for no-data (it is a real outcome),
  but never invisibly.
- resolver `LookupError` → surfaces as `failed: ...` (existing exception path)
  → nonzero exit via the existing end-of-batch RuntimeError.

Files: `driver.py` (+~30). Tests: `tests/test_driver.py` (fake rasterize
writing metadata with/without note; assert batch_status strings + warning).

## P2 — Staging phase: site manifest + project-selection aids (skeleton here)

New module `src/lidar_tools/staging.py` (new file → zero PR-stack conflict):

- `TESM_URL`; `load_tesm_tiles(aoi_gdf, tesm_source)` — bbox read;
  `attach_workunits(tesm_gdf, wesm_gdf)` — join by `workunit_id` ONLY
  (TESM project names drift from WESM's; verified 2026-07-18).
- `parse_grid_id(filename)` / `grid_origin(zone, square, northing_hint)` /
  `decode_tile_footprints(urls, utm_zone)` — the verified USGS 1-km grid-ID
  decode (0.00 m residual vs LAZ headers, zone 11); zone-generic MGRS lattice
  with unit tests pinned to the 7 verified zone-11 origins.
- `fetch_links_file(lpc_link)` → tile URL list (the payload-adjacent truth).
- `reconcile_tile_sources(tesm_tiles, decoded_tiles)` → per-workunit counts +
  disagreement warnings (TESM-missing-workunit à la Southern_4; count drift).
- `build_site_manifest(...)` / `write_site_manifest` / `load_site_manifest`:
  per-AOI YAML — pinned WESM snapshot rows, per-workunit EPT resolution record
  (tier, name) + boundary-overlap fracs, TESM tile counts, links-file counts,
  reconciliation verdicts, staged-metadata/OPR links, probe verdict slots (P3),
  `lpc_cache: <output>/lpc_cache/<workunit>/` convention.
- CLI: `lidar-tools prepare <aoi> <output> [--workunits ...]` writing
  `<output>/site_manifest.yaml`. `rasterize-projects` consumption of the
  manifest (offline WESM pin, resolver skip) is P2b — NOT in the skeleton, to
  keep the rasterize path untouched until the manifest schema settles.

Project-selection aids (P2c, design):
- incremental AOI coverage table (the 2026-07-18 manual analysis, productized):
  priority-ordered cumulative/incremental % using WESM polys (± footprints when
  prior runs exist) → printed + saved next to the manifest;
- survey map figure (workunit polygons colored by QL/date, EPT/TESM status
  hatching, AOI overlay) via the existing preview/figure conventions.
Both consume only manifest inputs → implement after P2a lands.

Files: `staging.py` (new, ~250 for skeleton), `cli.py` (+2), tests
(`tests/test_staging.py`, offline fixtures only).

## P3 — Staging-time probes (design; implement after P2)

- Vertical datum probe (README §8.5): K checkpoint windows from the
  fetch-reports-staged `vertical_accuracy/` GPKGs → tiny `readers.ept` bbox
  reads → median dz under geoid vs ellipsoid interpretation (~25 m separation)
  → pin `ept_vertical` in the manifest; `rasterize` consumes the pin (its
  `auto` empirical path stays as fallback). Prior art to harvest:
  `confirm_3dep_vertical` on `origin/datum-check` (not superseded by f3).
- EPT↔LAZ single-tile cross-check (source map §6): 1 AOI-interior staged tile
  (TESM/links) vs same-extent EPT read → point count, class histogram, dz
  median/NMAD, intensity scale → manifest verdict; catches naming
  misselection, EPT staleness vs republication, datum-branch errors.

Both are new probe functions in `staging.py` + small PDAL helpers; network +
compute optional and cheap (~min/workunit); no changes to the rasterize path.

## P4 — Provider abstraction (design only; sequence AFTER PR stack #73–#80)

Source providers (EPT / local LAZ dir / remote staged-LAZ via manifest tile
lists / future COPC slot) behind one "source → readers + native CRS + extent"
interface converging on `create_tile_pipelines`; auto-selection policy from
`laz_vs_ept_study.md` (EPT when resolvable+current, staged-LAZ fallback, loud).
Touches the same files as PRs #75–#77 → do not start until the stack lands.

## PR-stack conflict assessment

- `staging.py`, `tests/test_staging.py`: new files — no conflicts.
- `survey.py`, `driver.py`: additive; both files enter via PR #73/#74 — this
  branch is FROM the f3 tip, so it layers cleanly after #80 merges (or can be
  PR'd as the stack's successor). No rebase of existing stack commits needed.
- `pdal_pipeline.py`/`dsm_functions.py` edits: small, additive-parameter only;
  same layering argument.
- Recommended review path: adversarial review of this branch diff (P1+P2a)
  before it becomes PR #81; vertical probe (P3) as its own follow-up PR.

## Sequencing

1. This branch: P1a+P1b (+tests) → P2a skeleton (+tests) → review → PR after
   (or stacked on) #80.
2. Rerun `NV_LasVegas_QL2_2016` at 1 m the moment P1a merges (or from this
   branch with dshean's OK) — completes the LV 99.2% stack.
3. P2b manifest consumption + P2c selection aids.
4. P3 probes. 5. P4 provider abstraction (post-stack).
