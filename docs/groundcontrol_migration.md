# lidar_tools → groundcontrol geodesy migration design

Status: DRAFT for review — 2026-07-15
Companion doc: `groundcontrol/docs/consolidation_geodesy.md` (the symbol-by-symbol
port map, already landed on groundcontrol `stage2-finalize`).

## 1. Context and goal

Owner decision (2026-07-04): **groundcontrol is the canonical CRS/transformation
library for the uw-cryo ecosystem.** lidar_tools grew a parallel engine in
`src/lidar_tools/geodesy.py` (741 lines on the `f3-single-read-multiproduct` branch
@ `c1b9560`; the module does not exist on lidar_tools `main`).

The **port is already done on the groundcontrol side**: `groundcontrol.geodesy`
(branch `stage2-finalize` @ `9014b23`, in review → main) carries the
general-purpose subset with 15 dedicated tests. This document designs the
**lidar_tools side**: how and when lidar_tools deletes its duplicated code and
imports groundcontrol, without changing a single byte of product georeferencing.

## 2. Verified current state (2026-07-15)

**Signature parity** (AST comparison, lidar_tools `f3` @ c1b9560 vs groundcontrol
`stage2-finalize`): all 16 shared symbols have identical signatures; zero
groundcontrol-only drift. lidar_tools-only symbols are exactly the four designated
to stay:

| stays in lidar_tools | why |
|---|---|
| `build_3857_navd88_compound`, `build_ept_3857_navd88_compound`, `build_ept_3857_nad83_2011` | 3DEP-EPT-on-AWS null-datum-tie relabeling semantics — point-cloud source domain knowledge |
| `set_coordinate_epoch` | in-place raster stamping via `osgeo.gdal`/`osr`; groundcontrol is deliberately GDAL-free |

**Call sites on `f3`** (the only branch with the module):

| module | groundcontrol-bound symbols | stay-local symbols |
|---|---|---|
| `driver.py` | `build_utm_target`, `write_crs_file` | — |
| `dsm_functions.py` | `navd88_offset` (×2), `preflight_vertical_transform` | — |
| `pdal_pipeline.py` | `write_crs_file` (×5), `preflight_vertical_transform` (×4), `build_utm_target`, `epoch_pinned_pipeline`, `geographic_base_epsg`, `geoid_grid_hint`, `library_versions`, `NAD83_2011_EPSG`, `DEFAULT_COORDINATE_EPOCH` | `build_ept_*` (×6), `set_coordinate_epoch` |

**Tests on `f3`**: `tests/test_geodesy.py` (24 tests) + `tests/test_geodesy_epoch.py`.
Partially overlaps groundcontrol's 15-test suite; the overlap gets deduplicated in
Phase 2.

**Packaging constraint (the gate)**: lidar_tools is public and conda/pixi-first;
groundcontrol is a private setuptools repo, unpublished. A hard dependency would
break public lidar_tools installs until groundcontrol is public or published
(PyPI and/or conda-forge). Everything below is sequenced around this gate.

## 3. Design decisions

**D1 — Full re-point, no re-export shim.** `lidar_tools/geodesy.py` shrinks to the
four domain symbols (plus module docstring pointing at groundcontrol); the three
consuming modules import `from groundcontrol import geodesy as gc_geodesy` (or
symbol-level imports) directly. Rejected alternative: keeping lidar_tools/geodesy.py
as a thin `from groundcontrol.geodesy import *` shim — it hides which engine is
live, keeps a misleading module surface, and saves little (only 3 files import the
module). The module keeps its name/path so git history follows the remainder.

**D2 — Constants come from groundcontrol too.** `DEFAULT_COORDINATE_EPOCH`,
`NAD83_2011_EPSG`, `GEOID_GRID_HINTS` etc. are imported, not duplicated. The stay-local
EPT builders take `base_epsg` parameters and keep working against the imported
constants.

**D3 — `library_versions` needs no wrapper.** The groundcontrol version reports
GDAL only-if-importable; the lidar_tools pixi env always has GDAL, so recorded
run metadata gains/loses nothing.

**D4 — Golden-fixture parity tests, built from real run metadata.** The migration's
one real risk is a silent change in PROJ pipeline selection (PROJ version, grid
availability, preflight behavior). We have byte-exact expected values already on
disk: the `preflight` blocks (`proj_pipeline` strings, branch names, accuracies) in
`processing_metadata.yaml` from validated production runs (SF f3 G2139, SF NAD83,
Casa Grande GEOID18-pinned). Phase 1 copies representative blocks into
`tests/data/groundcontrol_parity/` and asserts `groundcontrol.geodesy` reproduces
the pipelines **byte-identically** (geoid / ellipsoid / intensity-2D /
epoch-pinned branches, both output datums). Priced by experience: the 2026-07-15
SF intensity-quarantine episode cost two agents most of a day to rule out exactly
this class of silent transform drift (`3DEP_20260705_f3/INTENSITY_GEOREF_VERIFICATION.md`).
These tests `pytest.importorskip("groundcontrol")` so public CI skips them until
Phase 3.

**D5 — Interim dev-only dependency, hard dependency at the gate.** Until
groundcontrol publishes: a pixi feature (e.g. `[tool.pixi.feature.groundcontrol]`)
with a git+ssh pypi-dependency pinned to a groundcontrol tag, enabled only in a
local/dev environment — public CI and end-user installs are untouched. At the gate:
move to `[project.dependencies] groundcontrol>=0.1.0` (+ pixi pypi-dependency or
conda-forge once available) and promote the parity tests to always-run.
Ask on the groundcontrol side: **tag a release when `stage2-finalize` merges** so
the pin has a name. Environment note: `epoch_pinned_pipeline` shells out to
`projinfo` — present in any conda/pixi PROJ, no new constraint for lidar_tools.

**D6 — Freeze + canonicality rule during the transition.** From the moment
groundcontrol `stage2-finalize` merges, lidar_tools `geodesy.py` is frozen except
for critical fixes, and any CRS-logic change lands in groundcontrol **first**, then
(only if urgently needed pre-repoint) is copied back annotated with the
groundcontrol commit hash. The parity suite doubles as the drift alarm.

**D7 — Follow-ons are explicitly out of scope here** (tracked, not designed):
(a) rasterio-based raster epoch stamping in groundcontrol (its docs §6), after
which lidar_tools can drop `set_coordinate_epoch` and its `osgeo` import;
(b) adopting `groundcontrol.crs.transform_points` / `propagate_epoch` in
validation flows (two-stage CRS model from the audit plan);
(c) ITRF datum threading through rasterize-projects;
(d) the EPT compound builders stay in lidar_tools permanently — they are source
domain knowledge, not general geodesy.

## 4. Sequencing

```
gc: stage2-finalize ──merge+tag──▶ gc main (v0.1.x)          [in review now]
lt: f3 branch ──briefing──▶ lt main (geodesy.py arrives)      [existing plan, independent]
lt: groundcontrol-port-strategy branch
    Phase 1  parity harness (no dependency for CI, dev-only install)   [unblocked NOW]
    Phase 2  re-point PR, stacked on post-f3 main                      [needs f3 merged]
    ── PACKAGING GATE: groundcontrol public/published ──
    Phase 3  flip to hard dependency, parity tests always-run          [closes migration]
```

- **Phase 0 (groundcontrol side, in flight)**: merge `stage2-finalize`, tag.
- **Phase 1 (unblocked, this branch)**: dev pixi feature installing groundcontrol
  from git; `tests/test_groundcontrol_parity.py` + fixtures per D4; signature-parity
  test over the 16 symbols. Works against lidar_tools `main` — the fixtures are
  recorded strings, not calls into lidar_tools' own geodesy.
- **Phase 2 (after f3 → main)**: the re-point PR —
  1. swap imports in `driver.py`, `dsm_functions.py`, `pdal_pipeline.py`;
  2. shrink `geodesy.py` to the 4 domain symbols; grep-guard that no ported symbol
     is defined locally anymore;
  3. dedupe tests: drop lidar_tools tests now covered by groundcontrol's suite,
     keep stay-local and integration-level tests (`test_geodesy_epoch.py` mostly
     stays — it exercises lidar_tools' stamping + pipeline wiring);
  4. **verification gate**: smoke-run a small EPT AOI (e.g.
     `sandbox/20260705_f3_sf_debug/sf_tiny.geojson`) for both output datums,
     pre- and post-repoint, and byte-diff the `.wkt` sidecars and the
     `processing_metadata.yaml` preflight/transform blocks. Zero diff required.
- **Phase 3 (at the packaging gate)**: dependency + CI flip; delete the dev
  feature; update `consolidation_geodesy.md` to mark the re-point map executed.

## 5. Rollback

Phase 2 is one revert away at any point before Phase 3: the shrink and the import
swap are separate commits, and reverting them restores the self-contained module
byte-for-byte from git. After Phase 3, rollback additionally means restoring the
pixi/pyproject dependency stanzas — still mechanical, no data products depend on
which engine computed identical pipelines.

## 6. Acceptance criteria (whole migration)

1. Parity suite green: recorded production pipelines reproduced byte-identically
   by `groundcontrol.geodesy` on the pinned tag.
2. Smoke-run diff (Phase 2.4) shows zero change in WKT sidecars and metadata
   transform blocks.
3. No `def <ported-symbol>` remains under `src/lidar_tools/`; `osgeo` imports in
   `geodesy.py` limited to `set_coordinate_epoch`.
4. Public `pixi install` + tests pass with groundcontrol as a first-class
   dependency (post-gate).
5. `groundcontrol/docs/consolidation_geodesy.md` updated to reflect completion.
