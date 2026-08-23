# Multi-epoch merge: plate motion across projects — design plan

Status: DRAFT for review — 2026-07-16. Branch context: `f3-single-read-multiproduct`.
All file:line citations are against this branch; all PROJ behavior was verified
in the pinned pixi dev env (PROJ **9.7.1**, `pixi run -e dev projinfo`).

## 1. Problem, and one load-bearing clarification

When the output frame is dynamic (ITRF20xx/WGS84 realizations), projects surveyed
years apart *can* sit at different coordinate epochs, and rigid plate motion
(~2–7 cm/yr) turns the epoch spread into horizontal offsets between merged
neighbors. The open question from dshean: correct at rasterize time or merge time?

**Clarification that bounds the problem:** for today's only production source
(3DEP EPT), the multi-epoch offset between projects **does not exist by
construction**. Every 3DEP survey is adjusted to NAD83(2011), *epoch-reduced to
2010.0* (`src/lidar_tools/geodesy.py:32-36`), a plate-fixed static frame. Our
dynamic-frame outputs apply the *same* epoch-pinned NAD83(2011)↔ITRF Helmert to
every project (`src/lidar_tools/pdal_pipeline.py:788-805`), so all projects in a
batch land at one shared target epoch (stamped, `pdal_pipeline.py:907-931`)
regardless of collect-date spread. Casa Grande confirms the same-era floor:
inter-project dz +0.02–0.04 m, NMAD 0.03–0.07 m — survey error, not frame motion.

The problem dshean describes becomes real for sources that are **not**
epoch-reduced into a plate-fixed frame:
- local LAZ / vendor deliveries natively in a dynamic frame at survey epoch
  (`pdal_pipeline.py:507-536` local path; commercial stereo-derived products);
- beyond-EPT ingest (gh #72), international sources (ATRF vs GDA2020, NZ);
- mixed plate-fixed *realizations* (NAD83(HARN)/NSRS2007 vs (2011): 5–20 cm in
  deforming regions, already threaded via `base_epsg` — `geodesy.py:466-471`,
  survey record pinning at `pdal_pipeline.py:407-437`).

Plus true ground deformation between survey dates (subsidence, plate-boundary
strain) — which is *data*, not a coordinate error (see §5).

## 2. Decision framework and error budget

Rigid-plate horizontal rates vs epoch spread, against product GSD 0.5–1 m:

| plate (frame anchor) | rate (cm/yr) | 3 yr spread | 10 yr spread | 10 yr in px @1 m |
|---|---|---|---|---|
| N. America interior (NAD83) | ~1.7 (cited at `geodesy.py:698`) | 5 cm | 17 cm | 0.17 |
| Eurasia (ETRS89) | ~2.5 | 8 cm | 25 cm | 0.25 |
| Australia (GDA2020) | ~7.0 (from measured Helmert rates, §4) | 21 cm | 70 cm | 0.7 |
| Pacific (NAD83(PA11)) | ~6–8 | ~20 cm | ~70 cm | ~0.7 |

Decision rule:
1. **Plate-fixed static output (e.g. NAD83(2011), ETRS89, GDA2020) or all sources
   epoch-reduced to one plate-fixed realization** → no epoch motion between
   projects. Document + assert; nothing to compute. This is the entire current
   3DEP path, for both static and dynamic outputs (shared pinned epoch).
2. **Epoch propagation required** when any source is natively dynamic at its own
   survey epoch, or realizations are mixed, AND rate × spread exceeds a floor.
   Sensible floor: max(0.1 px, the measured same-era validation floor ~5 cm).
   On the NA plate that is ~3 yr of spread; on AU/Pacific ~1 yr.
3. **Never** treat intraplate deformation as a frame correction by default (§5).

## 3. Where to correct: rasterize-time vs merge-time

**(a) Rasterize-time (recommended).** Propagate each project from its source
epoch to one batch-level target epoch inside the existing final warp.
- The machinery is 90 % present: `geodesy.epoch_pinned_pipeline()` resolves an
  explicit `-ct` pipeline with the epoch baked in via `projinfo --t_epoch`
  (`geodesy.py:339-412`; enforced with fail-loud substring guards at
  `pdal_pipeline.py:788-805`). Missing: a *source*-epoch knob (`--s_epoch`,
  `geodesy.py:390-392` currently passes only `--t_epoch`), per-project source-epoch
  handling, and the within-frame motion composition (§4).
- Provenance is clean: the pinned pipeline and stamped epoch already land in
  `processing_metadata.yaml` (`pdal_pipeline.py:800-805`, `:930-931`).
- All products get identical treatment in one warp — including intensity, whose
  2D branch deferred epoch handling (commit `797a67b` message: "epoch handling
  for the 2D branch deferred"; branch at `pdal_pipeline.py:841-882`).
- Honest cost: changing the target epoch invalidates the batch. With
  `cleanup=True` the pre-warp intermediates are deleted, so a re-epoch is a full
  re-run (Casa Grande full AOI: ~15 h). Target-epoch policy must therefore be
  decided before the batch (open question Q1).

**(b) Merge-time post-hoc shift (rejected as a correction path).** The merge is
deliberately resample-free — VRT composite on one shared grid, refusing
mismatched grids (`src/lidar_tools/merge.py:5-9`, `:269-275`). A sub-pixel
horizontal shift requires resampling, breaking that contract; an integer-pixel
VRT offset quantizes the correction to ≥0.5–1 m, larger than nearly all of the
signal (§2 table). It also forks the georeference away from the per-project
metadata, and does nothing for the (small) vertical component. Merge-time work
is still valuable as *measurement*: offset validation and drift alarms (§6, Tier 2).

## 4. PROJ machinery survey (verified, PROJ 9.7.1)

- **Within-frame epoch change silently no-ops.** `projinfo -s ITRF2014 -t
  ITRF2014 --s_epoch 2015 --t_epoch 2025` → "Null geographic offset …
  `+proj=noop`" (same with an Australia `--bbox`; NAD83(CSRS)v7 pair → ballpark
  noop). The pinned proj.db has **no point-motion operation table at all**, so no
  operation-selection route can move a point between epochs.
- **The deeper trap — dynamic→dynamic with two epochs *relabels*.** `-s ITRF2008
  -t ITRF2020 --s_epoch 2015 --t_epoch 2025` returns the frame-tie Helmert
  evaluated at 2015 with a trailing `+proj=set +v_4=2025`: the epoch tag changes,
  the coordinates do not move. Ten years of unapplied plate motion, labeled 0.01 m
  accuracy. Any Tier-1 implementation must never trust projinfo for the epoch
  *change* itself — only for individual frame legs.
- **Frame ↔ plate-fixed legs are rigorous and epoch-parameterized.**
  `-s "NAD83(2011)" -t ITRF2014 --t_epoch 2025` → the full time-dependent Helmert
  (`+t_epoch=2010`, rate terms) inside `+proj=set +v_4=2025` bookends — the exact
  production route (commit `1b06ea4`), with the epoch as a plain knob.
  `-s ITRF2014 -t GDA2020 --s_epoch 2015` → rotation-rate-only Helmert
  (`drx=0.00150379 dry=0.00118346 drz=0.00120716` ″/yr, `t_epoch=2020`, 0.03 m):
  the AU plate-motion model expressed as Helmert rates (|ω| ≈ 2.27 mas/yr ≈
  7.0 cm/yr at the surface — the §2 number).
- **Therefore: within-frame propagation = compose two legs through a plate-fixed
  CRS.** `ITRF@e_src → plate-fixed → ITRF@e_dst` applies exactly
  rates × (e_dst − e_src) of rigid plate motion, each leg projinfo-resolvable and
  guardable like today's pipelines. Works wherever EPSG has a plate-fixed datum
  tied to the frame by a time-dependent Helmert: NAD83(2011) (NA), NAD83(PA11)/
  (MA11) (Pacific/Mariana), ETRS89 (Eurasia), GDA2020 (AU). **Euler-pole fallback**
  (hand-built rate-only `+proj=helmert` from ITRF plate poles) is needed only for
  plates without such a datum (Nazca, Somalia, …) — implement only on demand.
- **Deformation models exist for a few jurisdictions.** `-s ITRF96 -t NZGD2000
  --s_epoch 2020 --spatial-test intersects` → `+proj=defmodel
  +model=nz_linz_nzgd2000-20180701.json` (0.02 m): NZ's national deformation
  model is a first-class, epoch-aware operation. The `+proj=deformation` operator
  (3D velocity grids, e.g. Canada's `ca_nrc_NAD83v70VG.tif`) exists but must be
  hand-composed — no db operation selects it. NOAA HTDP/TRANS4D are standalone
  programs with no PROJ-consumable CONUS grid in proj-data.

## 5. Intraplate motion: when it matters, and the scope line

GIA (Hudson Bay, Fennoscandia): vertical up to ~1 cm/yr, horizontal a few mm/yr —
relevant only for decade-scale vertical stacking. Western-US plate-boundary
strain: up to ~5 cm/yr *relative motion across faults inside one AOI* — no rigid
correction can help. Subsidence (central AZ, Central Valley): vertical cm–dm/yr,
localized. Data exists (HTDP/TRANS4D, NAD83v70VG, GAGE/UNAVCO velocity fields,
NZ defmodel), but removing intraplate motion converts *real ground displacement*
into a coordinate convention — it changes the science content of a DSM.

**Out of scope until**: a concrete AOI where (intraplate rate × epoch spread)
exceeds the measured validation floor (NMAD 0.03–0.07 m) **and** dshean decides
the product contract is "geometry homogenized to the target epoch" rather than
"geometry as surveyed". Rigid plate motion (a pure coordinate artifact) is the
only correction the tiers below apply.

## 6. Validation: proving a correction worked

- **Horizontal**: same-modality phase correlation in project overlaps —
  hillshade↔hillshade and intensity↔intensity, never across modalities. SF
  evidence (INTENSITY_GEOREF_VERIFICATION.md, archived with the run
  artifacts on the project data volume): same-modality windows resolved a 1.34 m
  Helmert to 2–7 cm with ≤2 cm inter-window scatter at 0.5 m posting, while
  hillshade↔intensity cross-modality reported ~0.5 m offsets incoherent in
  direction — unreliable below ~0.5 m. Cross-*project* same-modality pairs sit
  between these (different sensors/dates); expect a few-cm to ~10 cm floor.
- **Vertical**: dz median/NMAD over the same overlaps (the Casa Grande numbers
  are exactly this measurement).
- **Reuse**: the merge stage already computes decimated per-source valid masks
  and their pairwise overlaps (`merge.py:57-79`, `:107-123`) — the window
  selection for offset measurement drops straight onto `comp_mask & mask`, with
  full-resolution reads only inside chosen windows. Predicted offset for the
  check comes from the same pipeline composition as the correction (rates × Δepoch),
  giving a closed loop: predicted ↔ measured ↔ ~0 after Tier 1.

## 7. Tiered roadmap

**Tier 0 — document + warn (now; ~1 day).**
- This document; document `coord_epoch` in the `rasterize` docstring — the
  parameter exists (`pdal_pipeline.py:208`) but has **no docstring entry** today
  (the Parameters block, `pdal_pipeline.py:213-274`, never mentions it).
- Pre-run warning in `rasterize_projects` (`src/lidar_tools/driver.py:100-118`):
  before the per-project loop, fetch each workunit's WESM record
  (`survey.workunit_record`, `src/lidar_tools/survey.py:108-116`; collect fields
  `survey.py:41-42`), compute collect-midpoint spread (reuse `_collect_midpoint`,
  `survey.py:203-210`), and warn when the target CRS is dynamic AND spread > N
  years (default N=3, §2), stating explicitly that 3DEP sources are epoch-reduced
  so the shared pinned epoch keeps projects consistent. Record the spread and the
  verdict in `batch_status.yaml` (`driver.py:125-131`). Must handle explicit
  `dst_crs` WKT too, not just `output_datum` (note: `driver.py:32` exposes only
  2 of the 6 datums in `geodesy.OUTPUT_DATUM_BUILDERS`, `geodesy.py:279-286`).
  This extends the existing fail-fast pre-run gate pattern (`pdal_pipeline.py:441-445`).
- Merge assertion: extend `_raster_signature` (`merge.py:42-54`) with the CRS
  coordinate epoch (`srs.GetCoordinateEpoch()`) so mixed-epoch mosaics refuse to
  merge exactly like mismatched grids (`merge.py:269-275`); verify empirically
  whether `gdal.BuildVRT` carries the epoch into the merge VRT and stamp it if not.
- Ownership: all lidar_tools. Tests: unit test the warning on fixture WESM
  records (monkeypatched fetch, static vs dynamic targets, spreads straddling N);
  merge test on tiny epoch-stamped COGs (matching/mismatched epochs).

**Tier 1 — rasterize-time epoch propagation.**
- groundcontrol (per the D6 gc-first rule, `docs/groundcontrol_migration.md:92-96`;
  `epoch_pinned_pipeline` is already flagged for migration, `geodesy.py:355-357`):
  add `source_epoch` (`projinfo --s_epoch`) to `epoch_pinned_pipeline`; add a
  `plate_motion_pipeline(frame, e_src, e_dst, plate_fixed_crs)` composer for the
  §4 round-trip, with the same fail-loud substring guards; reconcile with the
  existing `groundcontrol.crs.propagate_epoch` (migration doc D7(b),
  `groundcontrol_migration.md:99-105`) — it may already cover the point-domain half.
- lidar_tools: per-project `source_epoch` on `rasterize` (default None = source
  is static/epoch-reduced — 3DEP stays a no-op *by construction*, source epoch is
  2010.0, NOT the collect date); `target_epoch` on `rasterize_projects`
  (`driver.py:21-34` exposes neither today) passed through as `coord_epoch`;
  complete the intensity 2D-branch epoch handling deferred in `797a67b`
  (`pdal_pipeline.py:841-882`); record source/target epochs and the composed
  pipeline in the geodesy metadata block (`pdal_pipeline.py:540-545`).
- Tests: golden pipeline strings (migration doc D4 pattern); pyproj point
  round-trip asserting displacement ≈ rate × Δepoch per plate (NA ~1.7, AU
  ~7.0 cm/yr); dual-epoch smoke AOI whose product pair differs by the predicted
  shift (measured via the §6 same-modality correlator).

**Tier 2 — motion-model-aware merge validation/refinement.**
- Merge QA pass: same-modality phase correlation + dz stats per project overlap
  (§6), written to `merge_metadata.yaml` (`merge.py:328-347`), compared against
  the motion-model prediction (≈0 once Tier 1 ran); warn above a threshold tied
  to the SF-measured reliability floor. Ownership: lidar_tools (gc supplies
  predicted motion vectors).
- Optional, explicit opt-in: deformation-model warps (`+proj=defmodel` NZ,
  `+proj=deformation` + Canadian velocity grid) for jurisdictions where they are
  authoritative. HTDP/TRANS4D stay out of scope until a PROJ-consumable CONUS
  velocity grid is adopted (§5 scope line).

## 8. Open questions for dshean

1. **Target-epoch policy**: pin the batch epoch to the anchor project's survey
   epoch, keep 2010.0 (3DEP-native), or take it from the deliverable spec? This
   must be fixed before a batch since re-epoching = full re-run (§3a).
2. Is an integer-pixel post-hoc shift ever acceptable as an emergency knob for
   already-delivered batches, or is re-rasterization the only sanctioned path?
3. Source-epoch trust: for non-epoch-reduced sources, is the WESM/vendor collect
   midpoint an acceptable survey epoch, or do we require an explicit per-project
   epoch from delivery metadata?
4. Vertical: propagate horizontal-only (plate motion is ~horizontal), or carry
   vertical velocities too once deformation models enter (couples to §5)?
5. Sequencing vs gh #72 (beyond-EPT ingest): Tier 1 only pays off once
   non-3DEP sources are ingestable — build them together?
6. Does `groundcontrol.crs.propagate_epoch` already implement the plate-fixed
   round-trip of §4, and should the `--s_epoch` extension land there first (D6)?
7. Tier 0 threshold N: default 3 years (NA-calibrated) — or plate-rate-scaled so
   AU/Pacific AOIs warn at ~1 year?
