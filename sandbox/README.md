# sandbox/

Scratch area for run outputs, figures, profiling logs, and one-off analysis
scripts. **Entire directory is gitignored** — nothing here is under version
control, and nothing in the package may import from it.

Convention: one subdirectory per experiment/run, named `YYYYMMDD_short_slug/`,
ideally with the exact scripts used so results can be regenerated.

## Contents

### `20260704_uw_campus_smoke/`

End-to-end verification + profiling run for the Phase 1 geodesy batch
(PR #71, branch `phase1-geodesy`).

Command (from repo root, run twice — first for verification with default
cleanup, then instrumented with `--no-cleanup` for stage profiling):

```
/usr/bin/time -l pixi run lidar-tools rasterize \
    --quiet --num-process 4 --no-cleanup \
    --geometry notebooks/uw-campus.geojson \
    --output <outdir>/smoke-ept2 2>&1 | python3 -u scripts/ts.py > profile_run.log
```

Key numbers (3.49 km^2 AOI, 1 m, products=all, WA_KingCounty_2021 EPT,
M-series laptop, 4 dask workers):

- 137.5 s wall, 344 s CPU, peak RSS 1.2 GB
- tile compute = ~70% of wall, run as 4 sequential per-product dask batches,
  each re-reading the same EPT points from AWS (~4x redundant fetch/decode)
- 4 separate dask cluster spinups + idle gaps ~= 25 s (~18%)
- datum check (WorldCover + COP30 download) = 16 s (~12%), network every run
- finalize (mosaic/warp/epoch/overview+COG) = ~35 s
- vertical shift verification: finalize warp applied median -24.091 m vs
  pyproj-predicted (GEOID18 + time-dependent Helmert) -24.086 m -> 4 mm
  agreement; IQR 0.11 m from bilinear resampling on slopes

Files:

- `figs/fig1_uw_products.png` — DSM/DTM hillshades, DSM-DTM, intensity
- `figs/fig2_geodesy_tests.png` — CONUS field + per-site offsets removed by
  the §1.4 ellipsoid-branch fix (the Casa Grande regression-test values)
- `figs/fig3_stage_profile.png` — stage timeline from output-file mtimes
- `figs/fig4_vertical_shift.png` — applied vs predicted NAVD88->ellipsoidal shift
- `profile_run.log` — timestamped stdout (note: timestamps quantize at Python
  stdout buffer flushes; stage timing came from file mtimes instead)
- `smoke-ept2/` — full run directory incl. per-tile intermediates, temp
  mosaics, WKTs, processing_metadata.yaml (geodesy provenance section)
- `scripts/` — figure/profiling scripts, copied verbatim from the session
  scratchpad (hardcoded paths point at the original scratchpad locations;
  point them at `smoke-ept2/` to regenerate). `probe_crs.py`/`probe2.py` are
  the pyproj/GDAL capability probes that preceded the geodesy.py design.
