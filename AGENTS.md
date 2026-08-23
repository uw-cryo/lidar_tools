# AGENTS.md — lidar_tools

CLI tools for discovering, staging, and rasterizing USGS 3DEP lidar into
DSM/DTM/intensity/count rasters with rigorous CRS and epoch handling. The
thing a newcomer gets wrong: outputs are only comparable because every run
pins the full output CRS (datum realization, epoch, geoid model) into per-run
processing metadata — treat geodesy.py and that metadata contract as
load-bearing.

## Environment

pixi, configured in pyproject.toml (there is no pixi.toml). `pixi install`
for the runtime env; dev tooling (pytest, ruff, mypy) lives in the `dev`
feature and is reached through the tasks below.

## Commands

    pixi run test               # pytest (dev feature)
    pixi run lint               # ruff check (check-only, CI-safe)
    pixi run format             # ruff format (applies changes)
    pixi run typecheck          # mypy src/lidar_tools/
    pixi run check-regressions  # fail only on NEW findings vs baseline
    pixi run example            # small end-to-end rasterize (UW campus AOI)

CLI (cyclopts): `lidar-tools` with `search`, `prepare`, `rasterize`,
`merge`, `preview`, `fetch-reports`, `report-metrics`.

## Conventions and pitfalls

- `pixi run pytest` resolves to whatever pytest is on your outer PATH (a
  conda one, typically) because pytest is not in the default env — imports
  fail confusingly. Always use `pixi run test`.
- Lint/type gates are regression gates: CI compares against
  `.github/regression_baseline.json`. Run `pixi run update-baselines` only
  to ratchet the baseline down after fixing findings, never to admit new ones.
- The `example` and `test-example-*` tasks stream EPT tiles and WESM from
  the network.
- Merge-stage VRTs are written read-only on purpose: GDAL PAM write-back
  from a stale QGIS session once corrupted them. Do not chmod-and-edit.
- Output naming: per-project files are `<aoi>_<res>m_<workunit>-<product>`;
  composites drop the workunit token. Readers glob with a legacy bare-name
  fallback.
- Merges happen on green ubuntu CI; the macOS job is informational (slower
  runner hardware, not a different result).

## Workspace

- Experimental work goes in `sandbox/` (gitignored except
  `sandbox/README.md`), one dated dir per experiment:
  `sandbox/YYYYMMDD_slug/`, self-contained (scripts, figures, notes).
  Nothing in `sandbox/` is importable from package code.
- Figures and other run outputs live with the run artifacts that produced
  them, not in the repo.
