# lidar_tools

[![Actions Status][actions-badge]][actions-link]
[![DOI][zenodo-badge]][zenodo-link]


[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions
[zenodo-badge]:             https://zenodo.org/badge/916689740.svg
[zenodo-link]:              https://doi.org/10.5281/zenodo.15970111

Tools to process airborne and satellite LiDAR point clouds.



**Warning!** This package is under active development and may change rapidly!

## Datasets Supported
* [3DEP AWS Public Dataset](https://registry.opendata.aws/usgs-lidar/)
* Locally available, classified LiDAR point clouds in las/laz format

## Output Products
* Digital Surface Models: IDW interpolation based gridding of height values for `first` and `only` returns.
* Digital Terrain Models: IDW interpolation based gridding of height values for `ground` returns (Classification==2). We do not perform ground classification ourselves, input point clouds need to have ground returns labelled for terrain models generation. An additional gap-filled product using IDW interpolation with a 9 x 9 kernel can also be produced which is useful in reducing data gaps in areas with dense canopy or buildings.
* Surface Intensity: IDW interpolation based gridding of surface intensity values for `first` and `only` returns.

![UW Campus preview](https://github.com/user-attachments/assets/08798588-17d3-4e4b-b2c4-ee70a1ec0a7b)
*Sample of standard products created with lidar_tools `rasterize` utility for University of Washington Campus AOI, using publicly-available USGS 3DEP lidar point clouds ([WA_KingCounty_2021_B21](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WA_KingCounty_2021_B21/WA_KingCo_1_2021/reports/WA_KingCounty_2021_B21_Lidar_Delivery_1_Technical_Data_Report.pdf))*


## Quickstart

`lidar_tools` is a collection of CLI scripts to process LiDAR data. It should be installed into a stand-alone environment to ensure that scripts execute as intended. We recommend using [pixi](https://pixi.sh/latest/) to install a locked software environment.

We recommend using [pixi](https://pixi.sh/latest/) package manager to install a locked software environment for executing code in this repository.

Pixi can be installed following instructions from [here](https://pixi.sh/latest/#installation). For Linux and Mac OSX machines, pixi can be installed from the terminal by running the below command:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://github.com/uw-cryo/lidar_tools.git
cd lidar_tools
pixi install --frozen
```

### Example workflow
Run our example workflow to create DSM, DTM without interpolation, DTM with interpolation, and LiDAR Intensity for 3DEP data over a part of University of Washington Campus in Seattle, WA!
```
# NOTE: takes ~5 min to run
pixi run example
```

## CLI Commands

Once installed, you can run processing scripts from a terminal:

```bash
export PIXI_FROZEN=true # NOTE: set this to always use locked environment
pixi shell # NOTE: 'exit' deactivates the environment
lidar-tools --help            # all commands
lidar-tools rasterize --help  # options for one command
```

| Command | What it does |
| --- | --- |
| `search` | Search the lidar catalog: which collections cover an AOI, at what quality level and acquisition dates, with what declared CRS/datum/geoid, EPT availability, AOI overlap, and the uncovered fraction. |
| `prepare` | Stage discovery metadata for an AOI into `site_manifest.yaml`: pinned WESM records, EPT name resolution, TESM-vs-links tile reconciliation, staged-LAZ cache layout. |
| `rasterize` | Create DSM, DTM (with and without gap filling) and/or intensity rasters from 3DEP EPT or local LAS/LAZ: one subdirectory per selected survey (`--projects auto\|latest\|NAME\|A,B,C`), all on one shared target grid so `merge` can composite them without resampling. |
| `merge` | Merge a batch's per-project products into per-product VRT composites (priority order, no resampling), normalizing intensity to a common range. |
| `preview` | Write a one-page preview figure (shaded relief, scale bar, processing footer) for a run or for every project in a batch. |
| `fetch-reports` | Stage each project's vendor QA/QC, survey and mapping reports plus the USGS vertical-accuracy checkpoints next to its products. |
| `report-metrics` | Extract standardized metrics (acquisition period, tested vertical/horizontal accuracy, point density) from those reports into one record per project, with per-number source evidence. |

### Multi-project workflow

For an AOI covered by more than one 3DEP survey, the commands chain:

```bash
lidar-tools search aoi.geojson                       # what covers this AOI?
lidar-tools prepare aoi.geojson batch/               # pin the metadata once
lidar-tools rasterize aoi.geojson batch/ \
    --resolution 1                                   # co-registered per-project products
                                                     # (--projects auto is the default;
                                                     #  pass A,B,C for an explicit priority order)
lidar-tools merge batch/                             # per-product composites
lidar-tools preview batch/                           # QA figures
lidar-tools fetch-reports batch/                     # vendor reports
lidar-tools report-metrics batch/                    # standardized accuracy table
```

Projects are listed in priority order: the first one wins where they overlap.
See [docs/vendor_reports.md](https://github.com/uw-cryo/lidar_tools/blob/main/docs/vendor_reports.md)
for the report staging and metric extraction details (repo-only: `docs/` is not
shipped in the built package).

## Development

Use a developement environment (including pytest, ruff, mypy etc)
```
export PIXI_FROZEN=true # Disable this if you're changing dependency versions
pixi shell -e dev
```

Or run the test sweet
```
pixi run test
```

Run linting and formatting
```
pixi run lint
pixi run typecheck
```

### Packaging

To create a `conda` package that can be installed into any conda environment:

```
pixi build
```

This will output a `.conda` file named something like `lidar_tools-0.1.0-pyhbf21a9e_0.conda`

To actually install the package it needs to put in a conda "registry" or "channel" like conda-forge. For now we are using a public channel at https://prefix.dev/channels for development:

```
conda install -c https://repo.prefix.dev/uw-cryo lidar_tools
```

