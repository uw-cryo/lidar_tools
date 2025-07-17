# lidar_tools

[![Actions Status][actions-badge]][actions-link]
[![DOI][zenodo-badge]][zenodo-link]

[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions
[zenodo-badge]:             https://zenodo.org/badge/916689740.svg
[zenodo-link]:              https://doi.org/10.5281/zenodo.15970111

Tools to process airborne and satellite LiDAR point clouds

**Warning!** This package is under active development and may change rapidly!


## Datasets Supported
* [3DEP AWS Public Dataset](https://registry.opendata.aws/usgs-lidar/)
* Locally available, classified LiDAR point clouds in las/laz format


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

## CLI Commands:

Once installed, you can run processing scripts from a terminal:

```bash
export PIXI_FROZEN=true # NOTE: set this to always use locked environment
pixi shell # NOTE: 'exit' deactivates the environment
pdal_pipeline create-dsm --help
```

```console
Usage: pdal_pipeline create-dsm [ARGS] [OPTIONS]

Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and intensity raster from a given extent and
3DEP point cloud data.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  EXTENT-POLYGON --extent-polygon          Path to the polygon file defining the processing extent.          │
│                                             [required]                                                        │
│ *  OUTPUT-PREFIX --output-prefix            prefix with directory name and filename prefix for the project    │
│                                             (e.g., CO_ALS_proc/CO_3DEP_ALS) [required]                        │
│    TARGET-WKT --target-wkt                  Path to the WKT file defining the target coordinate reference     │
│                                             system (CRS).                                                     │
│    LOCAL-UTM --local-utm --no-local-utm     If true, compute the UTM zone from the extent polygon and use it  │
│                                             to create the output rasters. If false, use the CRS defined in    │
│                                             the target_wkt file. [default: False]                             │
│    SOURCE-WKT --source-wkt                  Path to the WKT file defining the source coordinate reference     │
│                                             system (CRS). If None, the CRS from the point cloud file is used. │
│    LOCAL-LAZ-DIR --local-laz-dir            If  the path to a local directory containing laz files is         │
│                                             specified, the laz files are processed. If not specified, the     │
│                                             function will process USGS 3DEP EPT tiles                         │
│    EPT-TILE-SIZE-KM --ept-tile-size-km      The size of the EPT tiles to be processed. This is only used if   │
│                                             local_laz_dir is not specified. The default is 1.0 km, which      │
│                                             means that the function will process 1 km x 1 km tiles. If you    │
│                                             want to process larger tiles, you can specify a larger value.     │
│                                             [default: 1.0]                                                    │
│    PROCESS-SPECIFIC-3DEP-SURVEY             If specified, only process the given 3DEP survey. This should be  │
│      --process-specific-3dep-survey         a string that matches the workunit name in the 3DEP metadata      │
│    PROCESS-ALL-INTERSECTING-SURVEYS         If true, process all available EPT surveys which intersect with   │
│      --process-all-intersecting-surveys     the input polygon. If false, and process_specific_3dep_survey is  │
│      --no-process-all-intersecting-surveys  not specified, only process the first available 3DEP EPT survey   │
│                                             that intersects the input polygon. [default: False]               │
│    NUM-PROCESS --num-process                Number of processes to use for parallel processing. Default is 1, │
│                                             which means all pdal and gdal processing will be done serially    │
│                                             [default: 1]                                                      │
│    CLEANUP --cleanup --no-cleanup           If true, remove the intermediate tif files for the output tiles,  │
│                                             leaving only the final mosaicked rasters. Default is True.        │
│                                             [default: True]                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

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
