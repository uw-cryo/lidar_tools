# lidar_tools

[![Actions Status][actions-badge]][actions-link]

[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions

Tools to process airborne and satellite LiDAR point clouds

**Warning!** This package is under active development and may change rapidly!


## Datasets Supported
* [3DEP AWS Public Dataset](https://registry.opendata.aws/usgs-lidar/)

  
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
Run our example workflow to create DSM, DEM, and LiDAR Intensity for 3DEP data over a part of University of Washington Campus in Seattle, WA!
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

Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and intensity
raster from a given extent and 3DEP point cloud data.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  EXTENT-POLYGON --extent-polygon          Path to the polygon file defining the processing extent. [required]           │
│ *  TARGET-WKT --target-wkt                  Path to the WKT file defining the target coordinate reference system (CRS).   │
│                                             [required]                                                                    │
│ *  OUTPUT-PREFIX --output-prefix            prefix with directory name and filename prefix for the project (e.g.,         │
│                                             CO_ALS_proc/CO_3DEP_ALS) [required]                                           │
│    SOURCE-WKT --source-wkt                  Path to the WKT file defining the source coordinate reference system (CRS).   │
│                                             If None, the CRS from the point cloud file is used.                           │
│    PROCESS-SPECIFIC-3DEP-SURVEY             If specified, only process the given 3DEP survey. This should be a string     │
│      --process-specific-3dep-survey         that matches the survey name in the 3DEP metadata                             │
│    PROCESS-ALL-INTERSECTING-SURVEYS         If true, process all intersecting surveys. If false, only process the first   │
│      --process-all-intersecting-surveys     LiDAR survey that intersects the extent defined in the GeoJSON file.          │
│      --no-process-all-intersecting-surveys  [default: False]                                                              │
│    CLEANUP --cleanup --no-cleanup           If true, remove the intermediate tif files for the output tiles [default:     │
│                                             True]                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

