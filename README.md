# lidar_tools

[![Actions Status][actions-badge]][actions-link]

[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions

Tools to process airborne and satellite LiDAR point clouds

**Warning!** This package is under active development!
## Datasets Supported
* [3DEP](https://www.usgs.gov/3d-elevation-program)
* 
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

## CLI Commands:

Once installed, you can run scripts from a terminal:

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
# Full dsm processing run (takes ~30min)
pixi run test-create-dsm
```

Run linting and formatting 
```
pixi run lint
pixi run typecheck
```