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

We recommend using [pixi](https://pixi.sh/latest/) package manager to install a locked software environment for executing code in this repository. 

Pixi can be installed following instructions from [here](https://pixi.sh/latest/#installation). For Linux and Mac OSX machines, pixi can be installed from the terminal by running the below command:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
**************************

Once installed, you can run scripts from a terminal:

```bash
git clone https://github.com/uw-cryo/lidar_tools.git
cd lidar_tools
pixi shell
pdal_pipeline create-dsm --help
```

```console
Usage: pdal_pipeline create-dsm [ARGS] [OPTIONS]

Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and intensity
raster from a given extent and 3DEP point cloud data.

╭─ Parameters ────────────────────────────────────────────────────────────────╮
│ *  EXTENT-POLYGON                Path to the polygon file defining the      │
│      --extent-polygon            processing extent. [required]              │
│ *  TARGET-WKT --target-wkt       Path to the WKT file defining the target   │
│                                  coordinate reference system (CRS).         │
│                                  [required]                                 │
│ *  OUTPUT-PREFIX                 prefix with directory name and filename    │
│      --output-prefix             prefix for the project (e.g.,              │
│                                  CO_ALS_proc/CO_3DEP_ALS) [required]        │
│    SOURCE-WKT --source-wkt       Path to the WKT file defining the source   │
│                                  coordinate reference system (CRS). If      │
│                                  None, the CRS from the point cloud file is │
│                                  used.                                      │
│    PROCESS-SPECIFIC-3DEP-SURVEY  If specified, only process the given 3DEP  │
│      --process-specific-3dep-su  survey. This should be a string that       │
│      rvey                        matches the survey name in the 3DEP        │
│                                  metadata                                   │
│    PROCESS-ALL-INTERSECTING-SUR  If true, process all intersecting surveys. │
│      VEYS --process-all-interse  If false, only process the first LiDAR     │
│      cting-surveys --no-process  survey that intersects the extent defined  │
│      -all-intersecting-surveys   in the GeoJSON file. [default: False]      │
│    CLEANUP --cleanup             If true, remove the intermediate tif files │
│      --no-cleanup                for the output tiles [default: True]       │
╰─────────────────────────────────────────────────────────────────────────────╯
```

## Development

Use a developement environment (including pytest)
```
pixi shell -e dev
```

Or run the test sweet
```
pixi run test
# Full dsm processing run (takes ~30min)
pixi run test-create-dsm
```


## Additional dependencies

By default, pixi only manages dependencies *required* to run scripts in this respository, such as [PDAL](https://pdal.io). However, you might also want to install optional libraries into the same environment such as [GDAL](https://github.com/OSGeo/gdal) command line tools:

```
pixi add gdal
```

Or [Ames Stereo pipeline](https://stereopipeline.readthedocs.io/en/latest/installation.html#conda-intro):
```
pixi project channel add nasa-ames-stereo-pipeline
pixi project channel add usgs-astrogeology
pixi add stereo-pipeline
```

### Using pip

If you already have an environment you can install just the code in this library with pip:

```
pip install git+https://github.com/uw-cryo/lidar_tools.git@main --no-deps
```
