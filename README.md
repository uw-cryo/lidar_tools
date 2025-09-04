# lidar_tools

[![Actions Status][actions-badge]][actions-link]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15970112.svg)](https://doi.org/10.5281/zenodo.15970112)

[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions

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
│ *  EXTENT-POLYGON --extent-polygon          Path to the vector dataset containing a polygon defining the      │
│                                             processing extent. [required]                                     │
│ *  OUTPUT-PREFIX --output-prefix            Path for output files, containing directory path and filename     │
│                                             prefix (e.g., /tmp/CO_3DEP_ALS). [required]                       │
│    TARGET-WKT --target-wkt                  Path to a text file containing WKT2 definition for the output     │
│                                             coordinate reference system (CRS). If unspecified, a local UTM    │
│                                             CRS will be used.                                                 │
│    LOCAL-UTM --local-utm --no-local-utm     If true, automatically compute the local UTM zone from the extent │
│                                             polygon for final output products. If false, use the CRS defined  │
│                                             in the target_wkt file. [default: False]                          │
│    SOURCE-WKT --source-wkt                  Path to a text file containing WKT2 definition for the coordinate │
│                                             reference system (CRS) of the input point cloud. If unspecified,  │
│                                             the CRS defined in the source point cloud metadata will be used.  │
│    LOCAL-LAZ-DIR --local-laz-dir            Path to directory containing source laz point cloud files. If not │
│                                             specified, the program will process USGS 3DEP EPT tiles.          │
│    EPT-TILE-SIZE-KM --ept-tile-size-km      The size of the EPT tiles to be processed. This is only used if   │
│                                             local_laz_dir is not specified. The default is 1.0 km, which      │
│                                             means that the function will process 1 km x 1 km tiles. If you    │
│                                             want to process larger tiles, you can specify a larger value.     │
│                                             [default: 1.0]                                                    │
│    PROCESS-SPECIFIC-3DEP-SURVEY             Only process the specified 3DEP project name. This should be a    │
│      --process-specific-3dep-survey         string that matches the workunit name in the 3DEP metadata.       │
│    PROCESS-ALL-INTERSECTING-SURVEYS         If true, process all available 3DEP EPT point clouds which        │
│      --process-all-intersecting-surveys     intersect with the input polygon. If false, and                   │
│      --no-process-all-intersecting-surveys  process_specific_3dep_survey is not specified, first 3DEP project │
│                                             encountered will be processed. [default: False]                   │
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
