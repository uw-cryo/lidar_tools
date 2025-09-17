# lidar_tools

[![Actions Status][actions-badge]][actions-link]
[![DOI][zenodo-badge]][zenodo-link]


[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions
[zenodo-badge]:             https://zenodo.org/badge/916689740.svg
[zenodo-link]:              https://doi.org/10.5281/zenodo.15970111

Tools to process airborne and satellite LiDAR point clouds.

![UW Campus preview](https://github.com/user-attachments/assets/08798588-17d3-4e4b-b2c4-ee70a1ec0a7b)
*Sample of standard products created with lidar_tools `pdal_pipeline` utility for University of Washington Campus AOI, using publicly-available USGS 3DEP lidar point clouds ([WA_KingCounty_2021_B21](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/WA_KingCounty_2021_B21/WA_KingCo_1_2021/reports/WA_KingCounty_2021_B21_Lidar_Delivery_1_Technical_Data_Report.pdf))*

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
lidar-tools rasterize --help
```

```console
Usage: lidar-tools rasterize [ARGS] [OPTIONS]

Create a Digital Surface Model (DSM), Digital Terrain Model (DTM) and/or Intensity raster from point cloud data.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  GEOMETRY --geometry                  Path to the vector dataset containing a single polygon that defines the processing extent. [required]                     │
│    INPUT --input                        Path to directory containing input LAS/LAZ files, otherwise uses USGS 3DEP EPT data on AWS. [default: EPT_AWS]            │
│    OUTPUT --output                      Path to output directory. [default: /tmp/lidar-tools-output]                                                              │
│    SRC-CRS --src-crs                    Path to file with PROJ-supported CRS definition to override CRS of input files.                                           │
│    DST-CRS --dst-crs                    Path to file with PROJ-supported CRS definition for the output. If unspecified, a local UTM CRS will be used.             │
│    RESOLUTION --resolution              Square output raster posting in units of dst_crs. [default: 1.0]                                                          │
│    PRODUCTS --products                  Which output products to generate: all products, digital surface model, digital terrain model, or intensity raster.       │
│                                         [choices: all, dsm, dtm, intensity] [default: all]                                                                        │
│    THREEDEP-PROJECT --threedep-project  "all" processes all available 3DEP EPT point clouds which intersect with the input polygon. "first" 3DEP project          │
│                                         encountered will be processed. "specific" should be a string that matches the "project" name in the 3DEP metadata.        │
│                                         [choices: all, latest] [default: latest]                                                                                  │
│    TILE-SIZE --tile-size                The size of rasterized tiles processed from input EPT point clouds in units of dst_crs. [default: 1.0]                    │
│    NUM-PROCESS --num-process            [default: 1]                                                                                                              │
│    OVERWRITE --overwrite                Overwrite output files if they already exist. [default: False]                                                            │
│    CLEANUP --cleanup                    Remove the intermediate tif files, keep only final mosaiced rasters. [default: False]                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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

