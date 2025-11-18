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

## CLI Commands:

Once installed, you can run processing scripts from a terminal:

```bash
export PIXI_FROZEN=true # NOTE: set this to always use locked environment
pixi shell # NOTE: 'exit' deactivates the environment
lidar-tools rasterize --help
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

