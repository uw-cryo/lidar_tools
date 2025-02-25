# lidar_tools

[![Actions Status][actions-badge]][actions-link]

[actions-badge]:            https://github.com/uw-cryo/lidar_tools/workflows/Tests/badge.svg
[actions-link]:             https://github.com/uw-cryo/lidar_tools/actions

Tools to process airborne and satellite LiDAR point clouds

**Warning!** This package is under active development!

## Quickstart

We recommend using [pixi](https://pixi.sh/latest/) to install a locked software environment for executing code in this repository. Once installed, you can run scripts from a terminal:

```bash
git clone https://github.com/uw-cryo/lidar_tools.git
cd lidar_tools
pixi shell
pdal_pipeline create-dsm --help
```

```console
Usage: pdal_pipeline create-dsm [ARGS] [OPTIONS]

Create a Digital Surface Model (DSM) from a given extent and point cloud data. This function
divides a region of interest into tiles and generates a DSM geotiff from EPT point clouds
for each tile using PDAL

╭─ Parameters ─────────────────────────────────────────────────────────────────────────────╮
│ *  EXTENT-GEOJSON --extent-geojson  Path to the GeoJSON file defining the processing     │
│                                     extent. [required]                                   │
│ *  TARGET-WKT --target-wkt          Path to the WKT file defining the target coordinate  │
│                                     reference system (CRS). [required]                   │
│ *  OUTPUT-PREFIX --output-prefix    prefix with directory name and filename prefix for   │
│                                     the project (e.g., CO_ALS_proc/CO_3DEP_ALS)          │
│                                     [required]                                           │
│    SOURCE-WKT --source-wkt          Path to the WKT file defining the source coordinate  │
│                                     reference system (CRS). If None, the CRS from the    │
│                                     point cloud file is used.                            │
│    MOSAIC --mosaic --no-mosaic      Mosaic the output tiles using a weighted average     │
│                                     algorithm [default: True]                            │
│    CLEANUP --cleanup --no-cleanup   If true, remove the intermediate tif files for the   │
│                                     output tiles [default: True]                         │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
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
