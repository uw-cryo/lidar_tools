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
pdal_pipeline notebooks/very_small_aoi.geojson notebooks/UTM_13N_WGS84_G2139_3D.wkt ../CO_processing/CO_3DEP_ALS --source-wkt notebooks/SRS_CRS.wkt
```

```console
> pdal_pipeline --help
                                                                                                                                                                                                    
 Usage: pdal_pipeline [OPTIONS] EXTENT_GEOJSON TARGET_WKT OUTPUT_PREFIX                                                                                                                             
                                                                                                                                                                                                    
 Create a Digital Surface Model (DSM) from a given extent and point cloud data. This function divides a region of interest into tiles and generates a DSM geotiff from EPT point clouds for each    
 tile using PDAL                                                                                                                                                                                    
 Parameters ---------- extent_geojson : str     Path to the GeoJSON file defining the processing extent. source_wkt : str or None     Path to the WKT file defining the source coordinate reference 
 system (CRS). If None, the CRS from the point cloud file is used. target_wkt : str     Path to the WKT file defining the target coordinate reference system (CRS). output_prefix : str     prefix  
 with directory name and filename prefix for the project (e.g., CO_ALS_proc/CO_3DEP_ALS) mosaic : bool     Mosaic the output tiles using a weighted average algorithm cleanup: bool     If true,    
 remove the intermediate tif files for the output tiles Returns ------- None                                                                                                                        
 Notes ----- After running this function, reproject the final DEM with the following commands: 1. gdalwarp -s_srs SRS_CRS.wkt -t_srs UTM_13N_WGS84_G2139_3D.wkt -r cubic -tr 1.0 1.0 merged_dsm.tif 
 merged_dsm_reprojected_UTM_13N_WGS84_G2139.tif None                                                                                                                                                
                                                                                                                                                                                                    
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    extent_geojson      TEXT  [default: None] [required]                                                                                                                                        │
│ *    target_wkt          TEXT  [default: None] [required]                                                                                                                                        │
│ *    output_prefix       TEXT  [default: None] [required]                                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --source-wkt                            TEXT  [default: None]                                                                                                                                    │
│ --mosaic                --no-mosaic           [default: mosaic]                                                                                                                                  │
│ --cleanup               --no-cleanup          [default: cleanup]                                                                                                                                 │
│ --install-completion                          Install completion for the current shell.                                                                                                          │
│ --show-completion                             Show completion for the current shell, to copy it or customize the installation.                                                                   │
│ --help                                        Show this message and exit.                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Installation

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
