#!/usr/bin/env python
"""
Recover final products from `*-temp.tif` mosaics left behind by an interrupted
`lidar_tools rasterize` run (EPT path grids in EPSG:3857; the final
reprojection/finalize stage never ran).

Runs the same finalize steps the pipeline would have
(dsm_functions.gdal_warp + gdal_add_overview) with two corrections:

1. Intensity is warped with a horizontal-only EPSG:3857 source SRS. Passing the
   compound 3857+NAVD88 SRS makes gdal.Warp apply the vertical datum shift to
   the intensity *values* (~-30 m in CONUS; values below the undulation clamp
   to 0 = nodata).
2. When --like is given, the output grid exactly matches the reference raster
   (targetAlignedPixels disabled), so recovered products stay pixel-aligned
   with previously recovered ones even if that grid is not tap-aligned.

Example (Las Vegas NV_ClarkCo_2_B22, intensity):
    python scripts/recover_temp_mosaic.py \
        /path/usa_lasvegas_cal-intensity_mos-temp.tif \
        --product intensity \
        --dst-wkt /path/UTM_11N_WGS84_G2139_3D.wkt \
        --like /path/usa_lasvegas_cal-DSM_mos.tif

Height products (DSM/DTM) from geoid-referenced surveys need the compound
source SRS so the vertical datum shift IS applied:
    python scripts/recover_temp_mosaic.py \
        /path/usa_lasvegas_cal-DTM_fill_window_size_4_mos-temp.tif \
        --product height --src-wkt /path/SRS_CRS.wkt \
        --dst-wkt /path/UTM_11N_WGS84_G2139_3D.wkt \
        --like /path/usa_lasvegas_cal-DSM_mos.tif
"""

import argparse
import os
import sys
from pathlib import Path

# set-if-unset so a deliberate PROJ_NETWORK=OFF is respected
os.environ.setdefault("PROJ_NETWORK", "ON")

from osgeo import gdal  # noqa: E402

from lidar_tools import dsm_functions  # noqa: E402

gdal.UseExceptions()


def reference_grid(like_fn: str) -> tuple[list, float]:
    """Return ([minx, miny, maxx, maxy], res) of an existing raster."""
    ds = gdal.Open(like_fn)
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    if gt[2] != 0 or gt[4] != 0 or abs(gt[1]) != abs(gt[5]):
        sys.exit(f"Reference raster {like_fn} is rotated or non-square-pixel")
    minx, maxy = gt[0], gt[3]
    maxx = minx + xsize * gt[1]
    miny = maxy + ysize * gt[5]
    return [minx, miny, maxx, maxy], abs(gt[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("temp_fn", help="Path to the *-temp.tif mosaic (EPSG:3857)")
    parser.add_argument(
        "--product",
        choices=["intensity", "height"],
        required=True,
        help="intensity: UInt16, horizontal-only source SRS (never vertical-shifted). "
        "height: Float32, source SRS from --src-wkt (compound for geoid surveys)",
    )
    parser.add_argument(
        "--dst-wkt", required=True, help="Target CRS (WKT file from the run dir, or any PROJ string)"
    )
    parser.add_argument(
        "--src-wkt",
        default=None,
        help="Source CRS for height products (e.g. the run's SRS_CRS.wkt compound "
        "3857+NAVD88). Defaults to EPSG:3857 (no vertical shift)",
    )
    parser.add_argument(
        "--like",
        default=None,
        help="Existing raster whose grid (extent + resolution) the output must match exactly",
    )
    parser.add_argument("--res", type=float, default=None, help="Output resolution (ignored with --like)")
    parser.add_argument("--out", default=None, help="Output path (default: temp name minus '-temp')")
    parser.add_argument(
        "--no-overviews", action="store_true", help="Skip Gaussian overviews + COG conversion"
    )
    args = parser.parse_args()

    temp_fn = Path(args.temp_fn)
    if not temp_fn.exists():
        sys.exit(f"Not found: {temp_fn}")
    out_fn = Path(args.out) if args.out else Path(str(temp_fn).split("-temp.tif")[0] + ".tif")
    if out_fn.exists():
        sys.exit(f"Refusing to overwrite existing {out_fn}")

    if args.product == "intensity":
        src_srs = "EPSG:3857"
        dtype = "UInt16"
        if args.src_wkt:
            sys.exit("--src-wkt is not valid with --product intensity (values are not heights)")
    else:
        src_srs = args.src_wkt if args.src_wkt else "EPSG:3857"
        dtype = "Float32"

    if args.like:
        out_extent, res = reference_grid(args.like)
        tap = False
    else:
        if args.res is None:
            sys.exit("Provide --like or --res")
        out_extent, res, tap = None, args.res, True

    print(f"src: {temp_fn}\ndst: {out_fn}\nsrc_srs: {src_srs}\ndst_srs: {args.dst_wkt}")
    print(f"grid: extent={out_extent} res={res} tap={tap} dtype={dtype}")

    dsm_functions.gdal_warp(
        str(temp_fn),
        str(out_fn),
        src_srs,
        args.dst_wkt,
        res=res,
        resampling_alogrithm="bilinear",
        out_extent=out_extent,
        dtype=dtype,
        target_aligned_pixels=tap,
    )

    if not args.no_overviews:
        dsm_functions.gdal_add_overview(str(out_fn))

    print(f"Recovered {out_fn}")


if __name__ == "__main__":
    main()
