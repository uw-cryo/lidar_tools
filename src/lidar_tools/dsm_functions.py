"""
Generate DSMs from 3DEP EPT data
"""
from rasterio.warp import transform_bounds
from pyproj import CRS
from shapely.geometry import Polygon
import geopandas as gpd
import requests
import subprocess
from shutil import which

def return_readers(input_aoi,
                   src_crs,
                   pointcloud_resolution = 1,
                   n_rows = 5,
                   n_cols=5,
                   buffer_value=0):
    """
    This method takes a raster file and finds overlapping 3DEP data. It then returns a series of readers
    corresponding to non overlapping areas that can be used as part of further PDAL processing pipelines
    The method also returns the CRS specified i
    """
    xmin, ymin, xmax, ymax = input_aoi.bounds
    x_step = (xmax - xmin) / n_cols
    y_step = (ymax - ymin) / n_rows

    dst_crs = CRS.from_epsg(4326)

    readers = []
    pointcloud_input_crs = []

    for i in range(int(n_cols)):
        for j in range(int(n_rows)):
            aoi = Polygon.from_bounds(xmin+i*x_step, ymin+j*y_step, xmin+(i+1)*x_step, ymin+(j+1)*y_step)

            src_bounds_transformed = transform_bounds(src_crs, dst_crs, *aoi.bounds)
            aoi_4326 = Polygon.from_bounds(*src_bounds_transformed)

            src_bounds_transformed_3857 = transform_bounds(src_crs, CRS.from_epsg(3857), *aoi.bounds)
            aoi_3857 = Polygon.from_bounds(*src_bounds_transformed_3857)
            print(aoi.bounds, src_bounds_transformed_3857)
            if buffer_value:
                aoi_3857.buffer(buffer_value)

            gdf = gpd.read_file('https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson').set_crs(4326)
            # in the eventuality that the above URL breaks, we store a local copy
            # gdf = gpd.read_file('../data/shapefiles/resources.geojson').set_crs(4326)

            for _, row in gdf.iterrows():
                if row.geometry.intersects(aoi_4326):
                    usgs_dataset_name = row['name']
                    break

            print("Dataset being used: ", usgs_dataset_name)
            url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{usgs_dataset_name}/ept.json"
            reader = {
            "type": "readers.ept",
            "filename": url,
            "resolution": pointcloud_resolution,
            "polygon": str(aoi_3857.wkt),
            }

            # SRS associated with the 3DEP dataset
            response = requests.get(url)
            data = response.json()
            srs_wkt = data['srs']['wkt']

            pointcloud_input_crs.append(CRS.from_wkt(srs_wkt))
            readers.append(reader)

    return readers, pointcloud_input_crs


def create_pdal_pipeline(filter_low_noise=False, filter_high_noise=False,
                         filter_road=False, reset_classes=False, reclassify_ground=False,
                         return_only_ground=False, percentile_filter=False, percentile_threshold=0.95,
                         group_filter="first,only", reproject=True, save_pointcloud=False,
                         pointcloud_file = 'pointcloud', input_crs=None,
                         output_crs=None, output_type='laz'):

    assert abs(percentile_threshold) <= 1, "Percentile threshold must be in range [0, 1]"
    assert output_type in ['las', 'laz'], "Output type must be either 'las' or 'laz'"
    assert output_crs is not None, "Argument 'output_crs' must be explicitly specified!"

    stage_filter_low_noise = {
        "type":"filters.range",
        "limits":"Classification![7:7]"
    }
    stage_filter_high_noise = {
        "type":"filters.range",
        "limits":"Classification![18:18]"
    }
    stage_filter_road = {
        "type":"filters.range",
        "limits":"Classification![11:11]"
    }
    stage_reset_classes = {
        "type":"filters.assign",
        "value":"Classification = 0"
    }
    stage_reclassify_ground = {
        "type":"filters.smrf",
        # added from pdal smrf documentation, in turn from Pingel, 2013
        "scalar":1.2,
        "slope":0.2,
        "threshold":0.45,
        "window":8.0
    }
    stage_group_filter = {
        "type":"filters.returns",
        "groups":group_filter
    }
    stage_percentile_filter =  {
        "type":"filters.python",
        "script":"filter_percentile.py",
        "pdalargs": {"percentile_threshold":percentile_threshold},
        "function":"filter_percentile",
        "module":"anything"
    }
    stage_return_ground = {
        "type":"filters.range",
        "limits":"Classification[2:2]"
    }

    stage_reprojection = {
        "type":"filters.reprojection",
        "out_srs":str(output_crs)
    }
    if input_crs is not None:
        stage_reprojection["in_srs"] = str(input_crs)

    stage_save_pointcloud_las = {
        "type": "writers.las",
        "filename": f"{pointcloud_file}.las"
    }
    stage_save_pointcloud_laz = {
        "type": "writers.las",
        "compression": "true",
        "minor_version": "2",
        "dataformat_id": "0",
        "filename": f"{pointcloud_file}.laz"
    }

    # Build pipeline
    pipeline = []

    # resetting the original classifications resets
    # all point classifications to 0 (Unclassified)
    if reset_classes:
        pipeline.append(stage_reset_classes)
        if reclassify_ground:
            pipeline.append(stage_reclassify_ground)
    else:
        # we apply the percentile filter first as it
        # classifies detected outliers as 'high noise'
        if group_filter is not None:
            pipeline.append(stage_group_filter)
        if percentile_filter:
            pipeline.append(stage_percentile_filter)
        if filter_low_noise:
            pipeline.append(stage_filter_low_noise)
        if percentile_filter or filter_high_noise:
            pipeline.append(stage_filter_high_noise)
        if filter_road:
            pipeline.append(stage_filter_road)

    # For creating DTMs, we want to process only ground returns
    if return_only_ground:
        pipeline.append(stage_return_ground)

    if reproject:
        pipeline.append(stage_reprojection)

    # the pipeline can save the pointclouds to a separate file if needed
    if save_pointcloud:
        if output_type == 'laz':
            pipeline.append(stage_save_pointcloud_laz)
        else:
            pipeline.append(stage_save_pointcloud_las)

    return pipeline



def create_dem_stage(dem_filename='dem_output.tif', pointcloud_resolution=1.,
                        gridmethod='idw', dimension='Z'):
    dem_stage = {
            "type":"writers.gdal",
            "filename":dem_filename,
            "gdaldriver":'GTiff',
            "nodata":-9999,
            "output_type":gridmethod,
            "resolution":float(pointcloud_resolution),
            "gdalopts":"COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
    }

    dem_stage.update({
            'dimension':dimension
        })

    return [dem_stage]

def dem_mosaic(img_list,outfn,tr=None,tsrs=None,stats=None,tile_size=None,extent=None):
    """
    From https://github.com/uw-cryo/skysat_stereo/blob/master/skysat_stereo/asp_utils.py
    mosaic  input image list using ASP's dem_mosaic program.
    See dem_mosaic documentation here: https://stereopipeline.readthedocs.io/en/latest/tools/dem_mosaic.html
    Parameters
    ----------
    img_list: list
        List of input images to be mosaiced
    outfn: str
        Path to output mosaicked image
    tr: float/int
        target resolution of output mosaic
    t_srs: str
        target projection of output mosaic (default: EPSG:4326)
    stats: str
        metric to use for mosaicing
    tile_size: int
        tile size for distributed mosaicing (if less on memory)
    Returns
    ----------
    out: str
        dem_mosaic log
    """

    dem_mosaic_opt = []
  
    if stats:
        dem_mosaic_opt.extend(['--{}'.format(stats)])
    if tr:
        dem_mosaic_opt.extend(['--tr', str(tr)])
    if tsrs:
        dem_mosaic_opt.extend(['--t_srs', tsrs])
    if extent:
        xmin,ymin,xmax,ymax = extent.split(' ')
        dem_mosaic_opt.extend(['--t_projwin', xmin,ymin,xmax,ymax])
    dem_mosaic_args = img_list
    if tile_size:
        # will first perform tile-wise vertical mosaicing
        # then blend the result
        dem_mosaic_opt.extend(['--tile-size',str(tile_size)])
        temp_fol = os.path.splitext(outfn)[0]+'_temp'
        dem_mosaic_opt.extend(['-o',os.path.join(temp_fol,'run')])
        out_tile_op = run_cmd('dem_mosaic',dem_mosaic_args+dem_mosaic_opt)
        # query all tiles and then do simple mosaic
        #print(os.path.join(temp_fol,'run-*.tif'))
        mos_tile_list = sorted(glob.glob(os.path.join(temp_fol,'run-*.tif')))
        print(f"Found {len(mos_tile_list)}")
        # now perform simple mosaic
        dem_mos2_opt = []
        dem_mos2_opt.extend(['-o',outfn])
        dem_mos2_args = mos_tile_list
        out_fn_mos = run_cmd('dem_mosaic',dem_mos2_args+dem_mos2_opt)
        out = out_tile_op+out_fn_mos
        print("Deleting tile directory")
        shutil.rmtree(temp_fol)

    else:
        # process all at once, no tiling
        dem_mosaic_opt.extend(['-o',outfn])
        out = run_cmd('dem_mosaic',dem_mosaic_args+dem_mosaic_opt)
    return out

def run_cmd(bin, args, **kw):
    """
    From https://github.com/uw-cryo/skysat_stereo/blob/master/skysat_stereo/asp_utils.py
    wrapper around subprocess function to excute bash commands
    Parameters
    ----------
    bin: str
        command to be excuted (e.g., stereo or gdalwarp)
    args: list
        arguments to the command as a list
    Retuns
    ----------
    out: str
        log (stdout) as str if the command executed, error message if the command failed
    """
    
    #from dshean/vmap.py
    
    binpath = which(bin)
    #if binpath is None:
        #msg = ("Unable to find executable %s\n"
        #"Install ASP and ensure it is in your PATH env variable\n"
       #"https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/" % bin)
        #sys.exit(msg)
    #binpath = os.path.join('/opt/StereoPipeline/bin/',bin)
    call = [binpath,]
    if args is not None: 
        call.extend(args)
    #print(call)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


    
    

