
import numpy as np
import pyproj

from lidar_tools import preview


def _make_product(fn, value=100.0, nodata_frac=0.25, size=64):
    from osgeo import gdal, osr

    ds = gdal.GetDriverByName("GTiff").Create(str(fn), size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((500000, 1, 0, 5270000, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(pyproj.CRS.from_epsg(32612).to_wkt())
    ds.SetSpatialRef(srs)
    arr = np.full((size, size), value, dtype=np.float32)
    arr[: int(size * nodata_frac)] = -9999.0
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(arr)
    ds = None


def test_product_preview(tmp_path):
    _make_product(tmp_path / "aoi-DSM_mos.tif", 105.0)
    _make_product(tmp_path / "aoi-DTM_no_fill_mos.tif", 100.0)
    _make_product(tmp_path / "aoi-intensity_mos.tif", 30000.0)
    out = preview.product_preview(tmp_path)
    assert out == tmp_path / "preview.png"
    assert out.exists() and out.stat().st_size > 0


def test_product_preview_no_products(tmp_path):
    assert preview.product_preview(tmp_path) is None


def test_preview_batch_dir(tmp_path, capsys):
    # batch layout: product mosaics live in per-project subdirectories
    proj = tmp_path / "AZ_Project_1"
    proj.mkdir()
    _make_product(proj / "aoi-DSM_mos.tif")
    preview.preview(str(tmp_path))
    assert (proj / "preview.png").exists()
    assert "Wrote preview" in capsys.readouterr().out
