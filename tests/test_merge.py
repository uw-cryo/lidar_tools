import numpy as np
import pytest
import pyproj
import yaml

from lidar_tools import merge


def _make_mosaic(fn, value, valid_cols, origin=(500000.0, 5270000.0), size=32):
    """Product mosaic with `value` in valid_cols (slice), nodata elsewhere."""
    from osgeo import gdal, osr

    fn.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(fn), size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((origin[0], 1, 0, origin[1], 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(pyproj.CRS.from_epsg(32612).to_wkt())
    ds.SetSpatialRef(srs)
    arr = np.full((size, size), -9999.0, dtype=np.float32)
    arr[:, valid_cols] = value
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(arr)
    ds = None


def _make_batch(tmp_path, origin_b=(500000.0, 5270000.0)):
    """Two projects: A valid in cols 0-19 (value 100), B in cols 12-31 (200);
    overlap = cols 12-19. A is listed first in batch_status = priority."""
    _make_mosaic(tmp_path / "proj_a" / "aoi-DSM_mos.tif", 100.0, slice(0, 20))
    _make_mosaic(
        tmp_path / "proj_b" / "aoi-DSM_mos.tif", 200.0, slice(12, 32), origin=origin_b
    )
    with open(tmp_path / "batch_status.yaml", "w") as f:
        yaml.dump(
            {"projects": {"proj_a": "completed", "proj_b": "completed"}},
            f,
            sort_keys=False,
        )


def _read(fn):
    from osgeo import gdal

    ds = gdal.OpenEx(str(fn))
    return ds.GetRasterBand(1).ReadAsArray(), ds.GetGeoTransform()


def test_merge_priority_and_union(tmp_path):
    _make_batch(tmp_path)
    written = merge.merge_projects(tmp_path)
    assert [fn.name for fn in written] == ["aoi-DSM_mos.vrt"]
    arr, gt = _read(written[0])
    assert gt == (500000.0, 1, 0, 5270000.0, 0, -1)  # no shift, no resampling
    # priority: proj_a (listed first) wins the overlap cols 12-19
    assert (arr[:, 0:20] == 100.0).all()
    assert (arr[:, 20:32] == 200.0).all()  # union: b fills beyond a
    # metadata records the priority order
    meta = yaml.safe_load((tmp_path / "merge" / "merge_metadata.yaml").read_text())
    assert meta["priority_order"] == ["proj_a", "proj_b"]


def test_merge_explicit_priority_override(tmp_path):
    _make_batch(tmp_path)
    written = merge.merge_projects(tmp_path, workunits=["proj_b", "proj_a"])
    arr, _ = _read(written[0])
    assert (arr[:, 12:32] == 200.0).all()  # proj_b now wins the overlap


def test_merge_refuses_misaligned_grids(tmp_path):
    # half-pixel origin shift -> merging would require resampling
    _make_batch(tmp_path, origin_b=(500000.5, 5270000.0))
    with pytest.raises(ValueError, match="not on one grid"):
        merge.merge_projects(tmp_path)


def test_merge_vrt_is_portable(tmp_path):
    # sources are stored relative to the VRT so a moved batch dir still opens
    _make_batch(tmp_path)
    written = merge.merge_projects(tmp_path)
    moved = tmp_path.parent / f"{tmp_path.name}_moved"
    tmp_path.rename(moved)
    arr, _ = _read(moved / "merge" / written[0].name)
    assert (arr[:, 0:20] == 100.0).all()
