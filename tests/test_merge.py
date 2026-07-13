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


def _make_intensity(fn, valid_cols, dn_lo, dn_hi, size=64):
    """UInt16 intensity mosaic (nodata 0): a horizontal 'ground' ramp mapped
    linearly into [dn_lo, dn_hi] within valid_cols — two surveys of the same
    ground differ only by their linear DN scale."""
    from osgeo import gdal, osr

    fn.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(fn), size, size, 1, gdal.GDT_UInt16)
    ds.SetGeoTransform((500000, 1, 0, 5270000, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(pyproj.CRS.from_epsg(32612).to_wkt())
    ds.SetSpatialRef(srs)
    ground = np.tile(np.linspace(0.0, 1.0, size), (size, 1))  # same everywhere
    arr = np.zeros((size, size), dtype=np.uint16)
    arr[:, valid_cols] = (dn_lo + ground * (dn_hi - dn_lo))[:, valid_cols].astype(
        np.uint16
    )
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(arr)
    band.GetStatistics(0, 1)
    ds = None


def _make_intensity_batch(tmp_path):
    """A (10000-40000 DN, cols 0-39) + B (45000-55000 DN, cols 24-63):
    overlap cols 24-39, same underlying ground ramp."""
    _make_intensity(
        tmp_path / "proj_a" / "aoi-intensity_mos.tif", slice(0, 40), 10000, 40000
    )
    _make_intensity(
        tmp_path / "proj_b" / "aoi-intensity_mos.tif", slice(24, 64), 45000, 55000
    )
    with open(tmp_path / "batch_status.yaml", "w") as f:
        yaml.dump(
            {"projects": {"proj_a": "completed", "proj_b": "completed"}},
            f,
            sort_keys=False,
        )


def test_merge_intensity_normalized(tmp_path, monkeypatch):
    from osgeo import gdal

    monkeypatch.setattr(merge, "MIN_OVERLAP_PX", 100)  # tiny synthetic overlap
    _make_intensity_batch(tmp_path)
    written = merge.merge_projects(tmp_path)
    ds = gdal.OpenEx(str(written[0]))
    band = ds.GetRasterBand(1)
    assert gdal.GetDataTypeName(band.DataType) == "Float32"
    assert band.GetNoDataValue() == merge.NORM_NODATA
    arr = band.ReadAsArray()
    valid = arr != merge.NORM_NODATA
    # the LUT clamps every source to the common target range: same-ground
    # matching would let un-shared terrain extend beyond it (B's brightest
    # ground exists only outside the overlap) and a mis-assigned map lands
    # entire spans away (the basename-collision bug sat 5+ spans below lo)
    lo, hi = merge.INTENSITY_TARGET
    span = hi - lo
    assert valid[:, 0:64].all()  # union coverage
    assert arr[valid].min() >= lo
    assert arr[valid].max() <= hi
    # same ground -> same normalized value across the seam: compare the
    # a-only region (col 20) with the b-only region at the same ground value
    # via the shared ramp: ground(col) is identical for all rows
    a_only = arr[:, 20].mean()
    # ground value at col 20 appears in b-only territory nowhere (cols differ)
    # so instead check overlap agreement: b was painted only where a is absent,
    # but the fitted maps must agree in the overlap cols
    meta = yaml.safe_load((tmp_path / "merge" / "merge_metadata.yaml").read_text())
    norm = meta["products"]["intensity_mos"]["intensity_normalization"]
    assert [s["method"] for s in norm["sources"]] == [
        "global-stretch",
        "overlap-refined",
    ]
    ga, oa = norm["sources"][0]["gain"], norm["sources"][0]["offset"]
    gb, ob = norm["sources"][1]["gain"], norm["sources"][1]["offset"]
    # the two linear maps must send the same ground DN pair to ~equal values
    # (ground g: a-DN = 10000+g*30000, b-DN = 45000+g*10000)
    for g in (0.4, 0.5, 0.6):
        va = ga * (10000 + g * 30000) + oa
        vb = gb * (45000 + g * 10000) + ob
        assert abs(va - vb) < 0.02 * span, (g, va, vb)
    assert a_only > lo  # sanity: normalized values live in target space


def test_merge_intensity_clamped_outliers_stay_valid(tmp_path, monkeypatch):
    """A radiometric regime the overlap fit never saw (dark lift block, like
    the full-AOI PimaCo_2) clamps to the target floor — it must not land
    spans below the range (washing out every default display stretch) nor
    hit NORM_NODATA and vanish."""
    from osgeo import gdal

    monkeypatch.setattr(merge, "MIN_OVERLAP_PX", 100)
    _make_intensity_batch(tmp_path)
    # dark block in proj_b's exclusive region, far below its fitted DNs
    fn = tmp_path / "proj_b" / "aoi-intensity_mos.tif"
    ds = gdal.OpenEx(str(fn), gdal.OF_UPDATE)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    arr[:, 56:64] = 300
    band.WriteArray(arr)
    ds = None
    written = merge.merge_projects(tmp_path)
    ds_out = gdal.OpenEx(str(written[0]))
    out = ds_out.GetRasterBand(1).ReadAsArray()
    lo, hi = merge.INTENSITY_TARGET
    assert (out[:, 56:64] == lo).all()  # clamped to the floor, still valid
    valid = out != merge.NORM_NODATA
    assert valid[:, 0:64].all()  # no holes
    assert out[valid].max() <= hi


def test_merge_intensity_normalization_off(tmp_path):
    from osgeo import gdal

    _make_intensity_batch(tmp_path)
    written = merge.merge_projects(tmp_path, normalize_intensity=False)
    ds = gdal.OpenEx(str(written[0]))
    assert gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) == "UInt16"
    meta = yaml.safe_load((tmp_path / "merge" / "merge_metadata.yaml").read_text())
    assert "intensity_normalization" not in meta["products"]["intensity_mos"]


def test_merge_intensity_single_source_stays_raw(tmp_path):
    from osgeo import gdal

    _make_intensity(
        tmp_path / "proj_a" / "aoi-intensity_mos.tif", slice(0, 64), 10000, 40000
    )
    with open(tmp_path / "batch_status.yaml", "w") as f:
        yaml.dump({"projects": {"proj_a": "completed"}}, f, sort_keys=False)
    written = merge.merge_projects(tmp_path)
    ds = gdal.OpenEx(str(written[0]))
    assert gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) == "UInt16"
