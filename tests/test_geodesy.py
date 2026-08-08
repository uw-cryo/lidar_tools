from pathlib import Path

import numpy as np
import pyproj
import pytest

from lidar_tools import dsm_functions, geodesy

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

# Casa Grande, AZ (UTM zone 12N): the site where the ellipsoid-branch
# offset was originally measured
CASA_GRANDE_LONLAT = (-111.75, 32.9)


def test_utm_zone_label():
    assert geodesy.utm_zone_label(32610) == "10N"
    assert geodesy.utm_zone_label(32719) == "19S"
    with pytest.raises(ValueError, match="not a WGS84 UTM"):
        geodesy.utm_zone_label(26910)


def test_build_utm_g2139_3d_matches_reference_wkt():
    # must be numerically identical to the hand-built notebook template that
    # the pipeline previously fetched from GitHub and text-substituted
    crs = geodesy.build_utm_g2139_3d(32610)
    template = pyproj.CRS.from_wkt(
        (NOTEBOOKS / "UTM_10N_WGS84_G2139_3D.wkt").read_text()
    )
    assert len(crs.axis_info) == 3
    assert "FRAMEEPOCH" in crs.to_wkt()  # dynamic WGS84 (G2139) datum
    src = pyproj.CRS.from_epsg(9754)  # WGS 84 (G2139) geographic 3D
    point = (-122.3, 47.6, 100.0)
    result = pyproj.Transformer.from_crs(src, crs, always_xy=True).transform(*point)
    expected = pyproj.Transformer.from_crs(src, template, always_xy=True).transform(
        *point
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_build_utm_g2139_3d_southern_hemisphere():
    # the old WKT text substitution kept the northern false northing (0)
    # for southern zones
    crs = geodesy.build_utm_g2139_3d(32719)
    params = {
        p["name"]: p["value"] for p in crs.to_json_dict()["conversion"]["parameters"]
    }
    assert params["False northing"] == 10000000
    assert params["Longitude of natural origin"] == -69  # zone 19
    assert "19S" in crs.name


def test_build_utm_nad83_2011_3d():
    # native 3DEP output datum: static NAD83(2011), ellipsoidal heights, no
    # dynamic-frame epoch (contrast build_utm_g2139_3d's FRAMEEPOCH)
    crs = geodesy.build_utm_nad83_2011_3d(32610)
    assert crs.name == "NAD83(2011) / UTM zone 10N"
    assert len(crs.axis_info) == 3
    assert crs.axis_info[-1].name == "Ellipsoidal height"
    assert crs.geodetic_crs.to_epsg() == 6319  # NAD83(2011) geographic 3D
    assert "FRAMEEPOCH" not in crs.to_wkt()  # static, unlike G2139
    # southern zones keep the correct false northing (not the northern 0)
    south = geodesy.build_utm_nad83_2011_3d(32719)
    params = {
        p["name"]: p["value"]
        for p in south.to_json_dict()["conversion"]["parameters"]
    }
    assert params["False northing"] == 10000000
    assert "19S" in south.name


def test_build_utm_target_dispatch():
    crs, name = geodesy.build_utm_target(32610, "nad83_2011")
    assert crs.name == "NAD83(2011) / UTM zone 10N"
    assert name == "UTM_10N_NAD83_2011_3D.wkt"
    crs, name = geodesy.build_utm_target(32610)  # default realization
    assert crs.name == "WGS 84 (G2139) / UTM zone 10N"
    assert name == "UTM_10N_WGS84_G2139_3D.wkt"
    with pytest.raises(ValueError, match="Unknown output_datum"):
        geodesy.build_utm_target(32610, "bogus")


def test_nad83_2011_target_is_native_no_helmert():
    # NAD83(2011) is the EPT source realization, so an ellipsoid-branch warp
    # to a NAD83(2011) target is a pure projection change: accuracy 0, no
    # grids, and (unlike the G2139 target) no ITRF Helmert in the pipeline.
    # This is why the output is static and carries no coordinate epoch.
    record = geodesy.preflight_vertical_transform(
        geodesy.build_ept_3857_nad83_2011(),
        geodesy.build_utm_nad83_2011_3d(32610),
        download=False,
    )
    assert record["accuracy_m"] == 0.0
    assert record["grids"] == []
    assert "helmert" not in record["proj_pipeline"]


def test_build_3857_navd88_compound_matches_reference():
    # equivalent to the SRS_CRS.wkt previously fetched from GitHub at runtime
    crs = geodesy.build_3857_navd88_compound()
    reference = pyproj.CRS.from_wkt((NOTEBOOKS / "SRS_CRS.wkt").read_text())
    assert crs.equals(reference)


def test_build_ept_3857_nad83_2011_axes():
    crs3d = geodesy.build_ept_3857_nad83_2011()
    assert len(crs3d.axis_info) == 3
    assert crs3d.geodetic_crs.name == "NAD83(2011)"
    crs2d = geodesy.build_ept_3857_nad83_2011(three_d=False)
    assert len(crs2d.axis_info) == 2
    # projection parameters must still be pseudo-Mercator (EPSG:3857 numbers)
    assert "Pseudo-Mercator" in crs2d.to_wkt()


def test_ellipsoid_branch_source_engages_helmert():
    # Declaring the true NAD83(2011) datum of EPT "3857" coordinates must
    # engage the ITRF<->NAD83(2011) Helmert; the previous plain EPSG:3857
    # source hit a null ensemble tie that relabeled the coordinates,
    # leaving them ~1.3 m horizontal / ~0.9 m vertical off at Casa Grande.
    x, y = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(
        *CASA_GRANDE_LONLAT
    )
    dst = geodesy.build_utm_g2139_3d(32612)
    fixed = pyproj.Transformer.from_crs(
        geodesy.build_ept_3857_nad83_2011(), dst, always_xy=True
    ).transform(x, y, 400.0)
    naive = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(3857).to_3d(), dst, always_xy=True
    ).transform(x, y, 400.0)
    dh = float(np.hypot(fixed[0] - naive[0], fixed[1] - naive[1]))
    dz = fixed[2] - naive[2]
    assert 0.8 < dh < 2.0
    assert -1.3 < dz < -0.5


def test_write_crs_file_roundtrip(tmp_path):
    crs = geodesy.build_utm_g2139_3d(32611)
    outfn = geodesy.write_crs_file(crs, tmp_path / "UTM_11N_WGS84_G2139_3D.wkt")
    assert pyproj.CRS.from_wkt(Path(outfn).read_text()).equals(crs)


def test_build_ept_3857_base_realization_parameterized():
    # older surveys are HARN/NSRS2007-based: the declared datum must be
    # selectable per survey, not hard-coded to NAD83(2011)
    harn = geodesy.build_ept_3857_nad83_2011(base_epsg=4152)  # NAD83(HARN)
    assert "HARN" in harn.name
    assert harn.geodetic_crs.name == "NAD83(HARN)"


def test_geographic_base_epsg():
    # declared horizontal CRS -> validated NAD83-family geographic base
    assert geodesy.geographic_base_epsg(7131) == 6318  # NAD83(2011)/SP CA-3 ftUS
    assert geodesy.geographic_base_epsg(6521) == 6318  # NAD83(2011)/NV East ftUS
    assert geodesy.geographic_base_epsg("26910") == 4269  # NAD83(1986)/UTM 10N
    assert geodesy.geographic_base_epsg(3740) == 4152  # NAD83(HARN)/UTM 10N
    # Pacific-plate PA11: the North-America Helmert is wrong there
    with pytest.raises(ValueError, match="NAD83-family"):
        geodesy.geographic_base_epsg(6322)  # NAD83(PA11) geographic
    with pytest.raises(ValueError, match="NAD83-family"):
        geodesy.geographic_base_epsg(32610)  # WGS84 UTM


def test_geoid_grid_hint():
    assert geodesy.geoid_grid_hint("GEOID18") == "g2018"
    assert geodesy.geoid_grid_hint("GEOID12B") == "g2012b"
    assert geodesy.geoid_grid_hint("geoid 09") == "geoid09"
    assert geodesy.geoid_grid_hint("Unknown") is None
    assert geodesy.geoid_grid_hint(None) is None


def test_preflight_aoi_scoping():
    src = geodesy.build_ept_3857_nad83_2011()
    dst = geodesy.build_utm_g2139_3d(32612)
    # CONUS AOI: passes and records the operation's area of use
    record = geodesy.preflight_vertical_transform(
        src, dst, download=False, aoi_bounds=(-112.0, 32.7, -111.5, 33.1)
    )
    assert record["area_of_use"]
    # AOI far outside any candidate operation's validity: refuse to select
    # a pipeline that would be enforced out-of-area (and never fall back
    # to a world-scope ballpark operation)
    with pytest.raises(RuntimeError, match="area of use|no non-ballpark"):
        geodesy.preflight_vertical_transform(
            src, dst, download=False, aoi_bounds=(130.0, -30.0, 135.0, -25.0)
        )


def test_preflight_vertical_transform_provenance():
    # NAD83(2011)-based source -> G2139 UTM needs only the Helmert from
    # proj.db (no grids), so this must pass offline and return provenance
    record = geodesy.preflight_vertical_transform(
        geodesy.build_ept_3857_nad83_2011(),
        geodesy.build_utm_g2139_3d(32612),
        download=False,
    )
    assert record["source_crs"] == "Pseudo-Mercator (NAD83(2011) based)"
    assert record["proj_pipeline"].startswith("proj=pipeline")
    # the whole point: the time-dependent Helmert must be in the chain
    assert "helmert" in record["proj_pipeline"]
    assert record["grids"] == []
    assert record["description"]


def test_preflight_vertical_transform_missing_grid(monkeypatch):
    # simulate PROJ reporting the best transformation unavailable due to a
    # missing geoid grid: must raise loudly, never continue toward a
    # silently unshifted (~31 m off) product
    class FakeGrid:
        short_name = "us_noaa_gfake.tif"
        full_name = ""
        available = False

    class FakeOperation:
        grids = [FakeGrid()]

    class FakeTransformerGroup:
        def __init__(self, *args, **kwargs):
            self.best_available = False
            self.transformers = []
            self.unavailable_operations = [FakeOperation()]

    monkeypatch.setattr(geodesy, "TransformerGroup", FakeTransformerGroup)
    with pytest.raises(RuntimeError, match="us_noaa_gfake.tif"):
        geodesy.preflight_vertical_transform(
            "EPSG:4326", "EPSG:4326", download=False
        )


def _make_raster(fn, crs_wkt, size=64):
    from osgeo import gdal, osr

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(fn), size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((500000, 1, 0, 5270000, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    ds.SetSpatialRef(srs)
    ds.GetRasterBand(1).WriteArray(np.ones((size, size), dtype=np.float32))
    ds = None


def _read_epoch(fn):
    from osgeo import gdal

    with gdal.OpenEx(str(fn)) as ds:
        return ds.GetSpatialRef().GetCoordinateEpoch()


def test_set_coordinate_epoch_dynamic_crs(tmp_path):
    fn = tmp_path / "dynamic.tif"
    _make_raster(fn, geodesy.build_utm_g2139_3d(32610).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is True
    assert _read_epoch(fn) == geodesy.DEFAULT_COORDINATE_EPOCH


def test_set_coordinate_epoch_2d_crs_via_authoritative(tmp_path):
    # the GeoTIFF round-trip drops the DYNAMIC property of the custom-datum
    # 2D demotion (file SRS reads back static): passing the authoritative
    # CRS must still stamp (this is the intensity-product case)
    crs2d = geodesy.build_utm_g2139_3d(32610).to_2d()
    fn = tmp_path / "intensity2d.tif"
    _make_raster(fn, crs2d.to_wkt())
    assert geodesy.set_coordinate_epoch(fn, crs=crs2d) is True
    assert _read_epoch(fn) == geodesy.DEFAULT_COORDINATE_EPOCH


def test_set_coordinate_epoch_static_crs_noop(tmp_path):
    # static plate-fixed CRS (NAD83(2011) UTM 12N): no epoch applies
    fn = tmp_path / "static.tif"
    _make_raster(fn, pyproj.CRS.from_epsg(6341).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is False
    assert _read_epoch(fn) == 0.0  # unset


def test_set_coordinate_epoch_built_nad83_2011_utm_noop(tmp_path):
    # the programmatically built NAD83(2011) UTM target is static: no epoch
    # is stamped, even through the GeoTIFF round-trip or when the
    # authoritative CRS is passed (the intensity-product path)
    fn = tmp_path / "nad83_utm.tif"
    _make_raster(fn, geodesy.build_utm_nad83_2011_3d(32610).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is False
    assert (
        geodesy.set_coordinate_epoch(
            fn, crs=geodesy.build_utm_nad83_2011_3d(32610).to_2d()
        )
        is False
    )
    assert _read_epoch(fn) == 0.0  # unset


def test_intensity_warp_helmert_without_value_shift(tmp_path):
    # The intensity finalize warp must (a) never touch band values — the
    # compound source's vertical leg subtracted ~28 m from UInt16 DNs
    # (issue #70) — and (b) still apply the ITRF<->NAD83(2011) horizontal
    # Helmert (E1: EPT "3857" numbers are null-tie NAD83(2011) values).
    # A 2D NAD83(2011)-based source gives both in a single warp.
    from osgeo import gdal, osr

    from lidar_tools import dsm_functions

    # step-edge UInt16 raster in EPSG:3857 near Las Vegas
    src_fn = tmp_path / "step3857.tif"
    n = 200
    arr = np.full((n, n), 100, dtype=np.uint16)
    arr[:, n // 2 :] = 200
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(src_fn), n, n, 1, gdal.GDT_UInt16)
    ds.SetGeoTransform((-12818000, 1, 0, 4325000, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    ds.SetSpatialRef(srs)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None

    # 2D target: a 3D target promotes the source and applies the ~-0.7 m
    # Helmert dz to the band values (200 -> 199 after UInt16 truncation)
    dst_crs_fn = geodesy.write_crs_file(
        geodesy.build_utm_g2139_3d(32611).to_2d(), tmp_path / "utm11_g2139_2d.wkt"
    )
    intensity_src = geodesy.write_crs_file(
        geodesy.build_ept_3857_nad83_2011(three_d=False), tmp_path / "src2d.wkt"
    )

    def warp(src_srs, out_fn, ct=None):
        dsm_functions.gdal_warp(
            str(src_fn), str(out_fn), src_srs, dst_crs_fn,
            res=1.0, dtype="UInt16", resampling_alogrithm="nearest",
            coordinate_operation=ct,
        )
        with gdal.OpenEx(str(out_fn)) as ds:
            a = ds.GetRasterBand(1).ReadAsArray()
            gt = ds.GetGeoTransform()
        return a, gt

    # GDAL's own operation selection silently uses the null tie for this
    # horizontal-only pair: the explicit pipeline (as selected/recorded by
    # the preflight) is required to engage the Helmert
    check = geodesy.preflight_vertical_transform(
        geodesy.build_ept_3857_nad83_2011(three_d=False),
        geodesy.build_utm_g2139_3d(32611).to_2d(),
        download=False,
    )
    fixed, gt_f = warp(
        intensity_src, tmp_path / "fixed.tif", ct=check["proj_pipeline"]
    )
    naive, gt_n = warp("EPSG:3857", tmp_path / "naive.tif")

    # (a) values pass through exactly: no vertical leg can touch them
    assert set(np.unique(fixed)) <= {0, 100, 200}

    def edge_x(a, gt):
        cols = (a == 200).sum(axis=0)
        edge_col = int(np.argmax(cols > (a != 0).sum(axis=0).max() * 0.5))
        return gt[0] + edge_col * gt[1]

    # (b) Helmert engaged: the datum-declared warp places the same content
    # ~1.3 m west of the null-tie warp (null coordinates sit +1.26 m east
    # of truth at Las Vegas)
    shift_east = edge_x(fixed, gt_f) - edge_x(naive, gt_n)
    assert -2.0 < shift_east < -0.5


def _make_const_float_raster(fn, value=700.0, size=80):
    from osgeo import gdal, osr

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(fn), size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((-12818000, 1, 0, 4325000, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    ds.SetSpatialRef(srs)
    ds.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype=np.float32))
    ds = None


def _warped_height_shift(tmp_path, src_crs, name):
    # warp a constant-height raster through the enforced pipeline and
    # return (band shift, pyproj-predicted shift)
    from osgeo import gdal

    from lidar_tools import dsm_functions

    src_fn = tmp_path / f"const_{name}.tif"
    _make_const_float_raster(src_fn)
    dst = geodesy.build_utm_g2139_3d(32611)
    dst_fn = geodesy.write_crs_file(dst, tmp_path / f"dst_{name}.wkt")
    src_wkt_fn = geodesy.write_crs_file(src_crs, tmp_path / f"src_{name}.wkt")
    check = geodesy.preflight_vertical_transform(src_crs, dst, download=False)
    out_fn = tmp_path / f"out_{name}.tif"
    dsm_functions.gdal_warp(
        str(src_fn), str(out_fn), src_wkt_fn, dst_fn,
        res=1.0, resampling_alogrithm="nearest",
        coordinate_operation=check["proj_pipeline"],
    )
    with gdal.OpenEx(str(out_fn)) as ds:
        a = ds.GetRasterBand(1).ReadAsArray()
    v = a[np.isfinite(a) & (a != 0)]
    predicted = (
        pyproj.Transformer.from_crs(src_crs, dst, always_xy=True).transform(
            -12817960.0, 4324960.0, 700.0
        )[2]
        - 700.0
    )
    return float(np.median(v) - 700.0), predicted


def test_height_warp_ellipsoid_branch_enforced_pipeline(tmp_path):
    # the 3D ellipsoid-branch warp must apply the Helmert dz to band values
    # (grid-free: runs everywhere)
    shift, predicted = _warped_height_shift(
        tmp_path, geodesy.build_ept_3857_nad83_2011(), "ellipsoid"
    )
    assert abs(shift - predicted) < 0.01
    assert -1.0 < shift < -0.4  # ~-0.7 m at Las Vegas


def test_height_warp_geoid_branch_enforced_pipeline(tmp_path):
    # the compound-source warp must apply the full geoid + Helmert shift to
    # band values through the enforced pipeline (needs the GEOID18 grid)
    src = geodesy.build_3857_navd88_compound()
    try:
        geodesy.preflight_vertical_transform(
            src, geodesy.build_utm_g2139_3d(32611), download=False
        )
    except RuntimeError:
        pytest.skip("GEOID grid not available offline")
    shift, predicted = _warped_height_shift(tmp_path, src, "geoid")
    assert abs(shift - predicted) < 0.02
    assert -32.0 < shift < -26.0  # ~-28.8 m at Las Vegas


def test_coordinate_epoch_survives_overviews(tmp_path):
    # the pipeline stamps before gdal_add_overview; the COG translate in
    # there must carry the epoch through to the final product
    fn = tmp_path / "dsm.tif"
    _make_raster(fn, geodesy.build_utm_g2139_3d(32610).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is True
    dsm_functions.gdal_add_overview(str(fn))
    assert _read_epoch(fn) == geodesy.DEFAULT_COORDINATE_EPOCH
