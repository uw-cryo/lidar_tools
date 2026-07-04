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


def test_set_coordinate_epoch_static_crs_noop(tmp_path):
    # static plate-fixed CRS (NAD83(2011) UTM 12N): no epoch applies
    fn = tmp_path / "static.tif"
    _make_raster(fn, pyproj.CRS.from_epsg(6341).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is False
    assert _read_epoch(fn) == 0.0  # unset


def test_coordinate_epoch_survives_overviews(tmp_path):
    # the pipeline stamps before gdal_add_overview; the COG translate in
    # there must carry the epoch through to the final product
    fn = tmp_path / "dsm.tif"
    _make_raster(fn, geodesy.build_utm_g2139_3d(32610).to_wkt())
    assert geodesy.set_coordinate_epoch(fn) is True
    dsm_functions.gdal_add_overview(str(fn))
    assert _read_epoch(fn) == geodesy.DEFAULT_COORDINATE_EPOCH
