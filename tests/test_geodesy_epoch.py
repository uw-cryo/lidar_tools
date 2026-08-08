"""Coordinate-epoch support + unambiguous-transform-routing regression tests.

Baseline captured 2026-07-10 (PROJ 9.7.1 / GDAL 3.12 / PDAL 2.9.3) during the
Las Vegas four-frame CRS/epoch validation. These encode routing facts that can
silently change as PROJ's EPSG database evolves — if one starts failing,
re-audit operation selection before trusting outputs (the failure modes are
silent: a null NAD83<->WGS84 tie instead of the time-dependent Helmert, or a
GEOID03/NADCON5 chain instead of the survey's GEOID18).
"""

import pyproj
import pytest
from pyproj.transformer import TransformerGroup

from lidar_tools import dsm_functions, geodesy

# Las Vegas AOI (degrees) — matches the validation site
AOI = pyproj.aoi.AreaOfInterest(-115.85, 35.66, -114.85, 36.66)


def _ops(tg):
    """All candidate operations, available or not (grid-less CI safe)."""
    return list(tg.transformers) + list(tg.unavailable_operations)


def test_output_datum_builders_are_3d_utm():
    for key, (builder, label) in geodesy.OUTPUT_DATUM_BUILDERS.items():
        crs = builder(32611)
        assert len(crs.axis_info) == 3, key
        assert "UTM zone 11N" in crs.name, key


def test_itrf_alias_builders_names():
    assert geodesy.build_utm_itrf2008_3d(32611).name.startswith("ITRF2008")
    assert geodesy.build_utm_itrf2014_3d(32611).name.startswith("ITRF2014")
    assert geodesy.build_utm_itrf2020_3d(32611).name.startswith("ITRF2020")


def test_navd88_compound_declares_true_base_datum():
    comp = geodesy.build_ept_3857_navd88_compound()
    assert comp.is_compound
    assert "NAD83(2011)" in comp.name
    assert "NAVD88" in comp.name


def test_navd88_route_collapses_with_true_base_datum():
    """NAD83(2011)-declared compound -> NAD83(2011) 3D must be a SINGLE
    operation (the survey-consistent GEOID18 route), never a ranking contest
    with GEOID03/NADCON5 chains."""
    comp = geodesy.build_ept_3857_navd88_compound()
    tg = TransformerGroup(comp, "EPSG:6319", area_of_interest=AOI,
                          allow_ballpark=False, always_xy=True)
    ops = _ops(tg)
    assert len(ops) == 1
    if tg.transformers:  # grid installed: assert it is the GEOID18 grid
        assert "g2018" in tg.transformers[0].definition


def test_navd88_route_geoid18_ranks_first_for_itrf_target():
    comp = geodesy.build_ept_3857_navd88_compound()
    tgt = pyproj.CRS.from_epsg(9000).to_3d()  # ITRF2014 3D
    tg = TransformerGroup(comp, tgt, area_of_interest=AOI,
                          allow_ballpark=False, always_xy=True)
    if tg.transformers:
        assert "g2018" in tg.transformers[0].definition


def test_legacy_wgs84_compound_ambiguity_baseline():
    """The legacy WGS84-based compound is the documented ambiguity trap
    (multi-candidate GEOID03/09/1999/18 zoo). If this collapses after a PROJ
    upgrade, the routing landscape changed — re-run the raster validation."""
    legacy = geodesy.build_3857_navd88_compound()
    tg = TransformerGroup(legacy, "EPSG:6319", area_of_interest=AOI,
                          allow_ballpark=False, always_xy=True)
    assert len(_ops(tg)) > 1


def test_epoch_pinned_pipeline_lv_itrf2008():
    """The pinned pipeline must carry the epoch bookends, the survey geoid,
    and the time-dependent Helmert — the exact contract the warps enforce."""
    comp = geodesy.build_ept_3857_navd88_compound()
    dst = geodesy.build_utm_itrf2008_3d(32611)
    pipe = geodesy.epoch_pinned_pipeline(
        comp, dst, 2005.0, aoi_bounds=(-115.85, 35.66, -114.85, 36.66),
        require_substrings=["+proj=helmert", "vgridshift"])
    assert "+proj=set +v_4=2005" in pipe
    assert "g2018u0" in pipe
    assert "+proj=utm +zone=11" in pipe


def test_epoch_pinned_pipeline_rejects_missing_component():
    comp = geodesy.build_ept_3857_navd88_compound()
    dst = geodesy.build_utm_itrf2008_3d(32611)
    with pytest.raises(RuntimeError, match="required component"):
        geodesy.epoch_pinned_pipeline(
            comp, dst, 2005.0, aoi_bounds=(-115.85, 35.66, -114.85, 36.66),
            require_substrings=["this_should_not_be_there"])


def test_gdal_warp_epoch_and_ct_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        dsm_functions.gdal_warp(
            "in.tif", "out.tif", "EPSG:6319", "EPSG:9989",
            coordinate_operation="+proj=pipeline",
            coord_epoch=2010.0,
        )
