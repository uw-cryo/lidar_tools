import geopandas as gpd
import numpy as np
import shapely

from lidar_tools import survey


def _square(x0, y0, x1, y1):
    return shapely.box(x0, y0, x1, y1)


def _aoi():
    return gpd.GeoDataFrame(geometry=[_square(0, 0, 1, 1)], crs="EPSG:4326")


def _wesm():
    # collection A covers the west half, collection B the full AOI
    return gpd.GeoDataFrame(
        {
            "workunit": ["A", "B"],
            "ql": ["QL 1", "QL 2"],
            "collect_end": ["2022-01-01", "2016-01-01"],
            "vert_crs": ["5703", "6360"],
        },
        geometry=[_square(-1, -1, 0.5, 2), _square(-1, -1, 2, 2)],
        crs="EPSG:4326",
    )


def test_summarize_surveys_overlap_and_ept():
    ept = gpd.GeoDataFrame(
        {"name": ["A_ept"]},
        geometry=[_square(-1, -1, 0.5, 2)],  # covers A fully, B's west half
        crs="EPSG:4326",
    )
    out = survey.summarize_surveys(_wesm(), _aoi(), ept)
    assert list(out["workunit"]) == ["A", "B"]  # sorted QL then recency
    a = out[out.workunit == "A"].iloc[0]
    b = out[out.workunit == "B"].iloc[0]
    np.testing.assert_allclose(a["aoi_overlap_frac"], 0.5)
    np.testing.assert_allclose(b["aoi_overlap_frac"], 1.0)
    # EPT coverage is fraction of the collection's AOI footprint
    np.testing.assert_allclose(a["ept_coverage_frac"], 1.0)
    np.testing.assert_allclose(b["ept_coverage_frac"], 0.5)
    assert a["ept_names"] == ["A_ept"]


def test_summarize_surveys_no_intersection():
    aoi = gpd.GeoDataFrame(geometry=[_square(10, 10, 11, 11)], crs="EPSG:4326")
    out = survey.summarize_surveys(_wesm(), aoi)
    assert out.empty


def test_record_from_wesm():
    import pytest

    wesm = _wesm()
    rec = survey.record_from_wesm(wesm, "A")
    assert rec["workunit"] == "A"
    assert rec["ql"] == "QL 1"
    assert rec["vert_crs"] == "5703"
    with pytest.raises(ValueError, match="not found"):
        survey.record_from_wesm(wesm, "NOPE")


def test_coverage_gaps():
    # only collection A (west half) selected: east half is a gap
    selected = survey.summarize_surveys(_wesm().iloc[[0]], _aoi())
    gaps = survey.coverage_gaps(selected, _aoi())
    assert len(gaps) == 1
    np.testing.assert_allclose(gaps["gap_frac"].iloc[0], 0.5)
    # both collections: no gap
    all_sel = survey.summarize_surveys(_wesm(), _aoi())
    assert survey.coverage_gaps(all_sel, _aoi()).empty
    # nothing selected: the whole AOI is the gap
    empty = survey.summarize_surveys(_wesm(), _aoi()).iloc[0:0]
    gaps = survey.coverage_gaps(empty, _aoi())
    np.testing.assert_allclose(gaps["gap_frac"].sum(), 1.0)
