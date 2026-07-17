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


def test_fetch_reports_stages_report_files(tmp_path, monkeypatch):
    """fetch_reports downloads only the included extensions from the
    workunit prefix, stages the project-level report and the full
    vertical_accuracy tree, inventories everything, records the staging in
    the processing metadata, and re-runs without re-downloading."""
    import requests
    import yaml

    pdir = tmp_path / "wu_a"
    pdir.mkdir()
    link = (
        "https://prd-tnm.s3.amazonaws.com/index.html?"
        "prefix=StagedProducts/Elevation/metadata/PROJ_X/wu_a"
    )
    meta_fn = pdir / "aoi_1m_wu_a-processing_metadata.yaml"
    meta_fn.write_text(
        yaml.dump({"survey_records": [{"workunit": "wu_a", "metadata_link": link}]})
    )
    (tmp_path / "batch_status.yaml").write_text(
        yaml.dump({"projects": {"wu_a": "completed"}})
    )

    ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
    pre = "StagedProducts/Elevation/metadata/PROJ_X/wu_a/"
    proj = "StagedProducts/Elevation/metadata/PROJ_X/"

    def listing(*contents, truncated=False, token=""):
        rows = "".join(
            f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>"
            for k, s in contents
        )
        nxt = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
        return (
            f'<?xml version="1.0"?><ListBucketResult {ns}>'
            f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{nxt}{rows}"
            "</ListBucketResult>"
        )

    class FakeResp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield self.content

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "list-type=2" in url:
            if "vertical_accuracy" in url:
                page = listing(
                    (f"{proj}vertical_accuracy/USGS/VA.gpkg", 3),
                    (f"{proj}vertical_accuracy/contractor_provided/jpg/M1.JPG", 9),
                )
            elif "delimiter" in url:  # project level, non-recursive
                page = listing((f"{proj}USGS_PROJ_X_Project_Report.pdf", 4))
            elif "continuation-token=tok1" in url:
                page = listing((f"{pre}reports/Survey_Report.pdf", 4))
            else:
                page = listing(
                    (f"{pre}reports/QC_Report.pdf", 4),
                    (f"{pre}reports/photos/GCP01.jpg", 2),
                    (f"{pre}reports/vendor_provided_xml/WU_CPC.xml", 6),
                    truncated=True,
                    token="tok1",
                )
            return FakeResp(page.encode())
        if url.endswith(".gpkg"):
            return FakeResp(b"GPK")
        if url.endswith(".xml"):
            return FakeResp(b"<xml/>")
        assert not url.lower().endswith(".jpg")  # photos never downloaded
        assert url.endswith(".pdf")
        return FakeResp(b"%PDF")

    monkeypatch.setattr(requests, "get", fake_get)

    survey.fetch_reports(str(tmp_path))

    outdir = pdir / "vendor_reports"
    assert (outdir / "reports/QC_Report.pdf").read_bytes() == b"%PDF"
    assert (outdir / "reports/Survey_Report.pdf").exists()
    # FGDC metadata XML staged by default: report-metrics layer 1 needs it
    assert (outdir / "reports/vendor_provided_xml/WU_CPC.xml").exists()
    assert not (outdir / "reports/photos/GCP01.jpg").exists()  # excluded ext
    assert (outdir / "project_level/USGS_PROJ_X_Project_Report.pdf").exists()
    # the vertical_accuracy tree is staged whole minus monument photos
    assert (outdir / "project_level/vertical_accuracy/USGS/VA.gpkg").exists()
    assert not (
        outdir / "project_level/vertical_accuracy/contractor_provided/jpg/M1.JPG"
    ).exists()
    inventory = (outdir / "remote_inventory.txt").read_text()
    assert "reports/photos/GCP01.jpg" in inventory  # never dropped silently
    # no temp files linger, and the pid-unique naming means an overlapping
    # run (orphaned session) can never rename this run's .part from under it
    assert not list(outdir.rglob("*.part*"))
    meta = yaml.safe_load(meta_fn.read_text())
    assert meta["vendor_reports"]["remote_objects_total"] == 7
    assert sorted(meta["vendor_reports"]["files"]) == [
        "project_level/USGS_PROJ_X_Project_Report.pdf",
        "project_level/vertical_accuracy/USGS/VA.gpkg",
        "reports/QC_Report.pdf",
        "reports/Survey_Report.pdf",
        "reports/vendor_provided_xml/WU_CPC.xml",
    ]
    # idempotent: sizes match, so a re-run lists but downloads nothing
    n_before = len(calls)
    survey.fetch_reports(str(tmp_path))
    assert not [
        c for c in calls[n_before:]
        if c.endswith((".pdf", ".gpkg", ".xml"))
    ]
