import geopandas as gpd
import numpy as np
import shapely

from lidar_tools import catalog


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
    out = catalog.summarize_surveys(_wesm(), _aoi(), ept)
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
    out = catalog.summarize_surveys(_wesm(), aoi)
    assert out.empty


def test_record_from_wesm():
    import pytest

    wesm = _wesm()
    rec = catalog.record_from_wesm(wesm, "A")
    assert rec["workunit"] == "A"
    assert rec["ql"] == "QL 1"
    assert rec["vert_crs"] == "5703"
    with pytest.raises(ValueError, match="not found"):
        catalog.record_from_wesm(wesm, "NOPE")


def test_coverage_gaps():
    # only collection A (west half) selected: east half is a gap
    selected = catalog.summarize_surveys(_wesm().iloc[[0]], _aoi())
    gaps = catalog.coverage_gaps(selected, _aoi())
    assert len(gaps) == 1
    np.testing.assert_allclose(gaps["gap_frac"].iloc[0], 0.5)
    # both collections: no gap
    all_sel = catalog.summarize_surveys(_wesm(), _aoi())
    assert catalog.coverage_gaps(all_sel, _aoi()).empty
    # nothing selected: the whole AOI is the gap
    empty = catalog.summarize_surveys(_wesm(), _aoi()).iloc[0:0]
    gaps = catalog.coverage_gaps(empty, _aoi())
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
            f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in contents
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

    catalog.fetch_reports(str(tmp_path))

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
    catalog.fetch_reports(str(tmp_path))
    assert not [c for c in calls[n_before:] if c.endswith((".pdf", ".gpkg", ".xml"))]


def test_resolve_ept_resource_tiers():
    import pandas as pd

    ept = pd.DataFrame(
        {
            "name": [
                "NV_ClarkCo_2_B22",
                "NV_LasVegasValley_2010",
                "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018",
                "AK_Fairbanks-NSBorough_2010",
                "ARRA-AK_EkluntaGlacier_2010",
            ],
            "count": [10, 20, 30, 40, 50],
        }
    )
    # tier 1: exact
    r = catalog.resolve_ept_resource("NV_ClarkCo_2_B22", ept)
    assert (r["ept_name"], r["tier"]) == ("NV_ClarkCo_2_B22", 1)
    # tier 2: WESM legacy ALL-CAPS vs mixed-case EPT build
    r = catalog.resolve_ept_resource("NV_LASVEGASVALLEY_2010", ept)
    assert (r["ept_name"], r["tier"]) == ("NV_LasVegasValley_2010", 2)
    # tier 3: FTP-era USGS_LPC_/_LAS_<yr> wrapper
    r = catalog.resolve_ept_resource("NV_LasVegas_QL2_2016", ept)
    assert (r["ept_name"], r["tier"]) == (
        "USGS_LPC_NV_LasVegas_QL2_2016_LAS_2018",
        3,
    )
    # tier 4: hyphen/underscore drift (WESM legacy names are underscored caps)
    r = catalog.resolve_ept_resource("AK_FAIRBANKS_NSBOROUGH_2010", ept)
    assert (r["ept_name"], r["tier"]) == ("AK_Fairbanks-NSBorough_2010", 4)
    # tier 4 must also cover the ARRA drift: WESM spells the funding prefix
    # ARRA_ (underscore), the EPT build spells it ARRA- (hyphen) — hyphen
    # folding on BOTH sides resolves the pair without stripping the prefix
    r = catalog.resolve_ept_resource("ARRA_AK_EKLUNTAGLACIER_2010", ept)
    assert (r["ept_name"], r["tier"]) == ("ARRA-AK_EkluntaGlacier_2010", 4)
    # and the un-stripped prefix must NOT capture an unrelated workunit
    import pytest

    with pytest.raises(LookupError):
        catalog.resolve_ept_resource("AK_EKLUNTAGLACIER_2010", ept)


def test_resolve_ept_resource_count_tiebreak_and_tier_precedence():
    import pandas as pd

    # re-released build: same workunit reachable at the same tier twice ->
    # the larger build wins the tie
    ept = pd.DataFrame(
        {
            "name": [
                "USGS_LPC_X_Co_2016_LAS_2016",
                "USGS_LPC_X_Co_2016_LAS_2018",
            ],
            "count": [5, 500],
        }
    )
    r = catalog.resolve_ept_resource("X_Co_2016", ept)
    assert r["ept_name"] == "USGS_LPC_X_Co_2016_LAS_2018"  # larger build
    assert sorted(r["candidates"]) == sorted(ept["name"])
    # tier precedence: an exact-name build short-circuits at tier 1, so a
    # larger re-release at a later tier is never even considered
    ept2 = pd.DataFrame(
        {"name": ["X_Co_2016", "USGS_LPC_X_Co_2016_LAS_2018"], "count": [5, 500]}
    )
    r2 = catalog.resolve_ept_resource("X_Co_2016", ept2)
    assert (r2["ept_name"], r2["tier"]) == ("X_Co_2016", 1)


def test_resolve_ept_resource_unresolvable_raises():
    import pandas as pd
    import pytest

    ept = pd.DataFrame(
        {"name": ["NV_Southern_5_D23", "CA_MountainPass_B1_2019"], "count": [1, 2]}
    )
    with pytest.raises(LookupError, match="NV_Southern_4_D23"):
        catalog.resolve_ept_resource("NV_Southern_4_D23", ept)
    # the message points at the fallback path and lists same-state names
    with pytest.raises(LookupError, match="staged-LAZ"):
        catalog.resolve_ept_resource("NV_Southern_4_D23", ept)


def test_fetch_reports_survives_listing_failure(tmp_path, monkeypatch):
    """A transient S3 listing failure skips the workunit — it must not
    truncate the previous run's remote_inventory.txt or abort the batch."""
    import requests
    import yaml

    pdir = tmp_path / "wu_a"
    outdir = pdir / "vendor_reports"
    outdir.mkdir(parents=True)
    inv = outdir / "remote_inventory.txt"
    inv.write_text("           4 reports/QC_Report.pdf\n")
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

    def fake_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(requests, "get", fake_get)
    catalog.fetch_reports(str(tmp_path))  # must not raise

    assert inv.read_text() == "           4 reports/QC_Report.pdf\n"
    assert not list(outdir.rglob("*.part*"))
    # staging metadata untouched: no vendor_reports section claiming success
    meta = yaml.safe_load(meta_fn.read_text())
    assert "vendor_reports" not in meta


def test_area_fractions_equal_area_not_degrees():
    """Coverage fractions must ratio true areas: the northern half of a
    10-degree-tall Alaska AOI is ~45% of its area, not the 50% a raw
    degree-squared ratio reports."""
    aoi = gpd.GeoDataFrame(geometry=[_square(-150, 60, -140, 70)], crs="EPSG:4326")
    north = gpd.GeoDataFrame(
        {"workunit": ["AK_North"]},
        geometry=[_square(-150, 65, -140, 70)],
        crs="EPSG:4326",
    )
    expected = (np.sin(np.radians(70)) - np.sin(np.radians(65))) / (
        np.sin(np.radians(70)) - np.sin(np.radians(60))
    )  # ~0.453; the degree-squared ratio would be exactly 0.5
    out = catalog.summarize_surveys(north, aoi)
    np.testing.assert_allclose(out["aoi_overlap_frac"].iloc[0], expected, atol=0.005)
    gaps = catalog.coverage_gaps(out, aoi)
    np.testing.assert_allclose(gaps["gap_frac"].sum(), 1 - expected, atol=0.005)


def test_zero_area_aoi_rejected():
    """A degenerate (point/line) AOI must raise a clear error, not
    ZeroDivisionError, from the coverage-fraction math."""
    import pytest

    point_aoi = gpd.GeoDataFrame(
        geometry=[shapely.Point(-115.0, 36.0)], crs="EPSG:4326"
    )
    wesm = gpd.GeoDataFrame(
        {"workunit": ["WU"]},
        geometry=[_square(-116, 35, -114, 37)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="zero area"):
        catalog.summarize_surveys(wesm, point_aoi)


def _wesm_dated():
    """Three overlapping collections; B is the most recent."""
    import pandas as pd

    return gpd.GeoDataFrame(
        {
            "workunit": ["WU_A_2019", "WU_B_2023", "WU_C_2021"],
            "collect_start": pd.to_datetime(["2019-03-01", "2023-05-01", "2021-06-01"]),
            "collect_end": pd.to_datetime(["2019-06-01", "2023-08-01", "2021-09-01"]),
            "ql": ["QL 2", "QL 1", "QL 2"],
        },
        geometry=[_square(-1, -1, 1, 1)] * 3,
        crs="EPSG:4326",
    )


def _ept_for(names):
    return gpd.GeoDataFrame(
        {"name": names},
        geometry=[_square(-1, -1, 1, 1)] * len(names),
        crs="EPSG:4326",
    )


def test_select_latest_workunit_picks_newest_acquisition():
    """'latest' must mean most-recently-collected, not first-in-index."""
    aoi = gpd.GeoDataFrame(geometry=[_square(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")
    ept = _ept_for(["WU_A_2019", "WU_B_2023", "WU_C_2021"])
    out = catalog.select_latest_workunit(aoi, _wesm_dated(), ept)
    assert out["workunit"] == "WU_B_2023"
    assert out["undated"] is False
    assert out["n_candidates"] == 3


def test_select_latest_workunit_ignores_collections_without_ept():
    """The newest survey is useless if it has no EPT build to read."""
    aoi = gpd.GeoDataFrame(geometry=[_square(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")
    ept = _ept_for(["WU_A_2019", "WU_C_2021"])  # newest (B) has no build
    out = catalog.select_latest_workunit(aoi, _wesm_dated(), ept)
    assert out["workunit"] == "WU_C_2021"
    assert out["n_candidates"] == 2


def test_select_latest_workunit_undated_falls_back_and_says_so():
    import pandas as pd
    import pytest

    aoi = gpd.GeoDataFrame(geometry=[_square(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")
    wesm = _wesm_dated()
    wesm["collect_end"] = pd.NaT
    out = catalog.select_latest_workunit(aoi, wesm, _ept_for(list(wesm["workunit"])))
    assert out["undated"] is True
    # recency is unknowable here, so the fallback is widest AOI coverage;
    # these three footprints are identical, so the name tie-break decides.
    # It must be a stated rule, not whatever order the frame arrived in.
    assert out["workunit"] == "WU_A_2019"

    # ... and with no EPT build anywhere, fail loudly instead of no-data
    with pytest.raises(LookupError, match="none resolves"):
        catalog.select_latest_workunit(aoi, _wesm_dated(), _ept_for(["UNRELATED"]))


def test_select_latest_workunit_undated_fallback_prefers_coverage():
    """The undated fallback is coverage-ordered, not input- or QL-ordered."""
    import pandas as pd
    import pytest

    aoi = gpd.GeoDataFrame(geometry=[_square(0, 0, 1, 1)], crs="EPSG:4326")
    wesm = gpd.GeoDataFrame(
        {
            "workunit": ["WU_A_SLIVER", "WU_B_FULL"],
            "ql": ["QL 1", "QL 2"],  # A sorts first on QL, and on name
            "collect_start": pd.to_datetime([None, None]),
            "collect_end": pd.to_datetime([None, None]),
        },
        geometry=[_square(0, 0, 0.1, 1), _square(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    out = catalog.select_latest_workunit(
        aoi, wesm, _ept_for(["WU_A_SLIVER", "WU_B_FULL"])
    )
    assert out["undated"] is True
    assert out["workunit"] == "WU_B_FULL"
    assert out["aoi_overlap_frac"] == pytest.approx(1.0, abs=1e-6)


def test_undated_inventory_does_not_crash():
    """gh #89: an inventory whose collections all lack WESM dates must rank
    and epoch-label as 'undated', not TypeError in the anchor/epoch math."""
    import pandas as pd

    wesm = gpd.GeoDataFrame(
        {
            "workunit": ["OLD_A", "OLD_B"],
            "ql": ["Other", "Other"],
            "collect_start": [pd.NaT, pd.NaT],
            "collect_end": [pd.NaT, pd.NaT],
        },
        geometry=[_square(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
    )
    s = catalog.summarize_surveys(wesm, _aoi())
    out = catalog.assign_epochs(catalog.rank_collections(catalog.relative_metrics(s)))
    assert list(out["epoch"]) == ["undated", "undated"]
    assert list(out["priority"]) == [1, 2]


def test_anchor_prefers_dated_collection():
    """gh #89: an undated collection cannot be the temporal reference frame,
    even when it covers more of the AOI than the dated candidate."""
    import pandas as pd

    wesm = gpd.GeoDataFrame(
        {
            "workunit": ["UNDATED_BIG", "DATED_SMALL"],
            "ql": ["QL 2", "QL 2"],
            "collect_start": [pd.NaT, pd.Timestamp("2020-01-01")],
            "collect_end": [pd.NaT, pd.Timestamp("2020-03-01")],
        },
        geometry=[_square(-1, -1, 2, 2), _square(0, 0, 0.6, 1)],
        crs="EPSG:4326",
    )
    s = catalog.summarize_surveys(wesm, _aoi())
    out = catalog.relative_metrics(s)
    assert out.loc[out["anchor"], "workunit"].iloc[0] == "DATED_SMALL"


def test_select_workunits_orders_by_rank_and_filters_ept():
    """auto-selection: every EPT-backed intersecting survey, rank_collections
    order (anchor first, then QL / temporal proximity)."""
    aoi = gpd.GeoDataFrame(geometry=[_square(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")
    out = catalog.select_workunits(
        aoi, _wesm_dated(), _ept_for(["WU_A_2019", "WU_B_2023", "WU_C_2021"])
    )
    assert [r["workunit"] for r in out] == ["WU_B_2023", "WU_C_2021", "WU_A_2019"]
    assert [r["priority"] for r in out] == [1, 2, 3]

    # the newest survey has no EPT build: it must not appear at all
    out = catalog.select_workunits(
        aoi, _wesm_dated(), _ept_for(["WU_A_2019", "WU_C_2021"])
    )
    assert [r["workunit"] for r in out] == ["WU_C_2021", "WU_A_2019"]


def test_select_workunits_fails_loud_when_nothing_resolves():
    import pytest

    aoi = gpd.GeoDataFrame(geometry=[_square(-0.5, -0.5, 0.5, 0.5)], crs="EPSG:4326")
    with pytest.raises(LookupError, match="none resolves"):
        catalog.select_workunits(aoi, _wesm_dated(), _ept_for(["UNRELATED"]))
