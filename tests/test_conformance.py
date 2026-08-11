"""Conformance checks run against synthetic catalogs, so a change in what
the code supports shows up here rather than at a new site."""

import pandas as pd

from lidar_tools import conformance


def _wesm(**overrides) -> pd.DataFrame:
    base = {
        "workunit": ["WU_A", "WU_B", "WU_C"],
        "horiz_crs": ["6340", "6340", "6340"],
        "geoid": ["GEOID18", "GEOID12B", "GEOID18"],
        "p_method": ["linear-mode lidar"] * 3,
        "lpc_link": ["https://x/a", "https://x/b", "https://x/c"],
        "metadata_link": ["https://m/a", "https://m/b", "https://m/c"],
        "collect_start": ["2020-01-01"] * 3,
        "collect_end": ["2020-06-01"] * 3,
        "ql": ["QL 1"] * 3,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _by_check(findings) -> dict:
    return {f.check: f for f in findings}


def test_clean_catalog_has_no_findings():
    result = conformance.audit_wesm(_wesm())
    assert result["n_workunits"] == 3
    assert result["findings"] == []


def test_esri_codes_are_reported_separately_from_unsupported_datums():
    """ESRI authority codes in the EPSG-shaped horiz_crs field are OUR gap
    (they are NAD83-family and resolve as ESRI:<code>); a Mariana-plate datum
    is a genuine limitation. Conflating them hides a fixable bug."""
    # 102247 = NAD_1983_CORS96_Alaska_Albers, 8693 = NAD83(MA11) / UTM 55N
    found = _by_check(
        conformance.check_horizontal_crs(pd.Series(["102247", "102247", "8693"]))
    )
    assert found["horiz_crs.esri_authority_code"].count == 2
    assert found["horiz_crs.esri_authority_code"].severity == "blocker"
    assert found["horiz_crs.non_nad83_datum"].count == 1
    assert found["horiz_crs.non_nad83_datum"].severity == "unsupported"


def test_bogus_numeric_crs_is_unparseable_not_esri():
    """A numeric value PROJ rejects under BOTH authorities must land in
    unparseable: classification asks the ESRI authority, it does not trust
    a numeric range."""
    found = _by_check(conformance.check_horizontal_crs(pd.Series(["999999"])))
    assert "horiz_crs.esri_authority_code" not in found
    assert found["horiz_crs.unparseable"].count == 1


def test_geoid_checks_split_unrecognized_from_undeclared():
    found = _by_check(
        conformance.check_geoid(pd.Series(["GEOID18", "NOT_A_GEOID", "Unknown", None]))
    )
    assert found["geoid.unrecognized_model"].count == 1
    assert found["geoid.unrecognized_model"].severity == "blocker"
    # 'Unknown' and null both silently disable enforcement
    assert found["geoid.undeclared"].count == 2
    assert found["geoid.undeclared"].severity == "degraded"


def test_geoid_accepts_the_case_and_short_forms_seen_in_wesm():
    """WESM carries 'Geoid18' and bare '12B' alongside canonical names."""
    assert conformance.check_geoid(pd.Series(["Geoid18", "12B", "GEOID12A"])) == []


def test_non_lidar_acquisition_is_flagged():
    found = _by_check(
        conformance.check_acquisition_method(
            pd.Series(["linear-mode lidar", "Ifsar", "Ifsar", "Sonar"])
        )
    )
    assert found["p_method.not_lidar"].count == 2
    assert found["p_method.unrecognized"].count == 1


def test_missing_metadata_fields_are_reported_with_examples():
    wesm = _wesm(lpc_link=["https://x/a", None, None])
    found = _by_check(conformance.audit_wesm(wesm)["findings"])
    finding = found["metadata.missing_lpc_link"]
    assert finding.count == 2
    assert finding.fraction == 2 / 3
    assert finding.examples == ["WU_B", "WU_C"]


def test_report_renders_and_sorts_most_severe_first():
    wesm = _wesm(
        horiz_crs=["102247", "6340", "6340"],
        lpc_link=[None, None, "https://x/c"],
    )
    result = conformance.audit_wesm(wesm)
    assert result["findings"][0].severity == "blocker"
    text = conformance.format_report(result)
    assert "3DEP archive conformance: 3 workunits" in text
    assert "horiz_crs.esri_authority_code" in text
