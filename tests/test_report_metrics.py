"""Pattern-layer tests against real (abbreviated) vendor report excerpts —
one per observed format family — so vendor-format drift shows up as a
test failure, not silent non-extraction."""

from lidar_tools import report_metrics


FGDC_XML = """<metadata><idinfo><timeperd><timeinfo><rngdates>
<begdate>20230618</begdate><enddate>20230722</enddate>
</rngdates></timeinfo></timeperd></idinfo>
<dataqual><posacc><vertacc>
<vertaccr>produced to meet ASPRS ... for a 10-cm RMSEz Vertical Accuracy Class.</vertaccr>
</vertacc><horizpa>
<horizpar>for a ___ (cm) RMSEx / RMSEy Horizontal Accuracy Class = +/- ___ cm.</horizpar>
</horizpa></posacc></dataqual></metadata>"""


def test_parse_fgdc_xml_dates_and_unfilled_template():
    out = report_metrics.parse_fgdc_xml(FGDC_XML)
    assert out["acquisition_start"] == "2023-06-18"
    assert out["acquisition_end"] == "2023-07-22"
    assert out["unfilled_template_fields"]  # the "___" blanks are surfaced


GEIGER_TABLE = """
                               Aggregate Nominal Pulse Spacing (m)              0.23
                               Aggregate Nominal Pulse Density (pls/m2)            18.4
             Broad Land Cover Type # of Points RMSEz 95% Confidence Level 95th Percentile
               NVA of Point Cloud      129      0.051      0.100
               VVA of Bare Earth       101      0.053                          0.107
"""


def test_parse_geiger_style_tables():
    m, ev = report_metrics.parse_report_text(GEIGER_TABLE, "vendor.pdf")
    assert m["anpd_ppsm"] == 18.4
    assert m["anps_m"] == 0.23
    assert m["nva_rmsez_m"] == 0.051
    assert m["nva_95pct_m"] == 0.100
    assert m["vva_95th_m"] == 0.107
    assert m["n_nva"] == 129 and m["n_vva"] == 101
    assert all(e["file"] == "vendor.pdf" and e["line"] for e in ev)


TARGET_MEASURED_TABLE = """
                                           Target        Measured          Point Count
                       Raw NVA            0.196 m          0.1058              191
                         NVA              0.196 m          0.1656              191
                         VVA              0.294 m          0.1466              138
                         RMSEr
                                                                                0.1078 m
                          ACCr
                                                                                 0.19 m
                       Density                            9.66 pts / m2
"""


def test_parse_target_measured_style_tables():
    m, _ = report_metrics.parse_report_text(TARGET_MEASURED_TABLE, "r.pdf")
    assert m["nva_95pct_m"] == 0.1656  # final NVA row, not the raw row
    assert m["vva_95th_m"] == 0.1466
    assert m["horizontal_rmser_m"] == 0.1078
    assert m["horizontal_acc95_m"] == 0.19
    assert m["measured_density_ppsm"] == 9.66


MARS_QC = """
Aggregate                     90,703,674,128                    17,593,412,570             5.156/0.479               0.440/1.444
Non-vegetated Vertical Accuracy (NVA) RMSE(z)                                                  0.039/0.128 PASS
Non-vegetated Vertical Accuracy (NVA) at the 95% Confidence Level +/-                          0.077/0.252 PASS
"""


def test_parse_mars_qc_style():
    m, _ = report_metrics.parse_report_text(MARS_QC, "qc.pdf")
    assert m["first_return_density_ppsm"] == 5.156
    assert m["nva_rmsez_m"] == 0.039
    assert m["nva_95pct_m"] == 0.077


USGS_VALIDATION = """              Data Validation Report
from the National Geospatial Technical Operations Center in
  Based on this review, the delivered data is EXPECTED
     TO MEET 3D Elevation Program requirements.
Quality Level: 1                          P-Method: 15 - Geiger Mode Lidar
Horizontal EPSG Code: 6341                Vertical EPSG Code: 5703    Geoid Model: GEOID 18
Mechanism: GPSC                           Lidar Base Spec: 2.1
"""


def test_parse_usgs_validation():
    out = report_metrics.parse_usgs_validation_text(USGS_VALIDATION)
    assert out["quality_level"] == "1"
    assert out["lidar_base_spec"] == "2.1"
    assert out["p_method"] == "15 - Geiger Mode Lidar"
    assert out["geoid"] == "GEOID 18"
    assert out["verdict"] == "EXPECTED TO MEET"


THREE_COL_DENSITY = """
                    Average Point
   Density                               4.7 pts / m2           8 pts / m2          9.64pts / m2
The average (mean) line to line relative vertical accuracy for the AZ project
was 0.035 feet (0.011 meters).
"""


def test_parse_three_column_density_and_swath_precision():
    m, _ = report_metrics.parse_report_text(THREE_COL_DENSITY, "r.pdf")
    assert m["measured_density_ppsm"] == 9.64  # achieved column, not planned
    assert m["swath_relative_dz_mean_m"] == 0.011
