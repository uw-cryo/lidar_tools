import geopandas as gpd
import pandas as pd
import shapely
import yaml

from lidar_tools import staging


def test_parse_grid_id():
    g = staging.parse_grid_id(
        "https://rockyweb.usgs.gov/.../USGS_LPC_NV_Southern_D23_11SMA760990.laz"
    )
    assert g == {
        "zone": 11,
        "band": "S",
        "square": "MA",
        "e": 760,
        "n": 990,
        "gridid": "11SMA760990",
    }
    assert staging.parse_grid_id("no_grid_here.laz") is None


def test_grid_origin_matches_laz_header_calibration():
    # all 7 squares of NV_Southern_4_D23, verified against LAZ header
    # min/max at 0.00 m residual (2026-07-18)
    verified = {
        "MA": (400000, 4000000),
        "MB": (400000, 4100000),
        "NA": (500000, 4000000),
        "NB": (500000, 4100000),
        "NV": (500000, 3900000),
        "PA": (600000, 4000000),
        "PV": (600000, 3900000),
    }
    hint = staging.band_northing("S")  # ~36N -> ~4.0e6
    for square, origin in verified.items():
        assert staging.grid_origin(11, square, hint) == origin


def test_decode_tile_footprints():
    # MA 870940 -> SW corner 487000, 4094000 (LAZ-header verified)
    urls = [
        "https://x/USGS_LPC_NV_Southern_D23_11SMA870940.laz",
        "https://x/not_a_tile.txt",
    ]
    gdf = staging.decode_tile_footprints(urls, utm_epsg=6340)
    assert len(gdf) == 1  # unparseable dropped
    assert gdf.crs.to_epsg() == 6340
    assert gdf.iloc[0].geometry.bounds == (487000.0, 4094000.0, 488000.0, 4095000.0)


def test_attach_workunits_joins_by_id_not_name():
    tesm = gpd.GeoDataFrame(
        {
            # TESM project names drift from WESM's — only the id is stable
            "project": ["NV_Southern_D23", "NV_Las_Vegas_Region_2016_A16"],
            "workunit_id": [300544, 63305],
        },
        geometry=[shapely.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4269",
    )
    wesm = gpd.GeoDataFrame(
        {
            "workunit": ["NV_Southern_5_D23", "NV_LasVegas_QL2_2016"],
            "workunit_id": [300544, 63305],
        },
        geometry=[shapely.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
    )
    out = staging.attach_workunits(tesm, wesm)
    assert list(out["workunit"]) == ["NV_Southern_5_D23", "NV_LasVegas_QL2_2016"]


def test_reconcile_tile_sources_flags_index_lag():
    # the Southern_4 case: staged LAZ exists, TESM has nothing
    v = staging.reconcile_tile_sources("WU", tesm_count=0, links_count=10626)
    assert v["warning"] and "lags" in v["warning"]
    ok = staging.reconcile_tile_sources("WU", tesm_count=195, links_count=195)
    assert ok["warning"] is None


def test_fetch_links_file_injectable_opener():
    calls = []

    def opener(url):
        calls.append(url)
        return "https://a/t1.laz\nhttps://a/t2.laz\n\n"

    links = staging.fetch_links_file("https://rockyweb/x/WU_A/", opener=opener)
    assert links == ["https://a/t1.laz", "https://a/t2.laz"]
    assert calls == ["https://rockyweb/x/WU_A/0_file_download_links.txt"]


def _wesm():
    return gpd.GeoDataFrame(
        {
            "workunit": ["WU_A", "WU_B"],
            "workunit_id": [1, 2],
            "ql": ["QL 1", "QL 2"],
            "geoid": ["GEOID18", "GEOID12A"],
            "lpc_link": ["https://rockyweb/x/WU_A", "https://rockyweb/x/WU_B"],
        },
        geometry=[shapely.box(0, 0, 1, 1)] * 2,
        crs="EPSG:4326",
    )


def test_build_site_manifest_roundtrip(tmp_path):
    ept = pd.DataFrame(
        {"name": ["USGS_LPC_WU_A_LAS_2018"], "count": [7]}
    )  # WU_A resolves tier 3; WU_B has no EPT
    manifest = staging.build_site_manifest(
        "aoi.geojson",
        ["WU_A", "WU_B"],
        _wesm(),
        str(tmp_path / "batch"),
        ept_gdf=ept,
        tesm_counts={"WU_A": 100},
        links_counts={"WU_A": 100, "WU_B": 42},
    )
    a, b = manifest["workunits"]["WU_A"], manifest["workunits"]["WU_B"]
    # the WESM pin and the declared geoid ride along untouched
    assert a["wesm"]["geoid"] == "GEOID18"
    assert (a["ept"]["ept_name"], a["ept"]["tier"]) == ("USGS_LPC_WU_A_LAS_2018", 3)
    assert a["tiles"]["warning"] is None
    # no EPT -> recorded loudly, not dropped; staged-LAZ cache path pinned
    assert "error" in b["ept"] and "staged-LAZ" in b["ept"]["error"]
    assert b["tiles"]["warning"] and "lags" in b["tiles"]["warning"]
    assert b["lpc_cache"].endswith("batch/lpc_cache/WU_B")

    fn = tmp_path / "batch" / "site_manifest.yaml"
    staging.write_site_manifest(manifest, fn)
    loaded = staging.load_site_manifest(fn)
    assert loaded["workunits"]["WU_B"]["lpc_cache"] == b["lpc_cache"]
    # YAML stays plain-typed (no numpy scalars etc.)
    assert yaml.safe_load(fn.read_text())
