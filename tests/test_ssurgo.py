"""
Offline tests for the SSURGO soil path (gNATSGO map units + Soil Data Access).

Network calls are replaced with fakes; the parsing, aggregation, gridding and
curve-number wiring are checked against hand-computed answers.
"""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
from affine import Affine  # noqa: E402
from shapely.geometry import box  # noqa: E402

from aihydro_watershed.terrain import ssurgo as ss  # noqa: E402

# Two map units. 1001: 60 % "B" (Kw .32), 40 % "C" (Kw .43), dominant B.
# 1002: 85 % dual "B/D" (Kw .37), 15 % minor component with no group.
SDA_ROWS = [
    {"mukey": "1001", "cokey": "1", "comppct_r": "60", "hydgrp": "B", "kwfact": ".32",
     "sandtotal_r": "20", "silttotal_r": "60", "claytotal_r": "20"},
    {"mukey": "1001", "cokey": "2", "comppct_r": "40", "hydgrp": "C", "kwfact": ".43",
     "sandtotal_r": "10", "silttotal_r": "55", "claytotal_r": "35"},
    {"mukey": "1002", "cokey": "3", "comppct_r": "85", "hydgrp": "B/D", "kwfact": ".37",
     "sandtotal_r": "15", "silttotal_r": "65", "claytotal_r": "20"},
    {"mukey": "1002", "cokey": "4", "comppct_r": "15", "hydgrp": None, "kwfact": None,
     "sandtotal_r": None, "silttotal_r": None, "claytotal_r": None},
]


# ── hydgrp parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("A", (1, 1)), ("B", (2, 2)), ("C", (3, 3)), ("D", (4, 4)),
    ("A/D", (1, 4)), ("B/D", (2, 4)), ("C/D", (3, 4)),
    (" b/d ", (2, 4)), ("", (None, None)), (None, (None, None)), ("X", (None, None)),
])
def test_parse_hydgrp(raw, expected):
    assert ss.parse_hydgrp(raw) == expected


# ── SDA query + response handling ─────────────────────────────────────────────

def test_component_query_selects_surface_horizon_and_dedupes_keys():
    q = ss.build_component_query([1002, "1001", 1001.0])
    assert "ch.hzdept_r = 0" in q
    assert "c.hydgrp" in q and "ch.kwfact" in q
    assert "IN ('1001','1002')" in q


def test_component_query_rejects_empty():
    with pytest.raises(ValueError):
        ss.build_component_query([])


def test_parse_sda_table():
    payload = {"Table": [["mukey", "hydgrp"], ["1", "A"], ["2", "B/D"]]}
    assert ss.parse_sda_table(payload) == [
        {"mukey": "1", "hydgrp": "A"}, {"mukey": "2", "hydgrp": "B/D"},
    ]
    assert ss.parse_sda_table({}) == []


def test_summarize_components_known_answer():
    out = ss.summarize_components(SDA_ROWS)
    a, b = out[1001], out[1002]
    assert a["kw"] == pytest.approx(0.6 * 0.32 + 0.4 * 0.43)   # 0.364
    assert a["clay"] == pytest.approx(0.6 * 20 + 0.4 * 35)     # 26.0
    assert a["hydgrp"] == "B" and (a["hsg_drained"], a["hsg_undrained"]) == (2, 2)
    assert not a["is_dual"]
    # Null-valued minor component is skipped, not averaged in as zero.
    assert b["kw"] == pytest.approx(0.37)
    assert b["hydgrp"] == "B/D" and (b["hsg_drained"], b["hsg_undrained"]) == (2, 4)
    assert b["is_dual"]


def test_dominant_group_skips_components_without_one():
    rows = [
        {"mukey": "7", "comppct_r": "70", "hydgrp": "", "kwfact": None},
        {"mukey": "7", "comppct_r": "30", "hydgrp": "C", "kwfact": ".28"},
    ]
    assert ss.summarize_components(rows)[7]["hydgrp"] == "C"


def test_fetch_mukey_properties_chunks_queries():
    seen = []

    def fake_query(sql):
        seen.append(sql)
        return SDA_ROWS

    props = ss.fetch_mukey_properties([1001, 1002, np.nan, 0], query_fn=fake_query, chunk=1)
    assert len(seen) == 2                       # NaN and 0 dropped, one key per chunk
    assert set(props) == {1001, 1002}


# ── Gridding + catchment summary ──────────────────────────────────────────────

def _mukey_grid():
    """3x4 grid in EPSG:5070, 10 m cells: left half 1001, right half 1002, one hole."""
    vals = np.array([[1001, 1001, 1002, 1002],
                     [1001, 1001, 1002, 1002],
                     [1001, np.nan, 1002, 1002]], dtype="float64")
    tr = Affine(10.0, 0, 500000.0, 0, -10.0, 1800000.0)
    da = xr.DataArray(
        vals, dims=("y", "x"),
        coords={"x": 500000.0 + 5 + 10 * np.arange(4), "y": 1800000.0 - 5 - 10 * np.arange(3)},
        name="mukey",
    ).rio.write_crs("EPSG:5070").rio.write_transform(tr)
    da.attrs["resolution_m"] = 10.0
    return da


def test_mukey_grid_to_dataset_maps_properties():
    props = ss.summarize_components(SDA_ROWS)
    ds = ss.mukey_grid_to_dataset(_mukey_grid(), props)
    assert ds.attrs["_adata_product"] == ss.SSURGO_PRODUCT_ID
    assert ds["hsg_drained"].values[0, 0] == 2 and ds["hsg_drained"].values[0, 3] == 2
    assert ds["hsg_undrained"].values[0, 3] == 4
    assert np.isnan(ds["hsg_drained"].values[2, 1])
    assert ds["kw_si"].values[0, 0] == pytest.approx(0.364 * ss.KW_US_TO_SI)


def test_fetch_soil_data_ssurgo_with_fakes_and_summary():
    ds = ss.fetch_soil_data_ssurgo(
        box(0, 0, 1, 1),                          # geometry ignored by the fake
        mukey_fn=lambda g: _mukey_grid(),
        query_fn=lambda sql: SDA_ROWS,
    )
    summ = ss.summarize_ssurgo(ds)                # whole window
    # 5 cells of 1001 (Kw .364), 6 of 1002 (Kw .37), 1 unmapped.
    assert summ["kw_mean"] == pytest.approx((5 * 0.364 + 6 * 0.37) / 11, abs=1e-4)
    assert summ["soil_coverage"] == pytest.approx(11 / 12, abs=1e-4)
    assert summ["hsg_drained_pct"] == {"B": 100.0}
    assert summ["hsg_undrained_pct"]["D"] == pytest.approx(100 * 6 / 11, abs=0.01)
    assert summ["pct_dual_hsg"] == pytest.approx(100 * 6 / 11, abs=0.01)
    assert summ["n_map_units"] == 2


def test_summary_respects_polygon_mask():
    import geopandas as gpd

    ds = ss.mukey_grid_to_dataset(_mukey_grid(), ss.summarize_components(SDA_ROWS))
    # Polygon covering only the right half (map unit 1002), built in 5070, given in 4326.
    right = gpd.GeoSeries([box(500020.0, 1799970.0, 500040.0, 1800000.0)], crs=5070).to_crs(4326)
    summ = ss.summarize_ssurgo(ds, right.iloc[0])
    assert summ["n_map_units"] == 1
    assert summ["kw_mean"] == pytest.approx(0.37, abs=1e-4)


# ── Curve number on recorded groups ───────────────────────────────────────────

def _lulc_on(grid, value=81.0):
    lulc = xr.full_like(grid, value).rename("cover_2021")
    return lulc.rio.write_crs("EPSG:5070").to_dataset()


def test_cn_grid_uses_recorded_groups_and_reports_other_condition():
    from aihydro_watershed.terrain.curve_number import _create_cn_grid_from_data

    grid = _mukey_grid()
    soil = ss.mukey_grid_to_dataset(grid, ss.summarize_components(SDA_ROWS))
    cn_grid, groups, stats = _create_cn_grid_from_data(
        _lulc_on(grid, 81.0), soil, 2021, 10, dual_hsg="drained",
    )
    v = cn_grid.values
    # Pasture: B -> 61 everywhere mapped when drained; unmapped hole -> NaN.
    assert np.nanmin(v) == 61 and np.nanmax(v) == 61
    assert np.isnan(v[2, 1])
    assert stats["hsg_method"] == "ssurgo_recorded"
    assert stats["pct_dual_hsg"] == pytest.approx(100 * 6 / 11, abs=0.01)
    # Undrained: the 6 B/D cells become D (pasture D = 80).
    assert stats["cn_mean_undrained"] == pytest.approx((5 * 61 + 6 * 80) / 11)
    assert stats["kw_mean"] == pytest.approx((5 * 0.364 + 6 * 0.37) / 11, abs=1e-4)


def test_cn_grid_undrained_condition():
    from aihydro_watershed.terrain.curve_number import _create_cn_grid_from_data

    grid = _mukey_grid()
    soil = ss.mukey_grid_to_dataset(grid, ss.summarize_components(SDA_ROWS))
    cn_grid, _, stats = _create_cn_grid_from_data(
        _lulc_on(grid, 82.0), soil, 2021, 10, dual_hsg="undrained",
    )
    # Row crops: B -> 78 on 1001, D -> 89 on the B/D unit.
    assert cn_grid.values[0, 0] == 78 and cn_grid.values[0, 3] == 89
    assert stats["cn_mean_drained"] == pytest.approx(78.0)


def test_cn_grid_rejects_bad_dual_hsg():
    from aihydro_watershed.terrain.curve_number import _create_cn_grid_from_data

    grid = _mukey_grid()
    soil = ss.mukey_grid_to_dataset(grid, ss.summarize_components(SDA_ROWS))
    with pytest.raises(ValueError):
        _create_cn_grid_from_data(_lulc_on(grid), soil, 2021, 10, dual_hsg="wet")


# ── Region routing ────────────────────────────────────────────────────────────

def test_soil_router_prefers_ssurgo_in_conus(monkeypatch):
    from aihydro_watershed.terrain import _soil

    calls = []
    monkeypatch.setattr(ss, "fetch_soil_data_ssurgo", lambda g: calls.append("ssurgo") or "S")
    monkeypatch.setattr(_soil, "fetch_soil_data_polaris",
                        lambda g, product=None: calls.append(("polaris", product)) or "P")
    indiana = box(-86.95, 40.40, -86.94, 40.41)
    assert _soil.fetch_soil_data(indiana) == "S"
    assert _soil.fetch_soil_data(indiana, product="POLARIS") == "P"
    assert _soil.fetch_soil_data(box(8.0, 50.0, 8.1, 50.1)) == "P"   # Germany
    assert calls == ["ssurgo", ("polaris", "POLARIS"), ("polaris", None)]


def test_soil_router_falls_back_to_polaris_and_records_reason(monkeypatch):
    from aihydro_watershed.terrain import _soil

    def boom(g):
        raise RuntimeError("SDA down")

    monkeypatch.setattr(ss, "fetch_soil_data_ssurgo", boom)
    monkeypatch.setattr(_soil, "fetch_soil_data_polaris",
                        lambda g, product=None: xr.Dataset(attrs={"_adata_product": "POLARIS"}))
    ds = _soil.fetch_soil_data(box(-86.95, 40.40, -86.94, 40.41))
    assert ds.attrs["_adata_product"] == "POLARIS"
    assert "SDA down" in ds.attrs["_soil_fallback_reason"]


def test_soil_router_pinned_ssurgo_raises(monkeypatch):
    from aihydro_watershed.terrain import _soil

    def boom(g):
        raise RuntimeError("SDA down")

    monkeypatch.setattr(ss, "fetch_soil_data_ssurgo", boom)
    with pytest.raises(RuntimeError):
        _soil.fetch_soil_data(box(-86.95, 40.40, -86.94, 40.41), product="SSURGO")


@pytest.mark.live
def test_live_ssurgo_small_indiana_catchment():
    """Tippecanoe County, IN: mapped soils with recorded groups and Kw."""
    geom = box(-86.93, 40.42, -86.92, 40.43)
    res = ss.ssurgo_catchment_attributes(geom)
    s = res["summary"]
    assert s["soil_coverage"] > 0.9
    assert s["hsg_coverage"] > 0.5
    assert s["kw_mean"] is not None and 0.02 < s["kw_mean"] < 0.7
