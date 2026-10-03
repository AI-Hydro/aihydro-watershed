"""Regression for defect P2-D0: q5 / q95 must follow CAMELS (Addor et al. 2017, Table 3).

CAMELS ``q5`` is the 5% flow quantile (low flow) and ``q95`` the 95% flow quantile
(high flow); they are plain non-exceedance quantiles. Before this fix the platform
computed them the other way round (q5 = 95th percentile).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from aihydro_watershed.signatures import signatures as sig
from aihydro_watershed.signatures.signatures import compute_flow_stats_camels

FIXTURE = Path(__file__).parent / "fixtures" / "served_streamflow_01013500.csv"
FIXTURE_SHA256 = "683ccb1619efe82e3c124f8a540bdc9af93269af12efe5c3681d88d9438d359a"
AREA_KM2 = 2258.5163954900077  # session.json of e2e proof 1


def _series(values) -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_q5_is_low_flow_q95_is_high_flow_on_a_known_series():
    # 1..1000: linear-interpolated quantiles are exact, 1 + p * 999.
    out = compute_flow_stats_camels(_series(np.arange(1, 1001)))
    assert abs(out["q5"] - 50.95) < 1e-9
    assert abs(out["q95"] - 950.05) < 1e-9
    assert out["q5"] < out["q_median"] < out["q95"]


def test_q5_q95_are_input_order_independent_non_exceedance_quantiles():
    rng = np.random.default_rng(7)
    q = rng.lognormal(0.0, 1.0, size=2000)
    out = compute_flow_stats_camels(_series(q))
    assert out["q5"] == np.quantile(q, 0.05)
    assert out["q95"] == np.quantile(q, 0.95)


def test_proof1_served_csv_reproduces_camels_orientation():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    q_cms = pd.read_csv(FIXTURE, parse_dates=["date"]).set_index("date")["q_cms"]
    out = compute_flow_stats_camels(sig._to_mm_per_day(q_cms, AREA_KM2))
    # Proof 1 frozen record (pre-fix, swapped): q5 = 6.3566, q95 = 0.2405.
    assert abs(out["q5"] - 0.2405) < 5e-4
    assert abs(out["q95"] - 6.3566) < 5e-4
    # CAMELS camels_hydro.txt, gauge 01013500: q5 0.2411, q95 6.3730 (within 0.3 %).
    assert abs(out["q5"] / 0.241106126475711 - 1) < 0.01
    assert abs(out["q95"] / 6.37302139711473 - 1) < 0.01


def test_extract_signatures_carries_orientation_marker_and_bootstrap_matches(monkeypatch):
    days = pd.date_range("2000-01-01", periods=800, freq="D")
    q_cms = pd.Series(np.random.default_rng(3).lognormal(1.0, 0.8, size=800), index=days)
    monkeypatch.setattr(
        sig, "_fetch_precipitation_data_bygeom",
        lambda *a, **k: pd.Series(2.0, index=days),
    )
    square = {"type": "Polygon", "coordinates": [[
        [-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2],
    ]]}
    res = sig.extract_hydrological_signatures(
        gauge_id=None, watershed_geojson=square, area_km2=250.0, q_cms_series=q_cms,
    )
    d = res.data
    assert d["_flow_quantile_convention"] == sig.FLOW_QUANTILE_CONVENTION == "camels_nonexceedance_v1"
    assert d["q5"] < d["q_median"] < d["q95"]
    u = d["_uncertainty"]
    # The bootstrap CI belongs to the same quantile as the point value it labels.
    assert u["q5"]["ci_low"] <= d["q5"] <= u["q5"]["ci_high"]
    assert u["q95"]["ci_low"] <= d["q95"] <= u["q95"]["ci_high"]
    assert u["q5"]["ci_high"] < u["q95"]["ci_low"]
