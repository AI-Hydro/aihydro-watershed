"""watershed 0.2.0: slope_fdc definition, canonical BFI kernel, precipitation completeness.

Locked references: CAMELS v1.2 ``camels_hydro.txt`` (Zenodo 15529996, md5-verified; values quoted
in docs/vision-2040/evidence/signature-definitions-1). Streamflow fixtures are USGS NWIS daily
series, WY1990-2009, 7305 days. Offline.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures import signatures as sig
from aihydro_watershed.signatures.baseflow import (
    compute_bfi,
    lyne_hollick,
    lyne_hollick_ladson2013,
    lyne_hollick_legacy_clamp,
)
from aihydro_watershed.signatures.flow_duration import flow_duration_curve

FIX = Path(__file__).parent / "fixtures"
SHA = {
    "served_streamflow_01013500.csv": "683ccb1619efe82e3c124f8a540bdc9af93269af12efe5c3681d88d9438d359a",
    "served_streamflow_01031500.csv": "e01921cdd60e1750e3d27323aea4f8dc32632976c68fd0e961a3b64a444cbccc",
}
# camels_hydro.txt rows
CAMELS = {
    "01013500": {"slope_fdc": 1.52821853538976, "baseflow_index": 0.585225955779508},
    "01031500": {"slope_fdc": 1.49401920331253, "baseflow_index": 0.445090526012048},
}
AREA = 1000.0  # every statistic tested here is scale-invariant


def _q(gauge: str) -> pd.Series:
    f = FIX / f"served_streamflow_{gauge}.csv"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA[f.name]
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["q_cms"]


@pytest.mark.parametrize("gauge", ["01013500", "01031500"])
def test_slope_fdc_matches_camels_file(gauge):
    out = sig.compute_slope_fdc_camels(sig._to_mm_per_day(_q(gauge), AREA))
    assert out["slope_fdc"] == pytest.approx(CAMELS[gauge]["slope_fdc"], rel=1e-6)


def test_slope_fdc_sawicz2011_is_the_old_log_value():
    out = sig.compute_slope_fdc_camels(sig._to_mm_per_day(_q("01013500"), AREA))
    assert out["slope_fdc_sawicz2011"] == pytest.approx(2.696537802442423, rel=1e-9)


def test_slope_fdc_zero_structure():
    # >= 66% zero days: Q33 = Q66 = 0 -> exactly 0 (finite); the log form drops zeros.
    q = np.concatenate([np.zeros(700), np.linspace(1, 50, 300)])
    out = sig.compute_slope_fdc_camels(pd.Series(q, index=pd.date_range("2000-01-01", periods=1000)))
    assert out["slope_fdc"] == 0.0
    # zero mean -> undefined
    z = pd.Series(np.zeros(500), index=pd.date_range("2000-01-01", periods=500))
    assert np.isnan(sig.compute_slope_fdc_camels(z)["slope_fdc"])


def test_flow_duration_uses_the_same_definition():
    q = _q("01013500").to_numpy()
    fdc = flow_duration_curve(q)
    assert fdc["slope_fdc"] == pytest.approx(CAMELS["01013500"]["slope_fdc"], rel=1e-6)
    assert fdc["slope_fdc_sawicz2011"] == pytest.approx(2.696537802442423, rel=1e-9)


def test_signature_definitions_carry_source_and_formula():
    d = sig.SIGNATURE_DEFINITIONS
    src = d["slope_fdc"]["source"]
    assert "naddor/camels@88e7a0d2d2 hydro/hydro_signatures.R:97" in src
    assert "Not proven to be the script that produced the file" in src
    assert d["slope_fdc"]["log_transform"] is False and d["slope_fdc_sawicz2011"]["log_transform"] is True
    for k in ("formula", "quantile_type", "zero_handling"):
        assert d["slope_fdc"][k]
    assert "CAMELS-US v1.2 slope_fdc column only" in d["slope_fdc"]["comparable_to"]
    assert sig.SIGNATURE_UNITS["high_q_freq"] == sig.SIGNATURE_UNITS["low_q_freq"] == "days/year"


def test_zero_q_freq_counts_non_positive_days():
    idx = pd.date_range("2000-01-01", periods=400)
    q = pd.Series(np.r_[np.zeros(100), np.ones(300)], index=idx)
    assert sig.compute_event_stats_camels(q)["zero_q_freq"] == pytest.approx(0.25)


@pytest.mark.parametrize("gauge", ["01013500", "01031500"])
def test_bfi_kernel_matches_camels_file(gauge):
    q = _q(gauge).to_numpy()
    bf = lyne_hollick_ladson2013(q)
    assert bf.sum() / q.sum() == pytest.approx(CAMELS[gauge]["baseflow_index"], rel=1e-5)
    # the three entry points agree: signatures, separate_baseflow (lyne_hollick), bootstrap kernel
    assert sig.compute_flow_stats_camels(sig._to_mm_per_day(_q(gauge), AREA))["baseflow_index"] == \
        pytest.approx(CAMELS[gauge]["baseflow_index"], rel=1e-5)
    assert compute_bfi(q, lyne_hollick(q)) == pytest.approx(CAMELS[gauge]["baseflow_index"], rel=1e-5)
    assert np.allclose(sig._lyne_hollick_baseflow(q), bf)


def test_separate_baseflow_value_change_and_deprecated_legacy():
    q = _q("01031500").to_numpy()
    new = compute_bfi(q, lyne_hollick(q))
    with pytest.warns(DeprecationWarning):
        old = compute_bfi(q, lyne_hollick_legacy_clamp(q))
    assert new == pytest.approx(0.4450914, abs=1e-6)
    assert old == pytest.approx(0.36, abs=0.005)


def test_bfi_kernel_nan_gap_is_a_segment_boundary():
    rng = np.random.default_rng(0)
    q = rng.gamma(2.0, 1.0, 400)
    q_gap = q.copy()
    q_gap[200:205] = np.nan
    bf = lyne_hollick_ladson2013(q_gap)
    assert np.isnan(bf[200:205]).all() and np.isfinite(bf[:200]).all() and np.isfinite(bf[205:]).all()
    assert np.allclose(bf[:200], lyne_hollick_ladson2013(q[:200]))
    assert np.all(bf[np.isfinite(bf)] >= 0)


# ---- 3c: precipitation completeness -------------------------------------------------------

SQUARE = {"type": "Polygon", "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3],
                                               [-77.5, 39.3], [-77.5, 39.2]]]}
START, END = "2000-10-01", "2006-09-30"          # WY2001..WY2006
IDX = pd.date_range(START, END, freq="D")


def _qs():
    rng = np.random.default_rng(1)
    return pd.Series(2.0 + np.sin(np.linspace(0, 12 * np.pi, len(IDX))) + rng.random(len(IDX)), index=IDX)


def _p_full():
    return pd.Series(np.random.default_rng(2).gamma(0.5, 8.0, len(IDX)), index=IDX)


def _run(monkeypatch, p, cov=None):
    def fake(geom, start, end):
        sig._PRECIP_DIAG.d = {"product": "FAKE", **({"data_coverage": cov} if cov else {})}
        if p is not None:
            p.attrs["product"] = "FAKE"
        return p
    monkeypatch.setattr(sig, "_fetch_precipitation_data_bygeom", fake)
    return sig.extract_hydrological_signatures(None, SQUARE, 100.0, START, END, q_cms_series=_qs()).data


def test_complete_precipitation_is_used_and_reports_completeness(monkeypatch):
    d = _run(monkeypatch, _p_full())
    rec = d["_precipitation"]
    assert rec["status"] == "used" and rec["completeness"] == 1.0
    assert rec["n_expected_days"] == len(IDX) and rec["missing_ranges"] == []
    assert rec["water_years_used"] == [2001, 2002, 2003, 2004, 2005, 2006]
    assert rec["water_year_rule"]["min_paired_fraction"] == 0.95
    assert "engineering choice" in rec["water_year_rule"]["basis"]


def test_missing_year_and_leap_dec31_are_disclosed_and_excluded(monkeypatch):
    p = _p_full()
    wy2003 = (p.index >= "2002-10-01") & (p.index <= "2003-09-30")
    leap_dec31 = (p.index.month == 12) & (p.index.day == 31) & p.index.is_leap_year
    p_gap = p[~wy2003 & ~leap_dec31]
    cov = {"missing_ranges": [{"start": "2003-12-31", "end": "2003-12-31", "days": 1,
                               "reason": "client_calendar_assumption"}]}
    d = _run(monkeypatch, p_gap, cov)
    rec = d["_precipitation"]
    assert rec["status"] == "used_with_gaps" and "WY2003" in rec["reason"]
    assert rec["n_expected_days"] == len(IDX)
    assert rec["completeness"] == pytest.approx(len(p_gap) / len(IDX), abs=1e-5)
    reasons = {(r["start"], r["end"]): r["reason"] for r in rec["missing_ranges"]}
    assert reasons[("2002-10-01", "2003-09-30")] == "absent_in_response"
    # reason code passed through from the data layer where it supplied one; Dec 31 2003 is
    # inside the missing WY2003 block, so the 2004 one (not covered) stays unexplained.
    assert ("2004-12-31", "2004-12-31") in reasons and reasons[("2004-12-31", "2004-12-31")] == "absent_in_response"
    assert [e["water_year"] for e in rec["water_years_excluded"]] == [2003]
    assert rec["water_years_used"] == [2001, 2002, 2004, 2005, 2006]
    # RR uses the complete years only
    q_mm = sig._to_mm_per_day(_qs(), 100.0)
    keep = np.isin(sig._year_series(IDX), rec["water_years_used"]) & IDX.isin(p_gap.index)
    assert d["runoff_ratio"] == pytest.approx(q_mm[keep].sum() / p_gap.reindex(IDX)[keep].sum(), rel=1e-12)
    assert rec["n_paired_days_used"] == int(keep.sum())


def test_data_layer_reason_code_is_passed_through(monkeypatch):
    p = _p_full()
    drop = (p.index.month == 12) & (p.index.day == 31) & p.index.is_leap_year
    cov = {"missing_ranges": [{"start": f"{y}-12-31", "end": f"{y}-12-31", "days": 1,
                               "reason": "client_calendar_assumption"} for y in (2000, 2004)]}
    rec = _run(monkeypatch, p[~drop], cov)["_precipitation"]
    assert rec["status"] == "used_with_gaps"
    assert {r["reason"] for r in rec["missing_ranges"]} == {"client_calendar_assumption"}
    assert rec["missing_range_reasons_from"] == "aihydro-data"
    assert rec["water_years_excluded"] == []          # 365/366 days is well above 95 %


def test_no_complete_water_year_is_rejected_with_reason(monkeypatch):
    p = _p_full()
    # keep only 6 months of each water year -> every year is 50 % complete
    p_half = p[p.index.month.isin([10, 11, 12, 1, 2, 3])]
    d = _run(monkeypatch, p_half)
    rec = d["_precipitation"]
    assert d["runoff_ratio"] is None and d["stream_elas"] is None
    assert rec["status"] == "rejected" and "no water year" in rec["reason"]
    assert len(rec["water_years_excluded"]) == 6
