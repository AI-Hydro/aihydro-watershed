"""watershed 0.2.0 PR-3b: water-year grouped CIs, served series, qualifier exposure, reference comparison.

Offline. Streamflow: the locked USGS 01013500 fixture (WY1990-2009, 20 water years). Precipitation is
a synthetic series derived from q (this file tests mechanics, not hydrology).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures import _uncertainty as gu
from aihydro_watershed.signatures import signatures as sig
from aihydro_watershed.signatures.comparison import reference_comparison

FIX = Path(__file__).parent / "fixtures"
AREA = 2258.5163954900077
SQUARE = {"type": "Polygon", "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3],
                                               [-77.5, 39.3], [-77.5, 39.2]]]}
START, END = "1989-10-01", "2009-09-30"

# camels_hydro.txt / camels_clim.txt / camels_topo.txt rows (CAMELS v1.2, Zenodo 15529996)
REF_01013500 = {"q_mean": 1.69915450753356, "runoff_ratio": 0.543437466590222, "slope_fdc": 1.52821853538976,
                "baseflow_index": 0.585225955779508, "stream_elas": 1.84532416164806,
                "q5": 0.241106126475711, "q95": 6.37302139711473, "high_q_freq": 6.1,
                "high_q_dur": 8.71428571428571, "low_q_freq": 41.35, "low_q_dur": 20.1707317073171,
                "zero_q_freq": 0.0, "hfd_mean": 207.25,
                "frac_snow": 0.313440357191799, "aridity": 0.63055865946247, "p_seasonality": 0.187940258706929}
AREA_GAGES2 = 2252.7


def _q():
    return pd.read_csv(FIX / "served_streamflow_01013500.csv", parse_dates=["date"]).set_index("date")["q_cms"]


def _p_for(q_cms):
    rng = np.random.default_rng(11)
    q_mm = q_cms * 86.4 / AREA
    p = q_mm.rolling(30, min_periods=1).mean() / 0.55 * rng.lognormal(0, 0.4, len(q_mm))
    p.attrs["product"] = "GRIDMET_PRECIP"
    return p


@pytest.fixture()
def run(monkeypatch):
    def _go(q=None, p="auto", flags=None, gauge=None):
        q = _q() if q is None else q
        pp = _p_for(q) if isinstance(p, str) else p

        def fake_p(g, s, e):
            sig._PRECIP_DIAG.d = {"product": "GRIDMET_PRECIP"}
            return pp
        monkeypatch.setattr(sig, "_fetch_precipitation_data_bygeom", fake_p)
        if gauge:
            monkeypatch.setattr(sig, "_fetch_streamflow_internal",
                                lambda *a, **k: {"q_cms": q, "flags": flags})
            monkeypatch.setattr(sig, "_gauge_area_check",
                                lambda *a, **k: {"area_check": "unavailable", "warnings": []})
            return sig.extract_hydrological_signatures(gauge, SQUARE, AREA, START, END).data
        return sig.extract_hydrological_signatures(None, SQUARE, AREA, START, END, q_cms_series=q).data
    return _go


def test_all_requested_cis_are_water_year_grouped_and_bracket_their_points(run):
    d = run()
    u = d["_uncertainty"]
    assert set(u) >= {"q_mean", "q_median", "q5", "q95", "flow_variability", "baseflow_index",
                      "runoff_ratio", "stream_elas"}
    for k in ("q_mean", "q_median", "q5", "q95", "flow_variability", "baseflow_index", "runoff_ratio"):
        e = u[k]
        assert e["method"] == "bootstrap_group" and e["group"] == "water_year"
        assert e["n_groups"] == 20 and e["B"] >= 2000
        assert e["ci_low"] <= d[k if k != "baseflow_index" else "baseflow_index"] <= e["ci_high"], k
        assert e["value"] == pytest.approx(d[k], rel=1e-9), k
    # elasticity: indicative; the CI value is the reported point estimate
    e = u["stream_elas"]
    assert e["indicative"] is True and e["value"] == pytest.approx(d["stream_elas"], rel=1e-9)
    assert "n_years_near_zero_dP" in e and e["near_zero_dP_threshold"] > 0


def test_core_diagnostics_and_machine_readable_scope_are_surfaced(run):
    u = run()["_uncertainty"]
    for k, e in u.items():
        for key in ("ci_degraded", "valid_frac", "low_n", "coverage_note", "ci_reason", "scope"):
            assert key in e, (k, key)
        assert e["low_n"] is True                       # 20 groups < 30
        assert "under-cover" in e["coverage_note"]
        s = e["scope"]
        assert s["varies"] == ["water_year_sample"]
        assert "forcing_product" in s["excludes"] and "interannual_persistence" in s["excludes"]
        assert "exchangeable_years" in s["assumes"]
        assert e["annual_lag1_autocorrelation"]["threshold"] > 0
    json.dumps(run())               # JSON-safe (no NaN leaks past the sanitiser)


def test_flow_cis_use_the_same_days_as_the_point_estimates(run):
    d = run()
    q_mm = sig._to_mm_per_day(_q(), AREA)
    assert d["_uncertainty"]["q_mean"]["n"] == int(q_mm.notna().sum())
    rr = d["_uncertainty"]["runoff_ratio"]
    assert rr["n"] == d["_precipitation"]["n_paired_days_used"]


def test_too_few_water_years_refuses_the_interval_with_a_reason(run):
    q = _q().iloc[: 6 * 365 + 5]
    d = run(q=q, p=_p_for(_q()).iloc[: 6 * 365 + 5])
    e = d["_uncertainty"]["q_mean"]
    assert e["ci_refused"] and e["ci_low"] is None and "groups" in e["ci_reason"]
    assert d["q_mean"] is not None          # the point estimate is still reported


def test_incomplete_water_year_is_in_neither_point_nor_ci(run):
    p = _p_for(_q())
    gap = (p.index >= "1999-10-01") & (p.index <= "2000-09-30")
    d = run(p=p[~gap])
    rec = d["_precipitation"]
    assert [e["water_year"] for e in rec["water_years_excluded"]] == [2000]
    rr = d["_uncertainty"]["runoff_ratio"]
    assert rr["n_groups"] == 19 and rr["ci_low"] <= d["runoff_ratio"] <= rr["ci_high"]
    assert rr["value"] == pytest.approx(d["runoff_ratio"], rel=1e-9)


def test_runoff_ratio_ci_is_paired_ratio_of_sums():
    # q and p constructed so that RR is exactly 0.5 in every year: the CI must be degenerate at 0.5
    idx = pd.date_range("1990-10-01", periods=15 * 365, freq="D")
    p = np.random.default_rng(0).gamma(1.0, 3.0, len(idx)) + 0.1
    q = 0.5 * p
    out = gu.water_balance_cis(q, p, idx, with_elasticity=False)["runoff_ratio"]
    assert out["value"] == pytest.approx(0.5) and out["ci_low"] == pytest.approx(0.5)
    assert out["ci_high"] == pytest.approx(0.5)


def test_elasticity_blocks_split_repeated_and_adjacent_years():
    yr = np.repeat([2001, 2002, 2003], 4)
    # year 2001 drawn twice in a row, then 2002 directly after the repeat
    idx = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7])
    assert gu._blocks(idx, yr).tolist() == [0, 4, 8]
    # adjacent years drawn in order (3 -> 4 continues the run but changes the label)
    assert gu._blocks(np.arange(12), yr).tolist() == [0, 4, 8]


def test_near_zero_dp_year_is_reported_as_degraded():
    idx = pd.date_range("1990-10-01", periods=12 * 365, freq="D")
    yrs = gu.water_year(idx)
    rng = np.random.default_rng(3)
    scale = 1 + 0.3 * rng.standard_normal(12)
    scale[0] = 1.0                       # force a year whose P equals the mean of means roughly
    pw = (np.array([scale[y - yrs.min()] for y in yrs]) * 3.0)
    p = pw + 0.0
    q = 0.5 * p + rng.random(len(idx)) * 0.2
    e = gu.water_balance_cis(q, p, idx)["stream_elas"]
    assert e["indicative"] and isinstance(e["n_years_near_zero_dP"], int)
    if e["n_years_near_zero_dP"]:
        assert e["ci_degraded"] is True and "unstable" in e["ci_reason"]


def test_lag1_autocorrelation_reported_and_flags_persistence():
    idx = pd.date_range("1990-10-01", periods=25 * 365, freq="D")
    yrs = gu.water_year(idx)
    n = yrs.max() - yrs.min() + 1
    ann = np.linspace(10, 30, n) + np.random.default_rng(1).standard_normal(n)   # a drifting annual series
    q = ann[yrs - yrs.min()] + 0.01
    e = gu.flow_statistic_cis(q, idx)["q_mean"]
    assert e["annual_lag1_autocorrelation"]["r1"] > 0.5 and e["annual_lag1_autocorrelation"]["high"] is True


# ---- served series + qualifier exposure ----------------------------------------------------

def _flags(q, frac_estimated=0.3, seed=5):
    rng = np.random.default_rng(seed)
    idx = q.index
    q_mm = q.to_numpy()
    est = q_mm <= np.quantile(q_mm, 0.10)         # estimate every low-flow day
    codes = np.where(est, "e", "").astype(object)
    codes[rng.random(len(idx)) < 0.02] = "ICE"
    return pd.DataFrame({"approval_code": np.where(idx < "2008-10-01", "A", "P"),
                         "qualifier_code": codes}, index=idx)


def test_served_series_and_qualifier_exposure_for_a_fetched_gauge(run):
    q = _q()
    d = run(q=q, flags=_flags(q), gauge="01013500")
    s = d["_served_series"]
    assert len(s["dates"]) == len(s["q_cms"]) == len(s["qualifier_codes"]) == 7305
    assert s["dates"][0] == "1989-10-01" and s["q_cms"][0] == pytest.approx(q.iloc[0])
    assert set(s["qualifier_codes"]) >= {"", "e", "ICE"} and set(s["approval_codes"]) == {"A", "P"}
    qe = d["qualifier_exposure"]["per_signature"]
    # every low-flow day is estimated -> q5 rests (almost) entirely on estimated days; q_mean does not
    assert qe["q5"]["estimated_share"] > 0.9
    assert qe["q_mean"]["estimated_share"] == pytest.approx(0.10, abs=0.01)
    assert qe["q5"]["determining_days"] == "low_tail" and qe["q5"]["definition"]
    assert qe["zero_q_freq"]["n_days"] == 0 or qe["zero_q_freq"]["share"] is not None
    assert d["qualifier_exposure"]["qualifier_token_counts"]["ICE"] > 0
    json.dumps(d)


def test_caller_supplied_series_has_served_series_but_no_exposure(run):
    d = run()
    assert d["_served_series"]["qualifier_codes"] is None
    assert "qualifier_exposure" not in d


# ---- reference_comparison ------------------------------------------------------------------

def _sigs(run, **over):
    d = dict(run())
    d.update(over)
    return d


def test_reference_comparison_flags_area_forcing_definition_and_period(run):
    d = _sigs(run)
    cmp_ = reference_comparison(d, REF_01013500, AREA, AREA_GAGES2,
                                (START, END), ("1989-10-01", "2009-09-30"))
    t = cmp_["table"]
    assert t["q_mean"]["area_dependent"] and "area_differs" in t["q_mean"]["blockers"]
    assert t["q_mean"]["value_at_ref_area"] == pytest.approx(d["q_mean"] * AREA / AREA_GAGES2)
    assert not t["baseflow_index"]["area_dependent"]
    assert t["slope_fdc"]["definition_comparable"] is True
    assert t["baseflow_index"]["definition_comparable"] is True
    assert t["baseflow_index"]["blockers"] == [] and t["baseflow_index"]["can_claim_difference"]
    assert t["slope_fdc"]["rel_diff"] == pytest.approx(0.0, abs=1e-6)
    assert t["runoff_ratio"]["forcing_differs"] is True and "forcing_differs" in t["runoff_ratio"]["blockers"]
    assert not t["runoff_ratio"]["can_claim_difference"] and not t["stream_elas"]["can_claim_difference"]
    assert cmp_["context"]["period_differs"] is False
    assert cmp_["climate_context"] == {k: REF_01013500[k] for k in ("frac_snow", "aridity", "p_seasonality")}


def test_reference_comparison_period_differs_blocks_everything(run):
    cmp_ = reference_comparison(_sigs(run), REF_01013500, AREA_GAGES2, AREA_GAGES2,
                                ("1999-10-01", "2020-09-30"), ("1989-10-01", "2009-09-30"))
    assert cmp_["context"]["period_differs"] is True
    assert all("period_differs" in r["blockers"] and not r["can_claim_difference"]
               for r in cmp_["table"].values())


def test_reference_comparison_unknown_definition_for_pre_020_results(run):
    d = _sigs(run)
    d.pop("signature_definitions")
    t = reference_comparison(d, REF_01013500, AREA_GAGES2, AREA_GAGES2, (START, END))["table"]
    assert t["slope_fdc"]["definition_comparable"] is None
    assert "definition_unknown" in t["slope_fdc"]["blockers"]


def test_reference_comparison_sawicz_slope_is_not_comparable(run):
    d = _sigs(run)
    d["signature_definitions"] = dict(d["signature_definitions"])
    d["signature_definitions"]["slope_fdc"] = sig.SIGNATURE_DEFINITIONS["slope_fdc_sawicz2011"]
    t = reference_comparison(d, REF_01013500, AREA_GAGES2, AREA_GAGES2, (START, END))["table"]
    assert t["slope_fdc"]["definition_comparable"] is False
