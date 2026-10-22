"""Water-year grouped confidence intervals for the signatures (Science Wave 3, PR-3b).

Every interval here resamples whole water years with replacement (aihydro-core
``grouped_bootstrap``, percentile, B >= 2000). What that does and does not cover travels with
each interval as the machine-readable ``scope`` (``varies`` / ``assumes`` / ``excludes``).

The point estimate is ``fn(all items)`` on the SAME item set the reported signature was
computed from, so the interval is built around the value that is reported (critique M7):

* flow statistics and BFI: every valid streamflow day;
* ``runoff_ratio`` and ``stream_elas``: the paired q/p days of the COMPLETE water years only
  (>= 95% paired days; see ``signatures.WATER_YEAR_MIN_PAIRED_FRACTION``).

Core's diagnostics (``valid_frac``, ``ci_degraded``, ``ci_reason``, ``low_n``,
``coverage_note``) are passed through unchanged; fewer than ``MIN_GROUPS`` water years means no
interval is produced and the entry says why.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from aihydro_core.science._bootstrap import COVERAGE_NOTE
from aihydro_core.science import (
    annual_lag1_autocorrelation,
    grouped_bootstrap,
    water_year,
    water_year_sampling_scope,
)

log = logging.getLogger(__name__)

N_BOOTSTRAP = 2000
CI_LEVEL = 0.90
MIN_GROUPS = 10          # critique OQ4: fewer than 10 COMPLETE water years -> refuse the interval
#: A water year counts as a complete group when it has at least this fraction of its days.
#: ENGINEERING CHOICE; SOURCE TO VERIFY (same rule as the water-balance years).
COMPLETE_GROUP_FRACTION = 0.95

#: |dP / mean P| below this marks a year whose elasticity ratio is unstable (the ratio blows
#: up as dP -> 0). ENGINEERING THRESHOLD; SOURCE TO VERIFY.
NEAR_ZERO_DP_FRACTION = 0.02

SCOPE_NOTE = ("inter-annual sampling only: resamples water years as exchangeable draws of a fixed "
              "period; excludes forcing product, rating/measurement error, catchment area, method "
              "definition, between-year persistence and nonstationarity (see scope)")

METRIC_FAMILY_NOTE = {
    "baseflow_index": ("the filter is applied once to the observed series (re-running it on a spliced "
                       "resample would read every splice as quick flow); the interval excludes "
                       "filter-state and parameter uncertainty"),
    "stream_elas": ("indicative: median of ~one ratio per water year, so its bootstrap distribution is "
                    "discrete and years with dP near zero make ratios unstable"),
}


def _clean(o):
    """NaN/inf -> None, numpy scalars -> python, recursively (JSON-safe)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _wy_days(label: int) -> int:
    return int((pd.Timestamp(year=label, month=10, day=1) - pd.Timestamp(year=label - 1, month=10, day=1)).days)


def group_completeness(groups: np.ndarray) -> dict:
    """Complete / partial water-year groups of an item-level label array (one item = one day).

    A group is complete when it holds >= ``COMPLETE_GROUP_FRACTION`` of its calendar days.
    Partial groups (a window that does not start on 1 Oct, a stub year) stay in the point estimate
    and the resample (critique M7) but do not count towards ``MIN_GROUPS`` or the reported ``n_groups``."""
    labels, counts = np.unique(groups, return_counts=True)
    complete = [int(c) / _wy_days(int(l)) >= COMPLETE_GROUP_FRACTION - 1e-12 for l, c in zip(labels, counts)]
    n_c = int(sum(complete))
    return {"n_groups_complete": n_c, "n_partial_groups": int(labels.size - n_c),
            "n_groups_all": int(labels.size)}


def _refusal(name: str, value: Optional[float], n_items: int, comp: dict, reason: str) -> dict:
    """Same keys as a computed entry (so consumers need no special case), with the interval withheld."""
    return {"value": value, "ci_low": None, "ci_high": None, "method": "bootstrap_group",
            "n": int(n_items), "ci_level": CI_LEVEL, "n_groups": comp["n_groups_complete"],
            "group": "water_year", "B": N_BOOTSTRAP, "B_valid": 0, "valid_frac": None,
            "ci_refused": True, "ci_degraded": True, "ci_reason": reason, "low_n": True,
            "coverage_note": COVERAGE_NOTE, "scope": water_year_sampling_scope(), **comp}


def _run_group(fns: Dict[str, Callable], n_items: int, groups: np.ndarray, lag1: dict,
               point: Dict[str, Optional[float]]) -> Dict[str, dict]:
    """Grouped bootstrap of ``fns`` with refusal records instead of exceptions.

    The ``MIN_GROUPS`` gate counts COMPLETE water years only; every valid item still takes part
    in the point estimate and the resample."""
    comp = group_completeness(groups)
    extras = {"annual_lag1_autocorrelation": lag1, "scope_note": SCOPE_NOTE}
    if comp["n_groups_complete"] < MIN_GROUPS:
        why = (f"needs >={MIN_GROUPS} complete water years (>= {COMPLETE_GROUP_FRACTION:.0%} of days), got "
               f"{comp['n_groups_complete']} complete + {comp['n_partial_groups']} partial. "
               "Refusing to report an interval from too few groups.")
        log.warning("signatures: grouped CI refused: %s", why)
        return {k: {**_refusal(k, point.get(k), n_items, comp, why), **extras} for k in fns}
    try:
        res = grouped_bootstrap(fns, n_items, groups, n=N_BOOTSTRAP, ci=CI_LEVEL, min_groups=MIN_GROUPS)
    except ValueError as exc:
        log.warning("signatures: grouped CI refused: %s", exc)
        return {k: {**_refusal(k, point.get(k), n_items, comp, str(exc)), **extras} for k in fns}
    out = {}
    for k, v in res.items():
        d = dict(v)
        d.update(extras)
        d.update(comp)
        d["n_groups"] = comp["n_groups_complete"]       # complete groups only; n_groups_all has every group
        d["low_n"] = bool(comp["n_groups_complete"] < 30)
        if k in METRIC_FAMILY_NOTE:
            d["note"] = METRIC_FAMILY_NOTE[k]
        out[k] = d
    return out


def flow_statistic_cis(q: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, dict]:
    """q_mean, q_median, q5, q95, flow_variability over water-year groups of the valid days."""
    groups = water_year(dates)

    def _cv(idx):
        x = q[idx]
        m = float(np.mean(x))
        return float(np.std(x) / m) if m > 0 else float("nan")

    fns = {
        "q_mean": lambda idx: float(np.mean(q[idx])),
        "q_median": lambda idx: float(np.median(q[idx])),
        "q5": lambda idx: float(np.quantile(q[idx], 0.05)),
        "q95": lambda idx: float(np.quantile(q[idx], 0.95)),
        "flow_variability": _cv,
    }
    lag1 = annual_lag1_autocorrelation(q, groups)
    point = {k: f(np.arange(q.size)) for k, f in fns.items()}
    return _clean(_run_group(fns, q.size, groups, lag1, point))


def baseflow_index_ci(bf: np.ndarray, q: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, dict]:
    """BFI = sum(bf)/sum(q) over water-year groups; ``bf``/``q`` are paired arrays of the days that
    have a baseflow value (filter applied once to the observed series)."""
    groups = water_year(dates)
    fns = {"baseflow_index": lambda idx: float(np.sum(bf[idx]) / np.sum(q[idx]))}
    lag1 = annual_lag1_autocorrelation(q, groups)
    point = {k: f(np.arange(q.size)) for k, f in fns.items()}
    return _clean(_run_group(fns, q.size, groups, lag1, point))


def _blocks(idx: np.ndarray, yr: np.ndarray) -> np.ndarray:
    """Start offsets of each drawn water-year occurrence inside a concatenated index array.
    A boundary is a change of year label or a break in the index run (a year drawn twice in a
    row repeats its first index, so it is split too)."""
    lab = yr[idx]
    cut = (lab[1:] != lab[:-1]) | (np.diff(idx) != 1)
    return np.concatenate([[0], np.flatnonzero(cut) + 1])


def _annual_ratios(q: np.ndarray, p: np.ndarray, idx: np.ndarray, yr: np.ndarray):
    """Per drawn-year (mean q, mean p, (dQ/mean Q) / (dP/mean P)), anomalies against the
    long-term mean over all drawn DAYS (Sankarasubramanian et al. 2001; exactly CAMELS ``e_qp_sanka``:
    ``mp_tot = mean(p[avail])``, day-weighted, not the mean of the annual means). Duplicated years
    contribute duplicated ratios, and their days duplicated weight."""
    starts = _blocks(idx, yr)
    sizes = np.diff(np.concatenate([starts, [idx.size]]))
    mq = np.add.reduceat(q[idx], starts) / sizes
    mp = np.add.reduceat(p[idx], starts) / sizes
    mq_tot, mp_tot = float(np.mean(q[idx])), float(np.mean(p[idx]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ((mq - mq_tot) / mq_tot) / ((mp - mp_tot) / mp_tot)
        dp_rel = (mp - mp_tot) / mp_tot
    return mq, mp, ratio, dp_rel


def water_balance_cis(q: np.ndarray, p: np.ndarray, dates: pd.DatetimeIndex,
                      with_elasticity: bool = True) -> Dict[str, dict]:
    """runoff_ratio (sum q / sum p, paired) and stream_elas over water-year groups of the paired
    days of the complete years. ``q``, ``p`` and ``dates`` are the exact arrays the point
    estimates were computed from."""
    groups = water_year(dates)
    years = np.unique(groups)
    yr = groups

    def _rr(idx):
        return float(np.sum(q[idx]) / np.sum(p[idx]))

    def _elas(idx):
        _, _, ratio, _ = _annual_ratios(q, p, idx, yr)
        ratio = ratio[np.isfinite(ratio)]
        return float(np.median(ratio)) if ratio.size else float("nan")

    fns: Dict[str, Callable] = {"runoff_ratio": _rr}
    if with_elasticity:
        fns["stream_elas"] = _elas
    # annual runoff-ratio series for the persistence diagnostic (consecutive complete years)
    span = range(int(years.min()), int(years.max()) + 1)
    rr_by = {int(y): float(q[groups == y].sum() / p[groups == y].sum()) for y in years}
    lag1 = annual_lag1_autocorrelation(np.array([rr_by.get(y, np.nan) for y in span]))
    all_idx = np.arange(q.size)
    point = {k: f(all_idx) for k, f in fns.items()}
    out = _run_group(fns, q.size, groups, lag1, point)
    if with_elasticity and "stream_elas" in out:
        _, _, _, dp_rel = _annual_ratios(q, p, all_idx, yr)
        n_near = int(np.sum(np.abs(dp_rel) < NEAR_ZERO_DP_FRACTION))
        out["stream_elas"].update(
            indicative=True, n_years=int(years.size), n_years_near_zero_dP=n_near,
            near_zero_dP_threshold=NEAR_ZERO_DP_FRACTION,
            near_zero_dP_threshold_basis="|dP / mean P| below this; engineering threshold, source to verify")
        if n_near:
            # Reported, never hidden: those years' ratios are unstable, so the interval is degraded.
            why = (f"{n_near} water year(s) have |dP/mean P| < {NEAR_ZERO_DP_FRACTION:g}; "
                   "their elasticity ratios are unstable")
            e = out["stream_elas"]
            e["ci_degraded"] = True
            e["ci_reason"] = (e.get("ci_reason") + "; " if e.get("ci_reason") else "") + why
    return _clean(out)


def elasticity_point(q: np.ndarray, p: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """Elasticity by the same code path as the CI (used by tests to prove CI/point agreement)."""
    yr = water_year(dates)
    _, _, ratio, _ = _annual_ratios(q, p, np.arange(q.size), yr)
    ratio = ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else float("nan")
