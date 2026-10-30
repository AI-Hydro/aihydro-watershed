"""
Flow-Duration Curve (FDC), percentile flows, and low-flow design statistics.

Pure-compute kernels over a daily streamflow series — no I/O, no session, no
network. The MCP wrapper in tools_analysis.py supplies the array from
session.streamflow.

A flow-duration curve plots discharge against the fraction of time it is equalled
or exceeded. It is one of the most informative single summaries of a flow regime:
its high-exceedance tail (Q90–Q99) characterises droughts/baseflow, its
low-exceedance tail (Q1–Q10) characterises floods, and its mid-section slope
indicates flashiness (steep = flashy, flat = damped/groundwater-fed).

Conventions
-----------
Percentile flow ``Qxx`` is the discharge **equalled or exceeded xx% of the time**.
So Q95 is a low flow (exceeded 95% of days → small) and Q5 is a high flow
(exceeded only 5% of days → large). This is the standard hydrologic convention
(WMO-168), and is the inverse of a statistical "95th percentile".

Do not confuse these exceedance flows with the CAMELS signatures ``q5`` / ``q95``
produced by ``signatures.compute_flow_stats_camels``: there ``q5`` is the 5% flow
quantile (low flow) and ``q95`` the 95% quantile (high flow), i.e. CAMELS ``q5``
equals the exceedance flow ``Q95`` and CAMELS ``q95`` equals ``Q5`` here.

The mid-section ``slope_fdc`` matches ``signatures.compute_slope_fdc_camels`` (one kernel,
defined here): since 0.2.0 it is the CAMELS attributes v2.0 distributed definition
``(Q66 - Q33) / (mean(q) * 0.33)`` (non-exceedance percentiles, type 7, zeros kept, no
logarithm; naddor/camels@88e7a0d2d2 ``hydro_signatures.R:97``). The log-space value of Sawicz
et al. (2011), which was ``slope_fdc`` before 0.2.0, is ``slope_fdc_sawicz2011``.

``7Q10`` is the classic design low flow: the annual minimum 7-day mean discharge
with a 10-year recurrence interval (i.e. a 10% annual non-exceedance probability).
Widely used for water-quality permitting and minimum-flow regulation.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Standard percentile-flow exceedance points (percent of time exceeded).
_STANDARD_EXCEEDANCE = (5, 10, 25, 50, 75, 90, 95)


def flow_duration_curve(
    q: Sequence[float],
    exceedance_points: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Compute the flow-duration curve and standard percentile flows.

    Parameters
    ----------
    q : sequence of float
        Daily mean discharge (any consistent unit, typically m³/s). NaNs and
        negatives are dropped before computation.
    exceedance_points : sequence of float, optional
        Exceedance percentages to report as percentile flows. Defaults to
        (5, 10, 25, 50, 75, 90, 95).

    Returns
    -------
    dict with keys:
        exceedance_prob   : list[float]  — exceedance probability (0–1) per
                            sorted (descending) flow, via Weibull plotting position
                            rank / (n + 1).
        sorted_flows      : list[float]  — flows sorted descending (the FDC y-axis).
        percentile_flows  : dict[str, float]  — {"Q5": ..., "Q50": ..., "Q95": ...}.
        slope_fdc         : float  — (Q66 - Q33) / (mean(q) * 0.33), the CAMELS attributes v2.0 definition.
        slope_fdc_sawicz2011 : float — log-FDC slope between 33% and 66% exceedance (Sawicz 2011).
        n_days            : int
    """
    arr = np.asarray(q, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    n = arr.size
    if n < 2:
        raise ValueError(
            f"flow_duration_curve needs at least 2 valid discharge values, got {n}."
        )

    points = tuple(exceedance_points) if exceedance_points is not None else _STANDARD_EXCEEDANCE

    # FDC y-axis: flows sorted descending; x-axis: Weibull exceedance prob.
    sorted_desc = np.sort(arr)[::-1]
    ranks = np.arange(1, n + 1)
    exceedance = ranks / (n + 1)

    # Percentile flow Qxx = flow exceeded xx% of the time = quantile at (1 - xx/100).
    percentile_flows: Dict[str, float] = {}
    for p in points:
        qval = float(np.quantile(arr, 1.0 - p / 100.0))
        percentile_flows[f"Q{int(p)}"] = round(qval, 6)

    # Mid-section slope (same kernels as signatures.compute_slope_fdc_camels).
    slope = _fdc_slope(arr)
    slope_sawicz = fdc_slope_sawicz2011(arr)

    return {
        "exceedance_prob": exceedance.tolist(),
        "sorted_flows": sorted_desc.tolist(),
        "percentile_flows": percentile_flows,
        "slope_fdc": slope,
        "slope_fdc_sawicz2011": slope_sawicz,
        "n_days": int(n),
    }


def fdc_slope_camels_file(arr: np.ndarray) -> float:
    """``(Q66 - Q33) / (mean(q) * 0.33)``: CAMELS attributes v2.0 distributed ``slope_fdc``.

    Non-exceedance percentiles with linear interpolation (R type 7 / numpy default); zero-flow
    days kept; NaN unless there are >= 100 finite values and ``mean(q) > 0``.
    """
    q = np.asarray(arr, dtype=float)
    q = q[np.isfinite(q)]
    if q.size < 100:
        return float("nan")
    mean_q = float(np.mean(q))
    if not mean_q > 0:
        return float("nan")
    return float((np.quantile(q, 0.66) - np.quantile(q, 0.33)) / (mean_q * 0.33))


def fdc_slope_sawicz2011(arr: np.ndarray) -> float:
    """Log-space slope between 33% and 66% exceedance (Sawicz et al., 2011); zeros dropped.

    ``(ln Q33exc - ln Q66exc) / 0.33`` with the exceedance points taken as non-exceedance
    quantiles 0.67 and 0.34 (type 7). NaN unless there are >= 100 positive values.
    """
    q = np.asarray(arr, dtype=float)
    pos = q[np.isfinite(q) & (q > 0)]
    if pos.size < 100:
        return float("nan")
    q33 = np.quantile(pos, 0.67)   # flow exceeded 33% of the time
    q66 = np.quantile(pos, 0.34)   # flow exceeded 66% of the time
    if q33 <= 0 or q66 <= 0:
        return float("nan")
    return float((np.log(q33) - np.log(q66)) / (0.66 - 0.33))


def _fdc_slope(arr: np.ndarray) -> float:
    """Kept name: the CAMELS attributes v2.0 ``slope_fdc`` (see :func:`fdc_slope_camels_file`)."""
    return fdc_slope_camels_file(arr)


def seven_q10(
    q: Sequence[float],
    dates: Optional[Sequence] = None,
    min_years: int = 10,
    year_type: str = "climatic_year",
    min_fraction: Optional[float] = None,
    min_window_fraction: Optional[float] = None,
    calendar: Optional[str] = None,
    year_start_month: Optional[int] = None,
) -> Dict[str, object]:
    """Compute the 7Q10 design low flow (annual 7-day min, 10-yr recurrence).

    The year is the **climatic year** (1 Apr - 31 Mar) by default, the USGS
    convention for annual low flows (USGS SIR 2008-5126 and 2006-5035, section 3);
    it is labelled by the calendar year in which it BEGINS. A low-flow season that
    straddles 31 Dec is therefore one year, as it would not be in a calendar or
    water year. ``year_type`` may be 'water_year', 'calendar_year' or 'custom_year'
    (with ``year_start_month``, e.g. for Southern Hemisphere or monsoon regimes).

    The 7-day mean is trailing and is formed on the calendar-day axis: a window
    that contains a missing day is NOT evaluated (no splicing across a gap). A
    year is used only if BOTH (a) at least ``min_fraction`` (default 0.95) of its
    days are observed AND (b) at least ``min_window_fraction`` (default 0.95) of
    the 7-day windows ending in it were evaluable. (b) matters because one missing
    day removes 7 windows: 18 scattered missing days pass (a) yet can leave a third
    of the windows, and the annual minimum of what is left is biased HIGH (7Q10 read
    +108% in the methods review's test). Excluded years are listed with the reason;
    ``windows_per_year`` reports evaluated/possible windows for every year.
    ``calendar='no_leap_dec31'`` is for GridMET/Daymet-type calendars.

    Parameters
    ----------
    q : sequence of float
        Daily mean discharge (m³/s).
    dates : sequence of datetime-like, optional
        One date per discharge value. Required to group into years; if omitted,
        the series is chunked into consecutive 365-day blocks as a fallback
        (a warning is logged; no gap or completeness handling is possible).
    min_years : int
        Minimum number of annual minima required to fit (default 10). Below this,
        7Q10 is returned as None with a reason.
    year_type, min_fraction : see above.

    Returns
    -------
    dict with keys:
        seven_q10        : float | None  — the 7-day low flow with 10-yr recurrence.
        n_years          : int
        annual_7day_min  : list[float]
        years            : list[int] | None — label of each annual minimum
        method           : str
        note             : str | None
        year_type, year_convention, min_fraction, min_window_fraction, excluded_years,
        windows_per_year ({year: {evaluated, possible, fraction}})
    """
    arr = np.asarray(q, dtype=float)
    meta: Dict[str, object] = {"year_type": year_type, "year_convention": None,
                               "min_fraction": None, "min_window_fraction": None,
                               "excluded_years": [], "years": None, "windows_per_year": None}
    if arr.size < 7:
        return {"seven_q10": None, "n_years": 0, "annual_7day_min": [],
                "method": "log_pearson3", "note": "Fewer than 7 days of data.", **meta}

    if dates is not None:
        import pandas as pd
        from aihydro_watershed.signatures.periods import (
            complete_periods, _labels, day_index, absent_days, DEFAULT_MIN_FRACTION)
        comp = complete_periods(dates, arr, year_type, min_fraction, calendar=calendar,
                                year_start_month=year_start_month)
        wfrac = DEFAULT_MIN_FRACTION[year_type] if min_window_fraction is None else float(min_window_fraction)
        meta.update(year_convention=comp["convention"], min_fraction=comp["min_fraction"],
                    min_window_fraction=wfrac, excluded_years=[
                        {**e, "reason": "day_fraction"} for e in comp["excluded"]],
                    years_calendar_adjusted=comp["calendar_adjusted"], windows_per_year={})
        idx = day_index(dates)
        s = pd.Series(arr, index=idx)
        s = s[~s.index.duplicated(keep="first")]
        axis = pd.date_range(s.index.min(), s.index.max(), freq="D")
        ab = absent_days(axis[0], axis[-1], calendar)
        if len(ab):
            axis = axis.difference(ab)             # days the source never holds are not on the axis
        full = s.reindex(axis)
        # trailing 7-day mean on the calendar axis; a window with a missing day is NaN
        r7_all = full.rolling(7, min_periods=7).mean()
        r7_all = r7_all.iloc[6:]                   # windows with 6 predecessors on the axis
        yr = pd.Series(_labels(pd.DatetimeIndex(r7_all.index), year_type, year_start_month), index=r7_all.index)
        possible = yr.groupby(yr).size()
        evaluated = r7_all.notna().groupby(yr).sum()
        for y in possible.index:
            meta["windows_per_year"][int(y)] = {
                "evaluated": int(evaluated.get(y, 0)), "possible": int(possible[y]),
                "fraction": round(float(evaluated.get(y, 0)) / float(possible[y]), 4)}
        good = []
        for y in comp["complete"]:
            w = meta["windows_per_year"].get(int(y))
            if w is None or w["fraction"] < wfrac - 1e-12:
                meta["excluded_years"].append({
                    "period": int(y), "reason": "window_fraction",
                    "windows_evaluated": (w or {}).get("evaluated", 0),
                    "windows_possible": (w or {}).get("possible", 0),
                    "fraction": (w or {}).get("fraction", 0.0)})
            else:
                good.append(int(y))
        r7 = r7_all.dropna()
        ok = yr.reindex(r7.index).isin(good)
        grp = r7[ok].groupby(yr.reindex(r7.index)[ok]).min()
        years = [int(y) for y in grp.index]
        annual_min = grp.values.astype(float)
        meta["years"] = years
    else:
        log.warning("seven_q10: no dates supplied — chunking into 365-day blocks.")
        kernel = np.ones(7) / 7.0
        rolling7 = np.convolve(arr, kernel, mode="valid")
        n_full = rolling7.size // 365
        if n_full == 0:
            return {"seven_q10": None, "n_years": 0, "annual_7day_min": [],
                    "method": "log_pearson3", "note": "Less than one year of data.", **meta}
        annual_min = np.array([
            rolling7[i * 365:(i + 1) * 365].min() for i in range(n_full)
        ])
        meta["year_convention"] = "consecutive 365-day blocks (no dates supplied)"

    keep = ~np.isnan(annual_min)
    if meta["years"] is not None:
        meta["years"] = [y for y, k in zip(meta["years"], keep) if k]
    annual_min = annual_min[keep]
    n_years = annual_min.size
    if n_years < min_years:
        return {
            "seven_q10": None, "n_years": int(n_years),
            "annual_7day_min": [round(float(x), 6) for x in annual_min],
            "method": "log_pearson3",
            "note": f"Need ≥{min_years} years of annual minima for 7Q10; have {n_years}.",
            **meta,
        }

    val = _log_pearson3_low(annual_min, non_exceed_prob=0.10)
    return {
        "seven_q10": round(float(val), 6) if val is not None else None,
        "n_years": int(n_years),
        "annual_7day_min": [round(float(x), 6) for x in annual_min],
        "method": "log_pearson3",
        "note": None,
        **meta,
    }


def _log_pearson3_low(annual_min: np.ndarray, non_exceed_prob: float = 0.10) -> Optional[float]:
    """Log-Pearson III estimate of the low flow at a given non-exceedance prob.

    Fits a Pearson-III distribution to log10(annual minima) and returns the flow
    whose annual non-exceedance probability is ``non_exceed_prob`` (0.10 → 10-yr
    low-flow recurrence). Falls back to a Weibull plotting-position empirical
    quantile if the series has zeros (log undefined) or the fit is degenerate.
    """
    pos = annual_min[annual_min > 0]
    if pos.size < 2 or pos.size < annual_min.size:
        # Zeros present or too few positives → empirical Weibull quantile.
        srt = np.sort(annual_min)
        m = srt.size
        pp = np.arange(1, m + 1) / (m + 1)            # non-exceedance plotting pos
        return float(np.interp(non_exceed_prob, pp, srt))
    try:
        from scipy.stats import pearson3
        logs = np.log10(pos)
        log_std = logs.std(ddof=1)
        # Degenerate (near-)constant series: variance ~0 → skew undefined and the
        # distribution collapses to a point. The low flow is just that constant.
        if log_std < 1e-9:
            return float(10.0 ** logs.mean())
        skew = float(((logs - logs.mean()) ** 3).mean() / (log_std ** 3))
        # pearson3 is parameterised by skew; loc/scale from mean/std of logs.
        q_log = pearson3.ppf(non_exceed_prob, skew, loc=logs.mean(), scale=log_std)
        if not np.isfinite(q_log):
            raise ValueError("non-finite pearson3 quantile")
        return float(10.0 ** q_log)
    except Exception as exc:
        log.warning("Log-Pearson III fit failed (%s); using empirical quantile.", exc)
        srt = np.sort(pos)
        m = srt.size
        pp = np.arange(1, m + 1) / (m + 1)
        return float(np.interp(non_exceed_prob, pp, srt))
