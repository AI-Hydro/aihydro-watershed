"""
Period completeness: the one rule every period statistic shares.

A statistic over a calendar month (SPI), a water year (annual flood maximum) or a
climatic year (7-day low flow) describes that period only if the period is
(almost) fully observed. A month that ends on the 3rd, a year with a four-month
hole, or a record that starts in July is a different quantity from the full
period: a partial month reads as drought, a partial year lowers an annual
minimum or misses the annual flood.

``complete_periods`` splits the periods a daily series touches into **complete**
(kept) and **excluded** (dropped, with the count of days they had), and returns
a boolean mask over the input so a caller can restrict its statistic to the
complete periods. It never imputes, re-indexes or splices: a missing day is a
missing day.

Days a source can never hold
----------------------------
GridMET (as served by pygridmet) and Daymet have no 31 Dec in leap years. Counting
that day as missing would drop every leap-year December (and with it 24% of SPI-12
values). ``calendar="no_leap_dec31"`` (or an explicit ``known_absent`` list of
dates) removes such days from a period's EXPECTED days; the adjustment is reported
(``calendar_adjusted``). A day that is absent for any other reason still counts as
missing and the period is excluded at the default 1.0 for months.

Conventions (``PERIOD_CONVENTIONS``)
------------------------------------
- ``month``          calendar month, label ``"YYYY-MM"``.
- ``calendar_year``  1 Jan - 31 Dec, label = the year.
- ``water_year``     1 Oct - 30 Sep, label = the calendar year in which it ENDS
                     (USGS convention; Bulletin 17C, Appendix 10, p. 104: "The water
                     year is designated by the calendar year in which it ends").
- ``climatic_year``  1 Apr - 31 Mar, label = the calendar year in which it BEGINS
                     (USGS SIR 2008-5126 section 3 and SIR 2006-5035 section 3 use the
                     climatic year for annual 7-day lows; the methods review reports the
                     begin-year labelling from SIR 2006-5035; the section-3 pages
                     themselves state the dates only).
- ``custom_year``    a year starting on ``year_start_month`` (1-12), 12 months long,
                     labelled by the calendar year in which it BEGINS. For basins whose
                     low-flow or flood season does not fit the October or April starts
                     (Southern Hemisphere, monsoon, snowmelt regimes); the default
                     presets are unchanged.

Thresholds (``DEFAULT_MIN_FRACTION``)
-------------------------------------
- ``month``: 1.0 (every day present). SOURCED: WMO-No. 1203 (2017) section 4.4.1(c)
  ("can only be calculated if there are complete data over the month") and WMO-TD 341,
  as cited by the methods review. The SPI User Guide (WMO-No. 1090) itself marks a
  missing value -99, asks for 20-30 years and gives no within-month rule.
- years: 0.95 (>= 95% of days present). ENGINEERING CHOICE, not sourced: the fraction the
  package already uses for the water-balance and grouped-CI years
  (``signatures.WATER_YEAR_MIN_PAIRED_FRACTION``); one rule, not two. Bulletin 17C states
  no daily-coverage threshold (it treats missing years by censoring, EMA).
Both are arguments and are echoed in every result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PERIOD_CONVENTIONS: Dict[str, str] = {
    "month": "calendar month (label YYYY-MM)",
    "calendar_year": "calendar year, 1 Jan - 31 Dec (label = year)",
    "water_year": "water year, 1 Oct - 30 Sep, labelled by the calendar year in which it ends (USGS)",
    "climatic_year": "climatic year, 1 Apr - 31 Mar, labelled by the calendar year in which it begins (USGS)",
    "custom_year": "12-month year starting on year_start_month, labelled by the calendar year in which it begins",
}

DEFAULT_MIN_FRACTION: Dict[str, float] = {
    "month": 1.0,
    "calendar_year": 0.95,
    "water_year": 0.95,
    "climatic_year": 0.95,
    "custom_year": 0.95,
}

_YEAR_START_MONTH = {"calendar_year": 1, "water_year": 10, "climatic_year": 4}

CALENDARS = {None, "no_leap_dec31"}


def day_index(dates: Sequence) -> pd.DatetimeIndex:
    """Whole-day naive DatetimeIndex of ``dates``; a tz-aware stamp keeps its local calendar day."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def convention_text(freq: str, year_start_month: Optional[int] = None) -> str:
    if freq == "custom_year" and year_start_month:
        return (f"12-month year starting on day 1 of month {int(year_start_month)}, "
                "labelled by the calendar year in which it begins")
    return PERIOD_CONVENTIONS[freq]


def _start_month(freq: str, year_start_month: Optional[int]) -> int:
    if freq == "custom_year":
        if year_start_month is None or not 1 <= int(year_start_month) <= 12:
            raise ValueError("custom_year needs year_start_month in 1..12")
        return int(year_start_month)
    return _YEAR_START_MONTH[freq]


def _labels(idx: pd.DatetimeIndex, freq: str, year_start_month: Optional[int] = None) -> np.ndarray:
    """Period label of every timestamp (int year, or 'YYYY-MM' for months)."""
    if freq == "month":
        return np.asarray(idx.strftime("%Y-%m"))
    start = _start_month(freq, year_start_month)
    year = np.asarray(idx.year)
    month = np.asarray(idx.month)
    if start == 1:
        return year.astype(int)
    if freq == "water_year":                       # labelled by END year
        return np.where(month >= start, year + 1, year).astype(int)
    return np.where(month >= start, year, year - 1).astype(int)   # climatic/custom: START year


def period_bounds(label: Any, freq: str, year_start_month: Optional[int] = None) -> tuple:
    """(first_day, last_day) of the period ``label`` as Timestamps."""
    if freq == "month":
        p = pd.Period(str(label), freq="M")
        return p.start_time.normalize(), p.end_time.normalize()
    y, start = int(label), _start_month(freq, year_start_month)
    if freq == "water_year":
        first = pd.Timestamp(year=y - 1, month=start, day=1)
    else:
        first = pd.Timestamp(year=y, month=start, day=1)
    return first, first + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def period_days(label: Any, freq: str, year_start_month: Optional[int] = None) -> int:
    a, b = period_bounds(label, freq, year_start_month)
    return int((b - a).days) + 1


def absent_days(lo: pd.Timestamp, hi: pd.Timestamp, calendar: Optional[str] = None,
                known_absent: Optional[Sequence] = None) -> pd.DatetimeIndex:
    """Days in [lo, hi] that the source can never hold (see module docstring)."""
    if calendar not in CALENDARS:
        raise ValueError(f"unknown calendar {calendar!r}; choose from {sorted(c for c in CALENDARS if c)}")
    days = []
    if calendar == "no_leap_dec31":
        days += [pd.Timestamp(year=y, month=12, day=31) for y in range(lo.year, hi.year + 1)
                 if pd.Timestamp(year=y, month=12, day=31).is_leap_year]
    if known_absent is not None and len(known_absent):
        days += list(day_index(known_absent))
    out = pd.DatetimeIndex(sorted(set(days)))
    return out[(out >= lo) & (out <= hi)] if len(out) else out


def complete_periods(
    dates: Sequence,
    values: Sequence[float],
    freq: str,
    min_fraction: Optional[float] = None,
    window: Optional[tuple] = None,
    calendar: Optional[str] = None,
    known_absent: Optional[Sequence] = None,
    year_start_month: Optional[int] = None,
) -> Dict[str, Any]:
    """Split the periods of a daily series into complete and excluded ones.

    Parameters
    ----------
    dates, values : parallel sequences. A day counts as observed when its date
        is present AND its value is finite. Duplicate dates count once. A tz-aware
        stamp is taken at its local calendar day.
    freq : one of ``PERIOD_CONVENTIONS``.
    min_fraction : smallest observed/expected day fraction for a period to be
        complete. Default per ``DEFAULT_MIN_FRACTION``.
    window : optional requested ``(start, end)``. Every period the window touches
        is examined, so a leading or trailing period with no data at all is
        listed as excluded instead of being silently absent. Without it the span
        of the observed days is used.
    calendar, known_absent : days the source can never hold (``"no_leap_dec31"``
        for GridMET/Daymet, and/or explicit dates). They leave the period's
        expected days and are reported in ``calendar_adjusted``.
    year_start_month : for ``freq="custom_year"``.

    Returns
    -------
    dict with
      ``freq``, ``convention`` (text), ``min_fraction``;
      ``complete``  : sorted list of labels kept;
      ``excluded``  : list of ``{"period", "observed_days", "expected_days",
                      "fraction"}`` for every dropped period;
      ``calendar_adjusted`` : list of ``{"period", "days_removed", "expected_days"}``
                      for periods whose expected days were reduced;
      ``keep``      : bool array over the INPUT, True for an observed day that lies
                      in a complete period (use it to restrict a statistic);
      ``labels``    : period label of every input day (same length as input);
      ``n_periods`` : complete + excluded.
    """
    if freq not in PERIOD_CONVENTIONS:
        raise ValueError(f"unknown period convention {freq!r}; choose from {sorted(PERIOD_CONVENTIONS)}")
    frac = DEFAULT_MIN_FRACTION[freq] if min_fraction is None else float(min_fraction)
    if not 0.0 < frac <= 1.0:
        raise ValueError("min_fraction must be in (0, 1]")

    idx = day_index(dates)
    vals = np.asarray(values, dtype=float)
    if idx.size != vals.size:
        raise ValueError("dates and values must be the same length")
    finite = np.isfinite(vals)
    labels = _labels(idx, freq, year_start_month) if idx.size else np.array([])

    out: Dict[str, Any] = {
        "freq": freq, "convention": convention_text(freq, year_start_month), "min_fraction": frac,
        "complete": [], "excluded": [], "calendar_adjusted": [],
        "keep": np.zeros(idx.size, dtype=bool), "labels": labels, "n_periods": 0,
    }
    has_window = bool(window and window[0] and window[1])
    if not finite.any() and not has_window:
        return out

    lo_day = idx[finite].min() if finite.any() else None
    hi_day = idx[finite].max() if finite.any() else None
    if has_window:
        w0, w1 = pd.Timestamp(window[0]).normalize(), pd.Timestamp(window[1]).normalize()
        lo_day = w0 if lo_day is None else min(lo_day, w0)
        hi_day = w1 if hi_day is None else max(hi_day, w1)
    span = pd.date_range(lo_day, hi_day, freq="D")
    all_labels = list(dict.fromkeys(_labels(span, freq, year_start_month).tolist()))

    absent = absent_days(span[0] - pd.Timedelta(days=366), span[-1] + pd.Timedelta(days=366),
                         calendar, known_absent)
    absent_labels = _labels(absent, freq, year_start_month) if len(absent) else np.array([])
    absent_count = pd.Series(1, index=absent_labels).groupby(level=0).sum().to_dict() if len(absent) else {}

    # observed (unique) days per period; a day the source can never hold is not "observed" either
    obs = pd.DataFrame({"d": idx[finite], "l": labels[finite]}).drop_duplicates("d")
    if len(absent):
        obs = obs[~obs["d"].isin(absent)]
    counts = obs.groupby("l")["d"].size().to_dict() if len(obs) else {}

    complete: List[Any] = []
    excluded: List[dict] = []
    adjusted: List[dict] = []
    for lab in all_labels:
        full = period_days(lab, freq, year_start_month)
        removed = int(absent_count.get(lab, 0))
        expected = full - removed
        key = lab if freq == "month" else int(lab)
        if removed:
            adjusted.append({"period": key, "days_removed": removed, "expected_days": expected})
        n = int(counts.get(lab, 0))
        f = n / expected if expected else 0.0
        if f >= frac - 1e-12:
            complete.append(lab)
        else:
            excluded.append({"period": key, "observed_days": n, "expected_days": expected,
                             "fraction": round(f, 4)})
    complete_set = set(complete)
    out["complete"] = [lab if freq == "month" else int(lab) for lab in complete]
    out["excluded"] = excluded
    out["calendar_adjusted"] = adjusted
    out["keep"] = finite & np.fromiter((l in complete_set for l in labels), dtype=bool, count=labels.size)
    out["n_periods"] = len(all_labels)
    return out
