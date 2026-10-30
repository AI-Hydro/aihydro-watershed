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

Conventions (``PERIOD_CONVENTIONS``)
------------------------------------
- ``month``          calendar month, label ``"YYYY-MM"``.
- ``calendar_year``  1 Jan - 31 Dec, label = the year.
- ``water_year``     1 Oct - 30 Sep, label = the calendar year in which it ENDS
                     (USGS convention; Bulletin 17C, Appendix on EMA: "The water
                     year is designated by the calendar year in which it ends").
- ``climatic_year``  1 Apr - 31 Mar, label = the calendar year in which it BEGINS.
                     USGS SIR 2008-5126 section 3 uses the climatic year (April 1
                     to March 31) for annual 7-day low flows. The page does not
                     state how the year is labelled; labelling by the start year
                     is OUR choice and is reported in the result.

Thresholds (``DEFAULT_MIN_FRACTION``) - ENGINEERING CHOICES, not sourced:
- ``month``: 1.0 (every day present). A precipitation total with a missing day is
  a lower bound, and the missing day may be the wettest; a month is therefore
  excluded unless every one of its days has a finite value.
- years: 0.95 (>= 95% of days present). The same fraction the package already
  uses for the water-balance and grouped-CI years
  (``signatures.WATER_YEAR_MIN_PAIRED_FRACTION``); one rule, not two.
Neither is taken from the SPI User Guide (WMO-No. 1090) or Bulletin 17C, which
were not found to state a daily-coverage threshold. Both are exposed as
arguments and echoed in every result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PERIOD_CONVENTIONS: Dict[str, str] = {
    "month": "calendar month (label YYYY-MM)",
    "calendar_year": "calendar year, 1 Jan - 31 Dec (label = year)",
    "water_year": "water year, 1 Oct - 30 Sep, labelled by the calendar year in which it ends (USGS)",
    "climatic_year": "climatic year, 1 Apr - 31 Mar, labelled by the calendar year in which it begins",
}

DEFAULT_MIN_FRACTION: Dict[str, float] = {
    "month": 1.0,
    "calendar_year": 0.95,
    "water_year": 0.95,
    "climatic_year": 0.95,
}

_YEAR_START_MONTH = {"calendar_year": 1, "water_year": 10, "climatic_year": 4}


def _labels(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Period label of every timestamp (int year, or 'YYYY-MM' for months)."""
    if freq == "month":
        return np.asarray(idx.strftime("%Y-%m"))
    start = _YEAR_START_MONTH[freq]
    year = np.asarray(idx.year)
    month = np.asarray(idx.month)
    if start == 1:
        return year.astype(int)
    if freq == "water_year":                       # labelled by END year
        return np.where(month >= start, year + 1, year).astype(int)
    return np.where(month >= start, year, year - 1).astype(int)   # climatic: START year


def period_bounds(label: Any, freq: str) -> tuple:
    """(first_day, last_day) of the period ``label`` as Timestamps."""
    if freq == "month":
        p = pd.Period(str(label), freq="M")
        return p.start_time.normalize(), p.end_time.normalize()
    y, start = int(label), _YEAR_START_MONTH[freq]
    if freq == "water_year":
        first = pd.Timestamp(year=y - 1, month=start, day=1)
    else:
        first = pd.Timestamp(year=y, month=start, day=1)
    return first, first + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def period_days(label: Any, freq: str) -> int:
    a, b = period_bounds(label, freq)
    return int((b - a).days) + 1


def complete_periods(
    dates: Sequence,
    values: Sequence[float],
    freq: str,
    min_fraction: Optional[float] = None,
    window: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Split the periods of a daily series into complete and excluded ones.

    Parameters
    ----------
    dates, values : parallel sequences. A day counts as observed when its date
        is present AND its value is finite. Duplicate dates count once.
    freq : one of ``PERIOD_CONVENTIONS``.
    min_fraction : smallest observed/expected day fraction for a period to be
        complete. Default per ``DEFAULT_MIN_FRACTION``.
    window : optional requested ``(start, end)``. Every period the window touches
        is examined, so a leading or trailing period with no data at all is
        listed as excluded instead of being silently absent. Without it the span
        of the observed days is used.

    Returns
    -------
    dict with
      ``freq``, ``convention`` (text), ``min_fraction``;
      ``complete``  : sorted list of labels kept;
      ``excluded``  : list of ``{"period", "observed_days", "expected_days",
                      "fraction"}`` for every dropped period;
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

    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)).values)).normalize()
    vals = np.asarray(values, dtype=float)
    if idx.size != vals.size:
        raise ValueError("dates and values must be the same length")
    finite = np.isfinite(vals)
    labels = _labels(idx, freq) if idx.size else np.array([])

    out: Dict[str, Any] = {
        "freq": freq, "convention": PERIOD_CONVENTIONS[freq], "min_fraction": frac,
        "complete": [], "excluded": [], "keep": np.zeros(idx.size, dtype=bool),
        "labels": labels, "n_periods": 0,
    }
    if not finite.any() and not window:
        return out

    # observed (unique) days per period
    obs = pd.DataFrame({"d": idx[finite], "l": labels[finite]}).drop_duplicates("d")
    counts = obs.groupby("l")["d"].size().to_dict()

    # the labels to examine: every period between the first and last observed (or windowed) day
    lo_day = idx[finite].min() if finite.any() else None
    hi_day = idx[finite].max() if finite.any() else None
    if window and window[0] and window[1]:
        w0, w1 = pd.Timestamp(window[0]).normalize(), pd.Timestamp(window[1]).normalize()
        lo_day = w0 if lo_day is None else min(lo_day, w0)
        hi_day = w1 if hi_day is None else max(hi_day, w1)
    span = pd.date_range(lo_day, hi_day, freq="D")
    all_labels = list(dict.fromkeys(_labels(span, freq).tolist()))

    complete: List[Any] = []
    excluded: List[dict] = []
    for lab in all_labels:
        expected = period_days(lab, freq)
        n = int(counts.get(lab, 0))
        f = n / expected
        if f >= frac - 1e-12:
            complete.append(lab)
        else:
            excluded.append({"period": lab if freq == "month" else int(lab),
                             "observed_days": n, "expected_days": expected, "fraction": round(f, 4)})
    complete_set = set(complete)
    out["complete"] = [lab if freq == "month" else int(lab) for lab in complete]
    out["excluded"] = excluded
    out["keep"] = finite & np.fromiter((l in complete_set for l in labels), dtype=bool, count=labels.size)
    out["n_periods"] = len(all_labels)
    return out
