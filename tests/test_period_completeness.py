"""Period completeness (tool audit D08, D09, D26 part): hand-checkable synthetic cases.

No network. Numbers are constructed so they can be checked by counting days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures.drought_indices import _to_monthly, spi, spei
from aihydro_watershed.signatures.flood_frequency import annual_maxima, flood_frequency
from aihydro_watershed.signatures.flow_duration import seven_q10
from aihydro_watershed.signatures.periods import complete_periods, period_days


def _iso(d):
    return d.strftime("%Y-%m-%d").tolist()


# --- the shared rule --------------------------------------------------------

def test_period_day_counts():
    assert period_days("2020-02", "month") == 29
    assert period_days(2001, "water_year") == 365       # 1 Oct 2000 - 30 Sep 2001
    assert period_days(2000, "water_year") == 366       # contains 29 Feb 2000
    assert period_days(2000, "climatic_year") == 365    # 1 Apr 2000 - 31 Mar 2001
    assert period_days(2019, "climatic_year") == 366    # 1 Apr 2019 - 31 Mar 2020


def test_labels_water_year_ends_climatic_year_begins():
    d = pd.to_datetime(["2000-09-30", "2000-10-01", "2001-03-31", "2001-04-01"])
    wy = complete_periods(d, np.ones(4), "water_year", min_fraction=0.001)
    assert list(wy["labels"]) == [2000, 2001, 2001, 2001]
    cy = complete_periods(d, np.ones(4), "climatic_year", min_fraction=0.001)
    assert list(cy["labels"]) == [2000, 2000, 2000, 2001]


def test_excluded_period_reports_counts_and_window_edges():
    d = pd.date_range("2000-07-15", "2000-12-31", freq="D")        # 2000-07 starts mid-month
    r = complete_periods(_iso(d), np.ones(d.size), "month", window=("2000-06-01", "2001-01-31"))
    assert r["complete"] == ["2000-08", "2000-09", "2000-10", "2000-11", "2000-12"]
    ex = {e["period"]: e for e in r["excluded"]}
    assert ex["2000-07"]["observed_days"] == 17 and ex["2000-07"]["expected_days"] == 31
    assert ex["2000-06"]["observed_days"] == 0           # requested, nothing served
    assert ex["2001-01"]["observed_days"] == 0


def test_nan_day_is_not_an_observed_day():
    d = pd.date_range("2001-01-01", "2001-01-31", freq="D")
    v = np.ones(d.size); v[4] = np.nan
    assert complete_periods(d, v, "month")["complete"] == []            # default: every day
    assert complete_periods(d, v, "month", min_fraction=0.95)["complete"] == ["2001-01"]  # 30/31 = 0.968


# --- D08: SPI ---------------------------------------------------------------

def test_monthly_total_is_the_sum_of_a_complete_month_only():
    d = pd.date_range("2019-01-01", "2019-03-10", freq="D")
    m, comp = _to_monthly(np.ones(d.size), d)
    assert m.iloc[0] == 31 and m.iloc[1] == 28           # Jan, Feb complete: 1 mm/day
    assert np.isnan(m.iloc[2])                           # 10 of 31 March days: no total (was 10.0)
    assert comp["excluded"] == [{"period": "2019-03", "observed_days": 10, "expected_days": 31,
                                 "fraction": round(10 / 31, 4)}]


def _spi_series():
    d = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    p = np.random.default_rng(1).gamma(0.5, 6, d.size)
    return d, p


def test_incomplete_final_month_is_excluded_not_read_as_drought():
    d, p = _spi_series()
    full = spi(p, _iso(d), scale_months=1)
    cut = d[d <= "2019-12-03"]
    r = spi(p[:cut.size], _iso(cut), scale_months=1)
    assert r["index_months"][-1] == "2019-12" and r["index"][-1] is None     # was a number (audit: -3.09)
    assert [e["period"] for e in r["months_excluded"]] == ["2019-12"]
    # every other month is untouched by the partial one except December's own calibration
    i = full["index_months"].index("2019-11")
    assert r["index"][i] == full["index"][i]
    assert r["n_months_complete"] == full["n_months"] - 1


def test_interior_gap_is_nan_months_not_extreme_values():
    d, p = _spi_series()
    full = spi(p, _iso(d), scale_months=1)
    p2 = p.copy(); p2[(d >= "2010-06-01") & (d <= "2010-08-31")] = np.nan
    r = spi(p2, _iso(d), scale_months=1)
    i = r["index_months"].index("2010-07")
    assert r["index"][i] is None                         # audit: -1.83 against +0.39 on the real data
    assert [e["period"] for e in r["months_excluded"]] == ["2010-06", "2010-07", "2010-08"]
    assert full["index"][i] is not None


def test_accumulation_window_containing_an_incomplete_month_is_not_evaluated():
    d, p = _spi_series()
    p2 = p.copy(); p2[(d >= "2010-07-01") & (d <= "2010-07-31")] = np.nan
    r = spi(p2, _iso(d), scale_months=3)
    idx = r["index_months"]
    for m in ("2010-07", "2010-08", "2010-09"):          # the three 3-month windows that contain July
        assert r["index"][idx.index(m)] is None
    assert r["index"][idx.index("2010-10")] is not None


def test_spei_missing_pet_day_makes_the_month_incomplete():
    d, p = _spi_series()
    e = np.full(d.size, 2.0); e[(d == "2005-04-10")] = np.nan
    r = spei(p, e, _iso(d), scale_months=1)
    assert "2005-04" in [x["period"] for x in r["months_excluded"]]


def test_no_complete_month_is_an_error():
    d = pd.date_range("2000-01-01", "2000-01-20", freq="D")
    with pytest.raises(ValueError, match="no complete month"):
        spi(np.ones(d.size), _iso(d), scale_months=1)


# --- D09: annual maxima -----------------------------------------------------

def _flood_record():
    """2000-07-15 .. 2015-01-03, base 5 m3/s, one March flood a year 2001-2014 of 100+10k."""
    d = pd.date_range("2000-07-15", "2015-01-03", freq="D")
    q = np.full(d.size, 5.0)
    floods = {}
    for k, y in enumerate(range(2001, 2015)):
        floods[y] = 100.0 + 10 * k
        q[np.where(d == f"{y}-03-10")[0][0]] = floods[y]
    # the partial first and last years carry their own larger 'floods' in the observed part
    q[np.where(d == "2000-08-01")[0][0]] = 5.0
    return d, q, floods


def test_partial_years_are_excluded_from_annual_maxima():
    d, q, floods = _flood_record()
    am = annual_maxima(q, dates=_iso(d))
    assert am["years"] == list(range(2001, 2015)) and am["n_years"] == 14
    assert am["annual_max"] == [floods[y] for y in range(2001, 2015)]
    ex = {e["period"]: e for e in am["excluded_years"]}
    assert set(ex) == {2000, 2015}
    assert ex[2000]["observed_days"] == 17 + 31 + 30      # 15-31 Jul, Aug, Sep
    assert ex[2015]["observed_days"] == 31 + 30 + 31 + 3  # Oct, Nov, Dec 2014 + 1-3 Jan 2015
    assert am["year_type"] == "water_year" and "daily mean" in am["statistic"]


def test_partial_year_maxima_no_longer_inflate_the_design_flood():
    d, q, floods = _flood_record()
    q[np.where(d == "2000-09-01")[0][0]] = 5.0            # partial WY2000 and WY2015 hold only base flow
    old_style = list(floods.values()) + [5.0, 5.0]        # what calendar/partial grouping admits
    good = flood_frequency(annual_maxima(q, dates=_iso(d))["annual_max"], n_bootstrap=0)
    want = flood_frequency(list(floods.values()), n_bootstrap=0)
    bad = flood_frequency(old_style, n_bootstrap=0)
    q100 = lambda r: [x["value"] for x in r["return_levels"] if x["return_period"] == 100][0]
    assert q100(good) == q100(want)
    assert q100(bad) != q100(good)


def test_water_year_keeps_a_new_year_flood_in_one_year():
    d = pd.date_range("1999-10-01", "2009-09-30", freq="D")            # ten complete water years
    q = np.full(d.size, 5.0)
    q[(d == "2004-12-31") | (d == "2005-01-01")] = 900.0              # one storm over the new year
    wy = annual_maxima(q, dates=_iso(d))
    cy = annual_maxima(q, dates=_iso(d), year_type="calendar_year")
    assert wy["annual_max"].count(900.0) == 1                           # WY2005
    assert wy["n_years"] == 10
    # calendar years: 1999 (Oct-Dec only) and 2009 (Jan-Sep only) are partial and excluded; the storm is
    # in both 2004 and 2005 and is counted twice
    assert cy["annual_max"].count(900.0) == 2


# --- D26 (part): 7Q10 year and gaps ---------------------------------------

def test_7q10_uses_the_climatic_year_and_keeps_a_new_year_low_in_one_year():
    d = pd.date_range("1999-04-01", "2014-03-31", freq="D")            # 15 complete climatic years
    q = np.full(d.size, 10.0)
    for y in range(1999, 2014):
        q[(d >= f"{y}-12-28") & (d <= f"{y+1}-01-03")] = 1.0           # 7 days straddling 31 Dec
    r = seven_q10(q, dates=_iso(d))
    assert r["year_type"] == "climatic_year" and r["years"] == list(range(1999, 2014))
    assert r["annual_7day_min"] == [1.0] * 15
    assert r["excluded_years"] == []


def test_7q10_excludes_a_year_with_a_long_gap_and_does_not_splice():
    d = pd.date_range("1999-04-01", "2014-03-31", freq="D")
    q = np.full(d.size, 10.0)
    keep = ~((d >= "2005-06-01") & (d <= "2005-09-30"))                # 122 days dropped, as the slot does
    r = seven_q10(q[keep], dates=_iso(d[keep]))
    assert [e["period"] for e in r["excluded_years"]] == [2005]
    assert r["excluded_years"][0]["observed_days"] == 365 - 122
    assert 2005 not in r["years"] and r["n_years"] == 14
