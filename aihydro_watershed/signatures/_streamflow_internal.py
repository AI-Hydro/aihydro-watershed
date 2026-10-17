"""
Private streamflow helpers for aihydro-watershed signature computation.

These functions fetch USGS streamflow through the aihydro-data Water Data API
adapter (``aihydro_data.sources.usgs_waterdata``; the legacy waterservices
path is gone) and convert depth units for the hydrological signatures module.
Discharge units are converted once, inside the adapter. They are NOT part of the
public API - use aihydro-data for general streamflow acquisition.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)


# Why the last fetch returned None, for the caller on the same thread (the
# ``Optional[Dict]`` contract, which callers and tests patch, stays unchanged).
_FETCH_DIAG = threading.local()


def _fetch_diag(clear: bool = False) -> dict:
    """Failure record of the last :func:`_fetch_streamflow_internal` on this thread ({} if none)."""
    d = dict(getattr(_FETCH_DIAG, "d", None) or {})
    if clear:
        _FETCH_DIAG.d = {}
    return d


def _fetch_streamflow_internal(
    gauge_id: str,
    start_date: str,
    end_date: str,
    interval: str = "daily",
) -> Optional[Dict]:
    """Internal fetch for computation.

    Returns ``{q_cms, meta, quality, flags, upstream, site}`` or None. ``q_cms`` is a
    tz-naive daily (or UTC-instant) Series in m3/s converted ONCE by the adapter from the
    unit the API states; ``quality`` is the adapter's quality summary (approval counts,
    qualifier counts, provisional span, gaps); ``flags`` is a DataFrame aligned with
    ``q_cms`` (``approval_code`` A/P, ``qualifier_code``); ``upstream`` is the request
    provenance (URLs, body sha256, retrieval time). On any failure the reason is left in
    :func:`_fetch_diag` (``code``, ``message``, ``terminal``) - never raised, never
    answered with another product.
    """
    _FETCH_DIAG.d = {}
    try:
        from aihydro_data.exceptions import AihydroDataError, TerminalSourceError
        from aihydro_data.sources import usgs_waterdata as uw

        try:
            if interval == "daily":
                series = uw.get_daily(gauge_id, start_date, end_date)
            else:
                series = uw.get_continuous(gauge_id, start_date, end_date)
        except AihydroDataError as e:
            _FETCH_DIAG.d = {"code": getattr(e, "code", type(e).__name__),
                             "message": str(e), "terminal": isinstance(e, TerminalSourceError)}
            log.error("USGS streamflow fetch failed for %s: %s", gauge_id, e)
            return None

        frame = series.frame.dropna(subset=["value_m3s"])
        if frame.empty:
            _FETCH_DIAG.d = {"code": "USGS_NO_DATA", "message": "series has no non-null values",
                             "terminal": False}
            return None
        idx = pd.DatetimeIndex(frame["time"])
        if idx.tz is not None:                      # continuous: UTC instants -> naive UTC
            idx = idx.tz_convert("UTC").tz_localize(None)
        q_cms = pd.Series(frame["value_m3s"].to_numpy(dtype=float), index=idx, name="q_cms")
        flags = pd.DataFrame({"approval_code": frame["approval_code"].to_numpy(),
                              "qualifier_code": frame["qualifier_code"].to_numpy()}, index=idx)

        row_meta: dict = {"site_id": series.site_id, "time_series_id": series.time_series_id,
                          "units_native": series.units_native}
        try:   # site metadata is best-effort: signatures do not need it
            loc = uw.get_monitoring_locations(ids=[gauge_id])[0]
            row_meta.update(
                gauge_name=loc.name or "",
                latitude=loc.lat, longitude=loc.lon,
                horizontal_datum=loc.horizontal_datum,
                coordinate_note=loc.geometry_crs_note,
                hydrologic_unit_code=loc.hydrologic_unit_code,
            )
        except Exception as e:
            log.warning("monitoring-location metadata unavailable for %s: %s", gauge_id, e)
        return {"q_cms": q_cms, "meta": row_meta, "quality": series.quality_summary,
                "flags": flags, "upstream": series.upstream, "site": series.site_id}
    except Exception as e:
        _FETCH_DIAG.d = {"code": "FETCH_FAILED", "message": f"{type(e).__name__}: {e}",
                         "terminal": False}
        log.error("Internal streamflow fetch failed: %s", e)
        return None


def _to_mm_per_day(discharge_cms: pd.Series, area_km2: float) -> pd.Series:
    """
    Convert discharge from m³/s to basin-depth mm/day.

    Formula: q_mm_day = q_cms * 86.4 / area_km2
    """
    if discharge_cms is None or len(discharge_cms) == 0 or area_km2 <= 0:
        return pd.Series(dtype=float)

    q_mm_day = discharge_cms * (86.4 / area_km2)
    q_mm_day.index = pd.to_datetime(q_mm_day.index).tz_localize(None)
    return q_mm_day
