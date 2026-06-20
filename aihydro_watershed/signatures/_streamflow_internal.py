"""
Private streamflow helpers for aihydro-watershed signature computation.

These functions fetch USGS NWIS streamflow (via dataretrieval) and convert
units for use by the hydrological signatures module.  They are NOT part of
the public API — use aihydro-data for general streamflow acquisition.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)


def _fetch_streamflow_internal(
    gauge_id: str,
    start_date: str,
    end_date: str,
    interval: str = "daily",
) -> Optional[Dict]:
    """Internal fetch that returns {q_cms: pd.Series, meta: dict} for computation."""
    try:
        from dataretrieval import nwis as dr_nwis

        service = "dv" if interval == "daily" else "iv"
        if service == "dv":
            df, _ = dr_nwis.get_dv(
                sites=gauge_id, parameterCd="00060",
                start=start_date, end=end_date,
            )
        else:
            df, _ = dr_nwis.get_iv(
                sites=gauge_id, parameterCd="00060",
                startDT=start_date, endDT=end_date,
            )

        if df is None or df.empty:
            return None

        q_col = next((c for c in df.columns if "00060" in c and "Mean" in c), None)
        if q_col is None:
            q_col = df.select_dtypes(include="number").columns[0]

        q_cfs = pd.to_numeric(df[q_col], errors="coerce")
        q_cms = q_cfs * 0.0283168
        q_cms.index = pd.to_datetime(q_cms.index).tz_localize(None)
        q_cms = q_cms.dropna()
        if len(q_cms) == 0:
            return None

        row_meta: dict = {}
        try:
            site_df, _ = dr_nwis.get_info(sites=gauge_id)
            site_df["site_no"] = site_df["site_no"].astype(str)
            if gauge_id in site_df["site_no"].values:
                row = site_df.loc[site_df["site_no"] == gauge_id].iloc[0]
                row_meta = {
                    "gauge_name": str(row.get("station_nm", "")),
                    "latitude": float(row.get("dec_lat_va", float("nan"))),
                    "longitude": float(row.get("dec_long_va", float("nan"))),
                }
        except Exception:
            pass
        return {"q_cms": q_cms, "meta": row_meta}
    except Exception as e:
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
