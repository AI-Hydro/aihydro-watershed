"""
Hydrological Signatures
=======================

Extract CAMELS-style hydrological signatures from streamflow data.

Public Functions
----------------
extract_hydrological_signatures(gauge_id, watershed_geojson, area_km2,
                                start_date, end_date) -> HydroResult
    Extract 17 CAMELS-style hydrological signatures

compute_flow_stats_camels, compute_water_balance_camels,
compute_event_stats_camels, compute_timing_stats_camels,
compute_slope_fdc_camels — individual signature groups

References
----------
- Addor et al. (2017). The CAMELS data set. HESS.
- Ladson et al. (2013). Baseflow separation. J. Hydrol.
- Sawicz et al. (2011). Catchment classification. WRR.
- Sankarasubramanian et al. (2001). Streamflow elasticity. WRR.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

from aihydro_core import DataSource, HydroMeta, HydroResult, ToolError
from aihydro_watershed.signatures._streamflow_internal import (
    _fetch_streamflow_internal,
    _to_mm_per_day,
)
from aihydro_core.science.uncertainty import bootstrap_dict

_SOURCES_NWIS = [
    DataSource(
        name="USGS NWIS",
        url="https://waterservices.usgs.gov/nwis/dv/",
        citation=(
            "@misc{NWIS2024,\n"
            "  title={National Water Information System (NWIS)},\n"
            "  author={{USGS Water Resources Mission Area}},\n"
            "  year={2024},\n"
            "  url={https://waterdata.usgs.gov/nwis}\n"
            "}"
        ),
    )
]
_SOURCES_GRIDMET = [
    DataSource(
        name="GridMET",
        url="https://www.climatologylab.org/gridmet.html",
        citation=(
            "@article{Abatzoglou2013,\n"
            "  title={Development of gridded surface meteorological data for ecological "
            "applications and modelling},\n"
            "  author={Abatzoglou, John T},\n"
            "  journal={International Journal of Climatology},\n"
            "  volume={33}, number={1}, pages={121--131}, year={2013}\n"
            "}"
        ),
    )
]
_SOURCES_PRECIP_GLOBAL = [
    DataSource(
        name="aihydro-data precipitation router (GridMET / CHIRPS / ERA5-Land / CHIRPS_IRI)",
        url="https://github.com/AI-Hydro/aihydro-data",
        citation=(
            "@misc{AIHydroData2024,\n"
            "  title={aihydro-data: global hydrology dataverse},\n"
            "  author={{AI-Hydro Contributors}},\n"
            "  year={2024},\n"
            "  url={https://github.com/AI-Hydro/aihydro-data}\n"
            "}"
        ),
    )
]
_SOURCES_GEOGLOWS = [
    DataSource(
        name="GEOGLOWS v2 (ECMWF)",
        url="https://geoglows.ecmwf.int",
        citation=(
            "@misc{GEOGLOWS2024,\n"
            "  title={GEOGLOWS ECMWF Streamflow Services},\n"
            "  author={{ECMWF GEOGLOWS Team}},\n"
            "  year={2024},\n"
            "  url={https://geoglows.ecmwf.int}\n"
            "}"
        ),
    )
]
_SOURCES_OPENMETEO_FLOOD = [
    DataSource(
        name="Open-Meteo GloFAS v4",
        url="https://open-meteo.com/en/docs/flood-api",
        citation=(
            "@misc{OpenMeteoFlood2024,\n"
            "  title={Open-Meteo Flood API (GloFAS v4)},\n"
            "  author={{Open-Meteo Contributors}},\n"
            "  year={2024},\n"
            "  url={https://open-meteo.com/en/docs/flood-api}\n"
            "}"
        ),
    )
]

_TOOL_PATH_SIGNATURES = "aihydro_watershed.signatures.signatures.extract_hydrological_signatures"

# Baseflow separation method used for every baseflow_index this module
# computes (compute_flow_stats_camels -> _lyne_hollick_baseflow). Different
# methods (Lyne-Hollick vs Eckhardt vs UKIH) and different alpha/pass
# parameters shift BFI by up to ~0.1-0.2 for the same catchment — the method
# is part of the result's identity, not an implementation detail, so it is
# surfaced in HydroMeta.params (see extract_hydrological_signatures) and in
# aihydro-lsh's AttrProvenance.quality_flags (see recipes/hydrology.py).
# A SINGLE source of truth here means both call sites report the params the
# filter was actually run with, not a second hardcoded copy that can drift.
BASEFLOW_SEPARATION_METHOD = "lyne_hollick"
BASEFLOW_SEPARATION_PARAMS = {"alpha": 0.925, "passes": 3}
BASEFLOW_SEPARATION_REFERENCE = "Nathan & McMahon (1990); Ladson et al. (2013)"
BASEFLOW_METHOD_QUALITY_FLAG = (
    f"baseflow_method={BASEFLOW_SEPARATION_METHOD}"
    f"_alpha{BASEFLOW_SEPARATION_PARAMS['alpha']}"
    f"_passes{BASEFLOW_SEPARATION_PARAMS['passes']}"
)

log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

__all__ = [
    'extract_hydrological_signatures',
    'compute_flow_stats_camels',
    'compute_water_balance_camels',
    'compute_event_stats_camels',
    'compute_timing_stats_camels',
    'compute_slope_fdc_camels',
    'BASEFLOW_SEPARATION_METHOD',
    'BASEFLOW_SEPARATION_PARAMS',
    'BASEFLOW_SEPARATION_REFERENCE',
    'BASEFLOW_METHOD_QUALITY_FLAG',
]


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("aihydro-tools")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_hydrological_signatures(
    gauge_id: str | None,
    watershed_geojson: dict,
    area_km2: float,
    start_date: str = "1989-10-01",
    end_date: str = "2009-09-30",
    q_cms_series: list | pd.Series | None = None,
) -> HydroResult:
    """
    Extract 17 CAMELS-style hydrological signatures for a watershed.

    Parameters
    ----------
    gauge_id : str or None
        USGS gauge identifier (8-digit code). When ``None`` and no
        ``q_cms_series`` is supplied, streamflow is auto-fetched from
        GEOGLOWS v2 (anonymous AWS S3 Zarr, 1940→present) with
        Open-Meteo GloFAS v4 as the fallback — both require no auth.
    watershed_geojson : dict
        Watershed boundary as GeoJSON polygon dict (from delineate_watershed)
    area_km2 : float
        Watershed area in km²
    start_date : str, optional
        Start date YYYY-MM-DD (default: "1989-10-01" — CAMELS period)
    end_date : str, optional
        End date YYYY-MM-DD (default: "2009-09-30" — CAMELS period)
    q_cms_series : list[float] or pandas.Series or None, optional
        Pre-loaded daily streamflow (m³/s). A Series retains its DatetimeIndex;
        a list is assumed consecutive daily values from start_date. When supplied the USGS
        NWIS fetch is skipped — enables global / non-USGS workflows where
        streamflow was already fetched via ``data_fetch`` or another source.

    Returns
    -------
    HydroResult
        result.data keys (all float, NaN where insufficient data):
        q_mean, q_std, q5, q95, q_median, baseflow_index,
        runoff_ratio, stream_elas, high_q_freq, high_q_dur,
        low_q_freq, low_q_dur, zero_q_freq, flow_variability,
        hfd_mean, half_flow_date_std, slope_fdc

    Raises
    ------
    ToolError
        INVALID_AREA, INVALID_GEOMETRY, DEPENDENCY_ERROR
    """

    log.info("Extracting hydrological signatures for gauge %s (%s to %s)", gauge_id, start_date, end_date)

    if not np.isfinite(area_km2) or area_km2 <= 0:
        raise ToolError(
            code="INVALID_AREA",
            message=f"Invalid watershed area: {area_km2}. Must be positive and finite.",
            tool=_TOOL_PATH_SIGNATURES,
        )

    # Convert GeoJSON dict back to shapely for internal helpers
    try:
        from shapely.geometry import shape
        watershed_geom = shape(watershed_geojson)
    except Exception as e:
        raise ToolError(
            code="INVALID_GEOMETRY",
            message=f"Could not parse watershed_geojson: {e}",
            tool=_TOOL_PATH_SIGNATURES,
            recovery="Pass result.data['geometry_geojson'] from delineate_watershed().",
        ) from e

    try:
        # Streamflow source precedence:
        #  1. pre-loaded array (q_cms_series) — highest priority, skips all fetching
        #  2. USGS NWIS — when a gauge_id is supplied
        #  3. GEOGLOWS v2 → Open-Meteo GloFAS fallback — globally, no auth needed
        _global_product: str | None = None
        # Identity of the streamflow the signatures were computed from, read
        # from what was served — never inferred from the request. Consumers
        # (e.g. aihydro-lsh's HydrologyRecipe) record this as provenance.
        #   observed        — USGS NWIS daily values for gauge_id
        #   modelled        — reach-scale modelled discharge (GEOGLOWS / GloFAS)
        #   caller_supplied — q_cms_series passed in; origin unknown here
        #   none            — no usable streamflow; signatures are NaN
        if q_cms_series is not None:
            streamflow_result = {"q_cms": q_cms_series}
            _source = {"product": None, "observation": "caller_supplied"}
        elif gauge_id:
            streamflow_result = _fetch_streamflow_internal(gauge_id, start_date, end_date)
            _source = {"product": "NWIS_STREAMFLOW", "observation": "observed"}
        else:
            # Auto-route: try GEOGLOWS v2 (anonymous S3 Zarr, 1940→present),
            # fall back to Open-Meteo GloFAS v4 (no auth, ~1984→present).
            streamflow_result = _fetch_global_streamflow(watershed_geom, start_date, end_date)
            _global_product = (streamflow_result or {}).get("_product")
            _source = {"product": _global_product, "observation": "modelled"}

        _uncertainty: dict = {}
        if streamflow_result is None or len(streamflow_result.get("q_cms", [])) < 365:
            log.warning("Insufficient streamflow data for gauge %s", gauge_id)
            sigs = _get_default_hydrology()
            _source = {"product": None, "observation": "none"}
        else:
            q_cms = streamflow_result["q_cms"]
            # _to_mm_per_day expects a pd.Series with a DatetimeIndex.
            # When q_cms comes from a pre-loaded list (non-USGS path) it is a
            # plain Python list or numpy array; wrap it in a Series so the
            # conversion always receives the correct type. Use isinstance
            # (not hasattr) — plain lists also have .index().
            import pandas as _pd
            if not isinstance(q_cms, _pd.Series):
                _idx = _pd.date_range(start_date or "2000-01-01",
                                      periods=len(q_cms), freq="D")
                q_cms = _pd.Series(list(q_cms), index=_idx, dtype=float)
            q_mm_day = _to_mm_per_day(q_cms, area_km2)
            p_mm_day = _fetch_precipitation_data_bygeom(watershed_geom, start_date, end_date)

            sigs = {
                **compute_flow_stats_camels(q_mm_day),
                **compute_water_balance_camels(q_mm_day, p_mm_day),
                **compute_event_stats_camels(q_mm_day),
                **compute_timing_stats_camels(q_mm_day),
                **compute_slope_fdc_camels(q_mm_day),
            }
            log.info(
                "Extracted %d signatures (gauge=%s, preloaded_q=%s)",
                len(sigs),
                gauge_id or "none",
                q_cms_series is not None,
            )
            # Bootstrap CIs for scalar metrics that are IID-bootstrappable.
            # Uses block bootstrap (temporal autocorrelation in streamflow).
            try:
                q_arr = q_mm_day.dropna().values
                if len(q_arr) >= 30:
                    def _mean(x): return float(np.mean(x))
                    def _median(x): return float(np.median(x))
                    def _q5(x): return float(np.quantile(x, 0.95))
                    def _q95(x): return float(np.quantile(x, 0.05))
                    def _cv(x): return float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else float("nan")
                    _fns = {
                        "q_mean": _mean,
                        "q_median": _median,
                        "q5": _q5,
                        "q95": _q95,
                        "flow_variability": _cv,
                    }
                    _u = bootstrap_dict(_fns, q_arr, use_block=True, n=500, ci=0.90)
                    _uncertainty = {k: dict(v) for k, v in _u.items()}
                    _uncertainty.update(_baseflow_index_uncertainty(q_arr))
            except Exception as _ue:
                log.warning("signatures: uncertainty estimation failed (non-fatal): %s", _ue)

        # Ensure all values are JSON-serializable Python floats
        clean = {k: (float(v) if v is not None and np.isfinite(float(v)) else None)
                 for k, v in sigs.items()}
        if _uncertainty:
            clean["_uncertainty"] = _uncertainty
        clean["_streamflow_source"] = _source

        if _global_product == "GEOGLOWS_RETRO":
            _global_sources = _SOURCES_GEOGLOWS
        elif _global_product == "OPENMETEO_FLOOD":
            _global_sources = _SOURCES_OPENMETEO_FLOOD
        else:
            _global_sources = []
        _sources = _SOURCES_PRECIP_GLOBAL + (_SOURCES_NWIS if gauge_id else _global_sources)
        return HydroResult(
            data=clean,
            meta=HydroMeta(
                tool=_TOOL_PATH_SIGNATURES,
                version=_get_version(),
                gauge_id=gauge_id or "",
                sources=_sources,
                params={
                    "gauge_id": gauge_id or "global_watershed",
                    "area_km2": area_km2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "baseflow_method": BASEFLOW_SEPARATION_METHOD,
                    "baseflow_params": dict(BASEFLOW_SEPARATION_PARAMS),
                    "baseflow_reference": BASEFLOW_SEPARATION_REFERENCE,
                },
            ),
        )

    except ToolError:
        raise
    except ImportError as e:
        raise ToolError(
            code="DEPENDENCY_ERROR",
            message=str(e),
            tool=_TOOL_PATH_SIGNATURES,
            recovery="pip install 'ai-hydro[analysis]'",
        ) from e
    except Exception as e:
        log.error("Error extracting hydrological signatures: %s", e)
        return HydroResult(
            data={
                **_get_default_hydrology(),
                "_streamflow_source": {"product": None, "observation": "none"},
            },
            meta=HydroMeta(
                tool=_TOOL_PATH_SIGNATURES,
                version=_get_version(),
                gauge_id=gauge_id,
                # Cite NWIS only when a gauge was actually requested.
                sources=_SOURCES_NWIS if gauge_id else [],
                params={"gauge_id": gauge_id, "error": str(e)},
            ),
        )


# ---------------------------------------------------------------------------
# Signature computation groups
# ---------------------------------------------------------------------------

def compute_flow_stats_camels(q_mm_day: pd.Series) -> Dict[str, float]:
    """
    Compute basic flow statistics following CAMELS methodology.

    Returns: q_mean, q_std, q5, q95, q_median, baseflow_index
    """

    q = q_mm_day.dropna().values

    if len(q) < 365:
        log.warning(f"Insufficient data for flow stats: {len(q)} days (minimum 365)")
        return {k: np.nan for k in [
            "q_mean", "q_std", "q5", "q95", "q_median", "baseflow_index"
        ]}

    q_mean = float(np.mean(q))
    q_std = float(np.std(q))
    q5 = float(np.quantile(q, 0.95))   # High flow (95th percentile)
    q95 = float(np.quantile(q, 0.05))  # Low flow (5th percentile)
    q_med = float(np.median(q))

    # Baseflow index using Lyne-Hollick filter
    bf = _lyne_hollick_baseflow(q, alpha=0.925, passes=3)
    bfi = float(np.nansum(bf) / np.nansum(q)) if np.nansum(q) > 0 else np.nan
    bfi = max(0.0, min(1.0, bfi)) if np.isfinite(bfi) else np.nan

    log.debug(f"Flow stats: mean={q_mean:.2f}, BFI={bfi:.2f}")

    return {
        'q_mean': q_mean,
        'q_std': q_std,
        'q5': q5,
        'q95': q95,
        'q_median': q_med,
        'baseflow_index': bfi,
    }


def compute_water_balance_camels(
    q_mm_day: pd.Series,
    p_mm_day: Optional[pd.Series],
) -> Dict[str, float]:
    """
    Compute water balance metrics (runoff ratio, streamflow elasticity).

    Following Sankarasubramanian et al. (2001).
    Returns: runoff_ratio, stream_elas
    """

    if p_mm_day is None or len(p_mm_day) < 365:
        log.warning("Insufficient precipitation data for water balance")
        return {'runoff_ratio': np.nan, 'stream_elas': np.nan}

    q_aln, p_aln = _align_daily(q_mm_day, p_mm_day, min_days=365)

    if q_aln is None:
        log.warning("Failed to align Q and P time series")
        return {'runoff_ratio': np.nan, 'stream_elas': np.nan}

    mean_q, mean_p = float(q_aln.mean()), float(p_aln.mean())
    rr = mean_q / mean_p if mean_p > 0 else np.nan

    # Streamflow elasticity
    hy = _year_series(q_aln.index, hydro_year_start_month=10)
    mp = pd.Series(p_aln.values, index=hy).groupby(level=0).mean()
    mq = pd.Series(q_aln.values, index=hy).groupby(level=0).mean()

    if len(mp) < 3 or len(mq) < 3:
        log.warning("Insufficient years for elasticity calculation")
        return {'runoff_ratio': rr, 'stream_elas': np.nan}

    mp_tot, mq_tot = float(mp.mean()), float(mq.mean())
    dp, dq = (mp - mp_tot), (mq - mq_tot)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (dq / mq_tot) / (dp / mp_tot)

    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    elas = float(np.median(ratio)) if len(ratio) > 0 else np.nan

    log.debug(f"Water balance: RR={rr:.2f}, elasticity={elas:.2f}")

    return {'runoff_ratio': rr, 'stream_elas': elas}


def compute_event_stats_camels(q_mm_day: pd.Series) -> Dict[str, float]:
    """
    Compute extreme event statistics (high/low flow frequency and duration).

    Returns: high_q_freq, high_q_dur, low_q_freq, low_q_dur,
             zero_q_freq, flow_variability
    """

    q = q_mm_day.dropna().values

    if len(q) == 0:
        log.warning("No valid discharge data for event stats")
        return {k: np.nan for k in [
            "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur",
            "zero_q_freq", "flow_variability"
        ]}

    med_q, mean_q = np.median(q), np.mean(q)

    if mean_q <= 0:
        log.warning("Mean discharge <= 0, cannot compute event stats")
        return {k: np.nan for k in [
            "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur",
            "zero_q_freq", "flow_variability"
        ]}

    # Preserve calendar gaps: dropping NaNs before finding runs would merge
    # two separate events on either side of a missing observation. Use the
    # valid values for thresholds and denominators, but the full daily axis
    # for contiguous-run lengths.
    if isinstance(q_mm_day.index, pd.DatetimeIndex):
        daily = q_mm_day.sort_index().resample("D").asfreq()
        event_q = daily.to_numpy(dtype=float)
    else:
        event_q = q_mm_day.to_numpy(dtype=float)

    # High flow events (> 9x median)
    high_mask = np.isfinite(event_q) & (event_q > 9.0 * med_q)
    high_freq = float(np.sum(high_mask) / len(q) * 365.25)
    high_dur = _consecutive_event_lengths(high_mask)
    high_dur_mean = float(np.mean(high_dur)) if high_dur else np.nan

    # Low flow events (<= 0.2x mean)
    low_mask = np.isfinite(event_q) & (event_q <= 0.2 * mean_q)
    low_freq = float(np.sum(low_mask) / len(q) * 365.25)
    low_dur = _consecutive_event_lengths(low_mask)
    low_dur_mean = float(np.mean(low_dur)) if low_dur else np.nan

    # Zero flow frequency
    zero_freq = float(np.sum(q == 0) / len(q))

    # Flow variability (coefficient of variation)
    flow_var = float(np.std(q) / mean_q)

    log.debug(f"Event stats: high_freq={high_freq:.1f}, low_freq={low_freq:.1f}")

    return {
        'high_q_freq': high_freq,
        'high_q_dur': high_dur_mean,
        'low_q_freq': low_freq,
        'low_q_dur': low_dur_mean,
        'zero_q_freq': zero_freq,
        'flow_variability': flow_var,
    }


def compute_timing_stats_camels(q_mm_day: pd.Series) -> Dict[str, float]:
    """
    Compute flow timing statistics (half-flow date mean and variability).

    Returns: hfd_mean, half_flow_date_std
    """

    if q_mm_day is None or len(q_mm_day) < 365:
        log.warning("Insufficient data for timing stats")
        return {'hfd_mean': np.nan, 'half_flow_date_std': np.nan}

    df = q_mm_day.dropna().to_frame("q")
    hy = _year_series(df.index, hydro_year_start_month=10)
    hyd_start = pd.to_datetime([f"{y-1}-10-01" for y in hy])
    df["day"] = (df.index - hyd_start).days + 1
    df["hy"] = hy

    hfd_list = []
    for g, grp in df.groupby("hy"):
        qsum = grp["q"].sum()
        if len(grp) >= 300 and qsum > 0:
            csum = grp["q"].cumsum()
            idx = np.argmax(csum.values >= 0.5 * qsum)
            hfd_list.append(int(grp["day"].iloc[idx]))

    if len(hfd_list) >= 2:
        log.debug(f"Timing stats: {len(hfd_list)} years analyzed")
        return {
            'hfd_mean': float(np.mean(hfd_list)),
            'half_flow_date_std': float(np.std(hfd_list)),
        }

    log.warning(f"Insufficient years for timing stats: {len(hfd_list)}")
    return {'hfd_mean': np.nan, 'half_flow_date_std': np.nan}


def compute_slope_fdc_camels(q_mm_day: pd.Series) -> Dict[str, float]:
    """
    Compute slope of flow duration curve between 33% and 66% exceedance.

    Following Sawicz et al. (2011).
    Returns: slope_fdc
    """

    q = q_mm_day.dropna().values
    q = q[q > 0]

    if len(q) < 100:
        log.warning(f"Insufficient positive discharge values for FDC: {len(q)}")
        return {'slope_fdc': np.nan}

    q33 = np.quantile(q, 0.67)  # 33% exceedance
    q66 = np.quantile(q, 0.34)  # 66% exceedance

    if q66 <= 0 or q33 <= 0:
        log.warning("Invalid quantiles for FDC slope")
        return {'slope_fdc': np.nan}

    slope = (np.log(q33) - np.log(q66)) / (0.66 - 0.33)

    log.debug(f"FDC slope: {slope:.3f}")

    return {'slope_fdc': float(slope)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_global_streamflow(
    watershed_geom,
    start_date: str,
    end_date: str,
) -> Optional[dict]:
    """Auto-fetch global streamflow via GEOGLOWS v2 → Open-Meteo GloFAS fallback.

    Both backends are anonymous (no auth, no queue). Returns a dict with keys
    ``q_cms`` (pd.Series[float] with DatetimeIndex, m³/s) and ``_product``
    (the product ID aihydro-data actually *served*), or ``None`` if all
    backends fail.

    Each candidate is fetched with ``fallback=[]``: this loop is the whole
    fallback chain. Without it, a manual pin walks aihydro-data's routing
    policy on failure, so a GEOGLOWS request could be served by another
    product (in CONUS, observed NWIS) while being labelled GEOGLOWS.

    Install:  pip install aihydro-data[geoglows]   (GEOGLOWS backend)
              pip install aihydro-data              (Open-Meteo needs only requests)
    """
    try:
        import aihydro_data
        import geopandas as gpd
    except ImportError:
        log.warning("aihydro_data not installed — global streamflow auto-fetch unavailable.")
        return None

    gdf = gpd.GeoDataFrame(geometry=[watershed_geom], crs="EPSG:4326")

    for product in ("GEOGLOWS_RETRO", "OPENMETEO_FLOOD"):
        try:
            log.info("Global streamflow: trying product=%s (%s → %s)", product, start_date, end_date)
            fetch_result = aihydro_data.fetch(
                variable="streamflow",
                geometry=gdf,
                start=start_date,
                end=end_date,
                mode="manual",
                product=product,
                fallback=[],
            )
            df = fetch_result.data
            if not isinstance(df, pd.DataFrame) or df.empty:
                log.warning("Global streamflow: empty result from %s", product)
                continue
            if "streamflow" not in df.columns:
                log.warning("Global streamflow: 'streamflow' column missing in %s result", product)
                continue
            idx = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
            q_series = pd.Series(df["streamflow"].values, index=idx, dtype=float)
            q_series.name = "q_cms"
            served = getattr(fetch_result, "product", None) or product
            log.info("Global streamflow fetched via %s: %d days", served, len(q_series))
            return {"q_cms": q_series, "_product": served}
        except Exception as e:
            log.warning("Global streamflow fetch failed (product=%s): %s", product, e)

    log.warning("All global streamflow backends exhausted — hydrological signatures will be NaN.")
    return None


def _fetch_precipitation_data_bygeom(
    watershed_geom,
    start_date: str,
    end_date: str,
) -> Optional[pd.Series]:
    """Fetch daily precipitation (basin-mean) for water balance calculations.

    Routes through ``aihydro_data.fetch("precipitation")`` — globally capable.
    The auto-routing policy selects the best available product per region:

    * CONUS: GridMET → Daymet → CHIRPS (GEE) → ERA5-Land (GEE) → CHIRPS_IRI
    * Europe: ERA5-Land (GEE) → CHIRPS (GEE) → CHIRPS_IRI
    * global: CHIRPS (GEE) → IMERG (GEE) → ERA5-Land (GEE) → CHIRPS_IRI

    **CHIRPS_IRI** (IRI OPeNDAP, auth-free, 1981→present, 0.05°) is always the
    last fallback — it requires only ``pip install aihydro-data[opendap]``
    (xarray + netCDF4) and no API key.
    """
    try:
        import aihydro_data
        import geopandas as gpd
    except ImportError:
        log.warning("aihydro_data not installed — precipitation fetch unavailable.")
        return None

    gdf = gpd.GeoDataFrame(geometry=[watershed_geom], crs="EPSG:4326")

    try:
        log.info("Fetching precipitation via aihydro_data auto-routing (global)")
        result = aihydro_data.fetch(
            variable="precipitation",
            geometry=gdf,
            start=start_date,
            end=end_date,
            # aggregation="basin_mean" is the default — spatial mean over the watershed
        )
        df = result.data
        if not isinstance(df, pd.DataFrame) or df.empty:
            log.warning("Precipitation fetch returned empty result")
            return None
        # Both GEE and CHIRPS_IRI backends return DataFrame[date, precipitation]
        p_col = "precipitation" if "precipitation" in df.columns else None
        if p_col is None:
            p_col = next((c for c in df.columns if c != "date"), None)
        if p_col is None:
            log.warning("Precipitation DataFrame has no usable column")
            return None
        idx = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
        s = pd.Series(df[p_col].values, index=idx, dtype=float).dropna()
        s.index = s.index.tz_localize(None) if s.index.tzinfo is not None else s.index
        s.name = "precip_mm"
        log.info(
            "Retrieved %d days of precipitation (product=%s)",
            len(s), getattr(result, "product", "unknown"),
        )
        return s
    except Exception as e:
        log.warning("Precipitation fetch skipped (runoff_ratio/stream_elas will be NaN): %s", e)
        return None


def _lyne_hollick_baseflow(
    q: np.ndarray,
    alpha: float = 0.925,
    passes: int = 3,
) -> np.ndarray:
    """Lyne-Hollick recursive digital filter for baseflow separation.

    Each sweep estimates the quick-flow (high-frequency) component

        f(t) = alpha * f(t-1) + (1 + alpha) / 2 * (y(t) - y(t-1))

    and the baseflow is the residual ``b(t) = y(t) - f(t)``, clamped to
    ``0 <= b <= y``. Following Nathan and McMahon (1990) and Ladson et al.
    (2013, Australian Journal of Water Resources 17(1)), the filter is applied
    in ``passes`` alternating single-direction sweeps (forward, backward,
    forward, ...), with the baseflow output of each sweep used as the input to
    the next. The function returns the BASEFLOW series, so the baseflow index
    is ``sum(baseflow) / sum(q)``.

    Note: a previous implementation returned the quick-flow component ``f``
    rather than the baseflow ``q - f``, which inverted the baseflow index
    (reporting the quick-flow fraction). This version returns baseflow.
    """

    if q.size == 0 or np.all(~np.isfinite(q)):
        return np.full_like(q, np.nan, dtype=float)

    def _sweep(y: np.ndarray, forward: bool) -> np.ndarray:
        n = len(y)
        f = np.zeros(n, dtype=float)        # quick-flow component
        b = y.copy().astype(float)          # baseflow component (boundary: b = y)
        order = range(1, n) if forward else range(n - 2, -1, -1)
        prev = 0 if forward else n - 1
        for t in order:
            f[t] = alpha * f[prev] + (1.0 + alpha) / 2.0 * (y[t] - y[prev])
            b[t] = y[t] - f[t] if f[t] > 0.0 else y[t]
            # baseflow cannot be negative or exceed total flow
            b[t] = min(max(b[t], 0.0), y[t])
            prev = t
        return b

    bf = q.copy().astype(float)
    for i in range(passes):
        bf = _sweep(bf, forward=(i % 2 == 0))

    return np.clip(bf, 0, q)


BFI_BOOTSTRAP_BLOCK_DAYS = 365


def _baseflow_index_uncertainty(q: np.ndarray, n: int = 500, ci: float = 0.90) -> Dict[str, dict]:
    """Moving-block bootstrap CI for the baseflow index.

    The Lyne-Hollick filter is applied ONCE to the observed series (exactly as
    for the reported point value). Blocks of paired (baseflow, flow) days are
    then resampled and the index re-evaluated as ``sum(b) / sum(q)``.
    Re-running the filter on a spliced resample is deliberately avoided: every
    splice is an artificial step change that the filter reads as quick flow,
    which biases bootstrap BFI low. Consequence: the interval reflects
    sampling variability of the ratio over a ~annual-block dependence
    structure; it does NOT include filter-state/parameter uncertainty or
    observation (rating-curve) error.
    """
    if q.size < 2 * BFI_BOOTSTRAP_BLOCK_DAYS:
        return {}
    bf = _lyne_hollick_baseflow(q, alpha=BASEFLOW_SEPARATION_PARAMS["alpha"],
                                passes=BASEFLOW_SEPARATION_PARAMS["passes"])
    if not np.isfinite(bf).all() or float(np.sum(q)) <= 0:
        return {}

    def _bfi(idx: np.ndarray) -> float:
        i = idx.astype(int)
        return float(np.sum(bf[i]) / np.sum(q[i]))

    res = bootstrap_dict({"baseflow_index": _bfi}, np.arange(q.size, dtype=float),
                         use_block=True, block_size=BFI_BOOTSTRAP_BLOCK_DAYS, n=n, ci=ci)
    out = {k: dict(v) for k, v in res.items()}
    for v in out.values():
        v.setdefault("block_size", BFI_BOOTSTRAP_BLOCK_DAYS)
        v["scope"] = "sampling_variability_of_ratio; excludes filter and observation uncertainty"
    return out


def _align_daily(
    q_series: pd.Series,
    p_series: pd.Series,
    min_days: int = 365,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Align two daily time series to common index."""

    if q_series is None or p_series is None:
        return None, None

    qi = pd.to_datetime(q_series.index).tz_localize(None)
    pi = pd.to_datetime(p_series.index).tz_localize(None)

    q = pd.Series(q_series.values, index=qi, dtype=float).dropna()
    p = pd.Series(p_series.values, index=pi, dtype=float).dropna()

    common = q.index.intersection(p.index)

    if len(common) < min_days:
        return None, None

    return q.loc[common], p.loc[common]


def _year_series(dti: pd.DatetimeIndex, hydro_year_start_month: int = 10) -> np.ndarray:
    """Compute hydrologic year ID for each date."""
    return dti.year + (dti.month >= hydro_year_start_month).astype(int)


def _consecutive_event_lengths(mask: np.ndarray) -> list:
    """Calculate lengths of consecutive True values in boolean mask."""
    lengths = []
    count = 0
    for v in mask:
        if v:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return lengths


def _get_default_hydrology() -> Dict[str, float]:
    """Return default hydrological values when data unavailable."""
    return {
        'q_mean': np.nan,
        'q_std': np.nan,
        'q5': np.nan,
        'q95': np.nan,
        'q_median': np.nan,
        'baseflow_index': np.nan,
        'runoff_ratio': np.nan,
        'stream_elas': np.nan,
        'high_q_freq': np.nan,
        'high_q_dur': np.nan,
        'low_q_freq': np.nan,
        'low_q_dur': np.nan,
        'zero_q_freq': np.nan,
        'flow_variability': np.nan,
        'hfd_mean': np.nan,
        'half_flow_date_std': np.nan,
        'slope_fdc': np.nan,
    }
