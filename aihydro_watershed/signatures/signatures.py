"""
Hydrological Signatures
=======================

Extract CAMELS-style hydrological signatures from streamflow data.

Public Functions
----------------
extract_hydrological_signatures(gauge_id, watershed_geojson, area_km2,
                                start_date, end_date) -> HydroResult
    Extract 18 CAMELS-style hydrological signatures (17 + slope_fdc_sawicz2011)

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
import threading
import warnings

import numpy as np
import pandas as pd

from aihydro_core import DataSource, HydroMeta, HydroResult, ToolError
from aihydro_watershed.signatures._streamflow_internal import (
    _fetch_diag,
    _fetch_streamflow_internal,
    _to_mm_per_day,
)
from aihydro_core.science import water_year as _core_water_year
from aihydro_watershed.signatures import _uncertainty as _gu
from aihydro_watershed.signatures import qualifiers as _qual
from aihydro_watershed.signatures.flow_duration import fdc_slope_camels_file, fdc_slope_sawicz2011
from aihydro_watershed.signatures.baseflow import (
    LH_N_REFLECT,
    LYNE_HOLLICK_LADSON2013_REFERENCE,
    lyne_hollick_ladson2013,
)

_SOURCES_NWIS = [
    DataSource(
        name="USGS NWIS",
        url="https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items",
        citation=(
            "@misc{NWIS2024,\n"
            "  title={National Water Information System (NWIS), served by the USGS Water Data APIs},\n"
            "  author={{USGS Water Resources Mission Area}},\n"
            "  year={2024},\n"
            "  url={https://api.waterdata.usgs.gov/ogcapi/v1}\n"
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
# Since 0.2.0 the kernel is baseflow.lyne_hollick_ladson2013 (the CAMELS recipe),
# shared with separate_baseflow; the method name changed with it (BREAKING-VALUE).
BASEFLOW_SEPARATION_METHOD = "lyne_hollick_ladson2013"
BASEFLOW_SEPARATION_PARAMS = {"alpha": 0.925, "passes": 3, "n_reflect": LH_N_REFLECT}
BASEFLOW_SEPARATION_REFERENCE = LYNE_HOLLICK_LADSON2013_REFERENCE
BASEFLOW_METHOD_QUALITY_FLAG = (
    f"baseflow_method={BASEFLOW_SEPARATION_METHOD}"
    f"_alpha{BASEFLOW_SEPARATION_PARAMS['alpha']}"
    f"_passes{BASEFLOW_SEPARATION_PARAMS['passes']}"
)

log = logging.getLogger(__name__)

# q5 / q95 semantics of this module's output (CAMELS: Addor et al. 2017, Table 3).
FLOW_QUANTILE_CONVENTION = "camels_nonexceedance_v1"

# Units of every signature this module emits, derived from the code that computes it
# (not from CAMELS documentation). Streamflow is converted to basin-mean depth
# (mm/day = m3/s * 86.4 / area_km2) before any signature is computed, so the flow
# statistics are DEPTH rates, not discharge: 2.02 here is 2.02 mm/day, not 2.02 m3/s.
SIGNATURE_UNITS: Dict[str, str] = {
    "q_mean": "mm/day",                 # np.mean(q_mm_day)
    "q_std": "mm/day",                  # np.std(q_mm_day)
    "q5": "mm/day",                     # 5% non-exceedance quantile of q_mm_day
    "q95": "mm/day",                    # 95% non-exceedance quantile of q_mm_day
    "q_median": "mm/day",
    "baseflow_index": "dimensionless",  # sum(baseflow) / sum(q), in [0, 1]
    "runoff_ratio": "dimensionless",    # mean(q_mm_day) / mean(p_mm_day)
    "stream_elas": "dimensionless",     # median of (dQ/Q)/(dP/P) over hydrologic years
    "high_q_freq": "days/year",         # days with q > 9*median, scaled to 365.25
    "high_q_dur": "days",               # mean run length of those days
    "low_q_freq": "days/year",          # days with q <= 0.2*mean, scaled to 365.25
    "low_q_dur": "days",
    "zero_q_freq": "dimensionless",     # FRACTION of days with q <= 0 (not days/year)
    "flow_variability": "dimensionless",  # std/mean of q (coefficient of variation)
    "hfd_mean": "day of water year",    # days since 1 Oct (1 = 1 Oct) of the half-flow date
    "half_flow_date_std": "days",
    # Since 0.2.0 the bare key is the CAMELS attributes v2.0 *distributed* definition (mean-normalised,
    # linear). The log-space Sawicz et al. (2011) value is slope_fdc_sawicz2011. See
    # SIGNATURE_DEFINITIONS for the exact formula, quantile type, zero handling and source.
    "slope_fdc": "dimensionless",       # (Q.66 - Q.33) / (mean(q) * 0.33)
    "slope_fdc_sawicz2011": "dimensionless",  # (ln Q33exc - ln Q66exc) / 0.33, zeros dropped
}

# Machine-readable definition of every signature whose meaning is a choice, so a consumer can
# tell whether two numbers with the same name are the same quantity. Plain key in the output
# (`signature_definitions`), carried through the tools layer like `units`.
_SLOPE_FDC_CAMELS_FILE_SOURCE = (
    "slope of the flow duration curve between the 33rd and 66th percentiles, mean-normalised, as in "
    "the CAMELS attributes v2.0 distributed file `camels_hydro.txt` (naddor/camels@88e7a0d2d2 hydro/hydro_signatures.R:97). "
    "Reproduces the file at <=2.3e-7 relative on 01013500 and 01031500, and 9.7e-5 on 01516500. "
    "Its zero-structure prediction holds on all 671 basins. Not proven to be the script that "
    "produced the file."
)
SIGNATURE_DEFINITIONS: Dict[str, dict] = {
    "slope_fdc": {
        "formula": "(Q66 - Q33) / (mean(q) * 0.33), Q66 and Q33 the 66th and 33rd percentiles of q",
        "quantile_type": "non-exceedance, linear interpolation (R type 7; numpy default)",
        "zero_handling": "zero-flow days kept; defined only when mean(q) > 0",
        "log_transform": False,
        "source": _SLOPE_FDC_CAMELS_FILE_SOURCE,
        "evidence": "docs/vision-2040/evidence/signature-definitions-1",
        "comparable_to": "CAMELS attributes v2.0 slope_fdc column only; not the Sawicz et al. (2011) signature "
                         "and not necessarily the same quantity as 'slope_fdc' in other CAMELS datasets",
    },
    "slope_fdc_sawicz2011": {
        "formula": "(ln Q33exc - ln Q66exc) / 0.33, Q33exc and Q66exc the flows exceeded 33% and 66% of the time",
        "quantile_type": "exceedance points as non-exceedance quantiles 0.67 and 0.34, linear interpolation (type 7)",
        "zero_handling": "zero-flow days dropped; needs >= 100 positive days",
        "log_transform": True,
        "source": "log-space FDC slope (Sawicz et al. 2011). This is the pre-0.2.0 aihydro 'slope_fdc' value.",
    },
    "baseflow_index": {
        "formula": "sum(baseflow) / sum(q)",
        "method": BASEFLOW_SEPARATION_METHOD,
        "parameters": dict(BASEFLOW_SEPARATION_PARAMS),
        "zero_handling": "baseflow clipped at zero; quick flow subtracted only where positive; "
                         "30-day reflection at both ends",
        "source": BASEFLOW_SEPARATION_REFERENCE,
        "evidence": "docs/vision-2040/evidence/signature-definitions-1 (CAMELS attributes v2.0 reproduced to <=2e-6 relative)",
    },
    "q5": {"formula": "5% quantile of daily q (mm/day)", "quantile_type": "non-exceedance, type 7",
           "source": "Addor et al. (2017) Table 3 (low flow)"},
    "q95": {"formula": "95% quantile of daily q (mm/day)", "quantile_type": "non-exceedance, type 7",
            "source": "Addor et al. (2017) Table 3 (high flow)"},
    "high_q_freq": {"formula": "days with q > 9 * median(q), divided by available days, times 365.25",
                    "units": "days/year", "source": "Addor et al. (2017) Table 3; naddor/camels"},
    "low_q_freq": {"formula": "days with q <= 0.2 * mean(q), divided by available days, times 365.25",
                   "units": "days/year", "source": "Addor et al. (2017) Table 3; naddor/camels"},
    "zero_q_freq": {"formula": "days with q <= 0, divided by available days", "units": "fraction",
                    "source": "naddor/camels compute_no_flow (sum(q<=0)/length)"},
    "runoff_ratio": {"formula": "sum(q) / sum(p) over paired days of complete water years",
                     "completeness": "a water year counts when >= 95% of its days have paired q and p "
                                     "(engineering choice; source to verify)",
                     "source": "Sankarasubramanian et al. (2001); naddor/camels comp_r_qp"},
    "stream_elas": {"formula": "median over complete water years of (dQ/mean Q) / (dP/mean P), annual means "
                               "minus the long-term day-weighted mean (CAMELS e_qp_sanka)",
                    "completeness": "complete water years only (>= 95% paired days; engineering choice)",
                    "source": "Sankarasubramanian et al. (2001); naddor/camels e_qp_sanka"},
}

STREAMFLOW_DIGEST_SCHEME = (
    "sha256 over 'YYYY-MM-DD<TAB>repr(float)<LF>' lines of the daily m3/s series the signatures were computed from"
)


def _series_digest(ser: pd.Series) -> dict:
    """Digest + extent of the (m3/s) streamflow series actually used."""
    import hashlib
    idx = pd.to_datetime(ser.index)
    h = hashlib.sha256()
    for d, v in zip(idx, ser.values):
        h.update(f"{d:%Y-%m-%d}\t{float(v)!r}\n".encode())
    return {
        "digest": "sha256:" + h.hexdigest(),
        "digest_scheme": STREAMFLOW_DIGEST_SCHEME,
        "n_days": int(len(ser)),
        "n_valid_days": int(np.isfinite(ser.astype(float).values).sum()),
        "start": f"{idx.min():%Y-%m-%d}",
        "end": f"{idx.max():%Y-%m-%d}",
        "units": "m3/s",
    }

__all__ = [
    'extract_hydrological_signatures',
    'compute_flow_stats_camels',
    'compute_water_balance_camels',
    'compute_event_stats_camels',
    'compute_timing_stats_camels',
    'compute_slope_fdc_camels',
    'SIGNATURE_UNITS',
    'SIGNATURE_DEFINITIONS',
    'WATER_YEAR_MIN_PAIRED_FRACTION',
    'precipitation_problem',
    'RUNOFF_RATIO_PLAUSIBLE',
    'BASEFLOW_SEPARATION_METHOD',
    'BASEFLOW_SEPARATION_PARAMS',
    'BASEFLOW_SEPARATION_REFERENCE',
    'BASEFLOW_METHOD_QUALITY_FLAG',
]


from aihydro_watershed._version import tool_version as _get_version, version_params as _version_params  # noqa: E402


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
    precipitation: str = "auto",
) -> HydroResult:
    """
    Extract 18 CAMELS-style hydrological signatures for a watershed (17 + ``slope_fdc_sawicz2011``).

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
    precipitation : {"auto", "skip"}, optional
        ``"auto"`` (default) fetches basin-mean precipitation through
        aihydro-data and computes ``runoff_ratio`` / ``stream_elas`` when the
        series passes the physical-validity gate. ``"skip"`` makes NO
        precipitation request at all (use for offline / network-guarded runs:
        some backends read through the netCDF C library, which Python-level
        network guards cannot see); both signatures are then None and
        ``_precipitation.status == "not_attempted"`` with the reason recorded.

    Returns
    -------
    HydroResult
        result.data keys (all float, NaN where insufficient data):
        q_mean, q_std, q5, q95, q_median, baseflow_index,
        (q5 / q95 = 5% / 95% flow quantiles, low / high flow, as in CAMELS),
        runoff_ratio, stream_elas, high_q_freq, high_q_dur,
        low_q_freq, low_q_dur, zero_q_freq, flow_variability,
        hfd_mean, half_flow_date_std, slope_fdc (CAMELS attributes v2.0 definition since 0.2.0),
        slope_fdc_sawicz2011 (log-space, the pre-0.2.0 slope_fdc)

    Raises
    ------
    ToolError
        INVALID_AREA, INVALID_GEOMETRY, DEPENDENCY_ERROR
    """

    log.info("Extracting hydrological signatures for gauge %s (%s to %s)", gauge_id, start_date, end_date)

    if precipitation not in ("auto", "skip"):
        raise ToolError(
            code="INVALID_PARAMETER",
            message=f"precipitation must be 'auto' or 'skip', got {precipitation!r}.",
            tool=_TOOL_PATH_SIGNATURES,
            recovery="Pass precipitation='auto' (default) or 'skip'.",
        )

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
        _gauge_check: dict | None = None
        if q_cms_series is not None:
            streamflow_result = {"q_cms": q_cms_series}
            _source = {"product": None, "observation": "caller_supplied"}
        elif gauge_id:
            _fetch_diag(clear=True)
            streamflow_result = _fetch_streamflow_internal(gauge_id, start_date, end_date)
            _source = {"product": "NWIS_STREAMFLOW", "observation": "observed"}
            # Served-series identity and quality, when the fetch supplies them
            # (the adapter does; a patched fetch may not).
            for _k_src, _k_res in (("site", "site"), ("quality_summary", "quality"),
                                   ("upstream", "upstream")):
                if streamflow_result and streamflow_result.get(_k_res) is not None:
                    _source[_k_src] = streamflow_result[_k_res]
            # ADR-003 A2 R0 (explicit id): the identity is honoured, but the caller's area is
            # compared with the gauge's and the outcome recorded. Fail warns, never refuses.
            if streamflow_result and streamflow_result.get("site"):
                _gauge_check = _gauge_area_check(gauge_id, area_km2)
                _source["site_resolution"] = _gauge_check
                for _w in _gauge_check.get("warnings", []):
                    log.warning("%s", _w)
        else:
            # Auto-route: try GEOGLOWS v2 (anonymous S3 Zarr, 1940→present),
            # fall back to Open-Meteo GloFAS v4 (no auth, ~1984→present).
            streamflow_result = _fetch_global_streamflow(watershed_geom, start_date, end_date)
            _global_product = (streamflow_result or {}).get("_product")
            _source = {"product": _global_product, "observation": "modelled"}

        _uncertainty: dict = {}
        _stream_digest: dict | None = None
        _qual_exposure: dict | None = None
        _served: dict | None = None
        if streamflow_result is None or len(streamflow_result.get("q_cms", [])) < 365:
            log.warning("Insufficient streamflow data for gauge %s", gauge_id)
            sigs = _get_default_hydrology()
            _prev = _source
            _source = {"product": None, "observation": "none"}
            # Say WHY; never substitute another product.
            if streamflow_result is not None:
                # A record was obtained but is too short (fetched or caller-supplied): keep what
                # identifies / qualifies it.
                _source["error"] = {"code": "INSUFFICIENT_RECORD",
                                    "n_days": len(streamflow_result.get("q_cms", [])),
                                    "terminal": False}
                for _k in ("site", "quality_summary", "site_resolution"):
                    if _k in _prev:
                        _source[_k] = _prev[_k]
            elif gauge_id and q_cms_series is None and _fetch_diag():
                _source["error"] = _fetch_diag()   # e.g. gauge refusal, rate limit
            _precip_record = _precipitation_record(
                None, {"reason": "not attempted: no usable streamflow"}, attempted=False)
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
            try:
                _stream_digest = _series_digest(q_cms)
            except Exception as _de:   # provenance must never break the computation
                log.warning("signatures: streamflow digest failed (non-fatal): %s", _de)
            q_mm_day = _to_mm_per_day(q_cms, area_km2)
            # The fetch reports product / failure reason through a
            # thread-local side channel so its (Series | None) contract, which
            # callers and tests patch, stays unchanged.
            if precipitation == "skip":
                p_mm_day = None
                _precip_record = _precipitation_record(
                    None, {"reason": "skipped by caller (precipitation='skip'); "
                                     "no precipitation request was made"},
                    attempted=False)
            else:
                _PRECIP_DIAG.d = {}
                p_mm_day = _fetch_precipitation_data_bygeom(watershed_geom, start_date, end_date)
                _precip_record = _precipitation_record(
                    p_mm_day, getattr(_PRECIP_DIAG, "d", {}), window=(start_date, end_date))

            _wbd = _water_balance_detail(
                q_mm_day, p_mm_day, _effective_units(_precip_record), window=(start_date, end_date))
            _wb, _wb_reason = _wbd["values"], _wbd["reason"]
            sigs = {
                **compute_flow_stats_camels(q_mm_day),
                **_wb,
                **compute_event_stats_camels(q_mm_day),
                **compute_timing_stats_camels(q_mm_day),
                **compute_slope_fdc_camels(q_mm_day),
            }
            if _precip_record["status"] == "used":
                _precip_record.update(
                    water_years_used=_wbd["water_years_used"],
                    water_years_excluded=_wbd["water_years_excluded"],
                    n_paired_days_used=_wbd["n_paired_days_used"],
                )
                if _wb_reason and _wbd["reason_code"] in _COMPLETENESS_REASON_CODES:
                    # Too little paired data to compute anything: refused, with the reason.
                    _precip_record.update(status="rejected", reason=_wb_reason)
                elif _wb_reason:
                    _precip_record.update(
                        status="unusable",
                        reason=f"water balance not computable: {_wb_reason}",
                    )
                else:
                    _note = _gap_note(_precip_record)
                    if _note:
                        # Computed, but not from a complete record: say so rather than "used".
                        _precip_record.update(status="used_with_gaps", reason=_note)
            log.info(
                "Extracted %d signatures (gauge=%s, preloaded_q=%s)",
                len(sigs),
                gauge_id or "none",
                q_cms_series is not None,
            )
            # Confidence intervals: whole water years resampled (aihydro-core grouped_bootstrap).
            # Each carries its machine-readable scope; flow statistics and BFI use every valid
            # streamflow day, runoff_ratio / stream_elas the paired days of the complete water
            # years (the very set their point estimates came from).
            try:
                _uncertainty = _grouped_cis(q_mm_day, _wbd)
            except Exception as _ue:
                log.warning("signatures: uncertainty estimation failed (non-fatal): %s", _ue)
            _flags_df = streamflow_result.get("flags") if isinstance(streamflow_result, dict) else None
            try:
                _qual_exposure = _qual.qualifier_exposure(q_mm_day, _flags_df)
                _served = _qual.served_series(q_cms, _flags_df)
            except Exception as _qe:
                log.warning("signatures: qualifier exposure failed (non-fatal): %s", _qe)

        # Ensure all values are JSON-serializable Python floats
        clean = {k: (float(v) if v is not None and np.isfinite(float(v)) else None)
                 for k, v in sigs.items()}
        if _uncertainty:
            clean["_uncertainty"] = _uncertainty
        clean["_streamflow_source"] = _source
        # The served daily series (dates, m3/s, qualifier / approval codes) the signatures were
        # computed from, so a caller can bind validators to THIS series instead of whatever sits in
        # a session slot. Underscore key: the tools layer retains it as a file and pops it.
        if _served is not None:
            clean["_served_series"] = _served
        # Share of estimated / ice-affected / provisional days among the days that determine each
        # signature. Plain key (survives the tools layer's "_" drop).
        if _qual_exposure is not None:
            clean["qualifier_exposure"] = _qual_exposure
        # Marks q5/q95 as CAMELS non-exceedance quantiles. Deliberately NOT
        # underscore-prefixed: the tools layer drops "_" keys before sealing, and
        # this one must reach the sealed record. Absent on results computed before
        # this fix (watershed < 0.1.1), whose q5/q95 were swapped
        # (docs/vision-2040/findings/defect-q5-q95.md).
        clean["flow_quantile_convention"] = FLOW_QUANTILE_CONVENTION
        clean["_precipitation"] = _precip_record
        # Units for every signature present in the output (see SIGNATURE_UNITS). A plain key,
        # not underscore-prefixed: the tools layer drops "_" keys before sealing, and the
        # unit must reach the sealed record the model and the auditor read.
        clean["units"] = {k: SIGNATURE_UNITS[k] for k in clean if k in SIGNATURE_UNITS}
        # What each signature MEANS (formula, quantile type, zero handling, source), for the
        # signatures whose definition is a choice. Plain key, same reason as `units`.
        clean["signature_definitions"] = {
            k: SIGNATURE_DEFINITIONS[k] for k in clean if k in SIGNATURE_DEFINITIONS}
        # Digest of the streamflow series used (precipitation already carries one).
        if _stream_digest:
            clean["streamflow_digest"] = _stream_digest["digest"]
            clean["streamflow_n_days"] = _stream_digest["n_days"]
            clean["streamflow_n_valid_days"] = _stream_digest["n_valid_days"]
            clean["streamflow_start"] = _stream_digest["start"]
            clean["streamflow_end"] = _stream_digest["end"]
            clean["_streamflow"] = _stream_digest
        # Non-underscore mirror of the gauge-vs-polygon area check (the tools layer drops "_"
        # keys before sealing; a runoff depth computed with a mismatched area must say so).
        if _gauge_check is not None:
            clean["gauge_area_check"] = _gauge_check["area_check"]
            clean["gauge_area_ratio"] = _gauge_check.get("area_ratio")
            clean["gauge_area_km2"] = _gauge_check.get("gauge_area_km2")
            clean["streamflow_warnings"] = list(_gauge_check.get("warnings", []))
        # Non-underscore mirrors: the tools layer drops "_" keys before sealing,
        # and a claim bound to runoff_ratio must carry its precipitation
        # provenance in the sealed evidence.
        clean["precipitation_status"] = _precip_record["status"]
        clean["precipitation_product"] = _precip_record["product"]
        clean["precipitation_digest"] = _precip_record["digest"]
        clean["precipitation_units"] = _effective_units(_precip_record)

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
                    **_version_params(),
                    "gauge_id": gauge_id or "global_watershed",
                    "area_km2": area_km2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "baseflow_method": BASEFLOW_SEPARATION_METHOD,
                    "baseflow_params": dict(BASEFLOW_SEPARATION_PARAMS),
                    "baseflow_reference": BASEFLOW_SEPARATION_REFERENCE,
                    "precipitation": precipitation,
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
                "units": dict(SIGNATURE_UNITS),
                "signature_definitions": dict(SIGNATURE_DEFINITIONS),
                "_streamflow_source": {"product": None, "observation": "none"},
                "_precipitation": _precipitation_record(
                    None, {"reason": "not attempted: signature extraction failed"},
                    attempted=False),
            },
            meta=HydroMeta(
                tool=_TOOL_PATH_SIGNATURES,
                version=_get_version(),
                gauge_id=gauge_id,
                # Cite NWIS only when a gauge was actually requested.
                sources=_SOURCES_NWIS if gauge_id else [],
                params={"gauge_id": gauge_id, "error": str(e), **_version_params()},
            ),
        )


# ---------------------------------------------------------------------------
# Signature computation groups
# ---------------------------------------------------------------------------

def _gauge_area_check(gauge_id: str, area_km2: float) -> dict:
    """ADR-003 A2 R0 area check for an explicit gauge id against the caller's ``area_km2``.

    The gauge's total drainage area comes from the adapter's ``resolve_gauge(gauge_id=...)``
    (mi2 x 2.589988110336); the pass / fail rule is the adapter's pre-registered
    ``|ln(A_gauge / A_request)| <= ln 1.10``. ``area_check`` is pass | fail | unavailable.
    Never raises: a lookup that cannot be made is ``unavailable`` with the reason recorded.
    """
    import math
    from aihydro_data.sources import usgs_waterdata as uw

    lim = uw.DEFAULT_THRESHOLDS["max_abs_log_area_ratio"]
    out: dict = {"method": "explicit_id", "area_basis": "caller_area_km2",
                 "request_area_km2": float(area_km2), "threshold_max_abs_log_area_ratio": lim,
                 "rule_version": uw.RULE_VERSION, "area_check": "unavailable", "area_ratio": None,
                 "log_area_ratio": None, "gauge_area_km2": None, "warnings": []}
    try:
        g = uw.resolve_gauge(gauge_id=gauge_id)
    except Exception as e:                      # lookup could not be made: unavailable, not a refusal
        out["reason"] = f"{getattr(e, 'code', type(e).__name__)}: {e}"
        return out
    out.update(site=g.site, gauge_area_km2=g.A_gauge_km2,
               gauge_contributing_area_km2=g.A_gauge_contributing_km2,
               area_units_note=g.area_units_note, upstream=g.upstream)
    a = g.A_gauge_km2
    if a is None or not a > 0 or not area_km2 > 0:
        out["reason"] = "gauge drainage area not published"
        return out
    out["area_ratio"] = a / area_km2
    out["log_area_ratio"] = math.log(a / area_km2)
    out["area_check"] = "pass" if abs(out["log_area_ratio"]) <= lim + 1e-12 else "fail"
    if out["area_check"] == "fail":
        out["warnings"].append(
            f"area_check=fail: gauge {g.site} total drainage area {a:.2f} km2 vs supplied area_km2 "
            f"{area_km2:.2f} km2 (ratio {a / area_km2:.3f}); runoff depths (mm/day) use the supplied "
            "area and are probably wrong. The gauge identity was honoured, not refused.")
    return out


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
    # CAMELS (Addor et al. 2017, Table 3): q5 = 5% flow quantile (low flow),
    # q95 = 95% flow quantile (high flow). These are plain non-exceedance
    # quantiles, NOT hydrologic exceedance flows (Q5 high / Q95 low); see
    # flow_duration.py for that other convention.
    q5 = float(np.quantile(q, 0.05))
    q95 = float(np.quantile(q, 0.95))
    q_med = float(np.median(q))

    # Baseflow index: canonical Lyne-Hollick / Ladson (2013) kernel, run on the daily axis so
    # calendar gaps are segment boundaries (not spliced).
    bfi = _baseflow_index(q_mm_day)

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
    p_units: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute water balance metrics (runoff ratio, streamflow elasticity).

    Following Sankarasubramanian et al. (2001).
    Returns: runoff_ratio, stream_elas (NaN when precipitation is unusable or
    the ratio is outside the plausible range; see :func:`_water_balance`).
    ``p_units``: the unit the precipitation series declares, if known; a unit
    that is not mm/day-equivalent makes the series unusable.
    """
    return _water_balance(q_mm_day, p_mm_day, p_units)[0]


#: A water year takes part in the water balance when at least this fraction of its days have BOTH
#: q and p. ENGINEERING CHOICE; SOURCE TO VERIFY: the CAMELS code takes a required `tol` argument
#: with no default and applies `length(x) > (1 - tol) * 365`; the value used for v1.2 is unverified.
WATER_YEAR_MIN_PAIRED_FRACTION = _gu.COMPLETE_GROUP_FRACTION   # 0.95

#: reason_code values of a water-balance refusal that are completeness failures (status "rejected").
_COMPLETENESS_REASON_CODES = ("overlap", "no_complete_water_year")


def _water_year_days(label: int) -> int:
    """Days in the water year labelled ``label`` (1 Oct label-1 .. 30 Sep label)."""
    return int((pd.Timestamp(year=label, month=10, day=1) - pd.Timestamp(year=label - 1, month=10, day=1)).days)


def complete_water_years(
    q_aln: pd.Series, p_aln: pd.Series, min_fraction: float = WATER_YEAR_MIN_PAIRED_FRACTION,
    window=None,
) -> Tuple[list, list]:
    """Split the water years of two already-aligned daily series into usable and excluded.

    Both series must share an index of days on which q AND p are finite (see :func:`_align_daily`).
    A water year (1 Oct - 30 Sep, labelled by the calendar year in which it ends) is usable when
    ``paired days / days in the water year >= min_fraction``. Returns ``(used, excluded)``;
    ``used`` is a sorted list of labels, ``excluded`` a list of
    ``{"water_year", "paired_days", "expected_days", "fraction"}``.

    ``window``: the requested ``(start, end)``. Every water year the window touches is examined,
    so a leading or trailing year with no paired day at all is reported as excluded rather than
    silently absent; without it, the span of the paired days is used.
    """
    del p_aln  # the alignment already guarantees one p per q; kept for call-site symmetry
    labels = _year_series(pd.DatetimeIndex(q_aln.index))
    counts = pd.Series(1, index=labels).groupby(level=0).sum()
    # Every water year between the first and last paired day is examined, including one with
    # no paired day at all (a missing year must be reported, not silently absent).
    lo, hi = int(counts.index.min()), int(counts.index.max())
    if window and window[0] and window[1]:
        lo = min(lo, int(_year_series(pd.DatetimeIndex([pd.Timestamp(window[0])]))[0]))
        hi = max(hi, int(_year_series(pd.DatetimeIndex([pd.Timestamp(window[1])]))[0]))
    counts = counts.reindex(range(lo, hi + 1), fill_value=0)
    used, excluded = [], []
    for wy, n in counts.items():
        expected = _water_year_days(int(wy))
        frac = float(n) / expected
        if frac >= min_fraction - 1e-12:
            used.append(int(wy))
        else:
            excluded.append({"water_year": int(wy), "paired_days": int(n),
                             "expected_days": expected, "fraction": round(frac, 4)})
    return used, excluded


def _water_balance(q_mm_day, p_mm_day, p_units=None):
    """``(signatures dict, reason | None)``; reason is set when NaN was returned
    because an input or the result was unusable."""
    det = _water_balance_detail(q_mm_day, p_mm_day, p_units)
    return det["values"], det["reason"]


def _water_balance_detail(q_mm_day, p_mm_day, p_units=None,
                          min_fraction: float = WATER_YEAR_MIN_PAIRED_FRACTION, window=None) -> dict:
    """Water balance on COMPLETE water years only, with the bookkeeping behind it.

    Returns a dict: ``values`` (``runoff_ratio``, ``stream_elas``), ``reason`` (None when
    computed), ``reason_code`` (``precipitation_unusable`` | ``overlap`` |
    ``no_complete_water_year`` | ``not_finite`` | ``implausible`` | None), ``water_years_used``,
    ``water_years_excluded`` (see :func:`complete_water_years`), ``n_paired_days_used``,
    and ``q_used`` / ``p_used`` (the paired series restricted to the usable years; what the point
    estimates were computed from, and what any CI must resample).
    """
    nan = {'runoff_ratio': np.nan, 'stream_elas': np.nan}
    det: dict = {"values": nan, "reason": None, "reason_code": None, "water_years_used": [],
                 "water_years_excluded": [], "n_paired_days_used": 0, "q_used": None, "p_used": None,
                 "min_paired_fraction": min_fraction}

    def _fail(reason: str, code: str) -> dict:
        det.update(reason=reason, reason_code=code)
        return det

    # A precipitation series that is absent, empty, all-zero, negative,
    # non-finite, in the wrong unit or outside physical bounds is never
    # divided into: the precipitation-dependent signatures are NaN (recorded as
    # None, with the reason in ``_precipitation``), not a number computed from
    # a placeholder.
    problem = precipitation_problem(p_mm_day, p_units)
    if problem is not None:
        log.warning("Precipitation unusable for water balance (%s)", problem)
        return _fail(problem, "precipitation_unusable")

    q_aln, p_aln = _align_daily(q_mm_day, p_mm_day, min_days=365)

    if q_aln is None:
        log.warning("Failed to align Q and P time series")
        return _fail("precipitation and streamflow overlap < 365 days", "overlap")

    problem = precipitation_problem(p_aln, p_units)
    if problem is not None:
        log.warning("Aligned precipitation unusable for water balance (%s)", problem)
        return _fail(problem, "precipitation_unusable")

    # Only complete water years take part: a year with a hole in P (or Q) would bias both the
    # ratio and the annual anomalies. The point estimates and any CI use this same set of years.
    used, excluded = complete_water_years(q_aln, p_aln, min_fraction, window)
    det.update(water_years_used=used, water_years_excluded=excluded)
    if not used:
        return _fail(
            f"no water year has >= {min_fraction:.0%} of its days with both streamflow and "
            f"precipitation ({len(excluded)} water years examined; engineering choice, source to verify)",
            "no_complete_water_year")
    hy_all = _year_series(pd.DatetimeIndex(q_aln.index))
    keep = np.isin(hy_all, used)
    q_aln, p_aln = q_aln[keep], p_aln[keep]
    det.update(q_used=q_aln, p_used=p_aln, n_paired_days_used=int(len(q_aln)))

    mean_q, mean_p = float(q_aln.mean()), float(p_aln.mean())
    rr = mean_q / mean_p
    if not np.isfinite(rr):
        return _fail("runoff ratio is not finite", "not_finite")
    # An impossible Q/P is never sealed, whatever produced it (wrong unit,
    # fill value, broken streamflow). Same bound the water-balance validator uses.
    if not (RUNOFF_RATIO_PLAUSIBLE[0] <= rr <= RUNOFF_RATIO_PLAUSIBLE[1]):
        reason = (f"runoff ratio {rr:.4g} outside the plausible range "
                  f"[{RUNOFF_RATIO_PLAUSIBLE[0]:g}, {RUNOFF_RATIO_PLAUSIBLE[1]:g}] "
                  "(engineering bound; source to verify)")
        log.warning("Water balance rejected: %s", reason)
        return _fail(reason, "implausible")

    # Streamflow elasticity
    elas = _stream_elasticity(q_aln.values, p_aln.values, pd.DatetimeIndex(q_aln.index))
    det["values"] = {'runoff_ratio': rr, 'stream_elas': elas}
    log.debug("Water balance: RR=%.2f, elasticity=%s", rr, elas)
    return det


def _stream_elasticity(q: np.ndarray, p: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """Median of (dQ/mean Q)/(dP/mean P) over water years (Sankarasubramanian et al. 2001;
    CAMELS ``e_qp_sanka``). Anomalies are annual means minus the long-term DAY-weighted mean of
    the paired days (as CAMELS; before this revision it was the mean of the annual means, which
    differs by at most ~3e-5 in tests). NaN with fewer than 3 years."""
    hy = _year_series(dates)
    mp = pd.Series(p, index=hy).groupby(level=0).mean()
    mq = pd.Series(q, index=hy).groupby(level=0).mean()
    if len(mp) < 3 or len(mq) < 3:
        log.warning("Insufficient years for elasticity calculation")
        return float("nan")
    # Same code path as the CI (day-weighted long-term mean, as CAMELS e_qp_sanka).
    return _gu.elasticity_point(np.asarray(q, dtype=float), np.asarray(p, dtype=float), dates)


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

    # Zero flow frequency (fraction of days with q <= 0)
    zero_freq = float(np.sum(q <= 0) / len(q))   # CAMELS compute_no_flow: sum(q<=0)/length

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
    Slope of the mid-section of the flow duration curve, in two definitions.

    ``slope_fdc`` (CAMELS attributes v2.0 distributed definition, since 0.2.0)::

        (Q66 - Q33) / (mean(q) * 0.33)

    with Q33, Q66 the 33rd / 66th percentiles (non-exceedance, type 7), zero-flow days
    kept, defined only when ``mean(q) > 0``. No logarithm. See ``SIGNATURE_DEFINITIONS``
    for the source and what it does and does not establish.

    ``slope_fdc_sawicz2011``: the log-space value ``(ln Q33exc - ln Q66exc) / 0.33`` of
    Sawicz et al. (2011) (zeros dropped), which was this function's ``slope_fdc`` before 0.2.0.

    Returns: slope_fdc, slope_fdc_sawicz2011
    """

    q = q_mm_day.dropna().values.astype(float)
    out = {'slope_fdc': np.nan, 'slope_fdc_sawicz2011': np.nan}

    if len(q) < 100:
        log.warning(f"Insufficient discharge values for FDC: {len(q)}")
        return out

    if not float(np.mean(q)) > 0:
        log.warning("Mean discharge <= 0: slope_fdc undefined")
    out['slope_fdc'] = fdc_slope_camels_file(q)
    out['slope_fdc_sawicz2011'] = fdc_slope_sawicz2011(q)
    log.debug("FDC slope: %s", out)
    return out


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

    The product the data layer reports it served (and, on failure, the reason)
    is left in ``_PRECIP_DIAG.d``; the returned series carries the product in
    ``.attrs["product"]``. Validity is NOT judged here — see
    :func:`precipitation_problem`.
    """
    diag: dict = {}
    _PRECIP_DIAG.d = diag
    try:
        import aihydro_data
        import geopandas as gpd
    except ImportError:
        log.warning("aihydro_data not installed — precipitation fetch unavailable.")
        diag["reason"] = "aihydro_data not installed"
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
            diag["reason"] = "precipitation fetch returned an empty result"
            diag["product"] = getattr(result, "product", None)
            return None
        # Both GEE and CHIRPS_IRI backends return DataFrame[date, precipitation]
        p_col = "precipitation" if "precipitation" in df.columns else None
        if p_col is None:
            p_col = next((c for c in df.columns if c != "date"), None)
        if p_col is None:
            log.warning("Precipitation DataFrame has no usable column")
            diag["reason"] = "precipitation result has no usable column"
            diag["product"] = getattr(result, "product", None)
            return None
        idx = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
        s = pd.Series(df[p_col].values, index=idx, dtype=float).dropna()
        s.index = s.index.tz_localize(None) if s.index.tzinfo is not None else s.index
        s.name = "precip_mm"
        diag["product"] = getattr(result, "product", None)
        diag["units"] = getattr(result, "units", None) or None
        diag["units_declared"] = getattr(result, "units_declared", None) or None
        diag["data_coverage"] = _data_coverage(result)
        s.attrs["product"] = diag["product"]
        log.info(
            "Retrieved %d days of precipitation (product=%s)",
            len(s), getattr(result, "product", "unknown"),
        )
        return s
    except Exception as e:
        log.warning("Precipitation fetch skipped (runoff_ratio/stream_elas will be NaN): %s", e)
        diag["reason"] = f"no precipitation source returned data: {type(e).__name__}: {e}"
        return None


def _data_coverage(result) -> Optional[dict]:
    """Coverage fields the data layer reported for the fetch, when it reports them.

    aihydro-data >= 0.2.5 sets ``missing_ranges`` (run-compressed ``{start, end, days, reason}``
    with reason codes such as ``client_calendar_assumption``, ``chunk_mismatch``,
    ``absent_in_response``) and ``days_missing_internal``. Older versions only know the head and
    tail. Only well-formed values are taken; anything else is ignored, never guessed.
    """
    cov: dict = {}
    for k in ("coverage_complete", "days_missing_head", "days_missing_tail", "days_missing_internal"):
        v = getattr(result, k, None)
        if isinstance(v, (bool, int)):
            cov[k] = v
    mr = getattr(result, "missing_ranges", None)
    if isinstance(mr, (list, tuple)):
        ok = [dict(r) for r in mr
              if isinstance(r, dict) and r.get("start") and r.get("end") and r.get("reason")]
        if ok:
            cov["missing_ranges"] = ok
    return cov or None


# ---------------------------------------------------------------------------
# Precipitation validity gate + provenance record
# ---------------------------------------------------------------------------

# Physical bounds for a basin-mean DAILY precipitation series (mm/day). They
# are deliberately generous: the point is to refuse fill values / placeholders
# (e.g. a ~1e33 netCDF fill, the likely cause of runoff_ratio = 2.3e-33 in
# e2e proof 2), not to second-guess real climate.
# ENGINEERING BOUNDS; SOURCE TO VERIFY (no literature value is asserted here).
#   daily max : 2000 mm/day - chosen to sit above any plausible basin-mean day.
#   mean max  : 35 mm/day (~12,800 mm/yr) - chosen above the wettest places.
#   mean min  : 1e-4 mm/day (~0.04 mm/yr) - an effectively-zero / denormal
#               series is a placeholder, not climate; the driest places on
#               Earth are expected to be well above it (source to verify).
_PRECIP_DIAG = threading.local()
PRECIP_MIN_DAYS = 365
PRECIP_MAX_DAILY_MM = 2000.0
PRECIP_MAX_MEAN_MM_DAY = 35.0
PRECIP_MIN_MEAN_MM_DAY = 1e-4
PRECIP_DIGEST_SCHEME = "sha256 over 'YYYY-MM-DD<TAB>repr(float)<LF>' lines of the received series"
PRECIP_DEPENDENT_SIGNATURES = ["runoff_ratio", "stream_elas"]


_MM_PER_DAY_UNITS = {
    "mm/day", "mm/d", "mmday-1", "mmd-1", "mmday^-1", "mm.day-1",
    "kg/m2/day", "kgm-2day-1", "kgm-2d-1", "kg/m^2/day",
}
#: Plausible long-term Q/P. ENGINEERING BOUND; SOURCE TO VERIFY. Same bounds as
#: aihydro-tools' water-balance validator.
RUNOFF_RATIO_PLAUSIBLE = (1e-3, 3.0)


def _unit_is_mm_per_day(units: str) -> bool:
    norm = "".join(str(units).lower().replace("\u00b2", "2").split())
    return norm in _MM_PER_DAY_UNITS


def _effective_units(rec: dict) -> Optional[str]:
    """The unit to judge: what the payload declared, else the product spec's."""
    return rec.get("units_declared") or rec.get("units") or None


def precipitation_problem(p, units: Optional[str] = None) -> Optional[str]:
    """Return ``None`` if ``p`` is a usable basin-mean daily precipitation
    series (mm/day), else a short reason string. Never raises.

    ``units``: the unit the series declares, if known. A declared unit that is
    not mm/day-equivalent (e.g. m/day, mm/month) is refused as malformed input:
    the magnitude gate cannot catch a plausible-looking wrong-unit series.
    Unknown (None / empty) units are not refused.
    """
    if units is not None and str(units).strip() and not _unit_is_mm_per_day(units):
        return f"precipitation declared in unit {str(units).strip()!r}, not mm/day"
    if p is None:
        return "no precipitation series (no source returned data)"
    try:
        vals = np.asarray(p, dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        return f"precipitation series not numeric: {exc}"
    if np.isinf(vals).any():
        return "precipitation contains non-finite (inf) values"
    vals = vals[~np.isnan(vals)]
    if len(vals) < PRECIP_MIN_DAYS:
        return f"fewer than {PRECIP_MIN_DAYS} valid precipitation days ({len(vals)})"
    if (vals < 0).any():
        return f"negative precipitation values (min {vals.min():.4g} mm/day)"
    if vals.max() > PRECIP_MAX_DAILY_MM:
        return (f"precipitation outside physical range: max {vals.max():.4g} mm/day "
                f"> {PRECIP_MAX_DAILY_MM:g} (fill value or unit error)")
    mean = float(vals.mean())
    if not mean >= PRECIP_MIN_MEAN_MM_DAY:
        return (f"precipitation is zero or effectively zero (mean {mean:.4g} mm/day "
                f"< {PRECIP_MIN_MEAN_MM_DAY:g}): placeholder series")
    if mean > PRECIP_MAX_MEAN_MM_DAY:
        return (f"precipitation outside physical range: mean {mean:.4g} mm/day "
                f"> {PRECIP_MAX_MEAN_MM_DAY:g}")
    return None


def _finite_or_none(x) -> Optional[float]:
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return None
    return x if np.isfinite(x) else None


#: Longest ``missing_ranges`` list recorded; beyond it the list is cut and flagged.
MISSING_RANGES_CAP = 100
ABSENT_IN_RESPONSE = "absent_in_response"


def _runs(days: pd.DatetimeIndex, reasons: list) -> list:
    """Run-compress per-day ``(day, reason)`` into ``{start, end, days, reason}`` ranges."""
    out: list = []
    for d, r in zip(days, reasons):
        if out and out[-1]["reason"] == r and (d - pd.Timestamp(out[-1]["end"])).days == 1:
            out[-1]["end"] = f"{d:%Y-%m-%d}"
            out[-1]["days"] += 1
        else:
            out.append({"start": f"{d:%Y-%m-%d}", "end": f"{d:%Y-%m-%d}", "days": 1, "reason": r})
    return out


def _completeness(valid_days: pd.DatetimeIndex, window, data_cov: Optional[dict]) -> dict:
    """``n_expected_days``, ``completeness`` and ``missing_ranges`` of a daily series.

    ``window`` is the requested ``(start, end)`` (inclusive); when absent, the span of the series
    itself, so internal gaps are still found. Reason codes come from the data layer when it
    supplied a range covering the day (``data_coverage.missing_ranges``); a gap it did not
    explain is ``absent_in_response``. This function never names a cause it was not told.
    """
    vd = pd.DatetimeIndex(valid_days).normalize()
    if window and window[0] and window[1]:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    else:
        lo, hi = vd.min(), vd.max()
    expected = pd.date_range(lo, hi, freq="D")
    present = expected.intersection(vd.unique())
    missing = expected.difference(present)
    given = (data_cov or {}).get("missing_ranges") or []

    def _why(d: pd.Timestamp) -> str:
        for r in given:
            try:
                if pd.Timestamp(r["start"]) <= d <= pd.Timestamp(r["end"]):
                    return str(r["reason"])
            except Exception:  # noqa: BLE001  (a malformed range from upstream is ignored)
                continue
        return ABSENT_IN_RESPONSE

    ranges = _runs(missing, [_why(d) for d in missing])
    return {
        "n_expected_days": int(len(expected)),
        "n_valid_expected_days": int(len(present)),
        "completeness": round(len(present) / len(expected), 6) if len(expected) else None,
        "missing_ranges": ranges[:MISSING_RANGES_CAP],
        "missing_ranges_truncated": len(ranges) > MISSING_RANGES_CAP,
        "missing_range_reasons_from": "aihydro-data" if given else "aihydro-watershed (computed from the series; cause unknown)",
    }


def _precipitation_record(p, diag: dict, attempted: bool = True, window=None) -> dict:
    """Provenance of the precipitation the water-balance signatures used (or
    did not use). ``status``: used | used_with_gaps | rejected | unavailable | not_attempted
    (| unusable, set by the caller when the water balance itself was not computable).

    ``window``: the requested ``(start, end)``; completeness is measured against it
    (``n_expected_days``, ``completeness``, ``missing_ranges``)."""
    import hashlib
    rec: dict = {
        "status": "not_attempted", "product": diag.get("product"),
        "units": diag.get("units"), "units_declared": diag.get("units_declared"),
        "digest": None, "digest_scheme": PRECIP_DIGEST_SCHEME,
        "n_days": 0, "start": None, "end": None,
        "mean_mm_day": None, "min_mm_day": None, "max_mm_day": None,
        "reason": diag.get("reason"),
        "dependent_signatures": list(PRECIP_DEPENDENT_SIGNATURES),
        "n_expected_days": None, "completeness": None, "missing_ranges": [],
        "data_coverage": diag.get("data_coverage"),
        "water_year_rule": {
            "min_paired_fraction": WATER_YEAR_MIN_PAIRED_FRACTION,
            "basis": "engineering choice; source to verify",
            "applies_to": list(PRECIP_DEPENDENT_SIGNATURES),
        },
        "water_years_used": [], "water_years_excluded": [], "n_paired_days_used": 0,
    }
    if not attempted:
        return rec
    if p is None or len(p) == 0:
        rec["status"] = "unavailable"
        rec["reason"] = rec["reason"] or "no precipitation source returned data"
        return rec
    ser = pd.Series(p)
    h = hashlib.sha256()
    idx = pd.to_datetime(ser.index)
    for d, v in zip(idx, ser.values):
        h.update(f"{d:%Y-%m-%d}\t{float(v)!r}\n".encode())
    vals = ser.astype(float).values
    finite = vals[np.isfinite(vals)]
    rec.update(
        digest="sha256:" + h.hexdigest(), n_days=int(len(ser)),
        start=f"{idx.min():%Y-%m-%d}", end=f"{idx.max():%Y-%m-%d}",
        mean_mm_day=_finite_or_none(finite.mean()) if len(finite) else None,
        min_mm_day=_finite_or_none(finite.min()) if len(finite) else None,
        max_mm_day=_finite_or_none(finite.max()) if len(finite) else None,
        product=rec["product"] or (getattr(p, "attrs", None) or {}).get("product"),
    )
    valid_idx = pd.DatetimeIndex(idx[np.isfinite(vals)])
    if len(valid_idx):
        rec.update(_completeness(valid_idx, window, rec["data_coverage"]))
    problem = precipitation_problem(p, _effective_units(rec))
    if problem is None:
        rec["status"], rec["reason"] = "used", None
    else:
        rec["status"], rec["reason"] = "rejected", problem
    return rec


def _gap_note(rec: dict) -> Optional[str]:
    """One-line disclosure of the gaps in a usable precipitation record, or None when there are none."""
    ex = rec.get("water_years_excluded") or []
    rngs = rec.get("missing_ranges") or []
    comp = rec.get("completeness")
    if not ex and not rngs and (comp is None or comp >= 1.0):
        return None
    parts = []
    if comp is not None and comp < 1.0:
        parts.append(f"precipitation covers {comp:.1%} of the {rec.get('n_expected_days')} requested days "
                     f"({sum(r['days'] for r in rngs)} days in {len(rngs)} missing range(s)"
                     + (", list truncated" if rec.get("missing_ranges_truncated") else "") + ")")
    if ex:
        parts.append("water years excluded as incomplete: "
                     + ", ".join(f"WY{e['water_year']} ({e['fraction']:.0%})" for e in ex))
    return "; ".join(parts) + (f"; runoff_ratio and stream_elas use {len(rec.get('water_years_used') or [])}"
                               " complete water years only")


def _lyne_hollick_baseflow(
    q: np.ndarray,
    alpha: float = 0.925,
    passes: int = 3,
) -> np.ndarray:
    """Baseflow series of ``q`` by the canonical Lyne-Hollick kernel.

    Since 0.2.0 this is :func:`aihydro_watershed.signatures.baseflow.lyne_hollick_ladson2013`
    (alpha 0.925, forward-backward-forward, 30-day reflection, quick flow subtracted only where
    positive, final clip at zero), i.e. the filter CAMELS applied (``comp_i_bf_landson``). The
    baseflow index is ``sum(baseflow) / sum(q)``.

    Before 0.2.0 each sweep clamped ``0 <= b <= y`` and there was no reflection; that variant
    read +3e-4 relative to CAMELS on the verified basins, this one reproduces them to <= 2e-6
    (docs/vision-2040/evidence/signature-definitions-1).
    """
    q = np.asarray(q, dtype=float)
    if q.size == 0 or np.all(~np.isfinite(q)):
        return np.full_like(q, np.nan, dtype=float)
    return np.clip(lyne_hollick_ladson2013(q, alpha=alpha, passes=passes), 0, q)


def _daily_values(q_series: pd.Series) -> np.ndarray:
    """Values on the full daily axis (NaN for calendar gaps) when the index is a DatetimeIndex."""
    if isinstance(q_series.index, pd.DatetimeIndex):
        return q_series.sort_index().resample("D").asfreq().to_numpy(dtype=float)
    return q_series.to_numpy(dtype=float)


def _baseflow_index(q_series: pd.Series) -> float:
    """Baseflow index of a daily series; calendar gaps are segment boundaries, not spliced."""
    x = _daily_values(q_series)
    bf = _lyne_hollick_baseflow(
        x, alpha=BASEFLOW_SEPARATION_PARAMS["alpha"], passes=BASEFLOW_SEPARATION_PARAMS["passes"])
    ok = np.isfinite(bf) & np.isfinite(x)
    den = float(np.sum(x[ok]))
    if not ok.any() or den <= 0:
        return float("nan")
    return float(max(0.0, min(1.0, float(np.sum(bf[ok])) / den)))


def _bfi_items(q_series: pd.Series):
    """``(bf, q, dates)``: paired baseflow / flow (mm/day) on the days that have a baseflow value,
    filter applied once to the observed daily series. ``None`` if there is none."""
    ser = q_series.sort_index() if isinstance(q_series.index, pd.DatetimeIndex) else q_series
    if isinstance(ser.index, pd.DatetimeIndex):
        ser = ser.resample("D").asfreq()
        dates = pd.DatetimeIndex(ser.index).tz_localize(None) if ser.index.tz is not None else ser.index
    else:
        return None
    x = ser.to_numpy(dtype=float)
    bf = _lyne_hollick_baseflow(
        x, alpha=BASEFLOW_SEPARATION_PARAMS["alpha"], passes=BASEFLOW_SEPARATION_PARAMS["passes"])
    ok = np.isfinite(bf) & np.isfinite(x)
    if not ok.any():
        return None
    return bf[ok], x[ok], dates[ok]


def _baseflow_index_uncertainty(q_series: pd.Series) -> Dict[str, dict]:
    """Water-year grouped percentile CI for the baseflow index.

    The Lyne-Hollick filter is applied ONCE to the observed series (exactly as for the reported
    point value); water years of paired (baseflow, flow) days are then resampled and the index
    re-evaluated as ``sum(b) / sum(q)``. Re-running the filter on a spliced resample is
    deliberately avoided: every splice is an artificial step that the filter reads as quick flow.
    The interval excludes filter-state / parameter uncertainty and observation (rating) error.
    Returns ``{}`` when the series is too short for any interval.
    """
    items = _bfi_items(q_series)
    if items is None or float(np.sum(items[1])) <= 0:
        return {}
    bf, x, dates = items
    return _gu.baseflow_index_ci(bf, x, dates)


def _grouped_cis(q_mm_day: pd.Series, wbd: Optional[dict]) -> Dict[str, dict]:
    """All grouped CIs of one signatures run (see ``_uncertainty``)."""
    out: Dict[str, dict] = {}
    qs = q_mm_day.dropna()
    if len(qs) < 30 or not isinstance(qs.index, pd.DatetimeIndex):
        return out
    dates = pd.DatetimeIndex(qs.index).tz_localize(None) if qs.index.tz is not None else qs.index
    out.update(_gu.flow_statistic_cis(qs.to_numpy(dtype=float), dates))
    out.update(_baseflow_index_uncertainty(q_mm_day))
    if wbd and wbd.get("q_used") is not None and np.isfinite(wbd["values"]["runoff_ratio"]):
        qu, pu = wbd["q_used"], wbd["p_used"]
        out.update(_gu.water_balance_cis(
            qu.to_numpy(dtype=float), pu.to_numpy(dtype=float), pd.DatetimeIndex(qu.index),
            with_elasticity=bool(np.isfinite(wbd["values"]["stream_elas"]))
            or len(wbd.get("water_years_used") or []) >= 3))
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
    """Hydrologic (water) year label of each date: the calendar year in which the year ends.

    Start month 10 (the default and the only one the signatures use) is aihydro-core's
    ``water_year``, the platform's single convention (USGS: 1 Oct starts the year; labelled by
    its end year)."""
    if hydro_year_start_month == 10:
        return _core_water_year(dti)
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
        'slope_fdc_sawicz2011': np.nan,
    }
