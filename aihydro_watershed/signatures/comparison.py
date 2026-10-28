"""Pure comparison of computed signatures with a CAMELS-US reference row (critique M6).

``reference_comparison`` does no I/O: the caller supplies the reference row (e.g. a line of
``camels_hydro.txt`` joined with ``camels_clim.txt``) and the two areas / periods. For each shared
signature it says whether a difference could be a *real* difference or is explained by something
other than the catchment:

* ``area_dependent`` - depth statistics are streamflow divided by drainage area; a different area
  rescales them (CAMELS uses ``area_gages2``). ``value_at_ref_area`` is the rescaled value.
* ``definition_comparable`` - ``slope_fdc`` and ``baseflow_index`` changed definition in 0.2.0;
  a result without ``signature_definitions`` predates that and is not known to be comparable.
* ``forcing_differs`` - ``runoff_ratio`` and ``stream_elas`` depend on the precipitation product;
  CAMELS attributes v2.0 uses Daymet.
* ``period_differs`` - CAMELS signatures are WY1990-2009. The period actually used is READ FROM THE
  RECORD (``streamflow_start`` / ``streamflow_end``); a caller's ``period_used`` is only a fallback
  and a disagreement with the record is reported, never trusted.
* for ``runoff_ratio`` and ``stream_elas`` the record's own precipitation completeness also blocks:
  ``water_years_excluded`` (an incomplete year took no part, so the sample is not the CAMELS one),
  ``precipitation_used_with_gaps`` (gaps with any reason other than ``client_calendar_assumption``;
  a skipped leap-year Dec 31 alone does not block) and ``streamflow_period_differs``.

``can_claim_difference`` is False whenever any of those blockers is present, so a gate can refuse
"differs from CAMELS" claims (and "agrees" claims too: agreement under a different forcing is
coincidence, not confirmation) without a per-tool carve-out.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

CAMELS_REFERENCE_PERIOD = ("1989-10-01", "2009-09-30")   # WY1990-2009 (proven by exact matches)
CAMELS_REFERENCE_FORCING = "Daymet"

#: Signatures of ``camels_hydro.txt`` this comparison covers.
CAMELS_SIGNATURES = (
    "q_mean", "runoff_ratio", "slope_fdc", "baseflow_index", "stream_elas", "q5", "q95",
    "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "hfd_mean",
)
AREA_DEPENDENT = ("q_mean", "q5", "q95", "q_std", "runoff_ratio")   # depth rates (mm/day) and Q/P
FORCING_DEPENDENT = ("runoff_ratio", "stream_elas")
CLIENT_CALENDAR = "client_calendar_assumption"   # reason code from aihydro-data (leap-year Dec 31 skipped)
DEFINITION_SENSITIVE = ("slope_fdc", "baseflow_index")
CLIMATE_CONTEXT_KEYS = ("frac_snow", "aridity", "p_seasonality")


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _definition_status(sigs: Mapping, name: str) -> tuple:
    """(comparable: True | False | None, note)."""
    defs = sigs.get("signature_definitions")
    if not isinstance(defs, Mapping) or name not in defs:
        return None, ("no signature_definitions in this result (computed before watershed 0.2.0 or "
                      "by another tool): comparability with the CAMELS attributes v2.0 definition is unknown")
    if name == "slope_fdc":
        comp = isinstance(defs["slope_fdc"], Mapping) and not defs["slope_fdc"].get("log_transform", True)
        return comp, ("CAMELS attributes v2.0 distributed definition (mean-normalised linear)" if comp
                      else "slope_fdc here is a log-space definition; the CAMELS attributes v2.0 column is not")
    if name == "baseflow_index":
        method = (defs["baseflow_index"] or {}).get("method") if isinstance(defs["baseflow_index"], Mapping) else None
        comp = method == "lyne_hollick_ladson2013"
        return comp, ("Lyne-Hollick / Ladson 2013 kernel, as CAMELS" if comp
                      else f"baseflow method {method!r} is not the CAMELS Ladson 2013 recipe")
    return True, ""


def _wy(date_str: str) -> int:
    d = str(date_str)[:10]
    y, m = int(d[:4]), int(d[5:7])
    return y + 1 if m >= 10 else y


def _precip_blockers(rec: Mapping, period_ref) -> list:
    """Blockers read from the record's own ``_precipitation`` block (RR / elasticity rows)."""
    out = []
    if not isinstance(rec, Mapping):
        return ["precipitation_record_missing"]
    if rec.get("water_years_excluded"):
        out.append("water_years_excluded")
    if rec.get("status") == "used_with_gaps":
        reasons = {r.get("reason") for r in (rec.get("missing_ranges") or []) if isinstance(r, Mapping)}
        if (not reasons) or (reasons - {CLIENT_CALENDAR}) or rec.get("missing_ranges_truncated"):
            out.append("precipitation_used_with_gaps")
    used = rec.get("water_years_used")
    if used and period_ref:
        expected = set(range(_wy(period_ref[0]), _wy(period_ref[1]) + 1))
        if set(used) != expected and "water_years_excluded" not in out:
            out.append("water_years_used_differ")
    return out


def reference_comparison(
    sigs: Mapping,
    ref_row: Mapping,
    area_used: Optional[float],
    area_ref: Optional[float],
    period_used: Optional[Sequence[str]],
    period_ref: Optional[Sequence[str]] = CAMELS_REFERENCE_PERIOD,
    forcing_ref: str = CAMELS_REFERENCE_FORCING,
) -> dict:
    """Compare ``sigs`` (``result.data``) with ``ref_row`` (CAMELS column -> value).

    ``area_used`` is the km2 the depths were computed with; ``area_ref`` the CAMELS area
    (``area_gages2``); ``period_used`` / ``period_ref`` are ``(start, end)`` ISO dates.
    Returns ``{"table": {signature: {...}}, "context": {...}, "climate_context": {...}}``.
    """
    ratio = None
    if _num(area_used) and _num(area_ref) and area_used > 0 and area_ref > 0:
        ratio = float(area_used) / float(area_ref)
    area_differs = None if ratio is None else abs(math.log(ratio)) > 1e-3
    sf_start, sf_end = sigs.get("streamflow_start"), sigs.get("streamflow_end")
    derived = (str(sf_start)[:10], str(sf_end)[:10]) if sf_start and sf_end else None
    caller = tuple(str(x)[:10] for x in period_used) if period_used else None
    caller_mismatch = bool(derived and caller and derived != caller)
    effective = derived or caller          # the record wins; the caller's period is a fallback
    ref_t = tuple(str(x)[:10] for x in period_ref) if period_ref else None
    period_differs = None if not (effective and ref_t) else (effective != ref_t)
    streamflow_period_differs = None if not (derived and ref_t) else (derived != ref_t)
    precip_blockers = _precip_blockers(sigs.get("_precipitation"), ref_t)
    product = sigs.get("precipitation_product") or (sigs.get("_precipitation") or {}).get("product")
    if product is None:
        forcing_differs = None
    else:
        forcing_differs = forcing_ref.lower() not in str(product).lower()

    table: Dict[str, dict] = {}
    for name in CAMELS_SIGNATURES:
        ref = _num(ref_row.get(name))
        val = _num(sigs.get(name))
        if ref is None and val is None:
            continue
        row: dict = {"value": val, "reference": ref, "diff": None, "rel_diff": None}
        blockers = []
        if name in AREA_DEPENDENT:
            row["area_dependent"] = True
            if ratio is None:
                blockers.append("area_unknown")
            elif area_differs:
                row["value_at_ref_area"] = None if val is None else val * ratio
                blockers.append("area_differs")
        else:
            row["area_dependent"] = False
        if name in DEFINITION_SENSITIVE:
            comp, note = _definition_status(sigs, name)
            row["definition_comparable"] = comp
            row["definition_note"] = note
            if comp is not True:
                blockers.append("definition_not_comparable" if comp is False else "definition_unknown")
        else:
            row["definition_comparable"] = True
        row["forcing_differs"] = forcing_differs if name in FORCING_DEPENDENT else False
        if name in FORCING_DEPENDENT and forcing_differs is not False:
            blockers.append("forcing_differs" if forcing_differs else "forcing_unknown")
        row["period_differs"] = period_differs
        if name in FORCING_DEPENDENT:
            blockers.extend(precip_blockers)
            if streamflow_period_differs:
                blockers.append("streamflow_period_differs")
        if period_differs is not False:
            blockers.append("period_differs" if period_differs else "period_unknown")
        compare_val = row.get("value_at_ref_area", val)
        if compare_val is not None and ref is not None:
            row["diff"] = compare_val - ref
            row["rel_diff"] = (compare_val - ref) / ref if ref != 0 else None
        row["blockers"] = blockers
        row["can_claim_difference"] = not blockers and row["diff"] is not None
        table[name] = row

    return {
        "table": table,
        "context": {
            "area_used_km2": _num(area_used), "area_ref_km2": _num(area_ref), "area_ratio": ratio,
            "area_differs": area_differs,
            "period_used": list(effective) if effective else None,
            "period_used_source": "record (streamflow_start/end)" if derived else ("caller" if caller else None),
            "period_used_caller_mismatch": caller_mismatch,
            "period_ref": list(period_ref) if period_ref else None, "period_differs": period_differs,
            "precipitation_product": product, "forcing_ref": forcing_ref, "forcing_differs": forcing_differs,
            "note": ("CAMELS attributes v2.0 signatures are WY1990-2009 from Daymet forcing and area_gages2. "
                     "A signature with blockers cannot support a 'differs from CAMELS' (or 'agrees with "
                     "CAMELS') claim: the sampling-only confidence intervals exclude forcing product, "
                     "area, definition and period."),
        },
        "climate_context": {k: _num(ref_row.get(k)) for k in CLIMATE_CONTEXT_KEYS if k in ref_row},
    }
