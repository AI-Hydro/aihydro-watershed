"""Area verification of a delineated polygon against a reported drainage area, and the reliability summary.

Source: the delineation improvement audit of 2026-11-14 (``docs/vision-2040/findings/
delineation-improvement-audit-2026-11-14.md``), proposals P1 (reported-area check) and P3 (ambiguous outlet).

The check never changes the polygon. It compares the returned polygon area with a *reported* drainage area
and records the outcome as ``area_check``; a mismatch also marks the result ``unreliable``.

Reported-area providers, in order: (a) the caller's ``expected_area_km2``; (b) a lookup by gauge identifier
(USGS monitoring locations), only when the caller opts in. A provider that fails yields ``area_unverified``
with a reason, never an exception.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

#: |relative error| at or below this: ``verified_by_area``.
#: The audit measured the flag on a held-out CONUS sample against NWIS areas: at 35% 21 of 22 wrong basins
#: were caught with 10 of 178 correct ones flagged. Those numbers are for that sample only; NWIS and the
#: GAGES-II reference share lineage, so they are optimistic, and the check cannot see a wrong channel of
#: similar area. These defaults are not tuned on any new data; the audit's preregistered test applies.
TOL_VERIFY = 0.10

#: |relative error| above this: ``area_mismatch`` (result ``unreliable``, reason ``AREA_MISMATCH``).
#: Between ``TOL_VERIFY`` and ``TOL_FLAG`` the area neither verifies nor refutes: ``area_unverified``.
#: See the in-sample / held-out caveat at ``TOL_VERIFY``.
TOL_FLAG = 0.35

STATUS_VERIFIED = "verified_by_area"
STATUS_MISMATCH = "area_mismatch"
STATUS_UNVERIFIED = "area_unverified"

CODE_AREA_MISMATCH = "AREA_MISMATCH"
CODE_AMBIGUOUS_OUTLET = "AMBIGUOUS_OUTLET"

STATE_VERIFIED = "verified"
STATE_AMBIGUOUS = "ambiguous"
STATE_UNVERIFIED = "unverified"

SQMI_TO_KM2 = 2.589988110336

# USGS Water Data OGC API, monitoring-locations collection. Field ``drainage_area`` (number). The schema
# https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/schema?f=json titles it
# "Drainage area" without a unit; USGS documents square miles and the legacy site-file value for 01013500
# (drain_area_va = 870) equals the API value (aihydro-data, usgs_waterdata.AREA_UNITS_NOTE), so it is
# converted from square miles. API landing page: https://api.waterdata.usgs.gov/ogcapi/v1/
USGS_MONITORING_LOCATIONS_URL = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items"
USGS_LOOKUP_TIMEOUT_S = 10.0

_USGS_SITE = re.compile(r"^(?:USGS-)?(\d{8,15})$")


@dataclass(frozen=True)
class ReportedArea:
    """A reported drainage area (km2) with its source, or the reason none could be obtained."""

    area_km2: float | None
    source: str
    reason: str | None = None
    independent: bool = True  # False when the same number also steered the snap


#: A provider maps a gauge identifier to a :class:`ReportedArea`; it must not raise (failures are values).
Provider = Callable[[str], ReportedArea]


def usgs_reported_area(
    gauge_id: str,
    *,
    timeout: float = USGS_LOOKUP_TIMEOUT_S,
    http_get: Callable[..., Any] | None = None,
) -> ReportedArea:
    """Drainage area of a USGS site (km2) from the Water Data API. Never raises.

    ``http_get(url, params=..., timeout=...)`` is the network seam (default ``requests.get``); tests inject it.
    """
    source = "usgs_waterdata_monitoring_locations"
    m = _USGS_SITE.match(str(gauge_id).strip())
    if not m:
        return ReportedArea(None, source, f"{gauge_id!r} is not a USGS site number")
    site = f"USGS-{m.group(1)}"
    try:
        if http_get is None:
            import requests

            http_get = requests.get
        resp = http_get(
            USGS_MONITORING_LOCATIONS_URL,
            params={"id": site, "f": "json", "properties": "id,drainage_area"},
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        if getattr(resp, "status_code", 200) != 200:
            return ReportedArea(None, source, f"HTTP {resp.status_code} from the USGS Water Data API")
        feats = (resp.json() or {}).get("features") or []
        if not feats:
            return ReportedArea(None, source, f"{site} is not a monitoring location in the USGS Water Data API")
        da = (feats[0].get("properties") or {}).get("drainage_area")
        if da is None:
            return ReportedArea(None, source, f"{site} has no drainage_area recorded")
        da = float(da)
        if not da > 0:
            return ReportedArea(None, source, f"{site} drainage_area {da!r} is not positive")
        return ReportedArea(da * SQMI_TO_KM2, source)
    except Exception as exc:  # timeout, DNS, bad JSON: the lookup did not happen; not a claim about the basin
        log.info("USGS drainage-area lookup for %s failed: %s", gauge_id, exc)
        return ReportedArea(None, source, f"lookup failed: {type(exc).__name__}: {exc}")


def resolve_reported_area(
    *,
    expected_area_km2: float | None = None,
    gauge_id: str | None = None,
    lookup: bool = False,
    providers: list[Provider] | None = None,
) -> ReportedArea:
    """Reported area: caller-supplied first, then (opt-in) each provider for ``gauge_id``. Never raises."""
    if expected_area_km2 and expected_area_km2 > 0:
        # the caller's number also steers the snap, so agreement is weaker evidence than a separate source
        return ReportedArea(float(expected_area_km2), "caller_supplied", independent=False)
    if not lookup:
        return ReportedArea(None, "none", "no expected_area_km2 and lookup_reported_area is off")
    if not gauge_id:
        return ReportedArea(None, "none", "lookup_reported_area is on but no gauge_id was given")
    last = ReportedArea(None, "none", "no provider returned an area")
    for provider in providers if providers is not None else [usgs_reported_area]:
        try:
            got = provider(gauge_id)
        except Exception as exc:  # a provider that raises is a failed lookup
            got = ReportedArea(None, getattr(provider, "__name__", "provider"), f"provider raised: {exc}")
        if got.area_km2:
            return got
        last = got
    return last


def build_area_check(
    polygon_area_km2: float,
    reported: ReportedArea,
    *,
    tol_verify: float = TOL_VERIFY,
    tol_flag: float = TOL_FLAG,
) -> dict[str, Any]:
    """``{reported_area_km2, source, rel_err, status, ...}``; ``rel_err`` = (polygon - reported) / reported."""
    if not reported.area_km2 or reported.area_km2 <= 0:
        return {
            "reported_area_km2": None,
            "source": reported.source,
            "rel_err": None,
            "status": STATUS_UNVERIFIED,
            "reason": reported.reason or "no reported area",
            "independent_of_snap": None,
            "tol_verify": tol_verify,
            "tol_flag": tol_flag,
        }
    rel = (float(polygon_area_km2) - reported.area_km2) / reported.area_km2
    if abs(rel) <= tol_verify:
        status, reason = STATUS_VERIFIED, None
    elif abs(rel) > tol_flag:
        status = STATUS_MISMATCH
        reason = f"polygon {polygon_area_km2:.4g} km2 differs from reported {reported.area_km2:.4g} km2 by {rel:+.0%}"
    else:
        status = STATUS_UNVERIFIED
        reason = f"relative error {rel:+.0%} is between the verify ({tol_verify:.0%}) and flag ({tol_flag:.0%}) tolerances"
    return {
        "reported_area_km2": round(reported.area_km2, 3),
        "source": reported.source,
        "rel_err": round(rel, 4),
        "status": status,
        "reason": reason,
        "independent_of_snap": reported.independent,
        "tol_verify": tol_verify,
        "tol_flag": tol_flag,
    }


def reliability_state(area_check: dict[str, Any] | None, ambiguous_outlet: bool) -> str:
    """Three-way summary, never vacuous: ``verified`` | ``ambiguous`` | ``unverified``.

    ``verified``: the area check passed (it also resolves an ambiguous outlet, since the reported area
    selected the polygon). ``ambiguous``: ``AREA_MISMATCH`` or an unresolved ``AMBIGUOUS_OUTLET``.
    ``unverified``: no evidence either way. This is a summary of evidence, not a probability of correctness.
    """
    status = (area_check or {}).get("status")
    if status == STATUS_VERIFIED:
        return STATE_VERIFIED
    if status == STATUS_MISMATCH or ambiguous_outlet:
        return STATE_AMBIGUOUS
    return STATE_UNVERIFIED
