"""NLDI watershed at a pour point (CONUS) via nearest COMID."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import geopandas as gpd

from aihydro_watershed.delineation.types import area_km2
from aihydro_core import DataSource, HydroMeta, HydroResult, ToolError

log = logging.getLogger(__name__)

_TOOL_PATH = "aihydro_watershed.delineation.nldi_point.delineate_nldi_at_point"

_NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
_HYDROLOCATION_ATTEMPTS = 4
_HYDROLOCATION_BACKOFF_S = (3.0, 8.0, 15.0)
_HYDROLOCATION_TIMEOUT_S = 30.0

_SOURCES = [
    DataSource(
        name="USGS NLDI",
        url="https://api.water.usgs.gov/nldi/linked-data",
    ),
]

# Rough CONUS extent for NLDI hydrologic indexing
CONUS_LON = (-125.0, -66.0)
CONUS_LAT = (24.0, 50.0)

# Pour-point quick pick (no published drainage area): expand off tiny tributary reaches.
_QUICK_TRIBUTARY_MAX_KM2 = 80.0
_QUICK_DOWNSTREAM_KM = 30
_QUICK_MAX_CANDIDATES = 12
_QUICK_MAX_BASIN_KM2 = 15_000.0


@dataclass(frozen=True)
class HydrolocationHit:
    """The NLDI-indexed flowline nearest a point (``hydrolocation`` endpoint)."""

    comid: int
    lon: float  # snapped point on the flowline
    lat: float
    measure: float | None = None
    reachcode: str | None = None


def hydrolocation_url(lon: float, lat: float) -> str:
    """The ``hydrolocation`` request URL with the WKT space percent-encoded.

    pynhd (0.19.x) formats ``POINT(lon lat)`` with a bare space and lets its HTTP stack
    encode it, which reaches the service as ``+``. The USGS NLDI service answers some
    points (e.g. CAMELS 08155200) with HTTP 502 in the ``+`` form and 200 with ``%20``.
    We build the URL ourselves so the form is fixed.
    """
    return f"{_NLDI_BASE}/hydrolocation?coords=POINT({lon:.6f}%20{lat:.6f})"


def _parse_hydrolocation(payload: dict) -> HydrolocationHit | None:
    """First feature the NLDI itself indexed (``source == 'indexed'``), else None."""
    for feat in payload.get("features") or []:
        props = feat.get("properties") or {}
        if props.get("source") != "indexed" or props.get("comid") in (None, ""):
            continue
        x, y = feat["geometry"]["coordinates"][:2]
        measure = props.get("measure")
        return HydrolocationHit(
            comid=int(props["comid"]),
            lon=float(x),
            lat=float(y),
            measure=float(measure) if measure not in (None, "") else None,
            reachcode=props.get("reachcode") or None,
        )
    return None


def hydrolocation_comid(
    lon: float,
    lat: float,
    *,
    session=None,
    sleep=time.sleep,
) -> HydrolocationHit:
    """Nearest NLDI-indexed COMID for a point, via our own percent-encoded request.

    Retries 5xx / 429 / connection errors with back-off (a few calls, never a hammer).
    Raises ``ToolError`` ``NLDI_UNAVAILABLE`` (carrying the last HTTP status) when the
    service stays down, ``NLDI_NOT_FOUND`` when it answers but indexes nothing there.
    """
    import requests

    sess = session or requests
    url = hydrolocation_url(lon, lat)
    last = "no response"
    status: int | None = None
    for attempt in range(_HYDROLOCATION_ATTEMPTS):
        retry_after = None
        try:
            resp = sess.get(url, timeout=_HYDROLOCATION_TIMEOUT_S, headers={"Accept": "application/json"})
            status = resp.status_code
            if status == 200:
                hit = _parse_hydrolocation(resp.json())
                if hit is None:
                    raise ToolError(
                        code="NLDI_NOT_FOUND",
                        message=f"No NHD COMID indexed near ({lat}, {lon}).",
                        tool=_TOOL_PATH,
                    )
                return hit
            last = f"HTTP {status}"
            if status != 429 and status < 500:
                break  # a 4xx is not transient
            retry_after = resp.headers.get("Retry-After")
        except requests.RequestException as e:
            status = None
            last = f"{type(e).__name__}: {e}"
        if attempt < _HYDROLOCATION_ATTEMPTS - 1:
            wait = _HYDROLOCATION_BACKOFF_S[min(attempt, len(_HYDROLOCATION_BACKOFF_S) - 1)]
            try:
                wait = max(wait, min(float(retry_after), 60.0))
            except (TypeError, ValueError):
                pass
            log.info("NLDI hydrolocation %s; retrying in %.0fs", last, wait)
            sleep(wait)
    raise ToolError(
        code="NLDI_UNAVAILABLE",
        message=f"NLDI hydrolocation failed for ({lat}, {lon}): {last} (host api.water.usgs.gov).",
        tool=_TOOL_PATH,
        recovery="The USGS NLDI service is down or throttling; retry later or use method='merit_gee'.",
    )


def is_conus(lat: float, lon: float) -> bool:
    return CONUS_LON[0] <= lon <= CONUS_LON[1] and CONUS_LAT[0] <= lat <= CONUS_LAT[1]


def snap_outlet_nldi(lat: float, lon: float) -> tuple[float, float, bool]:
    """Snap pour point to nearest NHD COMID location (CONUS). Uses our own encoded hydrolocation request."""
    if not is_conus(lat, lon):
        return lat, lon, False
    try:
        hit = hydrolocation_comid(lon, lat)
        return hit.lat, hit.lon, True
    except Exception as e:
        log.debug("NLDI outlet snap failed: %s", e)
        return lat, lon, False


def resolve_comid_for_area(
    lat: float,
    lon: float,
    expected_area_km2: float,
    *,
    search_distance_km: int = 120,
    max_candidates: int = 60,
) -> tuple[int, float]:
    """
    Pick an NHD COMID whose indexed basin area best matches ``expected_area_km2``.

    The nearest COMID from ``comid_byloc`` is often a small tributary; this searches
    upstream/downstream flowlines in the NLDI network for a better match.
    """
    from pynhd import NLDI

    nldi = NLDI()
    start = hydrolocation_comid(lon, lat).comid
    candidates: set[int] = {start}

    for navigation in ("upstreamMain", "upstreamTributaries", "downstreamMain"):
        try:
            flowlines = nldi.navigate_byid(
                "comid",
                start,
                navigation,
                "flowlines",
                distance=search_distance_km,
            )
            col = "nhdplus_comid" if "nhdplus_comid" in flowlines.columns else "comid"
            if col in flowlines.columns:
                candidates.update(int(c) for c in flowlines[col].dropna().astype(int))
        except Exception as e:
            log.debug("NLDI navigate %s skipped: %s", navigation, e)

    best_comid = start
    best_area = area_km2(nldi.get_basins(start, fsource="comid").dissolve())
    best_err = abs(best_area - expected_area_km2)

    for comid in list(candidates)[:max_candidates]:
        if comid == start:
            continue
        try:
            a = area_km2(nldi.get_basins(comid, fsource="comid").dissolve())
        except Exception:
            continue
        err = abs(a - expected_area_km2)
        if err < best_err:
            best_err = err
            best_comid = comid
            best_area = a

    log.info(
        "COMID area match: start=%s -> selected=%s (%.0f km², target %.0f km²)",
        start,
        best_comid,
        best_area,
        expected_area_km2,
    )
    return best_comid, best_area


def quick_comid_resolution(
    lat: float,
    lon: float,
    *,
    allow_downstream_expansion: bool = False,
    search_distance_km: int = _QUICK_DOWNSTREAM_KM,
    max_candidates: int = _QUICK_MAX_CANDIDATES,
) -> tuple[int, float, dict | None]:
    """
    Pick the COMID for a pour point with no expected drainage area.

    Returns ``(comid, area_km2, ambiguity)``. The outlet is the user's point, so the
    default is the nearest reach's own basin. When that basin is small
    (< ``_QUICK_TRIBUTARY_MAX_KM2``) the point could be a tributary reach or the head of
    a larger stem; the downstream ``downstreamMain`` reaches are looked up and reported in
    ``ambiguity`` (``candidates`` with areas) and the nearest reach is still returned.
    Moving the outlet downstream to the largest basin (the former behaviour, which turned
    CAMELS 10336660, 29.8 km2, into a 1414 km2 basin) happens only with
    ``allow_downstream_expansion=True``. ``ambiguity`` is None when the nearest reach is
    not small.
    """
    from pynhd import NLDI

    nldi = NLDI()
    start = hydrolocation_comid(lon, lat).comid
    start_area = area_km2(nldi.get_basins(start, fsource="comid").dissolve())

    if start_area >= _QUICK_TRIBUTARY_MAX_KM2:
        return start, start_area, None

    candidates: list[int] = []
    try:
        flowlines = nldi.navigate_byid(
            "comid",
            start,
            "downstreamMain",
            "flowlines",
            distance=search_distance_km,
        )
        col = "nhdplus_comid" if "nhdplus_comid" in flowlines.columns else "comid"
        if col in flowlines.columns:
            candidates = [int(c) for c in flowlines[col].dropna().astype(int) if int(c) != start]
    except Exception as e:
        log.debug("NLDI downstreamMain skipped: %s", e)

    listed: list[dict] = [{"comid": start, "area_km2": round(start_area, 3), "role": "nearest_reach"}]
    for comid in candidates[:max_candidates]:
        try:
            a = area_km2(nldi.get_basins(comid, fsource="comid").dissolve())
        except Exception:
            continue
        listed.append({"comid": comid, "area_km2": round(a, 3), "role": "downstream_main"})

    larger = [c for c in listed[1:] if c["area_km2"] <= _QUICK_MAX_BASIN_KM2]
    ambiguity = {
        "reason": (
            f"nearest NHDPlus reach drains {start_area:.1f} km2 (< {_QUICK_TRIBUTARY_MAX_KM2:.0f}); "
            "the point may be on a tributary reach or on a larger stem"
        ),
        "selected_comid": start,
        "selected_area_km2": round(start_area, 3),
        "candidates": listed,
        "resolve_with": "pass expected_area_km2, or allow_downstream_expansion=True to move the outlet downstream",
    }
    if allow_downstream_expansion and larger:
        best = max(larger, key=lambda c: c["area_km2"])
        ambiguity["selected_comid"] = best["comid"]
        ambiguity["selected_area_km2"] = best["area_km2"]
        ambiguity["outlet_moved_downstream"] = True
        log.info(
            "Quick COMID (expansion allowed): start=%s (%.0f km2) -> selected=%s (%.0f km2)",
            start, start_area, best["comid"], best["area_km2"],
        )
        return int(best["comid"]), float(best["area_km2"]), ambiguity
    log.info(
        "Quick COMID: nearest reach %s (%.1f km2) kept; %d downstream candidates reported as ambiguity",
        start, start_area, len(listed) - 1,
    )
    return start, start_area, ambiguity


def resolve_comid_for_quick(
    lat: float,
    lon: float,
    *,
    allow_downstream_expansion: bool = False,
    search_distance_km: int = _QUICK_DOWNSTREAM_KM,
    max_candidates: int = _QUICK_MAX_CANDIDATES,
) -> tuple[int, float]:
    """``(comid, area_km2)`` of :func:`quick_comid_resolution` (ambiguity dropped)."""
    comid, a, _ = quick_comid_resolution(
        lat,
        lon,
        allow_downstream_expansion=allow_downstream_expansion,
        search_distance_km=search_distance_km,
        max_candidates=max_candidates,
    )
    return comid, a


def _normalize_nldi_basins(basins: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if basins.empty:
        return basins
    if basins.geometry.name is None:
        if "geometry" in basins.columns:
            basins = basins.set_geometry("geometry")
        else:
            geom_cols = [c for c in basins.columns if basins[c].dtype.name == "geometry"]
            if geom_cols:
                basins = basins.set_geometry(geom_cols[0])
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    else:
        basins = basins.to_crs(4326)
    return basins.dissolve().reset_index(drop=True)


def nldi_basin_gdf(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
) -> gpd.GeoDataFrame | None:
    """Return dissolved NLDI basin polygon for (lat, lon), or None if unavailable."""
    try:
        result = delineate_nldi_at_point(
            lat, lon, expected_area_km2=expected_area_km2
        )
    except Exception:
        return None
    geom = result.data.get("geometry_geojson")
    if not geom:
        return None
    gdf = gpd.GeoDataFrame.from_features([geom], crs=4326)
    if gdf.empty or gdf.geometry.is_empty.all():
        return None
    return gdf.dissolve().reset_index(drop=True)


def delineate_nldi_at_point(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    allow_downstream_expansion: bool = False,
) -> HydroResult:
    """
    Delineate using NLDI: snap to nearest NHD COMID, then fetch indexed basin.

    When ``expected_area_km2`` is set, searches the local river network for a COMID
    whose basin area is a closer match (avoids tiny tributary mis-picks). Without it the
    nearest reach's basin is returned; if that basin is small the result carries
    ``outlet_ambiguity`` and the ``NLDI_OUTLET_AMBIGUOUS`` quality flag, and the outlet
    moves downstream only when ``allow_downstream_expansion`` is True.
    """
    if not is_conus(lat, lon):
        raise ToolError(
            code="OUT_OF_DOMAIN",
            message="NLDI point delineation is only available for CONUS.",
            tool=_TOOL_PATH,
            recovery="Use method='fast' or merit_basins outside CONUS.",
        )

    from pynhd import NLDI

    nldi = NLDI()
    ambiguity: dict | None = None
    if expected_area_km2 and expected_area_km2 > 0:
        comid, _ = resolve_comid_for_area(lat, lon, expected_area_km2)
    else:
        comid, _, ambiguity = quick_comid_resolution(
            lat, lon, allow_downstream_expansion=allow_downstream_expansion
        )

    basins = nldi.get_basins(comid, fsource="comid")
    if basins.empty:
        raise ToolError(
            code="DELINEATION_FAILED",
            message=f"NLDI returned empty basin for COMID {comid}.",
            tool=_TOOL_PATH,
        )

    dissolved = _normalize_nldi_basins(basins)
    a = area_km2(dissolved)
    fc = json.loads(dissolved.to_json())
    geom = fc["features"][0]
    geom.setdefault("properties", {})

    import aihydro_watershed

    data = {
        "geometry_geojson": geom,
        "area_km2": round(a, 3),
        "outlet_lat": lat,
        "outlet_lon": lon,
        "method_used": "nldi_comid",
        "comid": comid,
        "name": f"basin_comid_{comid}",
    }
    if ambiguity is not None:
        data["outlet_ambiguity"] = ambiguity
        data["quality_flags"] = ["NLDI_OUTLET_AMBIGUOUS"]

    return HydroResult(
        data=data,
        meta=HydroMeta(
            tool=_TOOL_PATH,
            version=getattr(aihydro_watershed, "__version__", "unknown"),
            params={
                "lat": lat,
                "lon": lon,
                "comid": comid,
                "expected_area_km2": expected_area_km2,
                "allow_downstream_expansion": allow_downstream_expansion,
            },
            sources=_SOURCES,
        ),
    )
