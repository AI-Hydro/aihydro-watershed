"""
Tiered watershed delineation router for global pour points.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import geopandas as gpd

from aihydro_watershed.delineation.nldi_point import is_conus
from aihydro_watershed.delineation.types import FastDelineationResult
from aihydro_core import DataSource, HydroMeta, HydroResult, ToolError

log = logging.getLogger(__name__)

Method = Literal[
    "auto",
    "nldi",
    "merit_gee",
    "local_merit",
    "merit_basins",
    "merit_hybrid",
    "dem_raw_fallback",
    "fast",
    "small_catchment",
    "3dep",
]

_TOOL_PATH = "aihydro_watershed.delineation.router.delineate_from_point"
_AREA_MISMATCH_FRAC = 0.25
_MIN_AREA_KM2 = 1.0
_LARGE_SNAP_M = 5000.0
_NLDI_QUICK_MIN_KM2 = 1.0
_NLDI_QUICK_MAX_KM2 = 15_000.0
# Culvert-scale tier (USGS 3DEP 10 m + embankment notch). In auto mode it runs
# first in CONUS when the caller's expected area is below this, and as the
# CONUS fallback when NLDI returns nothing usable. Its own acceptance floor is
# ten 10 m cells; results under 0.05 km2 are flagged LIKELY_DITCH_SNAP.
_SMALL_CATCHMENT_MAX_KM2 = 5.0
_SMALL_CATCHMENT_MIN_KM2 = 0.001
_SMALL_CATCHMENT_METHOD = "small_catchment_3dep"
# Quality flags the small-catchment tier raises when its own result is a known failure signature (basin cut by
# the 10 km window, or a roadside-ditch snap). When that tier is only a stand-in for a failed NLDI call, a result
# carrying either is not served (see ``_try_small_catchment(strict=True)``).
_SMALL_CATCHMENT_REJECT_FLAGS = ("BASIN_TOUCHES_MAX_WINDOW", "LIKELY_DITCH_SNAP")
# Two tiers' areas that differ by more than this fraction of the served polygon raise SOURCE_DISAGREEMENT.
# Same 25 % as ``_AREA_MISMATCH_FRAC``: in the CAMELS evaluation NLDI-vs-USGS area error had P90 4.6 %, so 25 %
# is far outside the noise between two honest sources and inside the errors of a wrong basin.
_SOURCE_DISAGREEMENT_FRAC = 0.25
# What a throttled or down NLDI looks like from the caller: HTTP 429 / 5xx, no response, timeouts, the fail-fast
# memo, and the two shapes pynhd raises when a throttled reply is not the expected JSON/GeoJSON
# ("Service returned no features", "string indices must be integers"). Seen in the CAMELS evaluation:
# every gauge with one of those messages had a 429 in the same run.
_TRANSIENT_NLDI_RE = re.compile(
    r"\bHTTP\s*(?:429|5\d\d)\b|\bstatus(?:\s*code)?[ =:]*(?:429|5\d\d)\b|\btimed?[ -]?out\b|"
    # aiohttp / async_retriever ("502, message='Bad Gateway'") and requests ("500 Server Error: ...") lead with the code.
    r"^\s*(?:429|5\d\d)\b[,:]|\b(?:429|5\d\d) Server Error\b|"
    r"\bconnection (?:reset|refused|aborted|closed|error)\b|\bcannot connect to host\b|\bDNS\b|failing fast|"
    r"\bdeadline\b|too many requests|\bservice unavailable\b|\bno response\b|"
    r"Service returned no features|string indices must be integers",
    re.I,
)


def _transient_exception_types() -> tuple[type, ...]:
    """Exception classes that mean "could not reach the service" whatever their (possibly empty) message says."""
    import asyncio

    types: list[type] = [TimeoutError, asyncio.TimeoutError, ConnectionError]
    for mod, names in (
        ("requests.exceptions", ("Timeout", "ConnectionError")),
        ("httpx", ("TimeoutException", "NetworkError")),
        ("aiohttp", ("ClientConnectionError", "ServerTimeoutError")),
    ):
        try:
            m = __import__(mod, fromlist=list(names))
        except Exception:  # optional dependency
            continue
        types.extend(getattr(m, n) for n in names if isinstance(getattr(m, n, None), type))
    return tuple(types)


_SOURCES_3DEP = [
    DataSource(
        name="USGS 3D Elevation Program (3DEP) 1/3 arc-second DEM",
        url="https://www.usgs.gov/3d-elevation-program",
    ),
    DataSource(
        name="pyflwdir",
        url="https://deltares.github.io/pyflwdir/",
    ),
]

_SOURCES_FAST = [
    DataSource(
        name="MERIT-Hydro (Planetary Computer)",
        url="https://planetarycomputer.microsoft.com/dataset/merit-hydro",
        citation="@dataset{MERITHydro2019}",
    ),
    DataSource(
        name="Copernicus DEM GLO-30",
        url="https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30",
    ),
]

_SOURCES_MERIT_GEE = [
    DataSource(
        name="MERIT Hydro via Google Earth Engine",
        url="https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1",
        citation="@article{Yamazaki2019MERITHydro}",
    ),
    DataSource(
        name="pyflwdir",
        url="https://deltares.github.io/pyflwdir/",
    ),
]

_SOURCES_MERIT = [
    DataSource(
        name="MERIT-Basins",
        url="https://www.reachhydro.org/home/params/merit-basins",
        citation="@article{Lin2019MERITBasins}",
    ),
    DataSource(
        name="MERIT-Basins / MERIT Hydro regional files (mghydro.com, SHA-256 pinned to delineator 2.2.4)",
        url="https://mghydro.com/watersheds/data/",
    ),
    DataSource(
        name="Heberger (2025) hybrid raster and vector delineation",
        url="https://mghydro.com/pages/Heberger_delineation_2025.pdf",
    ),
]


def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    subset = gdf[~gdf.geometry.is_empty]
    if subset.empty:
        raise ToolError(
            code="DELINEATION_FAILED",
            message="Empty geometry after delineation.",
            tool=_TOOL_PATH,
        )
    fc = json.loads(subset.to_json())
    feat = fc["features"][0]
    feat.setdefault("properties", {})
    return feat


def _should_escalate(
    fast_result,
    *,
    expected_area_km2: float | None,
) -> str | None:
    if fast_result.area_km2 < _MIN_AREA_KM2:
        return f"area {fast_result.area_km2:.2f} km² below minimum"
    if fast_result.scout_box_maxed:
        return "scout extent reached maximum (possible truncation)"
    if (
        expected_area_km2
        and expected_area_km2 > 0
        and abs(fast_result.area_km2 - expected_area_km2) / expected_area_km2
        > _AREA_MISMATCH_FRAC
    ):
        return (
            f"area {fast_result.area_km2:.0f} km² differs from expected "
            f"{expected_area_km2:.0f} km² by >{_AREA_MISMATCH_FRAC:.0%}"
        )
    if getattr(fast_result, "used_nldi_basin", False):
        return None
    if (
        fast_result.merit_snap_distance_m is None
        or fast_result.merit_snap_distance_m > _LARGE_SNAP_M
    ):
        if expected_area_km2 and expected_area_km2 > 10:
            return "MERIT vector snap unavailable or outlet far from river network"
        if fast_result.area_km2 >= max(_MIN_AREA_KM2 * 5, 10.0):
            return None
        return "MERIT vector snap unavailable or outlet far from river network"
    return None


def _workflow_steps(
    method_used: str,
    *,
    routing_dataset: str | None = None,
    expected_area_km2: float | None = None,
    escalation_reason: str | None = None,
) -> list[dict[str, Any]]:
    if method_used == _SMALL_CATCHMENT_METHOD:
        return [
            {
                "step": "fetch_3dep_dem",
                "data": "USGS 3DEP 1/3 arc-second DEM, reprojected to EPSG:5070 at 10 m",
                "purpose": "Bare-earth elevation around the pour point.",
            },
            {
                "step": "embankment_notch",
                "data": "DEM cells within 30 m of the pour point",
                "purpose": "Lower them to the minimum within 60 m so a road embankment does not act as a dam.",
            },
            {
                "step": "flow_routing",
                "data": "pyflwdir.from_dem (depression fill + D8)",
                "purpose": "Derive flow directions and upstream area.",
            },
            {
                "step": "outlet_snap",
                "data": "Upstream area within 40 m of the pour point",
                "purpose": "Move the outlet to the largest-drainage cell nearby.",
            },
            {
                "step": "window_expansion",
                "data": "2.5 / 5 / 10 km half-width windows",
                "purpose": "Enlarge the DEM window until the basin no longer touches its edge.",
            },
        ]
    if method_used == "nldi_comid":
        return [
            {
                "step": "nldi_comid_lookup",
                "data": "USGS NLDI / NHDPlus",
                "purpose": "Find upstream basin polygon for the outlet COMID.",
            },
            {
                "step": "area_validation",
                "data": "NLDI basin area",
                "purpose": "Accept basin when area is plausible for the request.",
            },
        ]
    if method_used == "merit_gee_pyflwdir":
        steps = [
            {
                "step": "gee_fetch_merit_hydro",
                "data": routing_dataset or "MERIT/Hydro/v1_0_1",
                "purpose": "Fetch conditioned MERIT flow direction, upstream area, and river-width bands.",
            },
            {
                "step": "outlet_snap",
                "data": "MERIT upa/wth",
                "purpose": "Move the pour point to a nearby MERIT drainage cell.",
            },
            {
                "step": "local_flow_routing",
                "data": "MERIT dir + pyflwdir",
                "purpose": "Trace the upstream basin on a conditioned global flow-direction raster.",
            },
            {
                "step": "polygonize_and_validate",
                "data": "Basin mask, MERIT upa, expected_area_km2",
                "purpose": "Vectorize the watershed and report area/quality checks.",
            },
        ]
        if expected_area_km2:
            steps[1]["purpose"] = "Use expected area as a prior while snapping to a MERIT drainage cell."
        return steps
    if method_used == "local_merit_pyflwdir":
        return [
            {
                "step": "gee_snap_reference",
                "data": "MERIT upa/wth",
                "purpose": "Snap outlet and obtain official MERIT upstream area for validation.",
            },
            {
                "step": "load_regional_flowdir",
                "data": "Cached regional MERIT flowdir",
                "purpose": "Use a staged Pfaf-region flow-direction raster without requiring accumulation.",
            },
            {
                "step": "local_flow_routing",
                "data": "MERIT flowdir + pyflwdir",
                "purpose": "Trace the upstream basin on conditioned local flow direction.",
            },
            {
                "step": "polygonize_and_validate",
                "data": "Basin polygon + official MERIT upa",
                "purpose": "Vectorize the watershed and validate primarily against GEE MERIT upa.",
            },
        ]
    if method_used == "merit_basins_hybrid":
        return [
            {
                "step": "gee_snap_reference",
                "data": "MERIT upa/wth",
                "purpose": "Snap outlet and obtain official MERIT upstream area for validation.",
            },
            {
                "step": "load_merit_basins_topology",
                "data": "Cached regional MERIT-Basins catchments",
                "purpose": "Resolve terminal catchment and traverse upstream unit-catchment topology.",
            },
            {
                "step": "vector_topology_assembly",
                "data": "MERIT-Basins catchment polygons",
                "purpose": "Assemble and dissolve upstream vector catchments for large-basin routing.",
            },
            {
                "step": "terminal_raster_refinement",
                "data": "Cached regional MERIT flowdir",
                "purpose": "Refine only the terminal outlet catchment when raster routing is safe.",
            },
            {
                "step": "polygonize_and_validate",
                "data": "Hybrid polygon + official MERIT upa",
                "purpose": "Validate hybrid area and return overflow provenance.",
            },
        ]
    if method_used == "dem_raw_fallback":
        return [
            {
                "step": "raw_dem_fetch",
                "data": "NASADEM / Copernicus DEM",
                "purpose": "Fetch elevation tiles for fallback routing.",
            },
            {
                "step": "pysheds_conditioning",
                "data": "Raw DEM",
                "purpose": "Fill pits/depressions and derive flow direction locally.",
            },
            {
                "step": "fallback_warning",
                "data": escalation_reason or "RAW_DEM_EXPERIMENTAL_FALLBACK",
                "purpose": "Flag that this is not the production global default.",
            },
        ]
    if method_used in ("merit_basins", "merit_basins_hybrid"):
        return [
            {
                "step": "load_merit_basins",
                "data": "Local MERIT-Basins vectors and flowdir",
                "purpose": "Use the expert local MERIT-Basins/upstream-delineator workflow.",
            }
        ]
    return []


def _lonlat(res: Any) -> tuple[float, float] | None:
    """(lon, lat) of a tier result's snapped pour point, if it reports one."""
    try:
        lo, la = float(res.outlet_lon), float(res.outlet_lat)
        return (lo, la) if -180 <= lo <= 180 and -90 <= la <= 90 else None
    except (AttributeError, TypeError, ValueError):
        return None


def _mint_ref_dict(
    *,
    method_used: str | None,
    geojson: dict[str, Any] | None,
    area_km2: float | None,
    lat: float | None,
    lon: float | None,
    snapped_outlet: tuple[float, float] | None = None,
    snap_grid_crs: str | None = None,
    snap_grid_resolution_m: float | None = None,
    snap_distance_m: float | None = None,
    comid: Any = None,
    terminal_catchment_id: Any = None,
    vector_dataset_version: str | None = None,
    raster_dataset_version: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Mint the canonical BasinRef dict. Returns ``(ref, reason)``.

    ``ref`` is None with a ``reason`` code when no honest identity exists:
    ``no_snapped_pour_point`` (a grid anchor needs the tier's snapped pour
    point and grid; the requested outlet is never used), ``unsupported_method``
    or ``mint_failed:<code>``. Minting never raises and never affects tier
    selection; downstream claim promotion fails closed on ``basin_ref=None``.
    No network I/O happens here.
    """
    try:
        from aihydro_watershed import identity

        if method_used == "nldi_comid" and comid is not None:
            anchor, ok = identity.comid_anchor(comid)
            out_ll = (lon, lat) if lon is not None and lat is not None else None
        elif method_used in ("merit_basins", "merit_basins_hybrid") and terminal_catchment_id not in (None, ""):
            anchor, ok = identity.merit_basins_anchor(terminal_catchment_id, vector_dataset_version)
            out_ll = snapped_outlet or ((lon, lat) if lon is not None and lat is not None else None)
        else:
            if method_used == _SMALL_CATCHMENT_METHOD:
                product, version = "3dep-dem", None
            elif method_used in ("merit_gee_pyflwdir", "local_merit_pyflwdir", "merit_basins_hybrid"):
                product, version = "merit-hydro", raster_dataset_version
            elif method_used == "dem_raw_fallback":
                product, version = "raw-dem", None
            else:
                return None, "unsupported_method"
            if snapped_outlet is None or not snap_grid_crs or not snap_grid_resolution_m:
                return None, "no_snapped_pour_point"
            anchor, ok = identity.grid_cell_anchor(
                product, version, snap_grid_resolution_m, snapped_outlet[0], snapped_outlet[1],
                crs=snap_grid_crs,
            )
            out_ll = snapped_outlet
        ref = identity.mint_basin_ref(
            anchor=anchor,
            version_verified=ok,
            method=str(method_used),
            geometry_geojson=geojson,
            outlet_lon=out_ll[0] if out_ll else None,
            outlet_lat=out_ll[1] if out_ll else None,
            snap_distance_m=snap_distance_m,
            area_km2=area_km2,
            comid=comid if method_used == "nldi_comid" else None,
            merit_catchment=terminal_catchment_id if anchor.network == identity.MERIT_BASINS_NETWORK else None,
        )
        return ref.to_dict(), None
    except Exception as exc:
        log.warning("BasinRef minting failed (%s): %s", method_used, exc)
        return None, f"mint_failed:{getattr(exc, 'code', type(exc).__name__)}"


def _attach_workflow_steps(
    result: HydroResult,
    *,
    method_used: str,
    routing_dataset: str | None = None,
    expected_area_km2: float | None = None,
    escalation_reason: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> HydroResult:
    result.data.setdefault(
        "workflow_steps",
        _workflow_steps(
            method_used,
            routing_dataset=routing_dataset,
            expected_area_km2=expected_area_km2,
            escalation_reason=escalation_reason,
        ),
    )
    result.data.setdefault("routing_dataset", routing_dataset or result.data.get("source"))
    result.data.setdefault("quality_flags", [])
    d = result.data
    if "basin_ref" not in d:
        # Pure side effect after tier selection: must never raise (a raise here
        # is caught by the tier try/except and would silently change routing).
        try:
            sn = (
                (d["snapped_outlet_lon"], d["snapped_outlet_lat"])
                if d.get("snapped_outlet_lon") is not None and d.get("snapped_outlet_lat") is not None
                else None
            )
            d["basin_ref"], d["basin_ref_unavailable"] = _mint_ref_dict(
                method_used=d.get("method_used") or method_used,
                geojson=d.get("geometry_geojson"),
                area_km2=d.get("area_km2"),
                lat=d.get("outlet_lat") if d.get("outlet_lat") is not None else lat,
                lon=d.get("outlet_lon") if d.get("outlet_lon") is not None else lon,
                comid=d.get("comid"),
                snapped_outlet=sn,
                snap_grid_crs=d.get("snap_grid_crs"),
                snap_grid_resolution_m=d.get("snap_grid_resolution_m"),
                terminal_catchment_id=d.get("terminal_catchment_id"),
                vector_dataset_version=d.get("vector_dataset_version"),
                raster_dataset_version=d.get("raster_dataset_version"),
            )
        except Exception as exc:  # pragma: no cover - _mint_ref_dict already guards
            log.warning("BasinRef attach failed: %s", exc)
            d["basin_ref"], d["basin_ref_unavailable"] = None, "mint_failed:attach"
    return result


def _nldi_failure_is_transient(exc: BaseException) -> bool:
    """True when an NLDI failure says "the service could not answer" rather than "NLDI has no data here".

    ``NLDI_UNAVAILABLE`` is transient by construction (429 / 5xx / no response / hung call / fail-fast memo);
    ``NLDI_NOT_FOUND`` is an answer (nothing indexed at the point). Any other exception is transient when its
    text matches a known throttling signature (``_TRANSIENT_NLDI_RE``).
    """
    code = getattr(exc, "code", None)
    if code == "NLDI_UNAVAILABLE":
        return True
    if code == "NLDI_NOT_FOUND":
        return False
    seen: BaseException | None = exc
    for _ in range(4):  # the exception itself, then its cause / context chain
        if seen is None:
            break
        if isinstance(seen, _transient_exception_types()):
            return True
        response = getattr(seen, "response", None)
        status = getattr(seen, "status", None) or getattr(response, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
        seen = seen.__cause__ or seen.__context__
    return bool(_TRANSIENT_NLDI_RE.search(str(exc)))


def _with_trail(result: HydroResult, history: list[dict[str, Any]], method: str, outcome: str, reason: str) -> HydroResult:
    """Append the winning tier's row to ``history`` and carry the whole history on a result that returns early."""
    history.append({"method": method, "outcome": outcome, "reason": reason})
    try:
        result.data["fallback_history"] = list(history)
    except Exception:  # pragma: no cover - data is a dict on every tier
        pass
    return result


def delineate_from_point(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    method: Method = "auto",
    verbose: bool = False,
    name: str | None = None,
    allow_downstream_expansion: bool = False,
) -> HydroResult:
    """
    Delineate a watershed from a pour point (WGS84 decimal degrees).

    Parameters
    ----------
    lat, lon : float
        Outlet coordinates (EPSG:4326).
    expected_area_km2 : float, optional
        Prior drainage area for validation / adaptive snapping. In CONUS, a
        value below 5 km2 sends ``auto`` to the small-catchment tier first.
    method : str
        ``auto`` (NLDI in CONUS; globally the MERIT-Basins hybrid first, then local MERIT
        routing when a regional flowdir is cached, else ``merit_gee`` (which falls back to
        local MERIT after staging), then raw DEM), ``nldi``, ``merit_gee``,
        ``local_merit``, ``merit_hybrid`` (alias ``merit_basins``; MERIT-Basins vector
        assembly with terminal-catchment refinement, also usable in CONUS),
        ``dem_raw_fallback``, or
        ``small_catchment`` (CONUS culvert-scale catchments on USGS 3DEP 10 m
        with an embankment notch; see
        :mod:`aihydro_watershed.delineation.small_catchment`). ``fast`` is an
        alias for ``dem_raw_fallback`` and ``3dep`` for ``small_catchment``.
        Without an expected area, ``auto`` in CONUS trusts NLDI, which returns
        whole NHDPlus catchments; for culverts pass ``expected_area_km2`` or
        ``method="small_catchment"``.
        If NLDI raises in CONUS, ``auto`` falls back to the MERIT-Basins hybrid (then the other MERIT tiers,
        then the 3DEP tier only when its result has no ``BASIN_TOUCHES_MAX_WINDOW`` / ``LIKELY_DITCH_SNAP``
        flag, then raw DEM) and flags the result ``FALLBACK_AFTER_TRANSIENT_NLDI``; an earlier tier's partial
        area that differs from the served area by more than 25% adds ``SOURCE_DISAGREEMENT``.
    allow_downstream_expansion : bool
        NLDI, no expected area, nearest reach draining < 80 km2: by default the
        nearest reach's basin is returned with ``outlet_ambiguity`` (candidate
        downstream COMIDs and areas) and the ``NLDI_OUTLET_AMBIGUOUS`` flag. True
        moves the outlet downstream to the largest nearby basin (explicit consent).
    """
    trail: list[dict[str, Any]] = []
    try:
        return _delineate_from_point(
            lat, lon, expected_area_km2=expected_area_km2, method=method,
            verbose=verbose, name=name, _trail=trail,
            allow_downstream_expansion=allow_downstream_expansion,
        )
    except ToolError as exc:
        # A failed delineation keeps the tiers it tried: the same ``fallback_history`` a success returns,
        # carried in ``details`` so it reaches the error envelope. Never overwrites a history already set.
        if trail and "fallback_history" not in exc.details:
            exc.details = {**exc.details, "fallback_history": list(trail)}
        raise
    except Exception as exc:
        # An unexpected exception keeps the trail too, as an attribute a consumer can read
        # (``exc.fallback_history``); the exception type and message are untouched.
        if trail:
            try:
                exc.fallback_history = list(trail)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - an exception type that refuses attributes
                pass
        raise


def _delineate_from_point(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None,
    method: Method,
    verbose: bool,
    name: str | None,
    _trail: list[dict[str, Any]],
    allow_downstream_expansion: bool = False,
) -> HydroResult:
    """Implementation of :func:`delineate_from_point`; ``_trail`` is the caller-owned ``fallback_history``."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ToolError(
            code="INVALID_COORDS",
            message="lat/lon out of valid WGS84 range.",
            tool=_TOOL_PATH,
        )

    if method == "fast":
        method = "dem_raw_fallback"
    if method == "3dep":
        method = "small_catchment"
    if method == "merit_hybrid":
        method = "merit_basins"

    method_used = method
    escalation_reason: str | None = None
    pfaf_code: str | None = None
    snap_distance_m: float | None = None
    gdf: gpd.GeoDataFrame | None = None
    area: float = 0.0
    routing_dataset: str | None = None
    routing_resolution_m: float | None = None
    snap_quality: str | None = None
    snap_validation: dict[str, Any] | None = None
    snapped_upa_km2: float | None = None
    official_merit_upa_km2: float | None = None
    local_upstream_area_km2: float | None = None
    polygon_area_km2: float | None = None
    area_validation: dict[str, Any] | None = None
    quality_flags: list[str] = []
    cache_key: str | None = None
    license_note: str | None = None
    citation: str | None = None
    routing_data_source: str | None = None
    snap_source: str | None = None
    validation_sources: list[str] = []
    regional_flowdir_cached: bool | None = None
    execution_mode: str | None = None
    regional_flowdir_file_size_bytes: int | None = None
    window_expansion_iterations: int | None = None
    final_window_bounds: dict[str, float] | None = None
    final_window_cell_count: int | None = None
    basin_touched_window_boundary: bool | None = None
    window_complete: bool | None = None
    peak_memory_mb: float | None = None
    runtime_seconds: float | None = None
    memory_telemetry: dict[str, float] | None = None
    fallback_history = _trail
    safe_envelope_version: str | None = None
    snapped_outlet: tuple[float, float] | None = None  # (lon, lat) of the snapped pour point, if the tier reports it
    snap_grid_crs: str | None = None  # CRS / resolution of the grid that point sits on
    snap_grid_resolution_m: float | None = None
    terminal_catchment_id: str | int | None = None
    upstream_catchment_count: int | None = None
    terminal_refinement_used: bool | None = None
    vector_assembly_area_km2: float | None = None
    refined_polygon_area_km2: float | None = None
    vector_dataset_version: str | None = None
    raster_dataset_version: str | None = None
    vector_data_source: str | None = None
    raster_data_source: str | None = None
    data_provenance: dict[str, Any] | None = None
    hybrid_tried = False
    # CONUS auto: set when the primary tier (NLDI) raised, so the served polygon comes from a fallback.
    nldi_failure: str | None = None      # "transient" | "not_usable" (NLDI raised without being throttled / down)
    nldi_failure_is_exception = False
    nldi_reason = "NLDI failure"
    # Partial evidence from tiers that ran but did not serve: [{source, kind, area_km2}].
    evidence: list[dict[str, Any]] = []
    # A 3DEP result the guarded tier refused; served only if every later tier also fails (LAST_RESORT_UNVERIFIED).
    rejected_sc: Any = None
    policy_flags: list[str] = []         # flags the router itself raises; merged at the end (tiers replace quality_flags)

    def _note_nldi_failure(exc: BaseException) -> str:
        """Record an NLDI exception for the fallback policy; returns the escalation reason to carry on the trail."""
        nonlocal nldi_failure, nldi_failure_is_exception, nldi_reason
        transient = _nldi_failure_is_transient(exc)
        nldi_failure = "transient" if transient else "not_usable"
        nldi_failure_is_exception = True
        nldi_reason = f"NLDI {'transient failure' if transient else 'failure'}: {exc}"
        row = {"method": "nldi_comid", "outcome": "failed", "reason": str(exc)}
        if transient:
            row["transient"] = True
        fallback_history.append(row)
        return nldi_reason

    def _try_small_catchment(reason: str | None, *, strict: bool = False, last_resort_sc: Any = None) -> bool:
        """Run the 3DEP small-catchment tier; on success fill the result state.

        ``last_resort_sc`` serves a result ``strict`` already rejected (no re-run), flagged ``LAST_RESORT_UNVERIFIED``.
        """
        nonlocal gdf, area, method_used, routing_dataset, routing_resolution_m
        nonlocal snap_distance_m, polygon_area_km2, area_validation, snapped_outlet
        nonlocal snap_grid_crs, snap_grid_resolution_m
        nonlocal window_expansion_iterations, final_window_bounds
        nonlocal basin_touched_window_boundary, window_complete
        nonlocal license_note, citation, rejected_sc
        try:
            from aihydro_watershed.delineation.small_catchment import (
                ROUTING_DATASET,
                delineate_small_catchment,
            )

            sc = last_resort_sc if last_resort_sc is not None else delineate_small_catchment(lat, lon)
        except Exception as exc:
            log.info("Small-catchment tier failed: %s", exc)
            fallback_history.append(
                {"method": _SMALL_CATCHMENT_METHOD, "outcome": "failed", "reason": str(exc)}
            )
            if method == "small_catchment":
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=f"Small-catchment (3DEP) delineation failed: {exc}",
                    tool=_TOOL_PATH,
                    recovery="Check network access to USGS 3DEP, or nudge the pour point onto the channel.",
                ) from exc
            return False
        if sc.area_km2 < _SMALL_CATCHMENT_MIN_KM2:
            fallback_history.append(
                {
                    "method": _SMALL_CATCHMENT_METHOD,
                    "outcome": "rejected",
                    "reason": f"area {sc.area_km2:.4f} km2 below {_SMALL_CATCHMENT_MIN_KM2} km2",
                }
            )
            return False
        if strict and last_resort_sc is None:
            bad = [f for f in _SMALL_CATCHMENT_REJECT_FLAGS if f in sc.quality_flags]
            if bad:
                fallback_history.append(
                    {
                        "method": _SMALL_CATCHMENT_METHOD,
                        "outcome": "rejected",
                        "reason": (
                            f"area {sc.area_km2:.3f} km2 carries {', '.join(bad)}; not served as a stand-in "
                            "for an unavailable NLDI"
                        ),
                    }
                )
                evidence.append(
                    {"source": _SMALL_CATCHMENT_METHOD, "kind": "rejected_polygon_area", "area_km2": sc.area_km2}
                )
                rejected_sc = sc
                return False
        gdf = sc.gdf
        area = sc.area_km2
        polygon_area_km2 = sc.area_km2
        method_used = _SMALL_CATCHMENT_METHOD
        routing_dataset = ROUTING_DATASET
        routing_resolution_m = sc.resolution_m
        snap_distance_m = sc.snap_distance_m
        snapped_outlet = (sc.outlet_lon, sc.outlet_lat)
        snap_grid_crs, snap_grid_resolution_m = "EPSG:5070", float(sc.resolution_m)
        window_expansion_iterations = sc.window_iterations
        if sc.window_bounds_5070:
            w = sc.window_bounds_5070
            final_window_bounds = {
                "crs": "EPSG:5070", "minx": w[0], "miny": w[1], "maxx": w[2], "maxy": w[3],
            }
        basin_touched_window_boundary = sc.status == "touches_max_window"
        window_complete = not basin_touched_window_boundary
        license_note = "USGS 3DEP elevation data: public domain"
        citation = "U.S. Geological Survey, 3D Elevation Program 1/3 arc-second DEM"
        for f in [*sc.quality_flags, *(["LAST_RESORT_UNVERIFIED"] if last_resort_sc is not None else [])]:
            if f not in quality_flags:
                quality_flags.append(f)
        if expected_area_km2 and expected_area_km2 > 0:
            ratio = sc.area_km2 / expected_area_km2
            area_validation = {
                "expected_area_km2": expected_area_km2,
                "ratio_to_expected": round(ratio, 3),
                "within_factor_2": 0.5 <= ratio <= 2.0,
            }
            if not area_validation["within_factor_2"]:
                quality_flags.append("AREA_OUTSIDE_FACTOR_2_OF_EXPECTED")
        fallback_history.append(
            {
                "method": _SMALL_CATCHMENT_METHOD,
                # "succeeded_unverified" reads as served in the tools attempt trail (outcome startswith "succeeded").
                "outcome": "succeeded_unverified" if last_resort_sc is not None else "succeeded",
                "reason": reason or "requested",
                "outlet_lat": sc.outlet_lat,
                "outlet_lon": sc.outlet_lon,
                "notch_elevation_m": sc.notch_elevation_m,
            }
        )
        return True

    if method == "small_catchment":
        if not is_conus(lat, lon):
            raise ToolError(
                code="DELINEATION_FAILED",
                message="The small-catchment tier uses USGS 3DEP and covers CONUS only.",
                tool=_TOOL_PATH,
                recovery="Outside CONUS use method='merit_gee' or method='auto'.",
            )
        _try_small_catchment("requested")
        if gdf is None:
            raise ToolError(
                code="DELINEATION_FAILED",
                message="Small-catchment (3DEP) delineation produced no usable basin.",
                tool=_TOOL_PATH,
                recovery="Nudge the pour point onto the channel at the culvert inlet or outlet.",
            )
    elif (
        method == "auto"
        and is_conus(lat, lon)
        and expected_area_km2
        and 0 < expected_area_km2 < _SMALL_CATCHMENT_MAX_KM2
    ):
        _try_small_catchment(
            f"expected area {expected_area_km2:g} km2 below {_SMALL_CATCHMENT_MAX_KM2:g} km2"
        )

    run_nldi = method in ("auto", "nldi") and gdf is None
    run_merit_gee = method in ("auto", "merit_gee")
    run_local_merit = method == "local_merit"
    run_fast = method == "dem_raw_fallback"
    run_merit = method in ("auto", "merit_basins")

    # CONUS auto with known drainage area: NLDI COMID search vs expected_area_km2.
    if run_nldi and is_conus(lat, lon) and expected_area_km2 and expected_area_km2 > 0:
        try:
            from aihydro_watershed.delineation.nldi_point import delineate_nldi_at_point, nldi_deadline

            with nldi_deadline():
                nldi_result = delineate_nldi_at_point(
                    lat, lon, expected_area_km2=expected_area_km2
                )
            nldi_area = nldi_result.data.get("area_km2", 0)
            nldi_ok = nldi_area >= _MIN_AREA_KM2
            if expected_area_km2 and expected_area_km2 > 0:
                nldi_ok = nldi_ok and (
                    abs(nldi_area - expected_area_km2) / expected_area_km2
                    <= _AREA_MISMATCH_FRAC
                )
            if nldi_ok:
                nldi_result = _with_trail(nldi_result, fallback_history, "nldi_comid", "served", "")
                return _attach_workflow_steps(
                    nldi_result,
                    method_used="nldi_comid",
                    routing_dataset="USGS NLDI / NHDPlus",
                    expected_area_km2=expected_area_km2,
                    lat=lat,
                    lon=lon,
                )
            escalation_reason = (
                f"NLDI COMID area {nldi_area:.0f} km² "
                f"vs expected {expected_area_km2:.0f} km²"
                if expected_area_km2
                else f"NLDI COMID area {nldi_area:.0f} km² too small"
            )
            log.info("NLDI COMID tier skipped: %s", escalation_reason)
            fallback_history.append(
                {"method": "nldi_comid", "outcome": "skipped", "reason": escalation_reason}
            )
            if method == "auto":
                nldi_failure = "not_usable"
                evidence.append({"source": "nldi_comid", "kind": "upstream_area", "area_km2": float(nldi_area)})
        except Exception as e:
            log.info("NLDI COMID tier unavailable: %s", e)
            escalation_reason = _note_nldi_failure(e)
            if method == "nldi":
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=str(e),
                    tool=_TOOL_PATH,
                    recovery="NLDI is CONUS-only and network-dependent; try method='merit_gee'.",
                ) from e

    # CONUS auto without expected area: fast NLDI (mainstem COMID), then cloud DEM if out of range.
    if (
        run_nldi
        and is_conus(lat, lon)
        and not (expected_area_km2 and expected_area_km2 > 0)
    ):
        try:
            from aihydro_watershed.delineation.nldi_point import delineate_nldi_at_point, nldi_deadline

            with nldi_deadline():
                nldi_result = delineate_nldi_at_point(
                    lat, lon, allow_downstream_expansion=allow_downstream_expansion
                )
            nldi_area = float(nldi_result.data.get("area_km2", 0))
            # The ambiguity flag rides on the NLDI row so the attempt trail shows the outlet was not moved.
            nldi_flag = "NLDI_OUTLET_AMBIGUOUS" if "NLDI_OUTLET_AMBIGUOUS" in (
                nldi_result.data.get("quality_flags") or []
            ) else ""
            if _NLDI_QUICK_MIN_KM2 <= nldi_area <= _NLDI_QUICK_MAX_KM2:
                nldi_result = _with_trail(nldi_result, fallback_history, "nldi_comid", "served", nldi_flag)
                return _attach_workflow_steps(
                    nldi_result,
                    method_used="nldi_comid",
                    routing_dataset="USGS NLDI / NHDPlus",
                    expected_area_km2=expected_area_km2,
                    lat=lat,
                    lon=lon,
                )
            escalation_reason = (
                f"NLDI quick area {nldi_area:.0f} km² outside "
                f"{_NLDI_QUICK_MIN_KM2:.0f}–{_NLDI_QUICK_MAX_KM2:.0f} km²"
                + (f" ({nldi_flag})" if nldi_flag else "")
            )
            log.info("NLDI quick tier skipped: %s", escalation_reason)
            fallback_history.append(
                {"method": "nldi_comid", "outcome": "skipped", "reason": escalation_reason}
            )
            if method == "auto":
                evidence.append({"source": "nldi_comid", "kind": "upstream_area", "area_km2": nldi_area})
                # NLDI answered with a sub-km2 reach: that is the culvert-scale case the 3DEP tier is for, and
                # serving it there is by design, not a fallback. Only a tier that does not serve it is "not usable".
                if nldi_area >= _NLDI_QUICK_MIN_KM2 or not _try_small_catchment(escalation_reason):
                    nldi_failure = "not_usable"
        except Exception as e:
            log.info("NLDI quick tier unavailable: %s", e)
            escalation_reason = _note_nldi_failure(e)
            if method == "nldi":
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=str(e),
                    tool=_TOOL_PATH,
                    recovery="NLDI is CONUS-only and network-dependent; try method='merit_gee'.",
                ) from e
            # No small-catchment hand-off here: the 3DEP tier is not known to be right for a basin of unknown
            # size (CAMELS: 19 of 27 such hand-offs were the wrong basin). ``hybrid_first`` below tries the
            # MERIT-Basins hybrid, and the 3DEP tier remains only as a guarded last resort.

    if method == "nldi" and gdf is None:
        raise ToolError(
            code="DELINEATION_FAILED",
            message="NLDI could not return a valid watershed for this point.",
            tool=_TOOL_PATH,
            recovery="For global or non-NLDI points, call method='merit_gee' or method='auto'.",
        )

    snap_reference = None
    local_cache_status: dict[str, Any] | None = None
    # The default global tier snaps and refines itself (staged rasters first), so the Earth Engine
    # snap reference is only prepared up front when a different tier runs first.
    hybrid_first = gdf is None and (
        method == "merit_basins"
        or (method == "auto" and not is_conus(lat, lon))
        or (method == "auto" and nldi_failure_is_exception)
    )

    def _prepare_merit_context() -> None:
        nonlocal snap_reference, local_cache_status, pfaf_code, regional_flowdir_cached
        try:
            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                merit_get_snap_reference,
            )

            snap_reference = merit_get_snap_reference(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
                # with a known area: the narrow 1 km / +/-10% gate on the small window, not the wide scorer
                area_gate=bool(expected_area_km2 and expected_area_km2 > 0),
            )
        except Exception as e:
            log.info("MERIT GEE snap-reference unavailable: %s", e)
            if method == "merit_gee":
                # Full GEE delineation can still initialize/fail with a more specific error below.
                pass

        try:
            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                merit_check_routing_region_cache,
                merit_resolve_pfaf_region,
            )

            pfaf_code = merit_resolve_pfaf_region(lat, lon)
            local_cache_status = merit_check_routing_region_cache(pfaf_code)
            regional_flowdir_cached = bool(local_cache_status.get("flowdir_ready"))
        except Exception as e:
            log.info("MERIT regional cache status unavailable: %s", e)

    if (method in ("auto", "local_merit", "merit_gee", "merit_basins")) and gdf is None and not hybrid_first:
        _prepare_merit_context()

    run_cached_local_merit = (
        (method in ("auto", "local_merit"))
        and gdf is None
        and bool(local_cache_status and local_cache_status.get("flowdir_ready"))
    )

    def _apply_merit_result(local: Any, *, used_method: str) -> None:
        nonlocal gdf, area, snap_distance_m, snap_quality, snap_validation
        nonlocal snapped_upa_km2, official_merit_upa_km2, local_upstream_area_km2
        nonlocal polygon_area_km2, area_validation, quality_flags, cache_key
        nonlocal pfaf_code, routing_dataset, routing_resolution_m, routing_data_source
        nonlocal snap_source, validation_sources, regional_flowdir_cached
        nonlocal execution_mode, regional_flowdir_file_size_bytes
        nonlocal window_expansion_iterations, final_window_bounds, final_window_cell_count
        nonlocal basin_touched_window_boundary, window_complete, peak_memory_mb, runtime_seconds
        nonlocal memory_telemetry
        nonlocal method_used, snapped_outlet, snap_grid_crs, snap_grid_resolution_m
        nonlocal safe_envelope_version, terminal_catchment_id, upstream_catchment_count
        nonlocal terminal_refinement_used, vector_assembly_area_km2, refined_polygon_area_km2
        nonlocal vector_dataset_version, raster_dataset_version
        nonlocal vector_data_source, raster_data_source, data_provenance

        gdf = local.gdf
        area = local.area_km2
        snap_distance_m = local.snap_distance_m
        snapped_outlet = _lonlat(local)
        snap_grid_crs, snap_grid_resolution_m = "EPSG:4326", local.routing_resolution_m or 90.0
        snap_quality = local.snap_quality
        snap_validation = local.snap_validation
        snapped_upa_km2 = local.snapped_upa_km2
        official_merit_upa_km2 = local.official_merit_upa_km2
        local_upstream_area_km2 = local.local_upstream_area_km2
        polygon_area_km2 = local.polygon_area_km2
        area_validation = local.area_validation
        quality_flags = local.quality_flags
        cache_key = local.cache_key
        pfaf_code = local.pfaf_region or pfaf_code
        routing_dataset = local.routing_dataset
        routing_resolution_m = local.routing_resolution_m
        routing_data_source = local.routing_data_source
        snap_source = local.snap_source
        validation_sources = local.validation_sources or []
        regional_flowdir_cached = local.regional_flowdir_cached
        execution_mode = local.execution_mode
        regional_flowdir_file_size_bytes = local.regional_flowdir_file_size_bytes
        window_expansion_iterations = local.window_expansion_iterations
        final_window_bounds = local.final_window_bounds
        final_window_cell_count = local.final_window_cell_count
        basin_touched_window_boundary = local.basin_touched_window_boundary
        window_complete = local.window_complete
        peak_memory_mb = local.peak_memory_mb
        runtime_seconds = local.runtime_seconds
        memory_telemetry = local.memory_telemetry
        safe_envelope_version = local.safe_envelope_version
        terminal_catchment_id = local.terminal_catchment_id
        upstream_catchment_count = local.upstream_catchment_count
        terminal_refinement_used = local.terminal_refinement_used
        vector_assembly_area_km2 = local.vector_assembly_area_km2
        refined_polygon_area_km2 = local.refined_polygon_area_km2
        vector_dataset_version = local.vector_dataset_version
        raster_dataset_version = local.raster_dataset_version
        vector_data_source = local.vector_data_source
        raster_data_source = local.raster_data_source
        data_provenance = local.data_provenance
        method_used = used_method

    def _try_hybrid_result(reason: str) -> bool:
        nonlocal license_note, citation, run_fast, quality_flags, hybrid_tried
        if hybrid_tried:  # one attempt per call; the failed row is already in fallback_history
            return False
        hybrid_tried = True
        try:
            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                MERIT_BASINS_LICENSE,
                MERIT_CITATION,
                merit_basins_hybrid_delineate,
            )

            hybrid = merit_basins_hybrid_delineate(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
                snap_reference=snap_reference,
                allow_offline=True,
            )
            _apply_merit_result(hybrid, used_method="merit_basins_hybrid")
            license_note = MERIT_BASINS_LICENSE
            citation = MERIT_CITATION
            # The hybrid's own steps (refinement, local-window fallback, outlet assembly, area-gated snap), so a
            # failed refinement keeps its reason in the record. Namespaced so they are not read as served tiers;
            # a step's "succeeded" is written "completed" for the same reason.
            for step in (hybrid.data_provenance or {}).get("hybrid_fallback_history") or []:
                outcome = str(step.get("outcome", ""))
                fallback_history.append(
                    {
                        "method": f"merit_basins_hybrid/{step.get('method', '')}",
                        "outcome": "completed" if outcome == "succeeded" else outcome,
                        "reason": str(step.get("reason", "")),
                    }
                )
            fallback_history.append(
                {
                    "method": "merit_basins_hybrid",
                    "outcome": "succeeded",
                    "reason": reason,
                }
            )
            run_fast = False
            return True
        except Exception as hybrid_exc:
            if (
                "MERIT_BASINS_NOT_STAGED" in str(hybrid_exc)
                and "MERIT_BASINS_NOT_STAGED" not in quality_flags
            ):
                quality_flags.append("MERIT_BASINS_NOT_STAGED")
            if (
                "HYBRID_ROUTING_REQUIRED" in str(hybrid_exc)
                and "HYBRID_ROUTING_REQUIRED" not in quality_flags
            ):
                quality_flags.append("HYBRID_ROUTING_REQUIRED")
            if (
                "MERIT_ASSET_INTEGRITY_FAILED" in str(hybrid_exc)
                and "MERIT_ASSET_INTEGRITY_FAILED" not in quality_flags
            ):
                quality_flags.append("MERIT_ASSET_INTEGRITY_FAILED")
            fallback_history.append(
                {
                    "method": "merit_basins_hybrid",
                    "outcome": "failed",
                    "reason": str(hybrid_exc),
                }
            )
            return False

    # Default global tier (and forced method="merit_hybrid"/"merit_basins", also in CONUS): MERIT-Basins
    # vector assembly with terminal-catchment refinement. If it cannot serve, the older tiers follow.
    if hybrid_first and not _try_hybrid_result(escalation_reason or "default_global_tier"):
        if method == "auto":
            _prepare_merit_context()
            run_cached_local_merit = bool(
                gdf is None and local_cache_status and local_cache_status.get("flowdir_ready")
            )

    if run_cached_local_merit or (run_local_merit and gdf is None):
        try:
            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                MERIT_CITATION,
                MERIT_GEE_DATASET,
                MERIT_LICENSE,
                local_merit_flowdir_pyflwdir,
            )

            local = local_merit_flowdir_pyflwdir(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
                snap_reference=snap_reference,
                allow_offline=True,
            )
            _apply_merit_result(local, used_method="local_merit_pyflwdir")
            license_note = MERIT_LICENSE
            citation = MERIT_CITATION
            fallback_history.append(
                {"method": "local_merit_flowdir_pyflwdir", "outcome": "succeeded"}
            )
        except Exception as e:
            if method == "auto" and nldi_failure_is_exception:
                # NLDI already failed, so this chain must reach merit_gee / 3DEP / raw DEM: record and fall through.
                fallback_history.append(
                    {"method": "local_merit_flowdir_pyflwdir", "outcome": "failed", "reason": str(e)}
                )
                if "HYBRID_ROUTING_REQUIRED" in str(e):
                    _try_hybrid_result(str(e))   # no-op when the hybrid already ran (one attempt per call)
            elif "HYBRID_ROUTING_REQUIRED" in str(e) and method in ("auto", "merit_basins"):
                if _try_hybrid_result(str(e)):
                    pass
                else:
                    raise ToolError(
                        code="HYBRID_ROUTING_REQUIRED",
                        message=str(e),
                        tool=_TOOL_PATH,
                        recovery="Stage MERIT-Basins vectors for this Pfaf region and retry method='merit_basins'.",
                    ) from e
            elif "LOCAL_MEMORY_THRESHOLD_EXCEEDED" in str(e):
                raise ToolError(
                    code="LARGE_BASIN_HYBRID_REQUIRED",
                    message="Cached regional flowdir exceeds the raster-only local processing envelope.",
                    tool=_TOOL_PATH,
                    recovery="Use the future MERIT-Basins hybrid vector traversal tier for this basin.",
                ) from e
            elif "ADAPTIVE_WINDOW_LIMIT_REACHED" in str(e) or "LARGE_BASIN_HYBRID_REQUIRED" in str(e):
                raise ToolError(
                    code="LARGE_BASIN_HYBRID_REQUIRED",
                    message=str(e),
                    tool=_TOOL_PATH,
                    recovery="Use the future MERIT-Basins hybrid vector traversal tier for this basin.",
                ) from e
            elif gdf is None:
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=str(e),
                    tool=_TOOL_PATH,
                    recovery="Install local MERIT flowdir raster or use method='merit_gee'.",
                ) from e

    if run_merit_gee and gdf is None:
        try:
            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                MERIT_CITATION,
                MERIT_GEE_DATASET,
                MERIT_LICENSE,
                delineate_merit_gee,
            )

            merit_gee = delineate_merit_gee(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
            )
            gdf = merit_gee.gdf
            area = merit_gee.area_km2
            snap_distance_m = merit_gee.snap_distance_m
            snapped_outlet = _lonlat(merit_gee)
            snap_grid_crs, snap_grid_resolution_m = "EPSG:4326", merit_gee.routing_resolution_m or 90.0
            snap_quality = merit_gee.snap_quality
            snap_validation = merit_gee.snap_validation
            snapped_upa_km2 = merit_gee.snapped_upa_km2
            official_merit_upa_km2 = merit_gee.official_merit_upa_km2
            local_upstream_area_km2 = merit_gee.local_upstream_area_km2
            polygon_area_km2 = merit_gee.polygon_area_km2
            area_validation = merit_gee.area_validation
            quality_flags = merit_gee.quality_flags
            cache_key = merit_gee.cache_key
            routing_dataset = MERIT_GEE_DATASET
            routing_resolution_m = merit_gee.routing_resolution_m
            routing_data_source = merit_gee.routing_data_source
            snap_source = merit_gee.snap_source
            validation_sources = merit_gee.validation_sources or []
            regional_flowdir_cached = merit_gee.regional_flowdir_cached
            if local_cache_status and not local_cache_status.get("flowdir_ready"):
                quality_flags.append("REGIONAL_FLOWDIR_NOT_CACHED")
            license_note = MERIT_LICENSE
            citation = MERIT_CITATION
            method_used = "merit_gee_pyflwdir"

            # Quality gate on the GEE result. delineate_merit_gee can return a
            # degenerate basin WITHOUT raising — e.g. for a very large river it
            # may snap onto a tiny tributary/coastal cell and yield an area
            # below the minimum (flagged VERY_SMALL_BASIN). With no exception,
            # escalation_reason/run_fast are never set, so the run would
            # dead-end at the final "no valid polygon" error with no fallback.
            # In auto mode, reject it and fall through to the MERIT-Basins
            # hybrid (vector topology — assembles the full upstream network)
            # and ultimately raw DEM. Mirrors _should_escalate for the fast
            # tier. Pinned method="merit_gee" keeps the strict raise downstream.
            gee_degenerate = (
                gdf is None
                or getattr(gdf, "empty", False)
                or (area is not None and area < _MIN_AREA_KM2)
            )
            if gee_degenerate and method == "auto":
                log.info(
                    "MERIT GEE result degenerate (area=%.3f km2); escalating "
                    "to MERIT-Basins hybrid.",
                    area or 0.0,
                )
                fallback_history.append(
                    {
                        "method": "merit_gee_pyflwdir",
                        "outcome": "rejected",
                        "reason": (
                            f"area {area or 0.0:.3f} km2 below "
                            f"{_MIN_AREA_KM2} km2 (likely snap failure)"
                        ),
                    }
                )
                escalation_reason = escalation_reason or "merit_gee_degenerate_result"
                if official_merit_upa_km2:
                    evidence.append(
                        {"source": "merit_gee_pyflwdir", "kind": "merit_uparea", "area_km2": float(official_merit_upa_km2)}
                    )
                gdf = None
                method_used = None
                run_fast = True
        except Exception as e:
            log.warning("MERIT GEE tier failed: %s", e)
            escalation_reason = str(e)
            if not regional_flowdir_cached:
                quality_flags.append("REGIONAL_FLOWDIR_STAGING_REQUIRED")
            reason = (
                "MERIT_GEE_WINDOW_LIMIT"
                if "MERIT_GEE_WINDOW_LIMIT" in str(e)
                else "GEE_MEMORY_LIMIT"
                if re.search(r"memory limit|User memory limit|HTTP 400", str(e), re.I)
                else str(e)
            )
            fallback_history.append(
                {"method": "merit_gee_pyflwdir", "outcome": "failed", "reason": reason}
            )
            if local_cache_status and local_cache_status.get("flowdir_ready"):
                try:
                    from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                        MERIT_CITATION,
                        MERIT_LICENSE,
                        local_merit_flowdir_pyflwdir,
                    )

                    local = local_merit_flowdir_pyflwdir(
                        lat,
                        lon,
                        expected_area_km2=expected_area_km2,
                        snap_reference=snap_reference,
                        allow_offline=True,
                    )
                    _apply_merit_result(local, used_method="local_merit_pyflwdir")
                    license_note = MERIT_LICENSE
                    citation = MERIT_CITATION
                    fallback_history.append(
                        {"method": "local_merit_flowdir_pyflwdir", "outcome": "succeeded"}
                    )
                    run_fast = False
                except Exception as local_exc:
                    fallback_history.append(
                        {
                            "method": "local_merit_flowdir_pyflwdir",
                            "outcome": "failed",
                            "reason": str(local_exc),
                        }
                    )
                    if "HYBRID_ROUTING_REQUIRED" in str(local_exc):
                        run_fast = not _try_hybrid_result(str(local_exc))
                    else:
                        run_fast = True
            else:
                # Design B: auto-stage MERIT flowdir on first encounter,
                # then retry local_merit_pyflwdir (tiered policy — never
                # blocks on >2 GB without user confirmation).
                #
                # Guard: only auto-stage for non-CONUS points.  CONUS has
                # adequate NLDI + raw-DEM fallbacks; staging Rhine/Ganges/etc.
                # on a CONUS failure would waste bandwidth and surprise users.
                run_fast = True  # default, overridden on successful stage
                if pfaf_code and not is_conus(lat, lon):
                    try:
                        from aihydro_watershed.merit.merit_manager import MeritDataManager

                        mgr = MeritDataManager()
                        stage_status = mgr.auto_stage_flowdir(
                            pfaf_code,
                            progress_cb=lambda msg, done, tot: log.info(
                                "MERIT staging [Pfaf %s] %s", pfaf_code, msg
                            ),
                        )
                        if stage_status.flowdir_ready:
                            from aihydro_watershed.delineation.merit_flowdir_pipeline import (
                                MERIT_CITATION,
                                MERIT_LICENSE,
                                local_merit_flowdir_pyflwdir,
                            )

                            local2 = local_merit_flowdir_pyflwdir(
                                lat,
                                lon,
                                expected_area_km2=expected_area_km2,
                                snap_reference=snap_reference,
                                allow_offline=True,
                            )
                            _apply_merit_result(local2, used_method="local_merit_pyflwdir")
                            license_note = MERIT_LICENSE
                            citation = MERIT_CITATION
                            fallback_history.append(
                                {
                                    "method": "local_merit_flowdir_pyflwdir",
                                    "outcome": "succeeded_after_auto_stage",
                                    "pfaf": pfaf_code,
                                }
                            )
                            run_fast = False
                        elif hasattr(stage_status, "action_required"):
                            # Large region — surface info for agent to confirm
                            quality_flags.append("MERIT_STAGE_CONFIRM_REQUIRED")
                            fallback_history.append(
                                {
                                    "method": "merit_auto_stage",
                                    "outcome": "confirm_required",
                                    "message": stage_status.message,
                                    "size_gb": getattr(stage_status, "size_gb", None),
                                    "next_tool": "merit_ensure_routing_region",
                                }
                            )
                        else:
                            fallback_history.append(
                                {
                                    "method": "merit_auto_stage",
                                    "outcome": "failed",
                                    "reason": stage_status.message,
                                }
                            )
                    except Exception as _stage_exc:
                        log.warning(
                            "MERIT auto-staging failed for Pfaf %s: %s — "
                            "continuing to raw DEM fallback.",
                            pfaf_code, _stage_exc,
                        )
                        fallback_history.append(
                            {
                                "method": "merit_auto_stage",
                                "outcome": "exception",
                                "reason": str(_stage_exc),
                            }
                        )

            if method == "merit_gee" and gdf is None:
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=str(e),
                    tool=_TOOL_PATH,
                    recovery=(
                        "Run delineation_doctor() to check GEE auth and pyflwdir, "
                        "or use method='dem_raw_fallback' for experimental DEM routing."
                    ),
                ) from e

    # Non-CONUS accurate fallback: prefer the MERIT-Basins hybrid (vector
    # topology, ~MERIT accuracy) over raw DEM when GEE + local_merit couldn't
    # serve. This restores the documented tier order
    # (… → merit_basins → dem_raw_fallback) for the no-expected-area case, where
    # escalation-from-fast never fires (it requires expected_area_km2). In auto mode
    # CONUS stays on NLDI + raw DEM; the hybrid there is opt-in (method="merit_hybrid").
    # _try_hybrid_result sets gdf + method_used + run_fast=False on success; on
    # failure (e.g. basins not staged → MERIT_BASINS_NOT_STAGED) run_fast stays
    # True so raw DEM remains the final fallback.
    if method == "auto" and gdf is None and run_fast and not is_conus(lat, lon):
        _try_hybrid_result(escalation_reason or "merit_gee_unavailable")

    # After a failed NLDI the MERIT tiers above come first; the guarded 3DEP tier is the next stand-in, ahead of
    # the experimental raw-DEM fallback. It is served only if its own result shows no known failure signature.
    if (
        method == "auto"
        and gdf is None
        and nldi_failure_is_exception
        and is_conus(lat, lon)
        and _try_small_catchment(f"last resort after {nldi_reason}", strict=True)
    ):
        run_fast = False

    if run_fast:
        try:
            from aihydro_watershed.delineation.pysheds_pipeline import delineate_fast

            fast = delineate_fast(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
                verbose=verbose,
            )
            pfaf_code = fast.pfaf_code
            snap_distance_m = fast.merit_snap_distance_m
            if fast.snapped_outlet_lon is not None and fast.snapped_outlet_lat is not None:
                snapped_outlet = (fast.snapped_outlet_lon, fast.snapped_outlet_lat)
                snap_grid_crs = fast.snap_grid_crs
                snap_grid_resolution_m = fast.snap_grid_resolution_m
            else:  # e.g. NLDI indexed basin: no DEM snap, so no grid identity
                snapped_outlet, snap_grid_crs, snap_grid_resolution_m = None, None, None
            escalation_reason = _should_escalate(
                fast, expected_area_km2=expected_area_km2
            )
            if method == "dem_raw_fallback" or not escalation_reason:
                gdf = fast.gdf
                area = fast.area_km2
                method_used = "dem_raw_fallback"
                routing_dataset = "raw DEM (NASADEM/Copernicus)"
                routing_resolution_m = 30.0
                quality_flags.append("RAW_DEM_EXPERIMENTAL_FALLBACK")
                fallback_history.append(
                    {"method": "dem_raw_fallback", "outcome": "served", "reason": "requested" if method == "dem_raw_fallback" else "no better tier served"}
                )
            elif method == "auto":
                log.info("Escalating to merit_basins: %s", escalation_reason)
                fallback_history.append(
                    {"method": "dem_raw_fallback", "outcome": "rejected", "reason": str(escalation_reason)}
                )
                run_merit = True
        except Exception as e:
            log.warning("Fast tier failed: %s", e)
            fallback_history.append({"method": "dem_raw_fallback", "outcome": "failed", "reason": str(e)})
            if method == "dem_raw_fallback":
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=str(e),
                    tool=_TOOL_PATH,
                ) from e
            run_merit = True
            escalation_reason = str(e)

    if run_merit and gdf is None:
        if (
            not _try_hybrid_result(escalation_reason or "requested_merit_basins")
            and not (
                method == "auto"
                and rejected_sc is not None
                and _try_small_catchment(
                    f"last resort after {nldi_reason}; every other tier failed", last_resort_sc=rejected_sc
                )
            )
        ):
            # The local MERIT-Basins hybrid (merit_basins_hybrid_delineate) IS the
            # accurate tier. If it can't run (e.g. basin vectors/flowdir not staged
            # locally), there is no further MERIT fallback to attempt — surface a
            # clear, actionable error. (A former second attempt via the optional
            # `upstream_delineator` GitHub adapter was removed: it depended on the
            # SAME local staging the hybrid requires, only stricter, so it could
            # never succeed where the hybrid failed — it only ever produced a
            # duplicate failure.)
            hybrid_reason = next(
                (
                    h.get("reason")
                    for h in reversed(fallback_history)
                    if h.get("method") == "merit_basins_hybrid"
                    and h.get("outcome") == "failed"
                ),
                "MERIT-Basins hybrid unavailable",
            )
            if method == "merit_basins":
                raise ToolError(
                    code="DELINEATION_FAILED",
                    message=hybrid_reason,
                    tool=_TOOL_PATH,
                    recovery="Run merit_ensure_basin to stage MERIT-Basins data, or nudge the outlet onto the main channel.",
                )
            raise ToolError(
                code="DELINEATION_FAILED",
                message=(
                    f"All tiers failed. Fast: {escalation_reason or 'skipped'}. "
                    f"MERIT-Basins: {hybrid_reason}"
                ),
                tool=_TOOL_PATH,
                recovery="Run merit_ensure_basin or nudge the outlet onto the main channel.",
            )

    min_area_km2 = (
        _SMALL_CATCHMENT_MIN_KM2 if method_used == _SMALL_CATCHMENT_METHOD else _MIN_AREA_KM2
    )
    if gdf is None or gdf.empty or area < min_area_km2:
        raise ToolError(
            code="DELINEATION_FAILED",
            message="No valid watershed polygon produced. Try moving the outlet slightly.",
            tool=_TOOL_PATH,
        )

    # Router-level honesty flags (merged after the tiers, which own and may replace ``quality_flags``).
    if method == "auto" and nldi_failure and method_used != "nldi_comid":
        policy_flags.append(
            "FALLBACK_AFTER_TRANSIENT_NLDI" if nldi_failure == "transient" else "FALLBACK_AFTER_NLDI_NOT_USABLE"
        )
    source_disagreement: list[dict[str, Any]] = []
    for ev in evidence:
        ev_area = ev.get("area_km2")
        if not ev_area or ev_area <= 0 or not area or ev["source"] == method_used:
            continue
        rel = abs(ev_area - area) / area
        if rel > _SOURCE_DISAGREEMENT_FRAC:
            source_disagreement.append(
                {
                    **ev,
                    "area_km2": round(float(ev_area), 3),
                    "served_method": method_used,
                    "served_area_km2": round(float(area), 3),
                    "relative_difference": round(rel, 3),
                    "threshold": _SOURCE_DISAGREEMENT_FRAC,
                }
            )
    if source_disagreement:
        policy_flags.append("SOURCE_DISAGREEMENT")
    for f in policy_flags:
        if f not in quality_flags:
            quality_flags.append(f)

    geojson = _gdf_to_geojson(gdf)
    # Surface the headline delineation stats on the Feature itself so the map's
    # click popup (FeatureIdentifier reads feature.properties) shows area/method
    # etc. — same UX as the MERIT-river popups. Keep the 6 most useful first;
    # the popup caps at 6 and rolls the rest into "+N more properties". Skip
    # None/empty values so the popup stays clean.
    _feature_props = {
        "area_km2": (round(area, 1) if area >= 10 else round(area, 3)) if area is not None else None,
        "method": method_used,
        "pfaf": pfaf_code,
        "routing": routing_dataset,
        "snap_quality": snap_quality,
        "snap_distance_m": (
            round(snap_distance_m, 1) if snap_distance_m is not None else None
        ),
        "outlet": f"{lat:.5f}, {lon:.5f}",
        "name": name,
    }
    # Put curated stats FIRST so they win the popup's 6-property cap, then keep
    # any geometry-derived columns after them.
    _existing_props = geojson.get("properties") or {}
    geojson["properties"] = {
        **{k: v for k, v in _feature_props.items() if v not in (None, "")},
        **_existing_props,
    }
    if method_used in ("merit_gee_pyflwdir", "local_merit_pyflwdir"):
        sources = _SOURCES_MERIT_GEE
    elif method_used in ("merit_basins", "merit_basins_hybrid"):
        sources = _SOURCES_MERIT
    elif method_used == _SMALL_CATCHMENT_METHOD:
        sources = _SOURCES_3DEP
    else:
        sources = _SOURCES_FAST

    import aihydro_watershed

    basin_ref, basin_ref_unavailable = _mint_ref_dict(
        method_used=method_used,
        geojson=geojson,
        area_km2=round(area, 3),
        lat=lat,
        lon=lon,
        snapped_outlet=snapped_outlet,
        snap_grid_crs=snap_grid_crs,
        snap_grid_resolution_m=snap_grid_resolution_m,
        snap_distance_m=snap_distance_m,
        terminal_catchment_id=terminal_catchment_id,
        vector_dataset_version=vector_dataset_version,
        raster_dataset_version=raster_dataset_version,
    )

    meta = HydroMeta(
        tool=_TOOL_PATH,
        version=getattr(aihydro_watershed, "__version__", "unknown"),
        params={
            "lat": lat,
            "lon": lon,
            "method": method,
            "method_used": method_used,
            "expected_area_km2": expected_area_km2,
            "name": name,
            "routing_dataset": routing_dataset,
        },
        sources=sources,
    )

    return HydroResult(
        data={
            "geometry_geojson": geojson,
            "area_km2": round(area, 3),
            "polygon_area_km2": round(polygon_area_km2 if polygon_area_km2 is not None else area, 3),
            "outlet_lat": lat,
            "outlet_lon": lon,
            "method_used": method_used,
            "pfaf_code": pfaf_code,
            "pfaf_region": pfaf_code,
            "snap_distance_m": snap_distance_m,
            "snapped_upa_km2": snapped_upa_km2,
            "official_merit_upa_km2": official_merit_upa_km2,
            "local_upstream_area_km2": local_upstream_area_km2,
            "snap_quality": snap_quality,
            "snap_source": snap_source,
            "snap_validation": snap_validation,
            "routing_dataset": routing_dataset,
            "routing_data_source": routing_data_source,
            "vector_dataset_version": vector_dataset_version,
            "raster_dataset_version": raster_dataset_version,
            "vector_data_source": vector_data_source,
            "raster_data_source": raster_data_source,
            "data_provenance": data_provenance,
            "routing_resolution_m": routing_resolution_m,
            "area_validation": area_validation,
            "validation_sources": validation_sources,
            "regional_flowdir_cached": regional_flowdir_cached,
            "execution_mode": execution_mode,
            "regional_flowdir_file_size_bytes": regional_flowdir_file_size_bytes,
            "window_expansion_iterations": window_expansion_iterations,
            "final_window_bounds": final_window_bounds,
            "final_window_cell_count": final_window_cell_count,
            "basin_touched_window_boundary": basin_touched_window_boundary,
            "window_complete": window_complete,
            "peak_memory_mb": peak_memory_mb,
            "runtime_seconds": runtime_seconds,
            "memory_telemetry": memory_telemetry,
            "safe_envelope_version": safe_envelope_version,
            "terminal_catchment_id": terminal_catchment_id,
            "upstream_catchment_count": upstream_catchment_count,
            "terminal_refinement_used": terminal_refinement_used,
            "vector_assembly_area_km2": (
                round(vector_assembly_area_km2, 3)
                if vector_assembly_area_km2 is not None
                else None
            ),
            "refined_polygon_area_km2": (
                round(refined_polygon_area_km2, 3)
                if refined_polygon_area_km2 is not None
                else None
            ),
            "fallback_history": fallback_history,
            "source_disagreement": source_disagreement or None,
            "quality_flags": quality_flags,
            "cache_key": cache_key,
            "license": license_note,
            "citation": citation,
            "workflow_steps": _workflow_steps(
                method_used,
                routing_dataset=routing_dataset,
                expected_area_km2=expected_area_km2,
                escalation_reason=escalation_reason,
            ),
            "escalation_reason": escalation_reason,
            "name": name or "watershed",
            "basin_ref": basin_ref,
            "basin_ref_unavailable": basin_ref_unavailable,
            "snapped_outlet_lon": snapped_outlet[0] if snapped_outlet else None,
            "snapped_outlet_lat": snapped_outlet[1] if snapped_outlet else None,
        },
        meta=meta,
    )
