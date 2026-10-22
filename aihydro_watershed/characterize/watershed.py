"""
Watershed Delineation Tool
===========================

Automated watershed boundary extraction for USGS gauge sites.

Returns a standardized HydroResult with GeoJSON geometry and full
FAIR provenance metadata.

Functions
---------
delineate_watershed(gauge_id, save_geojson, output_dir) -> HydroResult

Examples
--------
>>> from aihydro_watershed.characterize.watershed import delineate_watershed
>>> result = delineate_watershed('01031500')
>>> print(result.data['area_km2'])      # float
>>> print(result.data['gauge_name'])    # str
>>> print(result.meta.cite())           # BibTeX
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Optional

log = logging.getLogger(__name__)

__all__ = ["delineate_watershed", "validate_gauge_id"]

# ---------------------------------------------------------------------------
# Data source declarations (used for FAIR provenance)
# ---------------------------------------------------------------------------
from aihydro_core import DataSource, HydroMeta, HydroResult, ToolError

_SOURCES = [
    DataSource(
        name="USGS NLDI",
        url="https://api.water.usgs.gov/nldi/",
        citation=(
            "@misc{NLDI2024,\n"
            "  title={Network-Linked Data Index (NLDI)},\n"
            "  author={{USGS Water Resources}},\n"
            "  year={2024},\n"
            "  url={https://api.water.usgs.gov/nldi/}\n"
            "}"
        ),
    ),
    DataSource(
        name="USGS NWIS",
        url="https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations",
        citation=(
            "@misc{NWIS2024,\n"
            "  title={National Water Information System (NWIS), served by the USGS Water Data APIs},\n"
            "  author={{USGS Water Resources Mission Area}},\n"
            "  year={2024},\n"
            "  url={https://waterdata.usgs.gov/nwis}\n"
            "}"
        ),
    ),
]

_TOOL_PATH = "aihydro_watershed.characterize.watershed.delineate_watershed"

try:
    from pynhd import NLDI
    import geopandas as gpd
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

try:
    from aihydro_data.sources import usgs_waterdata as _uw
    _UW_OK = True
except ImportError:
    _uw = None
    _UW_OK = False


def _get_nldi_basin_for_gauge(nldi: NLDI, gauge_id: str) -> gpd.GeoDataFrame:
    """Fetch NLDI basin by NWIS site id, with COMID fallback when API shape fails."""
    return _get_nldi_basin_for_gauge_ex(nldi, gauge_id)[0]


def _gauge_location(gauge_id: str):
    """Monitoring location of a gauge from the USGS Water Data API (via aihydro-data).

    Replaces the legacy pygeohydro site-info lookup. A site the API does not list is a terminal
    ``GAUGE_NOT_FOUND``; an unreachable / throttled API is ``NETWORK_ERROR`` (the lookup did
    not happen, which is not the same claim).
    """
    from aihydro_data.exceptions import AihydroDataError, TerminalSourceError

    try:
        return _uw.get_monitoring_locations(ids=[gauge_id])[0]
    except TerminalSourceError as e:
        raise ToolError(
            code="GAUGE_NOT_FOUND",
            message=f"Gauge {gauge_id} not found in the USGS Water Data API ({getattr(e, 'code', '')}).",
            tool=_TOOL_PATH,
            recovery="Verify gauge ID at https://waterdata.usgs.gov/nwis",
        ) from e
    except AihydroDataError as e:
        raise ToolError(
            code="NETWORK_ERROR",
            message=f"USGS monitoring-location lookup for {gauge_id} failed: {e}",
            tool=_TOOL_PATH,
            recovery="Retry later; the gauge may exist but could not be looked up.",
        ) from e


def _get_nldi_basin_for_gauge_ex(nldi: NLDI, gauge_id: str):
    """As :func:`_get_nldi_basin_for_gauge`, also reporting which path was used.

    Returns ``(gdf, path, comid)``: ``path`` is ``"gauge_index"`` (NLDI
    ``get_basins(<site>)``) or ``"comid_fallback"`` (basin fetched by the COMID
    at the NWIS gauge coordinates, a different spatial support); ``comid`` is
    set only for the fallback. Identity depends on this (ADR-003).
    """
    from aihydro_watershed.delineation.nldi_point import _normalize_nldi_basins

    try:
        gdf = nldi.get_basins(gauge_id)
        return _normalize_nldi_basins(gdf), "gauge_index", None
    except Exception as e:
        first_error = e
        log.warning(
            "NLDI get_basins(%s) failed (%s); trying COMID at gauge coordinates",
            gauge_id,
            e,
        )

    try:
        loc = _gauge_location(gauge_id)
    except ToolError as e:
        raise e from first_error
    lat, lon = loc.lat, loc.lon
    comid = int(nldi.comid_byloc((lon, lat)).comid.iloc[0])
    gdf = nldi.get_basins(comid, fsource="comid")
    return _normalize_nldi_basins(gdf), "comid_fallback", comid


def _mint_gauge_ref(gauge_id, path, comid, geometry_geojson, area_km2, lon, lat):
    """BasinRef dict for a gauge delineation, or None if minting fails.

    ``gauge_index`` anchors to ``usgs:<site>``; the COMID fallback anchors to
    the COMID (a different support) with the usgs alias ``fallback_of`` and
    flag ``gauge_basin_comid_fallback``. Minting never breaks delineation; a
    ``None`` ref makes claim promotion fail closed downstream.
    """
    try:
        from aihydro_watershed import identity

        if path == "comid_fallback":
            anchor, ok = identity.comid_anchor(comid)
            kw = {"usgs_relation": "fallback_of", "comid": comid,
                  "extra_flags": ("gauge_basin_comid_fallback",)}
        else:
            anchor, ok = identity.gauge_anchor(gauge_id)
            kw = {}
        ref = identity.mint_basin_ref(
            anchor=anchor, version_verified=ok, method=f"nldi_{path}",
            geometry_geojson=geometry_geojson, outlet_lon=lon, outlet_lat=lat,
            area_km2=area_km2, usgs_site=gauge_id, **kw,
        )
        return ref.to_dict()
    except Exception as exc:
        log.warning("BasinRef minting failed for gauge %s: %s", gauge_id, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delineate_watershed(
    gauge_id: str,
    save_geojson: bool = False,
    output_dir: Optional[str] = None,
) -> HydroResult:
    """
    Delineate watershed boundary for a USGS gauge using NLDI.

    Parameters
    ----------
    gauge_id : str
        8-digit USGS gauge identifier (e.g., '01031500')
    save_geojson : bool, optional
        Save watershed boundary as GeoJSON file (default: False)
    output_dir : str, optional
        Directory to save GeoJSON if save_geojson=True

    Returns
    -------
    HydroResult
        result.data keys:
        - geometry_geojson : dict  — GeoJSON polygon (WGS84 / EPSG:4326)
        - area_km2         : float — Watershed drainage area in km²
        - gauge_id         : str   — USGS gauge identifier
        - gauge_name       : str   — Official USGS station name
        - gauge_lat        : float — Gauge latitude (°N)
        - gauge_lon        : float — Gauge longitude (°E)
        - huc_02           : str   — 2-digit hydrologic unit code

    Raises
    ------
    ToolError
        GAUGE_NOT_FOUND   — gauge_id not in NLDI or NWIS
        INVALID_GAUGE_ID  — bad format
        NETWORK_ERROR     — API unreachable
        DEPENDENCY_ERROR  — pynhd or aihydro-data not installed

    Examples
    --------
    >>> result = delineate_watershed('01031500')
    >>> result.data['area_km2']
    847.3
    >>> geojson = result.data['geometry_geojson']  # pass to other tools
    >>> result.meta.cite()  # BibTeX for NLDI + NWIS
    """
    _check_deps()

    if not validate_gauge_id(gauge_id):
        raise ToolError(
            code="INVALID_GAUGE_ID",
            message=f"Invalid gauge_id: '{gauge_id}'. Must be 8+ digit string (e.g., '01031500').",
            tool=_TOOL_PATH,
            recovery="Check gauge ID at https://waterdata.usgs.gov/nwis",
        )

    log.info("Delineating watershed for gauge %s", gauge_id)

    try:
        # ── Step 1: Watershed boundary from NLDI ─────────────────────────
        nldi = NLDI()
        watershed_gdf, basin_path, fallback_comid = _get_nldi_basin_for_gauge_ex(nldi, gauge_id)

        if watershed_gdf.empty:
            raise ToolError(
                code="GAUGE_NOT_FOUND",
                message=f"NLDI returned no basin for gauge {gauge_id}.",
                tool=_TOOL_PATH,
                recovery="Verify gauge ID or use delineate_watershed_from_point at gauge coordinates.",
            )

        if watershed_gdf.geometry.name is None and "geometry" in watershed_gdf.columns:
            watershed_gdf = watershed_gdf.set_geometry("geometry")
        elif watershed_gdf._geometry_column_name not in watershed_gdf.columns:
            geom_cols = [
                c for c in watershed_gdf.columns if watershed_gdf[c].dtype.name == "geometry"
            ]
            if geom_cols:
                watershed_gdf = watershed_gdf.set_geometry(geom_cols[0])

        if watershed_gdf.crs is None:
            watershed_gdf = watershed_gdf.set_crs("EPSG:4326")
        elif watershed_gdf.crs.to_epsg() != 4326:
            watershed_gdf = watershed_gdf.to_crs("EPSG:4326")

        watershed_geom = watershed_gdf.geometry.iloc[0]
        area_km2 = float(
            watershed_gdf.to_crs("EPSG:5070").geometry.area.iloc[0] / 1e6
        )
        log.info("Watershed area: %.1f km²", area_km2)

        # ── Step 2: Gauge metadata from the USGS Water Data API ─────────────────────────────
        loc = _gauge_location(gauge_id)
        gauge_lat = float(loc.lat)
        gauge_lon = float(loc.lon)
        gauge_name = str(loc.name or "")
        huc_full = str(loc.hydrologic_unit_code or "")
        huc_02 = huc_full[:2] if len(huc_full) >= 2 else "NA"

        # ── Step 3: Geometry → GeoJSON dict (JSON-serializable) ──────────
        # Reset index or convert to object to avoid "Unable to avoid copy" error in 
        # geopandas.to_json() with NumPy 2.x + Arrow-backed indices.
        watershed_gdf.index = watershed_gdf.index.astype(object)
        geometry_geojson = json.loads(watershed_gdf.to_json())["features"][0]["geometry"]

        # ── Step 4: Optional file export ──────────────────────────────────
        if save_geojson and output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"watershed_{gauge_id}.geojson")
            with open(out_path, "w") as f:
                json.dump(geometry_geojson, f)
            log.info("Saved GeoJSON: %s", out_path)

        basin_ref = _mint_gauge_ref(
            gauge_id, basin_path, fallback_comid, geometry_geojson, area_km2, gauge_lon, gauge_lat
        )

        return HydroResult(
            data={
                "geometry_geojson": geometry_geojson,
                "area_km2": area_km2,
                "gauge_id": gauge_id,
                "gauge_name": gauge_name,
                "gauge_lat": gauge_lat,
                "gauge_lon": gauge_lon,
                "huc_02": huc_02,
                # Gauge position as the Water Data API serves it. The API states no CRS for the
                # geometry; original_horizontal_datum is the source datum (NAD83 for most gauges),
                # recorded here, not asserted as the geometry's datum.
                "gauge_hydrologic_unit_code": huc_full or None,
                "gauge_horizontal_datum": loc.horizontal_datum,
                "gauge_coordinate_note": loc.geometry_crs_note,
                "gauge_site_id": loc.site_id,
                "delineation_path": basin_path,
                "comid": fallback_comid,
                "basin_ref": basin_ref,
            },
            meta=HydroMeta(
                tool=_TOOL_PATH,
                version=_get_version(),
                gauge_id=gauge_id,
                sources=_SOURCES,
                params={
                    **_version_params(),
                    "gauge_id": gauge_id,
                    "save_geojson": save_geojson,
                    "output_dir": output_dir,
                },
            ),
        )

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(
            code="NETWORK_ERROR" if "connection" in str(e).lower() else "COMPUTATION_ERROR",
            message=f"Failed to delineate watershed for gauge {gauge_id}: {e}",
            tool=_TOOL_PATH,
            recovery="Check network connection and verify gauge ID exists in USGS system.",
        ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_gauge_id(gauge_id: str) -> bool:
    """Return True if gauge_id is valid 8+ digit format."""
    return isinstance(gauge_id, str) and len(gauge_id) >= 8 and gauge_id.isdigit()


def _check_deps() -> None:
    if not _UW_OK:
        raise ToolError(
            code="DEPENDENCY_ERROR",
            message="The USGS Water Data adapter is unavailable: this needs aihydro-data>=0.2.3.",
            tool=_TOOL_PATH,
            recovery="pip install -U 'aihydro-data>=0.2.3'",
        )
    if not _DEPS_OK:
        raise ToolError(
            code="DEPENDENCY_ERROR",
            message="Required dependencies not installed.",
            tool=_TOOL_PATH,
            recovery="pip install 'ai-hydro[watershed]'",
        )


from aihydro_watershed._version import tool_version as _get_version, version_params as _version_params  # noqa: E402
