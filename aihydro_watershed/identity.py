"""
Canonical place identity minting (ADR-003 and its Amendment, slice-3 packet P3).

aihydro-watershed is the **only minter** of :class:`aihydro_core.records.place.BasinRef`.
The spec (types, ``aihydro.geom/1``) lives in core; this module decides which
network element each delineation method anchors to.

Anchor choice per method (``BasinRef.id`` depends on the anchor alone):

==============================  ==============  ==========================  ==========================
method                          anchor kind     network / version           element
==============================  ==============  ==========================  ==========================
NLDI ``get_basins(<site>)``     gauge_index     ``nhdplusv2`` / unversioned ``usgs:<site>``
NLDI COMID (point or fallback)  network_element ``nhdplusv2`` / unversioned ``<comid>``
MERIT-Basins hybrid             network_element ``merit-basins`` / vector   ``<terminal catchment id>``
                                                dataset version
MERIT-Hydro pyflwdir, raw DEM,  grid_cell       DEM product / version       ``<crs>|<step>|<ix>|<iy>``
3DEP small catchment                                                        (quantised snapped cell)
==============================  ==============  ==========================  ==========================

Different methods are different bases for a claim, so they give different ids.
Cross-method sameness is asserted only by a shared ``usgs:`` alias or an
explicit :func:`compare_realisations` record; this module never merges ids.
``network_version`` is ``"unversioned"`` with flag ``network_version_unverified``
whenever the product version is not reported by the tier.

Geoconnex: ``https://geoconnex.us/usgs/monitoring-location/<site>`` is
constructible and added with ``verified=False``. A gage PID
(``.../ref/gages/<n>``) is an opaque integer and is attached only from
:func:`resolve_geoconnex_gage` or its local cache, never constructed. Minting
itself performs no network I/O.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from aihydro_core.records.place import (
    UNVERSIONED,
    BasinAnchor,
    BasinRef,
    OutletRef,
    PlaceAlias,
    PlaceIdentityError,
    SnapRef,
    crosses_antimeridian,
    geometry_id,
)

log = logging.getLogger(__name__)

__all__ = [
    "NHDPLUS_NETWORK",
    "MERIT_BASINS_NETWORK",
    "GEOCONNEX_GAGES_URL",
    "GeoconnexLookup",
    "gauge_anchor",
    "comid_anchor",
    "merit_basins_anchor",
    "grid_cell_anchor",
    "mint_outlet_ref",
    "mint_basin_ref",
    "compare_realisations",
    "resolve_geoconnex_gage",
    "cached_geoconnex_gage",
    "valid_usgs_site",
]

NHDPLUS_NETWORK = "nhdplusv2"
MERIT_BASINS_NETWORK = "merit-basins"
GEOCONNEX_GAGES_URL = "https://reference.geoconnex.us/collections/gages/items"
_GEOCONNEX_ML = "https://geoconnex.us/usgs/monitoring-location/"
_GEOCONNEX_GAGE_RE = re.compile(r"^https://geoconnex\.us/ref/gages/\d+$")
_USGS_SITE_RE = re.compile(r"^\d{8,15}$")  # O3: one rule; 7-digit strings stay labels

FLAG_VERSION_UNVERIFIED = "network_version_unverified"
FLAG_GEOCONNEX_UNRESOLVED = "geoconnex_unresolved"
FLAG_ANTIMERIDIAN = "antimeridian_crossing"


def valid_usgs_site(site: Any) -> bool:
    return isinstance(site, str) and bool(_USGS_SITE_RE.match(site))


# ------------------------------------------------------------------ anchors
def _version(v: Optional[str]) -> Tuple[str, bool]:
    """(version, verified). Unknown versions become ``unversioned``."""
    if isinstance(v, str) and v.strip() and v.strip().lower() != UNVERSIONED:
        return v.strip(), True
    return UNVERSIONED, False


def gauge_anchor(site: str, network_version: Optional[str] = None) -> Tuple[BasinAnchor, bool]:
    """NLDI gauge-index anchor: element ``usgs:<site>`` on NHDPlusV2."""
    if not valid_usgs_site(site):
        raise PlaceIdentityError("INVALID_USGS_SITE", f"{site!r} is not 8-15 digits")
    ver, ok = _version(network_version)
    return BasinAnchor("gauge_index", NHDPLUS_NETWORK, ver, f"usgs:{site}"), ok


def comid_anchor(comid: Any, network_version: Optional[str] = None) -> Tuple[BasinAnchor, bool]:
    """NLDI COMID anchor: the NHDPlusV2 flowline whose catchment basin was fetched."""
    try:
        c = str(int(comid))
    except (TypeError, ValueError):
        raise PlaceIdentityError("INVALID_COMID", f"{comid!r} is not an integer") from None
    ver, ok = _version(network_version)
    return BasinAnchor("network_element", NHDPLUS_NETWORK, ver, c), ok


def merit_basins_anchor(terminal_catchment_id: Any, vector_version: Optional[str] = None) -> Tuple[BasinAnchor, bool]:
    """MERIT-Basins terminal unit-catchment anchor."""
    if terminal_catchment_id is None or str(terminal_catchment_id).strip() == "":
        raise PlaceIdentityError("MISSING_TERMINAL_CATCHMENT", "no MERIT terminal catchment id")
    ver, ok = _version(vector_version)
    return BasinAnchor("network_element", MERIT_BASINS_NETWORK, ver, str(terminal_catchment_id).strip()), ok


def grid_cell_anchor(
    product: str,
    version: Optional[str],
    resolution_m: float,
    lon: float,
    lat: float,
    *,
    crs: str = "EPSG:4326",
) -> Tuple[BasinAnchor, bool]:
    """Quantised snapped-cell anchor on a named DEM / flow-direction product.

    The outlet is reduced to integer cell indices, so any point inside the same
    cell of the same product gives the same id. ``EPSG:4326`` grids use a step
    of ``round(resolution_m / 30.87)`` arc-seconds (30 m -> 1", 90 m -> 3")
    aligned to (-180, -90); a projected ``crs`` uses ``resolution_m`` metres
    aligned to its origin. Cell indices come from ``floor``; a coordinate lying
    exactly on a cell edge is subject to float rounding (documented limit).
    """
    if not isinstance(product, str) or not product:
        raise PlaceIdentityError("INVALID_GRID_PRODUCT", "product must be a non-empty string")
    if not (isinstance(resolution_m, (int, float)) and math.isfinite(resolution_m) and resolution_m > 0):
        raise PlaceIdentityError("INVALID_RESOLUTION", f"{resolution_m!r}")
    if crs.upper() == "EPSG:4326":
        arcsec = max(1, int(round(resolution_m / 30.87)))
        step = arcsec / 3600.0
        ix = math.floor((lon + 180.0) / step + 1e-9)
        iy = math.floor((lat + 90.0) / step + 1e-9)
        step_label = f"{arcsec}as"
    else:
        from pyproj import Transformer

        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        ix = math.floor(x / resolution_m)
        iy = math.floor(y / resolution_m)
        step_label = f"{resolution_m:g}m"
    ver, ok = _version(version)
    return BasinAnchor("grid_cell", product, ver, f"{crs.upper()}|{step_label}|{ix}|{iy}"), ok


# ------------------------------------------------------------------ geoconnex
@dataclass(frozen=True)
class GeoconnexLookup:
    """Result of a geoconnex gage lookup. ``uri`` is set only for resolved/cached."""

    site: str
    status: str  # "resolved" | "cached" | "unresolved" | "not_found"
    uri: Optional[str] = None


def _home() -> Path:
    return Path(os.environ.get("AIHYDRO_HOME") or (Path.home() / ".aihydro"))


def _cache_path() -> Path:
    return _home() / "cache" / "place" / "geoconnex.json"


def _read_cache() -> Dict[str, Any]:
    try:
        data = json.loads(_cache_path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(cache: Dict[str, Any]) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, sort_keys=True, indent=1))
        os.replace(tmp, p)
    except OSError as exc:  # cache is an optimisation, never fatal
        log.debug("geoconnex cache write failed: %s", exc)


def cached_geoconnex_gage(site: str) -> Optional[str]:
    """Gage PID URI from the local cache only. No network. ``None`` if absent."""
    entry = _read_cache().get(site)
    uri = entry.get("uri") if isinstance(entry, dict) else None
    return uri if isinstance(uri, str) and _GEOCONNEX_GAGE_RE.match(uri) else None


def _extract_gage_uri(payload: Mapping[str, Any], site: str) -> Optional[str]:
    for feat in payload.get("features") or []:
        props = (feat or {}).get("properties") or {}
        uri = props.get("uri") or props.get("id") or (feat or {}).get("id")
        if isinstance(uri, str) and _GEOCONNEX_GAGE_RE.match(uri):
            if str(props.get("provider_id") or "") == f"USGS-{site}":
                return uri
    return None


def resolve_geoconnex_gage(
    site: str,
    timeout: float = 5.0,
    *,
    _get: Optional[Callable[..., Any]] = None,
) -> GeoconnexLookup:
    """Look up the Geoconnex gage PID for a USGS site (one GET, cached on success).

    Order: local cache, then one request to
    ``https://reference.geoconnex.us/collections/gages/items?provider_id=USGS-<site>&f=json``.
    Network failure returns ``status="unresolved"`` (caller flags
    ``geoconnex_unresolved``); an empty answer returns ``"not_found"``. The PID
    is only ever taken from the service response.
    """
    if not valid_usgs_site(site):
        return GeoconnexLookup(site, "unresolved")
    hit = cached_geoconnex_gage(site)
    if hit:
        return GeoconnexLookup(site, "cached", hit)
    try:
        if _get is None:
            import requests

            _get = requests.get
        resp = _get(GEOCONNEX_GAGES_URL, params={"provider_id": f"USGS-{site}", "f": "json"}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.info("geoconnex gage lookup for %s unresolved: %s", site, exc)
        return GeoconnexLookup(site, "unresolved")
    uri = _extract_gage_uri(payload if isinstance(payload, Mapping) else {}, site)
    if uri is None:
        return GeoconnexLookup(site, "not_found")
    cache = _read_cache()
    cache[site] = {"uri": uri}
    _write_cache(cache)
    return GeoconnexLookup(site, "resolved", uri)


# --------------------------------------------------------------------- refs
def mint_outlet_ref(
    lon: float,
    lat: float,
    *,
    snap: Optional[Tuple[str, str, str, Optional[float]]] = None,
    aliases: Sequence[PlaceAlias] = (),
) -> OutletRef:
    """Build an :class:`OutletRef`. ``snap`` = (network, version, element, distance_m)."""
    s = SnapRef(snap[0], snap[1], snap[2], snap[3]) if snap else None
    return OutletRef(float(lon), float(lat), snap=s, aliases=tuple(aliases))


def _geometry_of(geojson: Mapping[str, Any]) -> Mapping[str, Any]:
    if geojson.get("type") == "Feature":
        return geojson.get("geometry") or {}
    return geojson


def _site_aliases(site: str, source: str, lookup: Optional[GeoconnexLookup]) -> Tuple[List[PlaceAlias], List[str]]:
    aliases = [
        PlaceAlias("usgs", site, "same_as", source="nwis", verified=False),
        PlaceAlias("geoconnex", _GEOCONNEX_ML + site, "same_as", source="constructed", verified=False),
    ]
    flags: List[str] = []
    uri = lookup.uri if lookup and lookup.uri else cached_geoconnex_gage(site)
    if uri:
        aliases.append(PlaceAlias("geoconnex", uri, "same_as", source="geoconnex-reference", verified=True))
    else:
        flags.append(FLAG_GEOCONNEX_UNRESOLVED)
    return aliases, flags


def mint_basin_ref(
    *,
    anchor: BasinAnchor,
    version_verified: bool,
    method: str,
    geometry_geojson: Mapping[str, Any],
    outlet_lon: Optional[float] = None,
    outlet_lat: Optional[float] = None,
    snap_distance_m: Optional[float] = None,
    area_km2: Optional[float] = None,
    usgs_site: Optional[str] = None,
    usgs_relation: str = "same_as",
    comid: Any = None,
    merit_catchment: Any = None,
    resolve_geoconnex: bool = False,
    geoconnex_lookup: Optional[GeoconnexLookup] = None,
    extra_flags: Sequence[str] = (),
    tool_version: Optional[str] = None,
) -> BasinRef:
    """Mint a :class:`BasinRef` for one delineation. No network unless ``resolve_geoconnex``.

    ``anchor`` is built with one of :func:`gauge_anchor`, :func:`comid_anchor`,
    :func:`merit_basins_anchor` or :func:`grid_cell_anchor`; the id depends on it
    alone. ``usgs_relation="fallback_of"`` marks a COMID fallback for a gauge.
    """
    geom = _geometry_of(geometry_geojson)
    gdigest = geometry_id(geom)  # raises PlaceIdentityError on degenerate input
    flags: List[str] = []
    if not version_verified:
        flags.append(FLAG_VERSION_UNVERIFIED)
    if crosses_antimeridian(geom):
        flags.append(FLAG_ANTIMERIDIAN)
    flags.extend(f for f in extra_flags if f not in flags)

    aliases: List[PlaceAlias] = []
    outlet_aliases: List[PlaceAlias] = []
    if usgs_site is not None:
        if valid_usgs_site(usgs_site):
            lookup = geoconnex_lookup or (resolve_geoconnex_gage(usgs_site) if resolve_geoconnex else None)
            site_aliases, site_flags = _site_aliases(usgs_site, "nwis", lookup)
            if usgs_relation != "same_as":
                site_aliases[0] = PlaceAlias("usgs", usgs_site, usgs_relation, source="nwis", verified=False)
            aliases.extend(site_aliases)
            outlet_aliases.extend(site_aliases)
            flags.extend(f for f in site_flags if f not in flags)
        else:
            flags.append("usgs_site_invalid")
    if comid is not None:
        aliases.append(PlaceAlias("comid", str(int(comid)), "derived_from", source="nldi", verified=True))
    if merit_catchment is not None:
        aliases.append(PlaceAlias("merit", str(merit_catchment), "derived_from", source="merit-basins", verified=True))

    outlet = None
    if outlet_lon is not None and outlet_lat is not None:
        snap = (anchor.network, anchor.network_version, anchor.element, snap_distance_m) \
            if anchor.kind == "network_element" else None
        outlet = mint_outlet_ref(outlet_lon, outlet_lat, snap=snap, aliases=outlet_aliases)

    if tool_version is None:
        try:
            import aihydro_watershed

            tool_version = getattr(aihydro_watershed, "__version__", "unknown")
        except Exception:  # pragma: no cover
            tool_version = "unknown"
    return BasinRef(
        anchor=anchor,
        method=method,
        outlet=outlet,
        geometry_digest=gdigest,
        area_km2=None if area_km2 is None else float(area_km2),
        aliases=tuple(aliases),
        minted_by={"tool": "aihydro_watershed.identity.mint_basin_ref", "version": tool_version},
        quality_flags=tuple(flags),
    )


# ------------------------------------------------------------------ compare
def _shape(g: Any):
    from shapely.geometry import shape

    if hasattr(g, "geom_type"):
        return g
    return shape(_geometry_of(g))


def compare_realisations(
    a: Any,
    b: Any,
    outlet_a: Optional[Tuple[float, float]] = None,
    outlet_b: Optional[Tuple[float, float]] = None,
) -> Dict[str, Optional[float]]:
    """Compare two geometry realisations. Never merges or equates ids.

    ``a``/``b``: GeoJSON geometry or Feature (EPSG:4326) or shapely geometry.
    ``outlet_*``: ``(lon, lat)``. Areas use EPSG:6933 (equal area).
    Returns ``{area_ratio (a/b), iou, outlet_distance_m}``; outlet distance is
    ``None`` unless both outlets are given.
    """
    import shapely.ops
    from pyproj import Geod, Transformer

    ga, gb = _shape(a), _shape(b)
    to_ea = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
    pa, pb = shapely.ops.transform(to_ea, ga.buffer(0)), shapely.ops.transform(to_ea, gb.buffer(0))
    area_a, area_b = pa.area, pb.area
    union = pa.union(pb).area
    out: Dict[str, Optional[float]] = {
        "area_ratio": (area_a / area_b) if area_b > 0 else None,
        "iou": (pa.intersection(pb).area / union) if union > 0 else None,
        "outlet_distance_m": None,
    }
    if outlet_a is not None and outlet_b is not None:
        _, _, dist = Geod(ellps="WGS84").inv(outlet_a[0], outlet_a[1], outlet_b[0], outlet_b[1])
        out["outlet_distance_m"] = float(dist)
    return out
