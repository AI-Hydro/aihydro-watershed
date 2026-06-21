"""DEM fetch for the fast-tier delineation pipeline.

Uses odc-stac to load DEM tiles in the caller's projected UTM CRS — a
requirement of pysheds, which needs a raster in consistent metric units.
``stackstac`` (used by aihydro-data's generic STAC backend) works in
EPSG:4326 and would require an extra reprojection step; ``odc.stac.load``
handles the CRS directly, so we keep it here for the delineation path.

Robustness model (mirrors aihydro-data/sources/stac.py):
  - Exponential-backoff retry on transient network errors.
  - Provider fallback chain: Planetary Computer → Element84 Earth Search.
    Both host the same Copernicus DEM GLO-30 COGs on independent infra.
  - Local file-system cache keyed on bbox + CRS + resolution + collection.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import rioxarray  # noqa: F401 — activates .rio on xarray
import xarray as xr
from odc.stac import load as stac_load
from pystac_client import Client
from shapely.geometry import Polygon

from aihydro_watershed.delineation.utils import meters_to_degrees_bbox

log = logging.getLogger(__name__)

# ── DEM provider chain ────────────────────────────────────────────────────────
# Each entry: (stac_url, collection, asset_key, needs_pc_sign)
# Tried in order; first success wins.
_DEM_PROVIDERS: list[tuple[str, str, str, bool]] = [
    (
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        "cop-dem-glo-30",
        "data",
        True,   # needs planetary_computer token signing
    ),
    (
        "https://earth-search.aws.element84.com/v1",
        "cop-dem-glo-30",
        "data",
        False,  # public AWS S3 COGs, no signing required
    ),
]

# Legacy alias used by callers that set collection= explicitly.
COLLECTION_ALIASES: dict[str, tuple[str, str]] = {
    "nasadem":     ("nasadem",       "elevation"),
    "merit-hydro": ("merit-hydro",   "elevtn"),
    "copernicus":  ("cop-dem-glo-30","data"),
}

# Default collection when none is specified (Copernicus GLO-30).
DEM_COLLECTION = "copernicus"
DEM_ASSET_KEY  = "data"

_CACHE_DIR = Path.home() / ".aihydro" / "cache" / "dem"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Errors that indicate a transient infrastructure failure (retry / fallback).
_RETRYABLE_PATTERNS = (
    "timeout", "time", "connection", "network", "exceeded",
    "502", "503", "504", "gateway", "service unavailable",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


class StacItemCache:
    def __init__(self) -> None:
        self._cache_ids: set[str] = set()
        self._cache_items: list = []

    def add(self, items: Iterable) -> None:
        for it in items:
            if it.id not in self._cache_ids:
                self._cache_ids.add(it.id)
                self._cache_items.append(it)

    @property
    def items(self) -> list:
        return list(self._cache_items)


def _retry(func, *, max_retries: int = 3, base_delay: float = 2.0, label: str = ""):
    """Call *func* up to *max_retries* times, backing off on transient errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            if attempt < max_retries - 1 and _is_retryable(exc):
                wait = base_delay * (2 ** attempt)
                warnings.warn(
                    f"DEM STAC {label} transient error "
                    f"(attempt {attempt + 1}/{max_retries}, retry in {wait:.0f}s): {exc}"
                )
                time.sleep(wait)
            else:
                raise


def _open_catalog(stac_url: str, needs_sign: bool) -> Client:
    if needs_sign:
        try:
            import planetary_computer as pc
            return Client.open(stac_url, modifier=pc.sign_inplace)
        except ImportError:
            pass  # fall through to unsigned
    return Client.open(stac_url)


def _dem_cache_key(
    bbox_proj: Polygon, crs_proj: str, resolution_m: float, collection: str
) -> str:
    key_str = f"{bbox_proj.wkt}_{crs_proj}_{resolution_m}_{collection}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def _load_cached_dem(key: str) -> xr.DataArray | None:
    cache_file = _CACHE_DIR / f"{key}.tif"
    if not cache_file.exists():
        return None
    try:
        return xr.open_dataarray(cache_file)
    except Exception:
        return None


def _save_cached_dem(key: str, dem_da: xr.DataArray) -> None:
    cache_file = _CACHE_DIR / f"{key}.tif"
    try:
        dem_da.rio.to_raster(str(cache_file))
    except Exception:
        pass


def fetch_dem_bbox(
    bbox_proj: Polygon,
    crs_proj: str,
    resolution_m: float,
    item_cache: StacItemCache,
    collection: str = DEM_COLLECTION,
    asset_key: str | None = None,
    verbose: bool = False,
) -> xr.DataArray:
    """Fetch DEM for a projected bbox.

    Tries the ``_DEM_PROVIDERS`` chain in order.  Retries each provider up to
    3 times on transient errors, then moves to the next.  Raises
    ``aihydro_core.primitives.ToolError`` only when every provider fails.
    """
    # Resolve collection alias → (collection_name, default_asset).
    if collection in COLLECTION_ALIASES:
        collection, default_asset = COLLECTION_ALIASES[collection]
        asset_key = asset_key or default_asset
    asset_key = asset_key or "data"

    bbox_wgs84 = meters_to_degrees_bbox(bbox_proj, crs_proj)
    cache_key = _dem_cache_key(bbox_proj, crs_proj, resolution_m, collection)

    cached = _load_cached_dem(cache_key)
    if cached is not None:
        if verbose:
            log.info("Loaded DEM from cache (%sm)", resolution_m)
        return cached

    if verbose:
        log.info("Fetching DEM %s @ %sm", collection, resolution_m)

    # Build provider list: if the caller requested a collection that maps to a
    # specific provider URL, restrict to that provider chain; otherwise use the
    # default GLO-30 chain.
    provider_chain = _DEM_PROVIDERS

    last_exc: Exception | None = None

    for stac_url, coll_name, default_asset_key, needs_sign in provider_chain:
        # Allow explicit collection override from caller (e.g. "nasadem" direct).
        fetch_coll = collection
        fetch_asset = asset_key if asset_key != "data" else default_asset_key

        catalog = _open_catalog(stac_url, needs_sign)

        # ── Search ───────────────────────────────────────────────────────
        try:
            items = _retry(
                lambda c=catalog, fc=fetch_coll: list(
                    c.search(collections=[fc], bbox=bbox_wgs84).item_collection()
                ),
                label=f"search {fetch_coll}@{stac_url}",
            )
        except Exception as exc:
            last_exc = exc
            log.warning(
                "DEM search failed on %s (%s): %s — trying next provider",
                stac_url, fetch_coll, exc,
            )
            continue

        if not items:
            log.warning(
                "No %s items for bbox %s on %s — trying next provider",
                fetch_coll, bbox_wgs84, stac_url,
            )
            continue

        item_cache.add(items)

        # ── Load ─────────────────────────────────────────────────────────
        # Sign items for Planetary Computer if needed.
        load_items = item_cache.items
        if needs_sign:
            try:
                import planetary_computer as pc
                load_items = [pc.sign(it) for it in load_items]
            except ImportError:
                pass

        try:
            dem_ds = _retry(
                lambda li=load_items, fa=fetch_asset: stac_load(
                    li,
                    bands=[fa],
                    bbox=bbox_wgs84,
                    resolution=resolution_m,
                    crs=crs_proj,
                ),
                label=f"load {fetch_coll}@{stac_url}",
            )
        except Exception as exc:
            last_exc = exc
            log.warning(
                "DEM load failed on %s (%s): %s — trying next provider",
                stac_url, fetch_coll, exc,
            )
            continue

        # ── Success ───────────────────────────────────────────────────────
        if verbose:
            log.info("DEM loaded from %s", stac_url)

        dem_da = dem_ds[fetch_asset].squeeze().astype(np.float32)
        if dem_da.rio.crs is None:
            dem_da = dem_da.rio.write_crs(crs_proj)

        _save_cached_dem(cache_key, dem_da)
        return dem_da

    # All providers exhausted.
    from aihydro_core.primitives import ToolError
    raise ToolError(
        code="DELINEATION_FAILED",
        message=(
            f"DEM fetch failed for bbox {bbox_wgs84} after trying all providers "
            f"({[p[0] for p in provider_chain]}). "
            f"Last error: {last_exc}"
        ),
        recovery=(
            "Check network connectivity. The Copernicus DEM STAC endpoints "
            "(Planetary Computer and Element84) may be temporarily unavailable. "
            "Retry later or use a local MERIT-Hydro cache (method='local_merit')."
        ),
    )
