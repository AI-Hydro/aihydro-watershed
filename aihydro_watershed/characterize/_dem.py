"""
Shared DEM-fetch helpers for analysis modules.

Provides a single global DEM-fetching path through ``aihydro_data.fetch("dem", …)``.
``aihydro-data`` owns source choice, fallback ordering, provenance, and product
metadata (for example GLO-30 primary with 3DEP/MERIT fallbacks where appropriate).

The function returns a ``rioxarray``-compatible ``xr.DataArray`` so callers
(``twi.py``, ``geomorphic.py``) can keep using rioxarray-style operations
(``.rio.reproject``, ``.rio.to_raster``, ``.rio.bounds``, …) without
branching on the data source.

This is the Wave 2.5 Axis 3 enabler — wiring the analysis layer to
aihydro-data so that every analysis tool inherits the same global coverage
the delineator already has.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# ── CONUS bounding box (loose) ────────────────────────────────────────────
# Includes Alaska/Hawaii implicitly via the longitude span; the 3DEP service
# itself returns an HTTP error outside its actual coverage, which we trap
# and fall back from. So a slightly oversized bbox is safe.
_CONUS_BBOX = (-125.0, 24.0, -66.5, 50.0)


def _geom_in_conus(geometry: Any) -> bool:
    """Best-effort check whether *geometry*'s centroid lies inside CONUS.

    Accepts shapely geometry, GeoDataFrame, GeoJSON dict, ``(lat, lon)`` tuple,
    or a bbox 4-tuple. Returns False on any parsing failure (safer — picks
    the global path).
    """
    try:
        lat, lon = _centroid_latlon(geometry)
        west, south, east, north = _CONUS_BBOX
        return (west <= lon <= east) and (south <= lat <= north)
    except Exception:
        return False


def _centroid_latlon(geometry: Any) -> tuple[float, float]:
    """Best-effort centroid extraction. Returns (lat, lon)."""
    # shapely geometry
    if hasattr(geometry, "centroid") and hasattr(geometry, "geom_type"):
        c = geometry.centroid
        return float(c.y), float(c.x)
    # GeoDataFrame
    try:
        import geopandas as gpd  # local import to avoid hard dep at module load
        if isinstance(geometry, gpd.GeoDataFrame):
            c = geometry.geometry.unary_union.centroid
            return float(c.y), float(c.x)
    except Exception:
        pass
    # GeoJSON dict
    if isinstance(geometry, dict):
        from shapely.geometry import shape
        g = geometry.get("geometry", geometry)
        c = shape(g).centroid
        return float(c.y), float(c.x)
    # (lat, lon) tuple
    if isinstance(geometry, (tuple, list)) and len(geometry) == 2:
        return float(geometry[0]), float(geometry[1])
    # bbox (west, south, east, north)
    if isinstance(geometry, (tuple, list)) and len(geometry) == 4:
        west, south, east, north = geometry
        return float((south + north) / 2), float((west + east) / 2)
    raise ValueError(f"Cannot extract centroid from geometry of type {type(geometry).__name__}")


def fetch_dem(
    geometry: Any,
    resolution: int = 30,
    prefer: str = "auto",
) -> "xr.DataArray":  # type: ignore[name-defined]
    """Fetch a DEM clipped to *geometry* and return an ``xr.DataArray``.

    Parameters
    ----------
    geometry : shapely / GeoDataFrame / GeoJSON dict / (lat,lon) / bbox
        Watershed boundary (or point) the DEM should cover.
    resolution : int
        Target resolution in metres. py3dep honours this faithfully; the
        global path returns Copernicus GLO-30 at ~30 m regardless.
    prefer : {"auto", "py3dep", "aihydro_data"}
        Kept for backward compatibility. ``auto`` and ``aihydro_data`` route
        through ``aihydro-data``. ``py3dep`` pins the ``DEM3DEP_10M`` product
        through ``aihydro-data`` rather than importing py3dep directly.

    Returns
    -------
    xr.DataArray
        Elevation raster in WGS84 by default. Always carries CRS metadata
        via rioxarray's ``.rio`` accessor so downstream reprojection works.

    Raises
    ------
    RuntimeError
        Both backends failed. The error message lists the underlying causes.
    """
    errors: list[str] = []

    # aihydro-data owns DEM source policy. It currently prefers GLO-30 for
    # reliability, keeps 3DEP as a CONUS fallback, and exposes MERIT/global
    # fallbacks with product metadata and citations.
    try:
        from aihydro_data import fetch as _adata_fetch
        geom_arg = _to_shapely(geometry)
        kwargs = {}
        if prefer == "py3dep":
            kwargs.update({"mode": "manual", "product": "DEM3DEP_10M"})
        result = _adata_fetch("dem", geom_arg, "", "", **kwargs)
        dem = result.data  # xr.DataArray from gee/stac backend
        dem = _normalize_dem(dem)
        log.info(
            "DEM fetched via aihydro-data (%s, source=%s)",
            result.product, result.source,
        )
        return dem, result.product, result.source
    except Exception as exc:
        errors.append(f"aihydro-data: {exc}")

    raise RuntimeError(
        "Could not fetch a DEM for the requested geometry. "
        "Tried: " + " | ".join(errors)
    )


class DemInputError(ValueError):
    """A DEM raster is ambiguous or incomplete (extra slices, no CRS).

    Raised instead of guessing. ``fetch_dem`` surfaces it in its
    ``RuntimeError``; ``slope_from_dem`` re-raises it as
    ``SlopeUnavailableError``.
    """


def _normalize_dem(dem: "xr.DataArray") -> "xr.DataArray":  # type: ignore[name-defined]
    """Normalise a DEM DataArray from any backend so downstream code never
    has to worry about dimension naming, CRS, or band dimensions.

    Three transformations applied in order:

    1. **Rename latitude/longitude → y/x** — rioxarray requires 'x'/'y';
       GEE and STAC backends sometimes return 'lat'/'lon' or
       'latitude'/'longitude'.

    2. **Squeeze singleton non-spatial dimensions** — GEE rasters often carry
       an extra leading size-1 band axis: (1, H, W) → (H, W).  pysheds and
       np.gradient both need a 2-D array.  A non-spatial dimension with more
       than one slice is **refused** (``DemInputError``), never reduced to its
       first slice: that would silently drop the rest of the data.

    3. **Require a CRS** — a DEM without CRS metadata is refused
       (``DemInputError``); WGS84 is not assumed, because a wrong guess turns
       into wrong slopes and areas with no trace.
    """
    # 1. Rename non-standard dimension names
    dim_renames: dict[str, str] = {}
    for old, new in (("latitude", "y"), ("longitude", "x"), ("lat", "y"), ("lon", "x")):
        if old in dem.dims:
            dim_renames[old] = new
    if dim_renames:
        dem = dem.rename(dim_renames)

    # 2. Squeeze singleton band / time dimensions; refuse multi-slice ones.
    if "y" in dem.dims and "x" in dem.dims:
        extra = [d for d in dem.dims if d not in ("y", "x")]
    else:
        extra = list(dem.dims[:-2])
    multi = {d: int(dem.sizes[d]) for d in extra if dem.sizes[d] > 1}
    if multi:
        raise DemInputError(
            f"DEM has non-spatial dimension(s) with more than one slice {multi}; "
            "refusing to keep only the first slice. Select the intended band/time "
            "explicitly before passing the raster."
        )
    for d in extra:
        dem = dem.squeeze(d, drop=True)

    # 3. A CRS is required
    try:
        crs = dem.rio.crs
    except Exception as exc:
        raise DemInputError(f"DEM CRS could not be read ({exc}); refusing to assume one") from exc
    if crs is None:
        raise DemInputError("DEM has no CRS metadata; refusing to assume EPSG:4326")

    return dem


def reproject_elevation(dem: "xr.DataArray", crs: str) -> "xr.DataArray":  # type: ignore[name-defined]
    """Reproject a continuous elevation raster with **bilinear** resampling.

    rioxarray's default is nearest, which copies cells and so leaves stair-steps
    and a half-cell shift in the elevation surface (and in slopes derived from
    it). Elevation is continuous; categorical rasters must not use this.
    """
    from rasterio.enums import Resampling

    return dem.rio.reproject(crs, resampling=Resampling.bilinear)


def _to_shapely(geometry: Any) -> Any:
    """Coerce *geometry* to a shapely geometry for downstream backends."""
    # Already shapely
    if hasattr(geometry, "geom_type"):
        return geometry
    # GeoDataFrame
    try:
        import geopandas as gpd
        if isinstance(geometry, gpd.GeoDataFrame):
            return geometry.geometry.unary_union
    except Exception:
        pass
    # GeoJSON dict
    if isinstance(geometry, dict):
        from shapely.geometry import shape
        return shape(geometry.get("geometry", geometry))
    # Pass through tuples/lists — backends accept these
    return geometry


class SlopeUnavailableError(ValueError):
    """The DEM's CRS or pixel spacing does not allow a metric slope.

    ``slope_from_dem`` raises this instead of returning a number computed on
    degrees (or on unit spacing). Callers must treat it as "no slope", never
    fall back to the native grid.
    """


def _metric_spacing(dem: "xr.DataArray") -> tuple[float, float]:  # type: ignore[name-defined]
    """Pixel size ``(dx, dy)`` in metres, or raise ``SlopeUnavailableError``."""
    try:
        from pyproj import CRS

        crs = dem.rio.crs
        if crs is None:
            raise SlopeUnavailableError("DEM has no CRS; refusing to assume a metric grid")
        crs = CRS.from_user_input(crs)
        if not crs.is_projected:
            raise SlopeUnavailableError(
                f"DEM CRS {crs.to_string()} is not projected (spacing is in degrees); "
                "reproject to a metric CRS (EPSG:5070 in CONUS, UTM elsewhere) before computing slope"
            )
        factor = float(crs.axis_info[0].unit_conversion_factor)
        dx, dy = (abs(float(v)) for v in dem.rio.resolution())
    except SlopeUnavailableError:
        raise
    except Exception as exc:  # unreadable CRS / resolution: no silent unit spacing
        raise SlopeUnavailableError(f"DEM CRS or pixel spacing could not be determined ({exc})") from exc
    dx, dy = dx * factor, dy * factor
    if not (dx > 0 and dy > 0) or dx != dx or dy != dy or dx == float("inf") or dy == float("inf"):
        raise SlopeUnavailableError(f"DEM pixel spacing is not usable (dx={dx}, dy={dy})")
    return dx, dy


def slope_from_dem(dem: "xr.DataArray") -> "xr.DataArray":  # type: ignore[name-defined]
    """Compute slope (degrees) from a DEM using numpy's gradient.

    Used when ``py3dep.get_map("Slope Degrees", …)`` is unavailable (i.e.
    outside CONUS). The DEM must already be in a projected, metric CRS
    (EPSG:5070 for CONUS, UTM elsewhere): the gradient uses its pixel size in
    metres. **Fails closed**: if the CRS is missing, geographic (spacing in
    degrees) or the pixel size cannot be read, ``SlopeUnavailableError`` is
    raised. It never returns a degrees-based or unit-spacing "slope" (such
    values saturate towards 90 degrees and depend on pixel size).
    """
    import numpy as np
    import xarray as xr

    # Ensure DEM is 2-D before computing gradient.  rioxarray reproject and
    # some GEE backends preserve a leading band axis (1, H, W); np.gradient
    # with only two spacing arguments only works on 2-D arrays.
    try:
        dem = _normalize_dem(dem)  # rename dims, squeeze singleton band axis, require CRS
    except DemInputError as exc:
        raise SlopeUnavailableError(str(exc)) from exc

    # Metric pixel size, or raise. Checked before any array work.
    dx, dy = _metric_spacing(dem)

    arr = dem.values.astype("float64")
    # At this point arr must be exactly 2-D; guard just in case.
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    # np.gradient returns (d/dy, d/dx) when axis order is (y, x), which it
    # is for rioxarray-loaded rasters (y increases downward in pixel space,
    # so dy carries the sign; we use abs above for the spacing argument).
    gy, gx = np.gradient(arr, dy, dx)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)

    # Reconstruct a 2-D DataArray with the DEM's spatial coords/dims.
    spatial_dims = [d for d in dem.dims if d in ("y", "x")]
    spatial_coords = {k: v for k, v in dem.coords.items() if k in spatial_dims}
    out = xr.DataArray(
        slope_deg,
        coords=spatial_coords,
        dims=spatial_dims if len(spatial_dims) == 2 else dem.dims,
        attrs={**dem.attrs, "units": "degrees", "long_name": "slope"},
    )
    try:
        out.rio.write_crs(dem.rio.crs, inplace=True)
    except Exception:
        pass
    return out
