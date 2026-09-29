"""
Small-catchment delineation on USGS 3DEP bare-earth elevation (CONUS).

The other delineation tiers are built for river basins: NLDI returns whole
NHDPlus catchments (median a few km²), MERIT-Hydro is 90 m, and the raw-DEM
fallback reads Copernicus GLO-30, a surface model that keeps tree canopy and
buildings. None of them resolves a road-culvert catchment of ~1 km², and none
handles the road embankment, which on a DEM is a dam across the channel with
the culvert invisible beneath it.

Method
------
1. Fetch the 3DEP 1/3 arc-second DEM (~10 m) around the pour point and
   reproject it to EPSG:5070 (Albers equal area) at 10 m.
2. Carve a notch through the embankment: every cell within ``notch_radius_m``
   (30 m) of the pour point is set to the minimum elevation found within
   ``notch_source_radius_m`` (60 m). This stands in for the culvert barrel.
3. Fill depressions and derive D8 flow directions with ``pyflwdir.from_dem``.
4. Snap: the outlet is the cell with the largest upstream area within
   ``snap_radius_m`` (40 m) of the pour point.
5. If the basin touches the edge of the DEM window, enlarge the window
   (2.5, 5, 10 km half-width by default) and repeat.

Evidence
--------
On 67 INDOT culverts in Indiana (SPR-4926), delineated areas were within a
factor of 2 of the engineer-computed drainage area at 76 % of sites (median
ratio 1.01, Spearman 0.80). Every delineation under 0.05 km² was a snap onto a
roadside ditch rather than the culvert's channel, so those are flagged
``LIKELY_DITCH_SNAP``.

USGS StreamStats is not used: the old ``streamstats.usgs.gov/streamstatsservices``
API is retired (404), and the current ``/ss-delineate/v1/delineate/sshydro/{STATE}``
service does not snap points onto its stream grid for very small channels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

EQUAL_AREA_EPSG = 5070
DEFAULT_RESOLUTION_M = 10.0
NOTCH_RADIUS_M = 30.0
NOTCH_SOURCE_RADIUS_M = 60.0
SNAP_RADIUS_M = 40.0
HALF_WIDTHS_M: tuple[float, ...] = (2500.0, 5000.0, 10000.0)
DITCH_SNAP_KM2 = 0.05
SMALL_CATCHMENT_MAX_KM2 = 5.0

ROUTING_DATASET = "USGS 3DEP 1/3 arc-second (10 m) bare-earth DEM"

_NODATA = np.float32(-9999.0)

# (x, y, half_width_m, resolution_m) -> (z [2-D float32, NaN = nodata], affine transform)
DemFetcher = Callable[[float, float, float, float], tuple[np.ndarray, Any]]


@dataclass
class SmallCatchmentResult:
    gdf: Any                                  # GeoDataFrame, EPSG:4326, one polygon
    area_km2: float
    status: str                               # "ok" | "touches_max_window"
    snap_distance_m: float
    outlet_lat: float
    outlet_lon: float
    window_half_width_m: float
    window_iterations: int
    resolution_m: float
    notch_elevation_m: Optional[float]
    quality_flags: list[str] = field(default_factory=list)
    window_bounds_5070: Optional[tuple[float, float, float, float]] = None


def fetch_3dep_window(
    x: float, y: float, half_width_m: float, resolution_m: float = DEFAULT_RESOLUTION_M,
) -> tuple[np.ndarray, Any]:
    """3DEP DEM for a square window centred on (x, y) in EPSG:5070.

    Returns the elevation array (float32, NaN where missing) and its affine
    transform, at ``resolution_m`` in EPSG:5070.
    """
    import py3dep
    import rioxarray  # noqa: F401  (registers .rio)
    from pyproj import Transformer

    to_ll = Transformer.from_crs(EQUAL_AREA_EPSG, 4326, always_xy=True)
    # The request box comes from the lower-left and upper-right corners only.
    # This fixes the origin of the reprojected grid, and the INDOT validation
    # was run with exactly this convention. Changing it (e.g. to all four
    # corners) shifts the grid by a fraction of a cell, which is enough to
    # flip a marginal snap between a ditch and the culvert channel.
    lon0, lat0 = to_ll.transform(x - half_width_m, y - half_width_m)
    lon1, lat1 = to_ll.transform(x + half_width_m, y + half_width_m)
    dem = py3dep.get_dem((lon0, lat0, lon1, lat1), resolution=10, crs=4326)
    dem = dem.rio.reproject(EQUAL_AREA_EPSG, resolution=resolution_m)
    dem = dem.rio.clip_box(x - half_width_m, y - half_width_m, x + half_width_m, y + half_width_m)
    z = dem.values.astype("float32")
    nd = dem.rio.nodata
    if nd is not None and np.isfinite(nd):
        z = np.where(z == np.float32(nd), np.nan, z)
    return z, dem.rio.transform()


def delineate_on_dem(
    z: np.ndarray,
    transform: Any,
    x: float,
    y: float,
    *,
    notch_radius_m: float = NOTCH_RADIUS_M,
    notch_source_radius_m: float = NOTCH_SOURCE_RADIUS_M,
    snap_radius_m: float = SNAP_RADIUS_M,
    carve_notch: bool = True,
) -> dict:
    """Delineate the catchment draining to (x, y) on a projected DEM array.

    ``z`` is a 2-D elevation array (NaN = nodata) with a north-up metric
    ``transform``; (x, y) are in the same CRS. Returns a dict with ``basin``
    (bool mask), ``outlet_rc``, ``snap_distance_m``, ``outlet_xy``,
    ``touches_edge``, ``area_m2`` and ``notch_elevation_m``.
    """
    import pyflwdir

    z = np.asarray(z, dtype="float32")
    valid = np.isfinite(z)
    zc = np.where(valid, z, _NODATA).astype("float32")

    rows, cols = np.indices(z.shape)
    cx = transform.c + (cols + 0.5) * transform.a
    cy = transform.f + (rows + 0.5) * transform.e
    d = np.hypot(cx - x, cy - y)

    notch_elev: Optional[float] = None
    if carve_notch:
        src = valid & (d <= notch_source_radius_m)
        if not src.any():
            raise RuntimeError("No DEM data within the notch search radius of the pour point.")
        notch_elev = float(z[src].min())
        zc[valid & (d <= notch_radius_m)] = notch_elev

    flw = pyflwdir.from_dem(
        data=zc, nodata=_NODATA, transform=transform, latlon=False, outlets="edge",
    )
    upa = flw.upstream_area(unit="m2")
    cand = valid & (d <= snap_radius_m)
    if not cand.any():
        raise RuntimeError("No DEM cells within the snap radius of the pour point.")
    score = np.where(cand, upa, -1.0)
    idx = int(np.argmax(score))
    r, c = np.unravel_index(idx, z.shape)
    basin = flw.basins(idxs=np.array([idx])) > 0
    touches = bool(basin[0, :].any() or basin[-1, :].any()
                   or basin[:, 0].any() or basin[:, -1].any())
    cell_area = abs(transform.a * transform.e)
    return {
        "basin": basin,
        "outlet_rc": (int(r), int(c)),
        "outlet_xy": (float(cx[r, c]), float(cy[r, c])),
        "snap_distance_m": float(d[r, c]),
        "touches_edge": touches,
        "area_m2": float(basin.sum() * cell_area),
        "notch_elevation_m": notch_elev,
    }


def _basin_polygon(basin: np.ndarray, transform: Any):
    import rasterio.features
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    polys = []
    for geom, val in rasterio.features.shapes(
        basin.astype("uint8"), mask=basin, transform=transform,
    ):
        if val != 1:
            continue
        try:  # skip degenerate single-cell slivers GEOS cannot close
            polys.append(make_valid(shape(geom)))
        except Exception:
            continue
    return unary_union(polys)


def delineate_small_catchment(
    lat: float,
    lon: float,
    *,
    half_widths_m: Sequence[float] = HALF_WIDTHS_M,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    notch_radius_m: float = NOTCH_RADIUS_M,
    notch_source_radius_m: float = NOTCH_SOURCE_RADIUS_M,
    snap_radius_m: float = SNAP_RADIUS_M,
    carve_notch: bool = True,
    dem_fetcher: Optional[DemFetcher] = None,
) -> SmallCatchmentResult:
    """Delineate a culvert-scale catchment from a pour point (WGS84).

    See the module docstring for the method. ``dem_fetcher`` replaces the 3DEP
    download (used by tests); it takes ``(x, y, half_width_m, resolution_m)``
    in EPSG:5070 and returns ``(z, transform)``.
    """
    import geopandas as gpd
    from pyproj import Transformer

    if not half_widths_m:
        raise ValueError("half_widths_m must contain at least one window size")
    fetch = dem_fetcher or fetch_3dep_window
    to_m = Transformer.from_crs(4326, EQUAL_AREA_EPSG, always_xy=True)
    to_ll = Transformer.from_crs(EQUAL_AREA_EPSG, 4326, always_xy=True)
    x, y = to_m.transform(lon, lat)

    res = None
    transform = None
    hw = half_widths_m[0]
    iterations = 0
    for iterations, hw in enumerate(half_widths_m, start=1):
        z, transform = fetch(x, y, hw, resolution_m)
        res = delineate_on_dem(
            z, transform, x, y,
            notch_radius_m=notch_radius_m,
            notch_source_radius_m=notch_source_radius_m,
            snap_radius_m=snap_radius_m,
            carve_notch=carve_notch,
        )
        if not res["touches_edge"]:
            break
        log.info("small_catchment: basin touches the %.0f m window edge; enlarging", hw)

    poly = _basin_polygon(res["basin"], transform)
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=EQUAL_AREA_EPSG).to_crs(4326)
    area_km2 = res["area_m2"] / 1e6
    olon, olat = to_ll.transform(*res["outlet_xy"])

    flags: list[str] = []
    if carve_notch:
        flags.append("EMBANKMENT_NOTCH_APPLIED")
    if res["touches_edge"]:
        flags.append("BASIN_TOUCHES_MAX_WINDOW")
    if area_km2 < DITCH_SNAP_KM2:
        flags.append("LIKELY_DITCH_SNAP")

    return SmallCatchmentResult(
        gdf=gdf,
        area_km2=area_km2,
        status="touches_max_window" if res["touches_edge"] else "ok",
        snap_distance_m=res["snap_distance_m"],
        outlet_lat=float(olat),
        outlet_lon=float(olon),
        window_half_width_m=float(hw),
        window_iterations=iterations,
        resolution_m=float(abs(transform.a)),
        notch_elevation_m=res["notch_elevation_m"],
        quality_flags=flags,
        window_bounds_5070=(x - hw, y - hw, x + hw, y + hw),
    )
