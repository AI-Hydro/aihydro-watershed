"""Shared types for delineation tiers."""

from __future__ import annotations

from typing import NamedTuple

import geopandas as gpd

from aihydro_watershed.delineation.utils import EQUAL_AREA_CRS


class FastDelineationResult(NamedTuple):
    gdf: gpd.GeoDataFrame
    area_km2: float
    scout_box_maxed: bool
    outlet_lat: float
    outlet_lon: float
    merit_snap_distance_m: float | None
    pfaf_code: str | None
    used_nldi_basin: bool = False
    # Snapped pour point (EPSG:4326) of the DEM delineation and the grid it sits
    # on. ``outlet_lat/lon`` above may be the requested/NLDI/MERIT-snapped point;
    # identity (BasinRef grid anchors) uses ONLY these fields. None when no DEM
    # snap happened (e.g. the NLDI indexed basin was used).
    snapped_outlet_lon: float | None = None
    snapped_outlet_lat: float | None = None
    snap_grid_crs: str | None = None
    snap_grid_resolution_m: float | None = None


def area_km2(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty or gdf.geometry.is_empty.all():
        return 0.0
    return float(gdf.to_crs(EQUAL_AREA_CRS).area.sum() / 1e6)
