from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

from aihydro_watershed.delineation.types import area_km2
from aihydro_watershed.delineation.router import _gdf_to_geojson


def test_valid_polygon_area_is_positive_after_projection():
    poly = Polygon([(-86.0, 40.0), (-85.9, 40.0), (-85.9, 40.1), (-86.0, 40.1)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    assert poly.is_valid
    assert area_km2(gdf) > 0


def test_gdf_to_geojson_rejects_empty_geometry():
    gdf = gpd.GeoDataFrame(geometry=[Polygon()], crs="EPSG:4326")

    try:
        _gdf_to_geojson(gdf)
    except Exception as exc:
        assert getattr(exc, "code", None) == "DELINEATION_FAILED"
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("empty geometry should raise DELINEATION_FAILED")


def test_gdf_to_geojson_preserves_properties():
    poly = Polygon([(-86.0, 40.0), (-85.9, 40.0), (-85.9, 40.1), (-86.0, 40.1)])
    gdf = gpd.GeoDataFrame({"basin_id": ["demo"]}, geometry=[poly], crs="EPSG:4326")

    feature = _gdf_to_geojson(gdf)

    assert feature["type"] == "Feature"
    assert feature["properties"]["basin_id"] == "demo"
