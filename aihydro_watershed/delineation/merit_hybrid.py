"""
Raster helpers for the MERIT-Basins hybrid tier: outlet snapping and terminal-catchment refinement.

Both steps read a small raster window, either from the regional ``accum##.tif`` /
``flowdir##.tif`` that ship with the pinned vectors (same provider, so the dataset versions
match the vectors) or, when those are not staged, from one Earth Engine request no larger than
``EE_MAX_WINDOW_KM`` (the size class that never failed in the camels-delineation-1 pilot).
Earth Engine serves MERIT Hydro v1.0.1, so using it is recorded as ``MERIT_VERSION_MIX``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine

from aihydro_watershed.delineation.merit_flowdir_pipeline import (
    MERIT_GEE_DATASET,
    MERIT_INTERACTIVE_MAX_WINDOW_CELLS,
    MERIT_SCALE_M,
    MeritSnap,
    _polygonize_mask,
    bbox_from_point,
    fetch_merit_gee_bbox,
    merit_dir_to_pyflwdir,
    snap_outlet_on_merit,
)

# Largest Earth Engine window known to succeed (the pilot's 20 km half-side request).
EE_MAX_WINDOW_KM = 40.0
_KM_PER_DEG = 111.195

ORIGIN_LOCAL = "local_mghydro"
ORIGIN_EE = "gee_" + MERIT_GEE_DATASET.replace("/", "_")
ORIGIN_NONE = "none"


def snap_search_half_side_km(expected_area_km2: float | None) -> float:
    """Half-side of the snap window: the snap search radius plus a 20% margin."""
    if expected_area_km2 and expected_area_km2 > 0:
        radius = min(30.0, max(3.0, 1.5 * math.sqrt(expected_area_km2)))
    else:
        radius = 3.0
    return radius * 1.2


def _cell_area_km2(transform: Affine, lat: float) -> float:
    return abs(transform.a) * abs(transform.e) * _KM_PER_DEG**2 * math.cos(math.radians(lat))


def snap_on_local_accum(
    accum_path: Path, lat: float, lon: float, *, expected_area_km2: float | None = None
) -> MeritSnap:
    """Snap to a stream cell on the staged accumulation raster (cells -> km2 at the point's latitude)."""
    import rasterio
    from rasterio.windows import from_bounds

    bbox = bbox_from_point(lat, lon, snap_search_half_side_km(expected_area_km2))
    with rasterio.open(accum_path) as src:
        left, bottom = max(src.bounds.left, bbox[0]), max(src.bounds.bottom, bbox[1])
        right, top = min(src.bounds.right, bbox[2]), min(src.bounds.top, bbox[3])
        if left >= right or bottom >= top:
            raise ValueError("Outlet is outside the staged accumulation raster.")
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        acc = src.read(1, window=window).astype(float)
        transform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        acc = np.where(acc == nodata, np.nan, acc)
    acc = np.where(acc < 0, np.nan, acc)  # masked/negative cells are not drainage evidence
    upa = acc * _cell_area_km2(transform, lat)
    return snap_outlet_on_merit(
        lat, lon, upa=upa, transform=transform, expected_area_km2=expected_area_km2
    )


def refine_from_dir(
    terminal: gpd.GeoDataFrame,
    dir_arr: np.ndarray,
    transform: Affine,
    crs: str,
    snap: MeritSnap,
) -> gpd.GeoDataFrame:
    """Part of ``terminal`` upstream of the snapped outlet, routed on a MERIT D8 window.

    Flow is restricted to cells touching the terminal polygon so neighbouring catchments cannot
    feed it, and the result is clipped to the polygon.
    """
    import pyflwdir
    import rasterio.features

    d8 = merit_dir_to_pyflwdir(dir_arr)
    inside = rasterio.features.rasterize(
        [(g, 1) for g in terminal.to_crs(crs).geometry],
        out_shape=d8.shape,
        transform=transform,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    d8 = np.where(inside, d8, 247).astype(np.uint8)
    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        transform=transform,
        latlon=str(crs).upper() in ("EPSG:4326", "OGC:CRS84"),
    )
    ids = flw.basins(xy=([snap.lon], [snap.lat]), ids=np.array([1], dtype=np.uint32))
    refined = _polygonize_mask(ids == 1, transform, crs)
    if refined.empty:
        raise RuntimeError("Terminal raster refinement produced an empty mask.")
    clipped = gpd.overlay(
        refined.to_crs(terminal.crs), terminal[["geometry"]], how="intersection"
    )
    if clipped.empty:
        raise RuntimeError("Terminal raster refinement did not intersect the terminal catchment.")
    return clipped[["geometry"]]


def _window_bounds(terminal: gpd.GeoDataFrame, margin_deg: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = terminal.to_crs(4326).total_bounds
    return (xmin - margin_deg, ymin - margin_deg, xmax + margin_deg, ymax + margin_deg)


def refine_local(
    terminal: gpd.GeoDataFrame, flowdir_path: Path, snap: MeritSnap, *, margin_cells: int = 32
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Refine on a window of the staged regional ``flowdir##.tif`` (terminal bounds + margin)."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(flowdir_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        margin = margin_cells * abs(float(src.transform.a))
        xmin, ymin, xmax, ymax = _window_bounds(terminal, margin)
        left, bottom = max(src.bounds.left, xmin), max(src.bounds.bottom, ymin)
        right, top = min(src.bounds.right, xmax), min(src.bounds.top, ymax)
        if left >= right or bottom >= top:
            raise ValueError("Terminal catchment is outside the staged flowdir raster.")
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        cells = int(window.width * window.height)
        if cells > MERIT_INTERACTIVE_MAX_WINDOW_CELLS:
            raise RuntimeError(
                f"TERMINAL_CATCHMENT_REFINEMENT_FAILED: terminal window is {cells} cells "
                f"(limit {MERIT_INTERACTIVE_MAX_WINDOW_CELLS})."
            )
        dir_arr = src.read(1, window=window)
        transform = src.window_transform(window)
    out = refine_from_dir(terminal, dir_arr, transform, crs, snap)
    return out, {"origin": ORIGIN_LOCAL, "cells": cells, "bounds": (left, bottom, right, top)}


def refine_ee(
    terminal: gpd.GeoDataFrame, snap: MeritSnap, *, margin_cells: int = 8
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Refine on one small Earth Engine ``dir`` window; refuse windows beyond ``EE_MAX_WINDOW_KM``."""
    res_deg = MERIT_SCALE_M / 1000.0 / _KM_PER_DEG
    xmin, ymin, xmax, ymax = _window_bounds(terminal, margin_cells * res_deg)
    mid_lat = 0.5 * (ymin + ymax)
    width_km = (xmax - xmin) * _KM_PER_DEG * math.cos(math.radians(mid_lat))
    height_km = (ymax - ymin) * _KM_PER_DEG
    if max(width_km, height_km) > EE_MAX_WINDOW_KM:
        raise RuntimeError(
            "TERMINAL_REFINEMENT_WINDOW_TOO_LARGE: terminal catchment needs a "
            f"{max(width_km, height_km):.0f} km window (Earth Engine limit {EE_MAX_WINDOW_KM:.0f} km)."
        )
    bundle = fetch_merit_gee_bbox((xmin, ymin, xmax, ymax), bands=("dir",))
    out = refine_from_dir(terminal, np.asarray(bundle.bands["dir"]), bundle.transform, bundle.crs, snap)
    return out, {
        "origin": ORIGIN_EE,
        "bounds": (xmin, ymin, xmax, ymax),
        "window_km": (round(width_km, 2), round(height_km, 2)),
        "cache_key": bundle.cache_key,
    }


def version_consistency(
    *, vector_version: str, snap_origin: str, refine_origin: str
) -> dict[str, Any]:
    """The dataset each step used, and whether any step mixes MERIT Hydro versions."""
    mixed = ORIGIN_EE in (snap_origin, refine_origin)
    return {
        "vector_dataset": vector_version,
        "snap_dataset": snap_origin,
        "refinement_dataset": refine_origin,
        "mixed": mixed,
        "note": (
            "Earth Engine serves MERIT Hydro v1.0.1 while the vectors are MERIT-Basins "
            "v0.7/v1.0_bugfix1; Heberger (2025) reports that mixing raster and vector versions gives "
            "poor results. Stage the region's flowdir/accum rasters to remove the mix."
            if mixed
            else "All raster steps used the provider's regional files that accompany the vectors "
            "(the provider does not state their MERIT Hydro version, so consistency is by provenance, "
            "not verified)."
        ),
    }
