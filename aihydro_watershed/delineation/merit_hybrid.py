"""
Raster helpers for the MERIT-Basins hybrid tier: outlet snapping and terminal-catchment refinement.

Both steps read a small raster window, either from the regional ``accum##.tif`` /
``flowdir##.tif`` that ship with the pinned vectors (same provider, so the dataset versions
match the vectors) or, when those are not staged, from one Earth Engine request no larger than
``EE_MAX_WINDOW_KM`` (the size class that never failed in the camels-delineation-1 pilot).
Earth Engine serves MERIT Hydro v1.0.1, so using it is recorded as ``MERIT_VERSION_MIX``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine

from aihydro_watershed.delineation.merit_flowdir_pipeline import (
    MERIT_GEE_DATASET,
    MERIT_INTERACTIVE_MAX_WINDOW_CELLS,
    AREA_GATE_NO_MATCH_SUFFIX,
    MERIT_SCALE_M,
    MeritSnap,
    _polygonize_mask,
    basin_touches_edge,
    bbox_from_point,
    fetch_merit_gee_bbox,
    merit_dir_to_pyflwdir,
    snap_outlet_area_gated,
    snap_outlet_on_merit,
)

log = logging.getLogger(__name__)

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
    accum_path: Path,
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    area_gate: bool = False,
) -> MeritSnap:
    """Snap to a stream cell on the staged accumulation raster (cells -> km2 at the point's latitude).

    ``area_gate`` with an ``expected_area_km2`` uses the narrow 1 km, +/-10% gate
    (:func:`snap_outlet_area_gated`) on the default small window instead of the wide area scorer.
    """
    import rasterio
    from rasterio.windows import from_bounds

    gated = bool(area_gate and expected_area_km2 and expected_area_km2 > 0)
    bbox = bbox_from_point(lat, lon, snap_search_half_side_km(None if gated else expected_area_km2))
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
    if gated:
        return snap_outlet_area_gated(
            lat, lon, upa=upa, transform=transform, expected_area_km2=float(expected_area_km2)
        )
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


LOCAL_WINDOW_MAX_HALF_SIDE_KM = 20.0


def local_window_half_side_km(snapped_upa_km2: float) -> float:
    """Half-side of the first outlet-centred window: ``max(3 km, 2*sqrt(upa))``, capped at 20 km."""
    return min(LOCAL_WINDOW_MAX_HALF_SIDE_KM, max(3.0, 2.0 * math.sqrt(max(snapped_upa_km2, 0.0))))


def delineate_on_local_window(
    snap: MeritSnap, *, snapped_upa_km2: float
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Basin of the snapped cell routed on an Earth Engine ``dir`` window centred on the outlet.

    Used when the terminal unit catchment could not be refined and the outlet drains only part of it,
    so that the whole terminal catchment is not returned. The window starts at
    :func:`local_window_half_side_km` and grows by 1.6x while the basin touches its edge, up to
    ``LOCAL_WINDOW_MAX_HALF_SIDE_KM``; a basin that still touches the edge is refused (RuntimeError).
    """
    import pyflwdir

    half_side = local_window_half_side_km(snapped_upa_km2)
    while True:
        bbox = bbox_from_point(snap.lat, snap.lon, half_side)
        bundle = fetch_merit_gee_bbox(bbox, bands=("dir",))
        d8 = merit_dir_to_pyflwdir(np.asarray(bundle.bands["dir"]))
        flw = pyflwdir.from_array(
            d8,
            ftype="d8",
            mask=np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
            transform=bundle.transform,
            latlon=str(bundle.crs).upper() in ("EPSG:4326", "OGC:CRS84"),
        )
        ids = flw.basins(xy=([snap.lon], [snap.lat]), ids=np.array([1], dtype=np.uint32))
        mask = ids == 1
        if not mask.any():
            raise RuntimeError("Outlet-centred window produced an empty basin mask.")
        if not basin_touches_edge(mask):
            return _polygonize_mask(mask, bundle.transform, bundle.crs), {
                "origin": ORIGIN_EE,
                "half_side_km": round(half_side, 2),
                "cache_key": bundle.cache_key,
            }
        if half_side >= LOCAL_WINDOW_MAX_HALF_SIDE_KM:
            raise RuntimeError(
                "LOCAL_WINDOW_TOUCHES_EDGE: the basin of the snapped outlet still touches the "
                f"{2 * half_side:.0f} km window edge."
            )
        half_side = min(LOCAL_WINDOW_MAX_HALF_SIDE_KM, half_side * 1.6)


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


# ---------------------------------------------------------------------------------------------------------
# Offline snap on the staged flow-direction raster, inside the home unit catchment (no accumulation raster,
# no Earth Engine).

OFFLINE_FLOWDIR_SNAP_QUALITY = "offline_flowdir_home_catchment"

# A cell of the terminal catchment is credited with the upstream unit catchments' area when at least this
# many cells of those catchments drain through it (~1 km2 at 90 m). A choice made when the method was first
# tested (camels-delineation-1, Pfaf 74/77), not swept.
INFLOW_RECEIVES_MIN_CELLS = 120

# The snap search radius is the package default of :func:`snap_outlet_on_merit` (3 km). At 500 m and 1 km the
# first test gave the same number of wrong basins within one gauge, so it was left at the default.
OFFLINE_SNAP_RADIUS_M = 3_000.0


def snap_on_terminal_flowdir(
    terminal: gpd.GeoDataFrame,
    nonterminal: gpd.GeoDataFrame,
    flowdir_path: Path,
    lat: float,
    lon: float,
    *,
    inflow_km2: float,
    expected_area_km2: float | None = None,
    margin_cells: int = 32,
) -> tuple[MeritSnap, dict[str, Any]]:
    """Snap ``(lat, lon)`` to a stream cell of the home unit catchment using only the staged ``flowdir##.tif``.

    The raster is read on the terminal catchment's bounds plus ``margin_cells`` (the window
    :func:`refine_local` uses). Flow is routed only through cells touching ``terminal`` and each cell's
    local upstream area is computed from that routing. The upstream unit catchments (``nonterminal``) drain
    into the terminal catchment's main reach, so ``inflow_km2`` (their summed area) is added to every
    terminal cell that at least :data:`INFLOW_RECEIVES_MIN_CELLS` cells of theirs drain through. The
    result is the drainage area the same cell has on the full network, without an accumulation raster.

    The snap is the package's own: :func:`snap_outlet_on_merit` (3 km radius), or with a positive
    ``expected_area_km2`` the narrow :func:`snap_outlet_area_gated` on the same array. Only cells inside the
    home catchment are candidates, so a coordinate that falls in a neighbouring catchment than its river
    cannot reach that river.

    Returns the snap (``snap_quality`` is ``offline_flowdir_home_catchment``; ``area_gated`` when the gate
    found a cell, the plain quality plus ``AREA_GATE_NO_MATCH_SUFFIX`` when it found none) and a record of
    the window. ``snapped_upa_km2`` is the reconstructed area, in km2.
    """
    import pyflwdir
    import rasterio
    import rasterio.features
    from rasterio.windows import from_bounds

    from aihydro_watershed.delineation.merit_flowdir_pipeline import snap_outlet_area_gated

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
                f"OFFLINE_FLOWDIR_SNAP_FAILED: terminal window is {cells} cells "
                f"(limit {MERIT_INTERACTIVE_MAX_WINDOW_CELLS})."
            )
        dir_arr = src.read(1, window=window)
        transform = src.window_transform(window)
    latlon = str(crs).upper() in ("EPSG:4326", "OGC:CRS84")
    d8 = merit_dir_to_pyflwdir(dir_arr)
    valid = np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128])
    inside = rasterio.features.rasterize(
        [(g, 1) for g in terminal.to_crs(crs).geometry],
        out_shape=d8.shape, transform=transform, all_touched=True, dtype="uint8",
    ).astype(bool)
    if not inside.any():
        raise RuntimeError("OFFLINE_FLOWDIR_SNAP_FAILED: the terminal catchment covers no raster cell.")
    receives = np.zeros_like(inside)
    if inflow_km2 > 0 and len(nonterminal):
        feeders = rasterio.features.rasterize(
            [(g, 1) for g in nonterminal.to_crs(crs).geometry],
            out_shape=d8.shape, transform=transform, all_touched=False, dtype="uint8",
        ).astype(bool) & ~inside
        flw_full = pyflwdir.from_array(d8, ftype="d8", mask=valid, transform=transform, latlon=latlon)
        fed = flw_full.accuflux(np.where(feeders, 1.0, 0.0).astype(np.float32))
        receives = inside & (fed >= INFLOW_RECEIVES_MIN_CELLS)
    d8_in = np.where(inside, d8, 247).astype(np.uint8)
    flw_in = pyflwdir.from_array(
        d8_in, ftype="d8", mask=np.isin(d8_in, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        transform=transform, latlon=latlon,
    )
    local = np.asarray(flw_in.upstream_area("km2"), dtype=float)
    upa = np.where(inside & (local > 0), local, np.nan) + np.where(receives, float(inflow_km2), 0.0)
    upa = np.where(inside, upa, np.nan)
    gated = bool(expected_area_km2 and expected_area_km2 > 0)
    if gated:
        snap = snap_outlet_area_gated(
            lat, lon, upa=upa, transform=transform, expected_area_km2=float(expected_area_km2)
        )
        if snap.snap_quality != "area_gated":  # the gate found nothing; keep the plain snap's cell
            snap = replace(snap, snap_quality=OFFLINE_FLOWDIR_SNAP_QUALITY + AREA_GATE_NO_MATCH_SUFFIX)
    else:
        snap = snap_outlet_on_merit(
            lat, lon, upa=upa, transform=transform, search_radius_m=OFFLINE_SNAP_RADIUS_M
        )
        snap = replace(snap, snap_quality=OFFLINE_FLOWDIR_SNAP_QUALITY)
    if snap.snapped_upa_km2 is None or not inside[snap.row, snap.col]:
        raise RuntimeError(
            "OFFLINE_FLOWDIR_SNAP_FAILED: no drainage cell of the home unit catchment within the search radius."
        )
    info = {
        "origin": ORIGIN_LOCAL,
        "cells": cells,
        "bounds": (left, bottom, right, top),
        "inflow_km2": round(float(inflow_km2), 3),
        "inflow_cells": int(receives.sum()),
        "area_gated": gated,
    }
    return snap, info


# ---------------------------------------------------------------------------------------------------------
# Reliability of an outlet that could not be snapped (P1). The thresholds were fitted on the 671 CAMELS
# gauges (camels-delineation-1, condition B) and are IN-SAMPLE: ten-fold cross-validation of the same family
# of rules gave sensitivity 0.84 / specificity 0.89 for a wrong basin. They are a screen, not a guarantee.

#: returned area / along-reach area estimate above which the polygon is flagged unreliable
UNSNAPPED_OVER_RATIO_MAX = 1.25
#: distance (m) from the input point to the terminal catchment's river line above which it is flagged
UNSNAPPED_REACH_DISTANCE_MAX_M = 300.0


def _reach_geometry(line, lat: float, lon: float) -> tuple[float, float]:
    """(distance in m from the point to ``line``, fraction along ``line`` from its first vertex)."""
    from pyproj import CRS, Transformer
    from shapely.geometry import Point
    from shapely.ops import transform as shp_transform

    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    tf = Transformer.from_crs(4326, aeqd, always_xy=True)
    projected = shp_transform(tf.transform, line)
    origin = Point(0.0, 0.0)
    return float(projected.distance(origin)), float(projected.project(origin, normalized=True))


def assess_unsnapped_outlet(
    store: Any,
    terminal_id: int,
    lat: float,
    lon: float,
    *,
    returned_area_km2: float,
    terminal_area_km2: float,
    inflow_km2: float,
) -> dict[str, Any]:
    """Test a polygon whose outlet was not snapped (the whole terminal catchment plus all upstream).

    The polygon is right when the gauge sits at the terminal reach's downstream end. Anywhere else it
    includes local area below the gauge (partway up the reach) or belongs to another channel (a tributary,
    a neighbouring catchment). Two signals from the vectors already loaded, with no reference:

    * ``estimated_area_km2_on_reach`` = inflow + (1 - fraction along the reach) x terminal area, the area the
      gauge would drain if it sits on the terminal reach (reach lines run downstream to upstream and local
      area is taken to accrue evenly). ``over_ratio`` = returned area / that estimate.
    * ``distance_to_reach_m`` = distance from the input point to the terminal reach's line.

    ``unreliable`` when ``over_ratio`` > :data:`UNSNAPPED_OVER_RATIO_MAX` or the distance >
    :data:`UNSNAPPED_REACH_DISTANCE_MAX_M`, and also when the reach line is not available (the test cannot
    run, so the polygon cannot be called reliable). Otherwise ``provisional``: it passed the screen and the
    outlet is still not snapped.
    """
    checks: dict[str, Any] = {
        "over_ratio_max": UNSNAPPED_OVER_RATIO_MAX,
        "distance_to_reach_max_m": UNSNAPPED_REACH_DISTANCE_MAX_M,
        "thresholds_are_in_sample": True,
        "returned_area_km2": round(float(returned_area_km2), 3),
        "terminal_catchment_area_km2": round(float(terminal_area_km2), 3),
        "upstream_inflow_km2": round(float(inflow_km2), 3),
    }
    line = store.river_line(terminal_id)
    if line is None:
        return {
            "status": "unreliable",
            "reasons": ["the terminal reach's line is not available, so the polygon could not be tested"],
            "estimated_area_km2_on_reach": None,
            "checks": checks,
        }
    distance_m, frac = _reach_geometry(line, lat, lon)
    est = float(inflow_km2) + (1.0 - frac) * float(terminal_area_km2)
    over = float(returned_area_km2) / est if est > 0 else float("inf")
    checks.update(
        distance_to_reach_m=round(distance_m, 1),
        fraction_along_reach=round(frac, 4),
        over_ratio=round(over, 3) if math.isfinite(over) else None,
    )
    reasons = []
    if over > UNSNAPPED_OVER_RATIO_MAX:
        reasons.append(
            f"the polygon is {over:.2f}x the {est:.1f} km2 the point would drain if it sits on the terminal "
            f"reach (limit {UNSNAPPED_OVER_RATIO_MAX:g}x)"
        )
    if distance_m > UNSNAPPED_REACH_DISTANCE_MAX_M:
        reasons.append(
            f"the point is {distance_m:.0f} m from the terminal reach (limit {UNSNAPPED_REACH_DISTANCE_MAX_M:.0f} m)"
        )
    return {
        "status": "unreliable" if reasons else "provisional",
        "reasons": reasons,
        "estimated_area_km2_on_reach": round(est, 3),
        "checks": checks,
    }


def unsnapped_remedy(
    pfaf: str | None,
    *,
    flowdir_staged: bool,
    flowdir_size_mb: float | None,
    snap_error: str | None,
    stage_message: str | None,
) -> list[str]:
    """Ordered ways to get a snapped outlet, worded for what the code does on this path."""
    steps = [
        "Sign in to Google Earth Engine (earthengine authenticate, then set the project) so the outlet is "
        "snapped on the MERIT Hydro accumulation.",
    ]
    if flowdir_staged:
        steps.append(
            "The regional flow-direction raster is staged but the offline snap did not complete"
            + (f" ({snap_error})" if snap_error else "")
            + "; move the outlet onto the channel and try again."
        )
    else:
        size = f" (~{flowdir_size_mb:.0f} MB, one-off, non-commercial licence)" if flowdir_size_mb else ""
        why = f" Automatic staging did not run: {stage_message}" if stage_message else ""
        region = f"pfaf_region='{pfaf}', " if pfaf else ""
        steps.append(
            f"Stage the region's flow-direction raster{size} with merit_ensure_routing_region({region}"
            "acquisition_policy='download'); the next call then snaps offline inside the home catchment and "
            "expected_area_km2 steers that snap (within 1 km, +/-10%)." + why
        )
    steps.append(
        "expected_area_km2 does not change the outlet while it is unsnapped: it only checks the returned "
        "area (AREA_DIFFERS_FROM_EXPECTED), so it detects this error but does not fix it."
    )
    return steps
