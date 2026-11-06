"""MERIT Hydro flow-direction delineation with GEE fetch + local pyflwdir routing."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import resource
import threading
import tempfile
import time
import warnings
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from pyproj import Geod
from shapely.geometry import shape

from aihydro_watershed._paths import aihydro_home
from aihydro_watershed.delineation.types import area_km2

log = logging.getLogger(__name__)

MERIT_GEE_DATASET = "MERIT/Hydro/v1_0_1"
MERIT_SCALE_M = 92.77
MERIT_LICENSE = "CC-BY-NC 4.0 or ODbL 1.0; derived-data obligations may apply."
# Hybrid tier data: MERIT-Basins vectors + regional rasters as published for the `delineator` package
# (pinned by SHA-256 in merit/delineator_data_manifest.json). Non-commercial; sources disagree on variant.
MERIT_BASINS_LICENSE = (
    "Non-commercial and share-alike. MERIT-Basins/mghydro files: CC BY-NC-SA 4.0 (share-alike); MERIT Hydro: "
    "CC BY-NC 4.0 or ODbL 1.0 (ODbL derived data must be shared alike); the sources disagree on the exact "
    "variant; derived-data obligations may apply."
)
MERIT_CITATION = (
    "Yamazaki D., Ikeshima D., Sosa J., Bates P.D., Allen G.H., Pavelsky T.M. "
    "MERIT Hydro: A high-resolution global hydrography map based on latest "
    "topography datasets. Water Resources Research, 55, 5053-5073, 2019. "
    "https://doi.org/10.1029/2019WR024873"
)
MERIT_BASINS_VECTOR_VERSION = "MERIT-Basins v0.7/v1.0_bugfix1 (mghydro data v3, delineator 2.2.4 manifest)"
MERIT_BASINS_VECTOR_SOURCE = "https://mghydro.com/watersheds/data/basins{pfaf}.db"
MERIT_FLOWDIR_RASTER_VERSION = "mghydro regional flowdir{pfaf}.tif (MERIT Hydro version not stated by provider)"
MERIT_FLOWDIR_RASTER_SOURCE = "https://mghydro.com/watersheds/data/flowdir{pfaf}.tif"

_TOOL_PATH = "aihydro_watershed.delineation.merit_flowdir_pipeline.delineate_merit_flowdir"
_GEOD = Geod(ellps="WGS84")
_DEFAULT_BANDS = ("dir", "upa", "wth")
_SNAP_REFERENCE_BANDS = ("upa", "wth")
_EDGE_MARGIN_CELLS = 2

# MERIT Hydro native grid on Earth Engine (EPSG:4326, 3 arc-second cells; pixel
# EDGES sit half a cell off the whole-degree lattice). Requests are made on this
# grid so Earth Engine does not resample (see fetch_merit_gee_bbox).
MERIT_NATIVE_PX_DEG = 1.0 / 1200.0
MERIT_ORIGIN_X = -180.0 - 0.5 * MERIT_NATIVE_PX_DEG
MERIT_ORIGIN_Y = 85.0 - 0.5 * MERIT_NATIVE_PX_DEG
# Extent of the asset: lon -180..180, lat 85..-60 (the window is clamped to it).
MERIT_GRID_COLS = 360 * 1200
MERIT_GRID_ROWS = 145 * 1200
# Largest synchronous window requested (3 bands). A 4.1M-cell / 82 km half-side
# native-grid request succeeded in an unsaved console probe (see
# docs/evidence/merit-gee-native-grid-2026-11/README.md); 4.5M was not run live.
MERIT_GEE_MAX_WINDOW_CELLS = 4_500_000
_AREA_MISMATCH_WARN_FRAC = 0.25
_MAX_LOCAL_FLOWDIR_CELLS = 250_000_000
_LOCAL_ROUTING_MEMORY_RISK_MB = 1_500.0
MERIT_SAFE_ENVELOPE_VERSION = "benchmark_2026-05-26_v1"
MERIT_INTERACTIVE_MAX_WINDOW_CELLS = 60_000_000
MERIT_INTERACTIVE_MAX_RSS_DELTA_MB = 600.0
MERIT_SCIENTIFIC_MAX_WINDOW_CELLS = 120_000_000
MERIT_SCIENTIFIC_MAX_RSS_DELTA_MB = 1_500.0


class MeritGeeWindowLimit(RuntimeError):
    """The basin needs a synchronous Earth Engine window larger than the cap."""


@dataclass(frozen=True)
class MeritRasterBundle:
    bands: dict[str, np.ndarray]
    transform: Affine
    crs: str
    cache_key: str
    source: str
    bbox: tuple[float, float, float, float]
    resolution_m: float = MERIT_SCALE_M


@dataclass(frozen=True)
class MeritSnap:
    lat: float
    lon: float
    row: int
    col: int
    distance_m: float
    snapped_upa_km2: float | None
    snap_quality: str


@dataclass(frozen=True)
class MeritSnapReference:
    snap: MeritSnap
    source: str
    cache_key: str
    bbox: tuple[float, float, float, float]
    official_merit_upa_km2: float | None


@dataclass(frozen=True)
class MeritFlowdirResult:
    gdf: gpd.GeoDataFrame
    area_km2: float
    outlet_lat: float
    outlet_lon: float
    snap_distance_m: float
    snapped_upa_km2: float | None
    snap_quality: str
    snap_validation: dict[str, Any]
    cache_key: str
    bbox: tuple[float, float, float, float]
    scout_box_maxed: bool
    touches_edge: bool
    area_validation: dict[str, Any]
    quality_flags: list[str]
    source: str
    routing_dataset: str
    routing_resolution_m: float
    pfaf_region: str | None = None
    routing_data_source: str | None = None
    snap_source: str | None = None
    official_merit_upa_km2: float | None = None
    local_upstream_area_km2: float | None = None
    polygon_area_km2: float | None = None
    validation_sources: list[str] | None = None
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
    fallback_history: list[dict[str, Any]] | None = None
    memory_telemetry: dict[str, float] | None = None
    safe_envelope_version: str | None = None
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


def _peak_memory_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru / (1024 * 1024) if ru > 10_000_000 else ru / 1024)


def _rss_memory_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return _peak_memory_mb()


def merit_dir_to_pyflwdir(merit_dir: np.ndarray) -> np.ndarray:
    """Convert MERIT Hydro local drainage direction to pyflwdir D8 codes."""
    arr = np.asarray(merit_dir)
    out = np.full(arr.shape, 247, dtype=np.uint8)
    valid = np.isin(arr, [0, 1, 2, 4, 8, 16, 32, 64, 128])
    out[valid] = arr[valid].astype(np.uint8)
    return out


def bbox_from_point(lat: float, lon: float, half_side_km: float) -> tuple[float, float, float, float]:
    """Build a WGS84 bbox roughly half_side_km around a point."""
    lat_delta = half_side_km / 111.32
    lon_delta = half_side_km / max(111.32 * math.cos(math.radians(lat)), 1e-6)
    return (
        max(-180.0, lon - lon_delta),
        max(-90.0, lat - lat_delta),
        min(180.0, lon + lon_delta),
        min(90.0, lat + lat_delta),
    )


def basin_touches_edge(mask: np.ndarray, margin_cells: int = _EDGE_MARGIN_CELLS) -> bool:
    """Return True if a basin mask reaches the raster edge margin."""
    if mask.size == 0 or not np.any(mask):
        return True
    m = max(1, int(margin_cells))
    return bool(
        mask[:m, :].any()
        or mask[-m:, :].any()
        or mask[:, :m].any()
        or mask[:, -m:].any()
    )


def validate_area(
    area: float,
    *,
    snapped_upa_km2: float | None,
    expected_area_km2: float | None,
    touches_edge: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Compare polygon area to MERIT UPA and optional user-provided area."""
    validation: dict[str, Any] = {
        "area_km2": round(float(area), 3),
        "snapped_upa_km2": round(float(snapped_upa_km2), 3)
        if snapped_upa_km2 is not None and np.isfinite(snapped_upa_km2)
        else None,
        "expected_area_km2": expected_area_km2,
        "touches_edge": bool(touches_edge),
    }
    flags: list[str] = []
    if touches_edge:
        flags.append("BASIN_TOUCHES_WINDOW_EDGE")
    if area < 1.0:
        flags.append("VERY_SMALL_BASIN")
    if snapped_upa_km2 and snapped_upa_km2 > 0:
        rel = abs(area - snapped_upa_km2) / snapped_upa_km2
        validation["relative_error_vs_merit_upa"] = round(float(rel), 4)
        if rel > _AREA_MISMATCH_WARN_FRAC:
            flags.append("MERIT_UPA_AREA_MISMATCH")
    if expected_area_km2 and expected_area_km2 > 0:
        rel = abs(area - expected_area_km2) / expected_area_km2
        validation["relative_error_vs_expected"] = round(float(rel), 4)
        if rel > _AREA_MISMATCH_WARN_FRAC:
            flags.append("AREA_DIFFERS_FROM_EXPECTED")
    validation["ok"] = not flags
    return validation, flags


def validate_snap(
    snap: MeritSnap,
    *,
    expected_area_km2: float | None,
) -> tuple[dict[str, Any], list[str]]:
    validation: dict[str, Any] = {
        "snap_distance_m": round(float(snap.distance_m), 3),
        "snapped_upa_km2": round(float(snap.snapped_upa_km2), 3)
        if snap.snapped_upa_km2 is not None and np.isfinite(snap.snapped_upa_km2)
        else None,
        "expected_area_km2": expected_area_km2,
        "snap_quality": snap.snap_quality,
    }
    flags: list[str] = []
    if snap.distance_m > 5_000:
        flags.append("OUTLET_SNAP_FAR")
    if expected_area_km2 and expected_area_km2 > 0 and snap.snapped_upa_km2:
        rel = abs(snap.snapped_upa_km2 - expected_area_km2) / expected_area_km2
        validation["relative_error_vs_expected"] = round(float(rel), 4)
        if rel > _AREA_MISMATCH_WARN_FRAC:
            flags.append("SNAP_UPA_DIFFERS_FROM_EXPECTED")
    validation["ok"] = not flags
    return validation, flags


def classify_safe_envelope(
    *,
    final_window_cell_count: int | None,
    rss_delta_mb: float | None,
    window_complete: bool,
    scientific_mode: bool = False,
) -> tuple[str, list[str]]:
    """Classify adaptive-window routing against the current benchmark envelope."""
    cells = int(final_window_cell_count or 0)
    rss_delta = float(rss_delta_mb or 0.0)
    flags: list[str] = []
    if not window_complete:
        return "hybrid_required", ["HYBRID_ROUTING_REQUIRED", "HYBRID_ROUTING_RECOMMENDED"]
    if cells > MERIT_SCIENTIFIC_MAX_WINDOW_CELLS or rss_delta > MERIT_SCIENTIFIC_MAX_RSS_DELTA_MB:
        return "hybrid_required", [
            "LOCAL_ROUTING_MEMORY_RISK",
            "HYBRID_ROUTING_REQUIRED",
            "HYBRID_ROUTING_RECOMMENDED",
        ]
    if cells > MERIT_INTERACTIVE_MAX_WINDOW_CELLS or rss_delta > MERIT_INTERACTIVE_MAX_RSS_DELTA_MB:
        flags.append("HYBRID_ROUTING_RECOMMENDED")
        if not scientific_mode:
            flags.append("HYBRID_ROUTING_REQUIRED")
            return "hybrid_required", flags
        return "scientific_allowed", flags
    return "interactive_ok", flags


def _cell_centers(transform: Affine, shape_: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape_)
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _distance_m(lon: float, lat: float, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    _, _, dist = _GEOD.inv(
        np.full(lons.shape, lon, dtype=float),
        np.full(lats.shape, lat, dtype=float),
        lons,
        lats,
    )
    return np.asarray(dist, dtype=float)


def snap_outlet_on_merit(
    lat: float,
    lon: float,
    *,
    upa: np.ndarray,
    transform: Affine,
    wth: np.ndarray | None = None,
    expected_area_km2: float | None = None,
    search_radius_m: float | None = None,
) -> MeritSnap:
    """Snap a pour point to a nearby MERIT drainage cell using UPA/width evidence."""
    upa_arr = np.asarray(upa, dtype=float)
    if upa_arr.ndim != 2:
        raise ValueError("MERIT UPA band must be a 2D array.")
    col_f, row_f = ~transform * (lon, lat)
    row0, col0 = int(round(row_f)), int(round(col_f))
    if not (0 <= row0 < upa_arr.shape[0] and 0 <= col0 < upa_arr.shape[1]):
        raise ValueError("Outlet is outside the fetched MERIT raster window.")

    if search_radius_m is None:
        if expected_area_km2 and expected_area_km2 > 0:
            search_radius_m = min(30_000.0, max(3_000.0, 1_500.0 * math.sqrt(expected_area_km2)))
        else:
            search_radius_m = 3_000.0

    lons, lats = _cell_centers(transform, upa_arr.shape)
    dist = _distance_m(lon, lat, lons, lats)
    finite_upa = np.isfinite(upa_arr) & (upa_arr > 0)
    if wth is not None:
        wth_arr = np.asarray(wth, dtype=float)
        stream = finite_upa & ((wth_arr > 0) | (upa_arr >= 1.0))
    else:
        stream = finite_upa & (upa_arr >= 1.0)
    candidates = stream & (dist <= search_radius_m)
    if not candidates.any():
        candidates = finite_upa & (dist <= search_radius_m)
    if not candidates.any():
        candidates = dist <= search_radius_m

    if expected_area_km2 and expected_area_km2 > 0:
        area_err = np.abs(upa_arr - expected_area_km2) / max(expected_area_km2, 1.0)
        score = area_err + 0.35 * (dist / max(search_radius_m, 1.0))
        quality_base = "area_targeted"
    else:
        # Normalise by the candidates inside the search radius, not the whole
        # window: a window-wide maximum made the snapped cell change when the
        # adaptive window grew (a big river far outside the radius shrank the
        # UPA term for everything inside it).
        upa_norm = np.log1p(np.where(np.isfinite(upa_arr) & (upa_arr > 0), upa_arr, 0.0))
        max_upa = max(float(np.max(upa_norm[candidates])), 1.0)
        score = (dist / max(search_radius_m, 1.0)) - 0.20 * (upa_norm / max_upa)
        quality_base = "nearest_merit_stream"
    score = np.where(candidates, score, np.inf)
    snap_r, snap_c = np.unravel_index(int(np.argmin(score)), score.shape)
    snap_lon, snap_lat = transform * (snap_c + 0.5, snap_r + 0.5)
    snap_dist = float(dist[snap_r, snap_c])
    snap_upa = float(upa_arr[snap_r, snap_c]) if np.isfinite(upa_arr[snap_r, snap_c]) else None
    quality = quality_base
    if snap_dist > 5_000:
        quality = f"{quality_base}_far"
    return MeritSnap(
        lat=float(snap_lat),
        lon=float(snap_lon),
        row=int(snap_r),
        col=int(snap_c),
        distance_m=snap_dist,
        snapped_upa_km2=snap_upa,
        snap_quality=quality,
    )


# The snapped outlet is off the terminal reach when its drainage area is below this fraction of the
# upstream unit catchments' total area (they would all have to drain through it).
OFF_REACH_FRACTION = 0.9

AREA_GATE_RADIUS_M = 1_000.0
AREA_GATE_TOLERANCE = 0.10
AREA_GATE_DISTANT_M = 500.0
AREA_GATE_NO_MATCH_SUFFIX = "_area_gate_no_match"


def snap_outlet_area_gated(
    lat: float,
    lon: float,
    *,
    upa: np.ndarray,
    transform: Affine,
    expected_area_km2: float,
    wth: np.ndarray | None = None,
    gate_radius_m: float = AREA_GATE_RADIUS_M,
    tolerance: float = AREA_GATE_TOLERANCE,
) -> MeritSnap:
    """Snap to the nearest stream cell within ``gate_radius_m`` whose UPA is within ``tolerance`` of ``expected_area_km2``.

    This is deliberately narrow. The wide area-targeted scorer in :func:`snap_outlet_on_merit`
    (radius up to 30 km) moved snaps by up to 7.7 km on small basins and turned right basins wrong
    (camels-delineation-1 replay); GSIM likewise warns that area matching alone can pick a wrong
    channel of similar size under 50 km2. So the search never goes wider than the gate: when no cell
    qualifies the result is the plain no-area snap, unchanged, and the caller flags the mismatch.

    A gated snap has ``snap_quality == "area_gated"``; when the gate found nothing the plain snap's
    quality gets the suffix ``AREA_GATE_NO_MATCH_SUFFIX``.
    """
    if not (expected_area_km2 and expected_area_km2 > 0):
        raise ValueError("snap_outlet_area_gated needs a positive expected_area_km2.")
    upa_arr = np.asarray(upa, dtype=float)
    if upa_arr.ndim != 2:
        raise ValueError("MERIT UPA band must be a 2D array.")
    col_f, row_f = ~transform * (lon, lat)
    row0, col0 = int(round(row_f)), int(round(col_f))
    if not (0 <= row0 < upa_arr.shape[0] and 0 <= col0 < upa_arr.shape[1]):
        raise ValueError("Outlet is outside the fetched MERIT raster window.")
    lons, lats = _cell_centers(transform, upa_arr.shape)
    dist = _distance_m(lon, lat, lons, lats)
    finite_upa = np.isfinite(upa_arr) & (upa_arr > 0)
    if wth is not None:
        stream = finite_upa & ((np.asarray(wth, dtype=float) > 0) | (upa_arr >= 1.0))
    else:
        stream = finite_upa & (upa_arr >= 1.0)
    with np.errstate(invalid="ignore"):
        rel = np.abs(upa_arr - expected_area_km2) / expected_area_km2
    ok = stream & (dist <= gate_radius_m) & (rel <= tolerance)
    if not ok.any():
        plain = snap_outlet_on_merit(lat, lon, upa=upa_arr, transform=transform, wth=wth)
        # same cell as the plain snap; the suffix records that the gate ran and found nothing
        return replace(plain, snap_quality=plain.snap_quality + AREA_GATE_NO_MATCH_SUFFIX)
    # nearest qualifying cell; ties (equal to the metre) go to the closer area match
    key = np.where(ok, np.round(dist, 3) + 1e-6 * rel, np.inf)
    r, c = np.unravel_index(int(np.argmin(key)), key.shape)
    snap_lon, snap_lat = transform * (c + 0.5, r + 0.5)
    return MeritSnap(
        lat=float(snap_lat),
        lon=float(snap_lon),
        row=int(r),
        col=int(c),
        distance_m=float(dist[r, c]),
        snapped_upa_km2=float(upa_arr[r, c]),
        snap_quality="area_gated",
    )


def _cache_dir() -> Path:
    """GEE raster cache, under ``AIHYDRO_HOME`` like the rest of the package."""
    d = aihydro_home() / "cache" / "delineation" / "merit_gee"
    d.mkdir(parents=True, exist_ok=True)
    return d


_EE_READY: str | None = None
_EE_LOCK = threading.Lock()


def _ee_initialize(ee, project_id: str | None) -> None:
    """Initialise Earth Engine once per process (each call re-downloads the API discovery document).

    Thread-safe. Re-initialises when the project id changes; call :func:`reset_ee`
    after credentials change.
    """
    global _EE_READY
    key = project_id or ""
    with _EE_LOCK:
        if _EE_READY == key:
            return
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        _EE_READY = key


def reset_ee() -> None:
    """Forget the cached Earth Engine initialisation (use after credentials or project change)."""
    global _EE_READY
    with _EE_LOCK:
        _EE_READY = None


def native_window(bbox: tuple[float, float, float, float]) -> tuple[Affine, int, int]:
    """Round ``bbox`` out to whole MERIT cells: (transform, ncols, nrows) on the asset's own grid.

    Clamped to the asset's extent (lon -180..180, lat 85..-60); a bbox that does not
    intersect it raises ``ValueError``. No antimeridian wrap: a window is cut at +/-180.
    """
    xmin, ymin, xmax, ymax = bbox
    px = MERIT_NATIVE_PX_DEG
    c0 = max(0, math.floor((xmin - MERIT_ORIGIN_X) / px))
    c1 = min(MERIT_GRID_COLS, math.ceil((xmax - MERIT_ORIGIN_X) / px))
    r0 = max(0, math.floor((MERIT_ORIGIN_Y - ymax) / px))
    r1 = min(MERIT_GRID_ROWS, math.ceil((MERIT_ORIGIN_Y - ymin) / px))
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"bbox {bbox} does not intersect the MERIT Hydro grid (lon -180..180, lat -60..85).")
    tf = Affine(px, 0.0, MERIT_ORIGIN_X + c0 * px, 0.0, -px, MERIT_ORIGIN_Y - r0 * px)
    return tf, int(c1 - c0), int(r1 - r0)


def _matches_native_window(
    transform: Affine, shape_: tuple[int, int], want: Affine, ncols: int, nrows: int
) -> bool:
    return (
        tuple(shape_) == (nrows, ncols)
        and abs(transform.a - want.a) < 1e-12
        and abs(transform.e - want.e) < 1e-12
        and abs(transform.c - want.c) < 1e-9
        and abs(transform.f - want.f) < 1e-9
        and abs(transform.b) < 1e-12
        and abs(transform.d) < 1e-12
    )


def _cache_key(
    *,
    bbox: tuple[float, float, float, float],
    bands: tuple[str, ...],
    dataset: str,
) -> str:
    payload = json.dumps(
        {
            "bbox": [round(v, 7) for v in bbox],
            "bands": bands,
            # "grid" separates native-grid windows from the old resampled
            # (scale=92.77 m, clipped) cache entries, which must not be reused.
            "grid": "merit-native-v1",
            "dataset": dataset,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_bundle(
    key: str, expect: tuple[Affine, int, int] | None = None
) -> MeritRasterBundle | None:
    meta_path = _cache_dir() / f"{key}.json"
    if not meta_path.exists():
        return None
    try:
        import rasterio

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bands: dict[str, np.ndarray] = {}
        for band, rel in meta["band_files"].items():
            with rasterio.open(_cache_dir() / rel) as src:
                bands[band] = src.read(1)
                transform = src.transform
                crs = src.crs.to_string() if src.crs else "EPSG:4326"
                if expect is not None and not _matches_native_window(transform, src.shape, *expect):
                    log.warning("Discarding cached MERIT window %s: shape/transform is not the native window.", key)
                    return None
        return MeritRasterBundle(
            bands=bands,
            transform=transform,
            crs=crs,
            cache_key=key,
            source="gee_cache",
            bbox=tuple(meta["bbox"]),
            resolution_m=float(meta.get("resolution_m", MERIT_SCALE_M)),
        )
    except Exception as exc:
        log.debug("Could not load cached MERIT bundle %s: %s", key, exc)
        return None


def _write_cached_bundle(
    key: str,
    *,
    band_files: dict[str, Path],
    bbox: tuple[float, float, float, float],
    resolution_m: float,
) -> None:
    rels = {band: path.name for band, path in band_files.items()}
    (_cache_dir() / f"{key}.json").write_text(
        json.dumps({"band_files": rels, "bbox": bbox, "resolution_m": resolution_m}, indent=2),
        encoding="utf-8",
    )


def fetch_merit_gee_bbox(
    bbox: tuple[float, float, float, float],
    *,
    bands: tuple[str, ...] = _DEFAULT_BANDS,
    scale_m: float | None = None,
    project_id: str | None = None,
) -> MeritRasterBundle:
    """Fetch MERIT Hydro bands from Earth Engine as cached GeoTIFF rasters on the native grid.

    ``scale_m`` is deprecated and ignored: the grid is never resampled, and the
    returned window is ``bbox`` rounded out to whole MERIT cells.
    """
    if scale_m is not None:
        warnings.warn(
            "scale_m is deprecated and ignored: MERIT Hydro is always fetched on its native grid.",
            DeprecationWarning,
            stacklevel=2,
        )
    # Request the asset's own grid: same crs_transform, whole-cell window, no clip.
    # The earlier request (EPSG:4326 at scale=92.77 m, clipped to a rectangle)
    # forced Earth Engine to resample onto a grid 0.004% off the native cell and
    # failed with "User memory limit exceeded" from about 32 km half-side, while
    # the native-grid request succeeds to 82 km (live, 3 bands).
    transform, ncols, nrows = native_window(bbox)
    if ncols * nrows > MERIT_GEE_MAX_WINDOW_CELLS:
        raise MeritGeeWindowLimit(
            f"MERIT_GEE_WINDOW_LIMIT: the basin needs a {ncols}x{nrows}-cell "
            f"({ncols * nrows / 1e6:.1f}M cells) MERIT window, above the synchronous "
            f"Earth Engine limit of {MERIT_GEE_MAX_WINDOW_CELLS / 1e6:.1f}M cells. "
            "Use the MERIT-Basins hybrid or a staged regional flowdir (local_merit)."
        )
    bbox = (
        transform.c,
        transform.f + nrows * transform.e,
        transform.c + ncols * transform.a,
        transform.f,
    )
    key = _cache_key(bbox=bbox, bands=bands, dataset=MERIT_GEE_DATASET)
    cached = _load_cached_bundle(key, expect=(transform, ncols, nrows))
    if cached is not None:
        return cached

    try:
        import ee  # type: ignore
        import requests
        import rasterio
    except Exception as exc:
        raise RuntimeError(
            "MERIT GEE delineation requires earthengine-api, requests, and rasterio. "
            "Install with `pip install aihydro-tools[delineation,gee]`."
        ) from exc

    try:
        _ee_initialize(ee, project_id)
    except Exception as exc:
        raise RuntimeError(
            "Google Earth Engine is not initialized. Run `aihydro-gee status`, "
            "select a registered project, or call delineation_doctor()."
        ) from exc

    image = ee.Image(MERIT_GEE_DATASET).select(list(bands))
    url = image.getDownloadURL(
        {
            "name": f"aihydro_merit_{key[:12]}",
            "crs": "EPSG:4326",
            "crs_transform": [transform.a, 0, transform.c, 0, transform.e, transform.f],
            "dimensions": f"{ncols}x{nrows}",
            "filePerBand": True,
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=120)
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(
            f"GEE MERIT download failed with HTTP {response.status_code}: {detail}. "
            "This usually means the synchronous raster window exceeded Earth Engine "
            "memory/size limits; use a smaller basin/window or an async export path."
        )

    band_files: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="aihydro_merit_gee_") as tmp:
        zip_path = Path(tmp) / "merit.zip"
        zip_path.write_bytes(response.content)
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            for band in bands:
                matches = [
                    p
                    for p in Path(tmp).glob("*.tif")
                    if band in p.stem.split(".")[-1] or p.stem.endswith(band)
                ]
                if not matches:
                    matches = list(Path(tmp).glob(f"*{band}*.tif"))
                if not matches:
                    raise RuntimeError(f"GEE MERIT download did not include band {band}.")
                dest = _cache_dir() / f"{key}_{band}.tif"
                dest.write_bytes(matches[0].read_bytes())
                band_files[band] = dest
        else:
            tif_path = Path(tmp) / "merit.tif"
            tif_path.write_bytes(response.content)
            with rasterio.open(tif_path) as src:
                if src.count < len(bands):
                    raise RuntimeError(
                        f"GEE MERIT download returned {src.count} band(s), expected {len(bands)}."
                    )
                for idx, band in enumerate(bands, start=1):
                    dest = _cache_dir() / f"{key}_{band}.tif"
                    profile = src.profile.copy()
                    profile.update(count=1)
                    with rasterio.open(dest, "w", **profile) as dst:
                        dst.write(src.read(idx), 1)
                    band_files[band] = dest

    want = (transform, ncols, nrows)
    with rasterio.open(band_files[bands[0]]) as src:
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
    loaded: dict[str, np.ndarray] = {}
    for band, path in band_files.items():
        with rasterio.open(path) as src:
            if not _matches_native_window(src.transform, src.shape, *want):
                raise RuntimeError(
                    f"GEE MERIT {band} came back as {src.shape} with transform {tuple(src.transform)[:6]}, "
                    f"not the requested native window {(nrows, ncols)} at {tuple(want[0])[:6]}; "
                    "refusing a grid that differs from the asset's."
                )
            loaded[band] = src.read(1)
    _write_cached_bundle(key, band_files=band_files, bbox=bbox, resolution_m=MERIT_SCALE_M)
    return MeritRasterBundle(
        bands=loaded,
        transform=transform,
        crs=crs,
        cache_key=key,
        source="gee",
        bbox=bbox,
        resolution_m=MERIT_SCALE_M,
    )


def merit_resolve_pfaf_region(lat: float, lon: float) -> str:
    """Resolve the MERIT Pfafstetter level-2 region for a point."""
    from aihydro_watershed.merit.merit_manager import MeritDataManager

    return MeritDataManager().resolve_pfaf_region(lat, lon)


def merit_check_routing_region_cache(pfaf_region: str) -> dict[str, Any]:
    """Return flowdir-first regional cache status for a Pfaf region."""
    from dataclasses import asdict

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    return asdict(MeritDataManager().routing_region_cache(pfaf_region))


def merit_ensure_routing_region(
    lat: float | None = None,
    lon: float | None = None,
    *,
    pfaf_region: str | None = None,
    required_assets: tuple[str, ...] = ("flowdir",),
    acquisition_policy: str = "check_only",
) -> dict[str, Any]:
    """Ensure/report regional MERIT routing assets without requiring accumulation."""
    from dataclasses import asdict

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    status = MeritDataManager().ensure_routing_region(
        lat,
        lon,
        pfaf_region=pfaf_region,
        required_assets=required_assets,
        acquisition_policy=acquisition_policy,
    )
    return asdict(status)


def merit_check_basins_region_cache(pfaf_region: str) -> dict[str, Any]:
    """Return MERIT-Basins vector/topology cache status for a Pfaf region."""
    from dataclasses import asdict

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    return asdict(MeritDataManager().basins_region_cache(pfaf_region))


def merit_ensure_basins_region(
    pfaf_region: str,
    *,
    acquisition_policy: str = "check_only",
    force_retry: bool = False,
) -> dict[str, Any]:
    """Ensure/report regional MERIT-Basins vectors for hybrid routing (``force_retry`` overrides the 24 h integrity back-off)."""
    from dataclasses import asdict

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    return asdict(
        MeritDataManager().ensure_basins_region(
            pfaf_region,
            acquisition_policy=acquisition_policy,
            force_retry=force_retry,
        )
    )


def merit_get_snap_reference(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    project_id: str | None = None,
    half_side_km: float = 8.0,
    area_gate: bool = False,
) -> MeritSnapReference:
    """Fetch a tiny GEE MERIT upa/wth window and snap to an official drainage cell.

    With ``area_gate`` and an ``expected_area_km2`` the snap is :func:`snap_outlet_area_gated` (1 km,
    +/-10%) on the same small window; otherwise an expected area selects the wide area-targeted scorer.
    """
    gated = bool(area_gate and expected_area_km2 and expected_area_km2 > 0)
    if expected_area_km2 and expected_area_km2 > 0 and not gated:
        snap_radius_km = min(30.0, max(3.0, 1.5 * math.sqrt(expected_area_km2)))
        half_side_km = max(half_side_km, snap_radius_km * 1.2)
    bbox = bbox_from_point(lat, lon, half_side_km)
    bundle = fetch_merit_gee_bbox(
        bbox,
        bands=_SNAP_REFERENCE_BANDS,
        project_id=project_id,
    )
    if gated:
        snap = snap_outlet_area_gated(
            lat,
            lon,
            upa=np.asarray(bundle.bands["upa"], dtype=float),
            wth=bundle.bands.get("wth"),
            transform=bundle.transform,
            expected_area_km2=float(expected_area_km2),
        )
    else:
        snap = snap_outlet_on_merit(
            lat,
            lon,
            upa=np.asarray(bundle.bands["upa"], dtype=float),
            wth=bundle.bands.get("wth"),
            transform=bundle.transform,
            expected_area_km2=expected_area_km2,
        )
    return MeritSnapReference(
        snap=snap,
        source=bundle.source,
        cache_key=bundle.cache_key,
        bbox=bundle.bbox,
        official_merit_upa_km2=snap.snapped_upa_km2,
    )


def _polygonize_mask(mask: np.ndarray, transform: Affine, crs: str) -> gpd.GeoDataFrame:
    import rasterio.features

    geoms = [
        shape(geom)
        for geom, value in rasterio.features.shapes(
            mask.astype(np.uint8), mask=mask.astype(bool), transform=transform
        )
        if int(value) == 1
    ]
    if not geoms:
        return gpd.GeoDataFrame({"catchment": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame({"catchment": [1] * len(geoms)}, geometry=geoms, crs=crs)
    return gdf.dissolve(by="catchment").reset_index(drop=True).to_crs(4326)


def delineate_merit_bundle(
    bundle: MeritRasterBundle,
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
) -> MeritFlowdirResult:
    """Delineate a basin from an already-fetched MERIT raster bundle."""
    try:
        import pyflwdir
    except Exception as exc:
        raise RuntimeError(
            "pyflwdir is required for MERIT flow-direction delineation. "
            "Install with `pip install aihydro-tools[delineation]`."
        ) from exc

    d8 = merit_dir_to_pyflwdir(bundle.bands["dir"])
    upa = np.asarray(bundle.bands["upa"], dtype=float)
    wth = bundle.bands.get("wth")
    snap = snap_outlet_on_merit(
        lat,
        lon,
        upa=upa,
        wth=wth,
        transform=bundle.transform,
        expected_area_km2=expected_area_km2,
    )
    mask = np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128])
    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=mask,
        transform=bundle.transform,
        latlon=bundle.crs.upper() in ("EPSG:4326", "OGC:CRS84"),
    )
    basin_ids = flw.basins(xy=([snap.lon], [snap.lat]), ids=np.array([1], dtype=np.uint32))
    basin_mask = basin_ids == 1
    touches = basin_touches_edge(basin_mask)
    gdf = _polygonize_mask(basin_mask, bundle.transform, bundle.crs)
    area = area_km2(gdf)
    validation, flags = validate_area(
        area,
        snapped_upa_km2=snap.snapped_upa_km2,
        expected_area_km2=expected_area_km2,
        touches_edge=touches,
    )
    snap_validation, snap_flags = validate_snap(snap, expected_area_km2=expected_area_km2)
    flags.extend(flag for flag in snap_flags if flag not in flags)
    return MeritFlowdirResult(
        gdf=gdf,
        area_km2=area,
        outlet_lat=snap.lat,
        outlet_lon=snap.lon,
        snap_distance_m=snap.distance_m,
        snapped_upa_km2=snap.snapped_upa_km2,
        snap_quality=snap.snap_quality,
        snap_validation=snap_validation,
        cache_key=bundle.cache_key,
        bbox=bundle.bbox,
        scout_box_maxed=False,
        touches_edge=touches,
        area_validation=validation,
        quality_flags=flags,
        source=bundle.source,
        routing_dataset=MERIT_GEE_DATASET,
        routing_resolution_m=bundle.resolution_m,
        routing_data_source=bundle.source,
        snap_source=bundle.source,
        official_merit_upa_km2=snap.snapped_upa_km2,
        polygon_area_km2=area,
        validation_sources=["gee_official_upa"],
        regional_flowdir_cached=False,
    )


def delineate_merit_gee(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    project_id: str | None = None,
    initial_half_side_km: float = 20.0,
    max_half_side_km: float = 1200.0,
) -> MeritFlowdirResult:
    """Fetch MERIT Hydro from GEE adaptively and delineate with pyflwdir."""
    half_side = initial_half_side_km
    last: MeritFlowdirResult | None = None
    while half_side <= max_half_side_km:
        bbox = bbox_from_point(lat, lon, half_side)
        bundle = fetch_merit_gee_bbox(bbox, project_id=project_id)
        result = delineate_merit_bundle(bundle, lat, lon, expected_area_km2=expected_area_km2)
        last = result
        if not result.touches_edge:
            return result
        half_side *= 1.6
    if last is None:
        raise RuntimeError("MERIT GEE delineation did not produce a basin.")
    flags = list(last.quality_flags)
    if "SCOUT_WINDOW_MAXED" not in flags:
        flags.append("SCOUT_WINDOW_MAXED")
    return MeritFlowdirResult(
        **{**last.__dict__, "scout_box_maxed": True, "quality_flags": flags}
    )


def local_merit_flowdir_pyflwdir(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    snap_reference: MeritSnapReference | None = None,
    allow_offline: bool = False,
    compute_local_upstream_area: bool = False,
    initial_half_side_km: float | None = None,
    max_half_side_km: float = 1200.0,
    force_offline: bool = False,
    scientific_mode: bool = False,
) -> MeritFlowdirResult:
    """Delineate from a cached regional MERIT flowdir raster only."""
    started = time.perf_counter()
    rss_before = _rss_memory_mb()
    rss_after_window_read = rss_before
    rss_after_pyflwdir_build = rss_before
    rss_after_basin_mask = rss_before
    rss_after_vectorization = rss_before
    import rasterio
    from rasterio.windows import from_bounds

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    mgr = MeritDataManager()
    status = mgr.ensure_basin(lat, lon, download=False)
    pfaf = status.pfaf_code
    flowdir_path = mgr.flowdir_path(pfaf)
    if not flowdir_path:
        raise FileNotFoundError(f"Local MERIT flow direction raster missing for basin {pfaf}.")
    flowdir_size = flowdir_path.stat().st_size if flowdir_path.exists() else None

    snap_flags: list[str] = []
    if snap_reference is None:
        try:
            if force_offline:
                raise RuntimeError("forced offline local routing")
            snap_reference = merit_get_snap_reference(
                lat,
                lon,
                expected_area_km2=expected_area_km2,
            )
        except Exception:
            if not allow_offline:
                raise
            snap_reference = MeritSnapReference(
                snap=MeritSnap(
                    lat=lat,
                    lon=lon,
                    row=-1,
                    col=-1,
                    distance_m=0.0,
                    snapped_upa_km2=None,
                    snap_quality="offline_unsnapped",
                ),
                source="offline",
                cache_key="offline",
                bbox=(lon, lat, lon, lat),
                official_merit_upa_km2=None,
            )
            snap_flags.append("GEE_VALIDATION_UNAVAILABLE")

    try:
        import pyflwdir
    except Exception as exc:
        raise RuntimeError(
            "pyflwdir is required for local MERIT flow-direction delineation. "
            "Install with `pip install aihydro-tools[delineation]`."
        ) from exc

    snap = snap_reference.snap
    if initial_half_side_km is None:
        area_hint = snap_reference.official_merit_upa_km2 or expected_area_km2 or 100.0
        initial_half_side_km = min(120.0, max(20.0, 1.2 * math.sqrt(max(area_hint, 1.0))))

    local_upstream_area_km2: float | None = None
    validation_sources: list[str] = []
    if snap_reference.official_merit_upa_km2 is not None:
        validation_sources.append("gee_official_upa")

    half_side = initial_half_side_km
    last_error: str | None = None
    expansion_iterations = 0
    final_cell_count = 0
    with rasterio.open(flowdir_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        src_bounds = src.bounds
        result_bbox = (src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top)
        while half_side <= max_half_side_km:
            bbox = bbox_from_point(snap.lat, snap.lon, half_side)
            left = max(src_bounds.left, bbox[0])
            bottom = max(src_bounds.bottom, bbox[1])
            right = min(src_bounds.right, bbox[2])
            top = min(src_bounds.top, bbox[3])
            if left >= right or bottom >= top:
                raise ValueError("Snapped outlet is outside the cached regional flowdir bounds.")
            window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
            result_bbox = (left, bottom, right, top)
            final_cell_count = int(window.width * window.height)
            if (
                final_cell_count > MERIT_INTERACTIVE_MAX_WINDOW_CELLS
                and not scientific_mode
            ):
                raise RuntimeError(
                    "HYBRID_ROUTING_REQUIRED HYBRID_ROUTING_RECOMMENDED "
                    f"safe_envelope_version={MERIT_SAFE_ENVELOPE_VERSION}: requested local "
                    f"flowdir window has {final_cell_count} cells, exceeding the "
                    f"interactive limit {MERIT_INTERACTIVE_MAX_WINDOW_CELLS}."
                )
            if final_cell_count > MERIT_SCIENTIFIC_MAX_WINDOW_CELLS:
                raise RuntimeError(
                    "HYBRID_ROUTING_REQUIRED LOCAL_MEMORY_THRESHOLD_EXCEEDED "
                    f"safe_envelope_version={MERIT_SAFE_ENVELOPE_VERSION}: requested local "
                    f"flowdir window has {final_cell_count} cells, exceeding the "
                    f"scientific limit {MERIT_SCIENTIFIC_MAX_WINDOW_CELLS}."
                )
            if final_cell_count > _MAX_LOCAL_FLOWDIR_CELLS:
                raise RuntimeError(
                    "LOCAL_MEMORY_THRESHOLD_EXCEEDED ADAPTIVE_WINDOW_LIMIT_REACHED "
                    "HYBRID_ROUTING_RECOMMENDED: requested local flowdir window exceeds "
                    "the current raster-only processing envelope."
                )
            dir_arr = src.read(1, window=window)
            rss_after_window_read = _rss_memory_mb()
            transform = src.window_transform(window)
            d8 = merit_dir_to_pyflwdir(dir_arr)
            mask = np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128])
            try:
                flw = pyflwdir.from_array(
                    d8,
                    ftype="d8",
                    mask=mask,
                    transform=transform,
                    latlon=crs.upper() in ("EPSG:4326", "OGC:CRS84"),
                )
                rss_after_pyflwdir_build = _rss_memory_mb()
                basin_ids = flw.basins(
                    xy=([snap.lon], [snap.lat]), ids=np.array([1], dtype=np.uint32)
                )
                rss_after_basin_mask = _rss_memory_mb()
            except Exception as exc:
                last_error = str(exc)
                half_side *= 1.6
                continue
            basin_mask = basin_ids == 1
            touches = basin_touches_edge(basin_mask)
            gdf = _polygonize_mask(basin_mask, transform, crs)
            rss_after_vectorization = _rss_memory_mb()
            area = area_km2(gdf)
            if compute_local_upstream_area:
                col_f, row_f = ~transform * (snap.lon, snap.lat)
                row, col = int(round(row_f)), int(round(col_f))
                uparea = flw.upstream_area(unit="km2")
                if 0 <= row < uparea.shape[0] and 0 <= col < uparea.shape[1]:
                    local_upstream_area_km2 = float(uparea[row, col])
                    if "local_pyflwdir_upstream_area" not in validation_sources:
                        validation_sources.append("local_pyflwdir_upstream_area")
            if not touches:
                break
            half_side *= 1.6
            expansion_iterations += 1
        else:
            raise RuntimeError(
                "HYBRID_ROUTING_REQUIRED LARGE_BASIN_HYBRID_REQUIRED ADAPTIVE_WINDOW_LIMIT_REACHED "
                "HYBRID_ROUTING_RECOMMENDED: local flowdir window reached maximum "
                f"extent without containing the full basin. Last error: {last_error or 'basin touches edge'}"
            )

    area_validation, flags = validate_area(
        area,
        snapped_upa_km2=snap_reference.official_merit_upa_km2,
        expected_area_km2=expected_area_km2,
        touches_edge=touches,
    )
    snap_validation, more_snap_flags = validate_snap(snap, expected_area_km2=expected_area_km2)
    flags.extend(flag for flag in more_snap_flags if flag not in flags)
    flags.extend(flag for flag in snap_flags if flag not in flags)
    if expansion_iterations > 0 and "ADAPTIVE_WINDOW_EXPANDED" not in flags:
        flags.append("ADAPTIVE_WINDOW_EXPANDED")
    if touches and "BASIN_TOUCHES_ROUTING_WINDOW_BOUNDARY" not in flags:
        flags.append("BASIN_TOUCHES_ROUTING_WINDOW_BOUNDARY")
    peak_memory = _peak_memory_mb()
    rss_peak = max(
        rss_before,
        rss_after_window_read,
        rss_after_pyflwdir_build,
        rss_after_basin_mask,
        rss_after_vectorization,
    )
    rss_delta = max(0.0, rss_peak - rss_before)
    if (
        rss_delta > _LOCAL_ROUTING_MEMORY_RISK_MB
        or final_cell_count > int(_MAX_LOCAL_FLOWDIR_CELLS * 0.75)
    ) and "LOCAL_ROUTING_MEMORY_RISK" not in flags:
        flags.append("LOCAL_ROUTING_MEMORY_RISK")
    window_complete = not touches
    envelope_status, envelope_flags = classify_safe_envelope(
        final_window_cell_count=final_cell_count,
        rss_delta_mb=rss_delta,
        window_complete=window_complete,
        scientific_mode=scientific_mode,
    )
    for flag in envelope_flags:
        if flag not in flags:
            flags.append(flag)
    if envelope_status == "hybrid_required":
        raise RuntimeError(
            "HYBRID_ROUTING_REQUIRED HYBRID_ROUTING_RECOMMENDED "
            f"safe_envelope_version={MERIT_SAFE_ENVELOPE_VERSION}: adaptive local routing "
            f"outside safe envelope (cells={final_cell_count}, rss_delta_mb={rss_delta:.3f}, "
            f"window_complete={window_complete})."
        )
    key = hashlib.sha256(
        f"{flowdir_path}:{snap_reference.cache_key}:{compute_local_upstream_area}".encode("utf-8")
    ).hexdigest()
    return MeritFlowdirResult(
        gdf=gdf,
        area_km2=area,
        outlet_lat=snap.lat,
        outlet_lon=snap.lon,
        snap_distance_m=snap.distance_m,
        snapped_upa_km2=snap_reference.official_merit_upa_km2,
        snap_quality=snap.snap_quality,
        snap_validation=snap_validation,
        cache_key=key,
        bbox=result_bbox,
        scout_box_maxed=False,
        touches_edge=touches,
        area_validation=area_validation,
        quality_flags=flags,
        source="local_merit_flowdir",
        routing_dataset=MERIT_GEE_DATASET,
        routing_resolution_m=MERIT_SCALE_M,
        pfaf_region=pfaf,
        routing_data_source="local_flowdir",
        raster_dataset_version=MERIT_FLOWDIR_RASTER_VERSION.format(pfaf=pfaf),
        raster_data_source=MERIT_FLOWDIR_RASTER_SOURCE.format(pfaf=pfaf),
        snap_source=snap_reference.source,
        official_merit_upa_km2=snap_reference.official_merit_upa_km2,
        local_upstream_area_km2=local_upstream_area_km2,
        polygon_area_km2=area,
        validation_sources=validation_sources,
        regional_flowdir_cached=True,
        execution_mode="local_staged_flowdir_adaptive_window",
        regional_flowdir_file_size_bytes=flowdir_size,
        window_expansion_iterations=expansion_iterations,
        final_window_bounds={
            "xmin": float(result_bbox[0]),
            "ymin": float(result_bbox[1]),
            "xmax": float(result_bbox[2]),
            "ymax": float(result_bbox[3]),
        },
        final_window_cell_count=final_cell_count,
        basin_touched_window_boundary=touches,
        window_complete=window_complete,
        peak_memory_mb=round(peak_memory, 3),
        runtime_seconds=round(time.perf_counter() - started, 3),
        memory_telemetry={
            "rss_before_mb": round(rss_before, 3),
            "rss_after_window_read_mb": round(rss_after_window_read, 3),
            "rss_after_pyflwdir_build_mb": round(rss_after_pyflwdir_build, 3),
            "rss_after_basin_mask_mb": round(rss_after_basin_mask, 3),
            "rss_after_vectorization_mb": round(rss_after_vectorization, 3),
            "rss_peak_mb": round(rss_peak, 3),
            "rss_delta_mb": round(rss_delta, 3),
        },
        safe_envelope_version=MERIT_SAFE_ENVELOPE_VERSION,
    )


def delineate_local_merit(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
) -> MeritFlowdirResult:
    """Backward-compatible local MERIT entrypoint using flowdir-only routing."""
    return local_merit_flowdir_pyflwdir(
        lat,
        lon,
        expected_area_km2=expected_area_km2,
    )


def merit_basins_hybrid_delineate(
    lat: float,
    lon: float,
    *,
    expected_area_km2: float | None = None,
    snap_reference: MeritSnapReference | None = None,
    allow_offline: bool = False,
    force_offline: bool = False,
    auto_stage: bool = True,
) -> MeritFlowdirResult:
    """
    Delineate with MERIT-Basins unit catchments plus terminal-catchment raster refinement.

    The vector topology (``nextdown`` traversal of the pinned, hash-verified ``basins##.db``) provides
    the upstream assembly. Only the terminal unit catchment is refined on a small raster window, read
    from the staged regional rasters when present or from one small Earth Engine request otherwise.
    Snapping uses the staged ``accum##.tif`` when present (same provider as the vectors); otherwise the
    Earth Engine reference, which is recorded as ``MERIT_VERSION_MIX``. A file that fails its pinned
    SHA-256 is never used (``MERIT_ASSET_INTEGRITY_FAILED``).

    ``snap_reference`` is honoured only when no accumulation raster is staged.
    """
    import pandas as pd

    from aihydro_watershed.delineation import merit_hybrid as mh
    from aihydro_watershed.merit import verified_fetch as vf
    from aihydro_watershed.merit.basins_store import BasinsStore
    from aihydro_watershed.merit.merit_manager import MeritDataManager, _merit_auto_stage_policy

    started = time.perf_counter()
    rss_before = _rss_memory_mb()
    fallback_history: list[dict[str, Any]] = []
    quality_flags: list[str] = ["HYBRID_ROUTING_USED"]
    validation_sources: list[str] = []

    def _flag(name: str) -> None:
        if name not in quality_flags:
            quality_flags.append(name)

    mgr = MeritDataManager()
    stage_policy = _merit_auto_stage_policy()[0] if auto_stage else "check_only"
    downloaded_names: list[str] = []
    try:
        # -- region + pinned vectors (level-2 index, basins##.db, rivers##.db) --------------------
        try:
            pfaf = mgr.resolve_pfaf_region(lat, lon)
        except FileNotFoundError as exc:
            if not auto_stage or stage_policy == "never":
                raise RuntimeError(f"MERIT_BASINS_NOT_STAGED HYBRID_ROUTING_REQUIRED: {exc}") from exc
            if mgr.ensure_level2_index(download=True) is not None:
                downloaded_names.append("megabasins.db")
            pfaf = mgr.resolve_pfaf_region(lat, lon)
        status = (
            mgr.auto_stage_vectors(
                pfaf, progress_cb=lambda msg, done, tot: log.info("MERIT-Basins staging [Pfaf %s] %s", pfaf, msg)
            )
            if auto_stage
            else mgr.basins_region_cache(pfaf)
        )
        downloaded_names += status.downloaded or []
        if status.integrity_error:
            raise vf.AssetIntegrityError(
                status.integrity_error.removeprefix(f"{vf.AssetIntegrityError.code}: ")
            )
        if not status.catchments_ready:
            raise RuntimeError(
                f"MERIT_BASINS_NOT_STAGED HYBRID_ROUTING_REQUIRED: {status.message}"
            )
        basins_db = mgr.verified_asset("basins", pfaf)
        rivers_db = mgr.verified_asset("rivers", pfaf)
        flowdir_path = mgr.verified_asset("flowdir", pfaf)
        accum_path = mgr.verified_asset("accum", pfaf)
    except vf.AssetIntegrityError as exc:
        _flag("MERIT_ASSET_INTEGRITY_FAILED")
        raise RuntimeError(str(exc)) from exc
    store = BasinsStore(basins_db, rivers_db)

    # -- snap -----------------------------------------------------------------------------------
    snap_origin = mh.ORIGIN_NONE
    caller_supplied_snap = snap_reference is not None
    if accum_path is not None:
        caller_supplied_snap = False  # a staged accumulation raster overrides any supplied reference
        snap = mh.snap_on_local_accum(
            accum_path, lat, lon, expected_area_km2=expected_area_km2, area_gate=True
        )
        snap_reference = MeritSnapReference(
            snap=snap,
            source="local_accum_mghydro",
            cache_key=f"local_accum:{pfaf}",
            bbox=(lon, lat, lon, lat),
            official_merit_upa_km2=snap.snapped_upa_km2,
        )
        snap_origin = mh.ORIGIN_LOCAL
    else:
        if snap_reference is None:
            try:
                if force_offline:
                    raise RuntimeError("forced offline hybrid routing")
                snap_reference = merit_get_snap_reference(
                    lat, lon, expected_area_km2=expected_area_km2, area_gate=True
                )
            except Exception:
                if not allow_offline:
                    raise
                snap_reference = MeritSnapReference(
                    snap=MeritSnap(
                        lat=lat, lon=lon, row=-1, col=-1, distance_m=0.0,
                        snapped_upa_km2=None, snap_quality="offline_unsnapped",
                    ),
                    source="offline",
                    cache_key="offline",
                    bbox=(lon, lat, lon, lat),
                    official_merit_upa_km2=None,
                )
                _flag("GEE_VALIDATION_UNAVAILABLE")
        if snap_reference.source != "offline":
            snap_origin = mh.ORIGIN_EE
    snap = snap_reference.snap
    unsnapped = snap.snap_quality == "offline_unsnapped"
    area_gate_info: dict[str, Any] | None = None
    if expected_area_km2 and expected_area_km2 > 0 and not unsnapped:
        # A gated snap is tagged "area_gated". The gate was evaluated only when this function made the snap
        # (local accum, or a reference it fetched itself with ``area_gate=True``) or the snap carries the tag.
        # A caller-supplied reference made with another scorer was never gated: say so, and raise nothing.
        gate_hit = snap.snap_quality == "area_gated"
        gate_nomatch = snap.snap_quality.endswith(AREA_GATE_NO_MATCH_SUFFIX)
        gate_evaluated = gate_hit or gate_nomatch or not caller_supplied_snap
        area_gate_info = {
            "gate_applied": gate_evaluated,
            "radius_m": AREA_GATE_RADIUS_M,
            "tolerance": AREA_GATE_TOLERANCE,
            "expected_area_km2": expected_area_km2,
            "matched": gate_hit if gate_evaluated else None,
        }
        if gate_hit and snap.distance_m > AREA_GATE_DISTANT_M:
            # the gate accepts up to 1 km; a match this far from the gauge may be another river of similar size
            _flag("AREA_GATE_SNAP_DISTANT")
        if gate_evaluated and not gate_hit:
            _flag("AREA_GATE_NO_MATCH")
            snap_upa = snap.snapped_upa_km2
            if snap_upa and abs(snap_upa - expected_area_km2) / expected_area_km2 > AREA_GATE_TOLERANCE:
                _flag("SNAP_UPA_DIFFERS_FROM_EXPECTED")
            fallback_history.append(
                {
                    "method": "area_gated_snap",
                    "outcome": "no_match",
                    "reason": (
                        f"no stream cell within {AREA_GATE_RADIUS_M:.0f} m has an upstream area within "
                        f"{AREA_GATE_TOLERANCE:.0%} of {expected_area_km2:g} km2; the plain snap is kept"
                    ),
                }
            )
    if snap_reference.official_merit_upa_km2 is not None:
        validation_sources.append(
            "local_accum_upa" if snap_origin == mh.ORIGIN_LOCAL else "gee_official_upa"
        )

    # -- vector assembly --------------------------------------------------------------------------
    terminal_id = store.locate(snap.lon, snap.lat)
    upstream_ids = store.upstream_ids(terminal_id)
    upstream = store.geometries(upstream_ids)
    if upstream.empty:
        raise RuntimeError(f"Hybrid topology returned no upstream catchments for {terminal_id}.")
    missing_ids = upstream.attrs.get("missing_comids") or []
    if missing_ids:
        _flag("VECTOR_ASSEMBLY_INCOMPLETE")
        if terminal_id in missing_ids:
            raise RuntimeError(f"Terminal catchment {terminal_id} has no polygon in {basins_db.name}.")
    rss_after_vector_load = _rss_memory_mb()
    terminal = upstream[upstream["comid"] == terminal_id][["geometry"]].copy()
    nonterminal = upstream[upstream["comid"] != terminal_id][["geometry"]].copy()
    vector_gdf = upstream[["geometry"]].dissolve().reset_index(drop=True)
    vector_area = area_km2(vector_gdf)
    rss_after_vector_assembly = _rss_memory_mb()
    # Self-consistency: the assembled catchment areas against MERIT-Basins' own cumulative area.
    catchment_area_sum = float(upstream["area_km2"].sum()) if "area_km2" in upstream else None
    official_uparea = store.river_uparea_km2(terminal_id)
    if official_uparea and catchment_area_sum:
        rel = abs(catchment_area_sum - official_uparea) / official_uparea
        if rel > 0.02:
            _flag("VECTOR_ASSEMBLY_UPAREA_MISMATCH")
        validation_sources.append("merit_basins_rivers_uparea")
    else:
        rel = None

    # Does the snapped outlet sit on the terminal catchment's main reach? The upstream unit catchments
    # drain to the terminal's downstream end. An outlet on a side tributary inside the terminal catchment
    # is cut off from them: its basin cannot contain their area, so unioning them in over-counts it
    # (camels-delineation-1: 11 small basins up to 117x too large). Its drainage area is then smaller than
    # the inflow the upstream catchments would add.
    inflow_km2 = (
        float(upstream.loc[upstream["comid"] != terminal_id, "area_km2"].sum())
        if "area_km2" in upstream and not nonterminal.empty
        else (area_km2(nonterminal) if not nonterminal.empty else 0.0)
    )
    terminal_total_km2 = area_km2(terminal)
    snap_upa_vec = snap_reference.official_merit_upa_km2
    off_reach = bool(
        not unsnapped
        and inflow_km2 > 0
        and snap_upa_vec is not None
        and snap_upa_vec < OFF_REACH_FRACTION * inflow_km2
    )

    # -- terminal refinement ------------------------------------------------------------------------
    refine_origin = mh.ORIGIN_NONE
    terminal_refinement_used = False
    terminal_component = terminal
    refinement_area: float | None = None
    refine_info: dict[str, Any] = {}
    if unsnapped:
        _flag("TERMINAL_CATCHMENT_REFINEMENT_SKIPPED_UNSNAPPED")
        fallback_history.append(
            {
                "method": "terminal_refinement",
                "outcome": "skipped",
                "reason": "outlet could not be snapped; the whole terminal catchment is kept",
            }
        )
    else:
        refine_label = "terminal_local_merit_flowdir_refinement"
        try:
            if flowdir_path is not None:
                terminal_component, refine_info = mh.refine_local(terminal, flowdir_path, snap)
            elif force_offline:
                raise RuntimeError("TERMINAL_CATCHMENT_REFINEMENT_FAILED: offline and no flowdir staged")
            else:
                refine_label = "terminal_gee_window_refinement"
                terminal_component, refine_info = mh.refine_ee(terminal, snap)
            refine_origin = refine_info["origin"]
            terminal_refinement_used = True
            refinement_area = area_km2(terminal_component.to_crs(4326))
            fallback_history.append({"method": refine_label, "outcome": "succeeded"})
        except Exception as exc:
            _flag("TERMINAL_CATCHMENT_REFINEMENT_FAILED")
            if "WINDOW_TOO_LARGE" in str(exc):
                _flag("TERMINAL_REFINEMENT_WINDOW_TOO_LARGE")
            fallback_history.append(
                {"method": refine_label, "outcome": "failed", "reason": str(exc)}
            )
    rss_after_terminal_refinement = _rss_memory_mb()

    consistency = mh.version_consistency(
        vector_version=MERIT_BASINS_VECTOR_VERSION, snap_origin=snap_origin, refine_origin=refine_origin
    )
    if consistency["mixed"]:
        _flag("MERIT_VERSION_MIX")

    # -- outlet / polygon consistency ---------------------------------------------------------------------
    # Local-only: the polygon must be the basin of the snapped outlet, not the main stem's. That holds when
    # the outlet is off the terminal reach (no upstream catchments reach it) or when the terminal catchment is
    # a headwater; in both the polygon is the terminal-catchment component alone.
    outlet_assembly: dict[str, Any] = {"mode": "union_upstream", "off_terminal_reach": off_reach}
    local_only = off_reach or nonterminal.empty
    keep_upstream = not off_reach
    if off_reach:
        _flag("OUTLET_OFF_TERMINAL_REACH")
        outlet_assembly.update(
            {
                "upstream_inflow_km2": round(inflow_km2, 3),
                "terminal_catchment_area_km2": round(terminal_total_km2, 3),
                "snapped_upa_km2": round(float(snap_upa_vec), 3),
                "mainstem_alternative": {
                    "area_km2": round(official_uparea, 3) if official_uparea else round(vector_area, 3),
                    "assembly_area_km2": round(vector_area, 3),
                    "source": "merit_basins_rivers_uparea" if official_uparea else "vector_assembly",
                    "note": (
                        "The snapped outlet is on a tributary inside the terminal unit catchment, so the "
                        f"{len(upstream_ids) - 1} upstream unit catchment(s) ({inflow_km2:.1f} km2) do not drain "
                        "to it. The polygon is the local catchment only; the area above is the main-stem "
                        "basin at the terminal catchment's downstream end, which a gauge on the main reach "
                        "would have."
                    ),
                },
            }
        )
    if local_only and not terminal_refinement_used and snap_upa_vec and not unsnapped:
        # Refinement failed (any reason) and the outlet drains only part of the terminal catchment: do
        # not return the whole terminal catchment. Delineate on a window centred on the outlet instead.
        if snap_upa_vec < OFF_REACH_FRACTION * terminal_total_km2:
            try:
                terminal_component, win_info = mh.delineate_on_local_window(
                    snap, snapped_upa_km2=float(snap_upa_vec)
                )
                refine_origin = win_info["origin"]
                terminal_refinement_used = True
                refinement_area = area_km2(terminal_component.to_crs(4326))
                refine_info = {**win_info, "kind": "outlet_centred_window"}
                _flag("TERMINAL_CATCHMENT_LOCAL_WINDOW_USED")
                fallback_history.append(
                    {
                        "method": "outlet_centred_window",
                        "outcome": "succeeded",
                        "reason": "terminal refinement failed; the outlet drains part of the terminal catchment",
                    }
                )
                consistency = mh.version_consistency(
                    vector_version=MERIT_BASINS_VECTOR_VERSION,
                    snap_origin=snap_origin,
                    refine_origin=refine_origin,
                )
                if consistency["mixed"]:
                    _flag("MERIT_VERSION_MIX")
            except Exception as exc:
                fallback_history.append(
                    {"method": "outlet_centred_window", "outcome": "failed", "reason": str(exc)}
                )
    if off_reach:
        # Local-only is the answer for this outlet. With a known area, keep whichever of the two areas is
        # closer to it, so a gauge truly on the main stem whose snap landed on a tributary is still answered
        # with the main stem.
        local_area = refinement_area if refinement_area is not None else terminal_total_km2
        outlet_assembly["local_polygon"] = (
            "refined" if terminal_refinement_used else "whole_terminal_catchment_unrefined"
        )
        outlet_assembly["local_area_km2"] = round(local_area, 3)
        if expected_area_km2 and expected_area_km2 > 0 and local_area > 0 and vector_area > 0:
            if abs(math.log(vector_area / expected_area_km2)) < abs(math.log(local_area / expected_area_km2)):
                keep_upstream = True
                outlet_assembly["tiebreak"] = (
                    f"expected {expected_area_km2:g} km2 is closer to the main-stem basin "
                    f"({vector_area:.1f} km2) than to the local catchment ({local_area:.1f} km2)"
                )
        if keep_upstream:
            outlet_assembly["mode"] = "union_upstream_kept_by_expected_area"
        else:
            outlet_assembly["mode"] = "local_only"
            _flag("HYBRID_LOCAL_ONLY")
        fallback_history.append(
            {
                "method": "outlet_assembly",
                "outcome": outlet_assembly["mode"],
                "reason": "snapped outlet is off the terminal unit catchment's main reach",
            }
        )
    outlet_assembly["area_gate"] = area_gate_info

    # -- merge ---------------------------------------------------------------------------------------------
    pieces = []
    if not nonterminal.empty and keep_upstream:
        pieces.append(nonterminal.to_crs(4326))
    # The kept main-stem union is the whole assembly (upstream catchments plus the whole terminal catchment).
    # The refined terminal piece belongs to the other (local) basin; adding it to the inflow would build a
    # polygon that is neither basin.
    main_stem_kept = off_reach and keep_upstream
    pieces.append((terminal if main_stem_kept else terminal_component).to_crs(4326))
    merged = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(
            [gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=4326).union_all()], crs=4326
        )
    )
    rss_after_merge = _rss_memory_mb()
    polygon_area = area_km2(merged)
    rss_peak = max(
        rss_before, rss_after_vector_load, rss_after_vector_assembly,
        rss_after_terminal_refinement, rss_after_merge,
    )

    area_validation, area_flags = validate_area(
        polygon_area,
        snapped_upa_km2=snap_reference.official_merit_upa_km2,
        expected_area_km2=expected_area_km2,
        touches_edge=False,
    )
    snap_validation, more_snap_flags = validate_snap(snap, expected_area_km2=expected_area_km2)
    area_validation["outlet_assembly"] = outlet_assembly
    if official_uparea:
        area_validation["merit_basins_uparea_km2"] = round(official_uparea, 3)
        area_validation["catchment_area_sum_km2"] = round(catchment_area_sum or 0.0, 3)
        area_validation["relative_error_assembly_vs_uparea"] = round(rel, 4) if rel is not None else None
    if missing_ids:
        area_validation["catchments_without_polygon"] = len(missing_ids)
    for flag in area_flags + more_snap_flags:
        _flag(flag)
    if "MERIT_UPA_AREA_MISMATCH" in quality_flags:
        _flag("HYBRID_AREA_VALIDATION_MISMATCH")

    files = []
    for kind, path in (("basins", basins_db), ("rivers", rivers_db), ("flowdir", flowdir_path), ("accum", accum_path)):
        rec = vf.read_provenance(path) if path else None
        if rec:
            files.append(
                {
                    "kind": kind,
                    "name": rec["name"],
                    "sha256": rec["sha256"],
                    "size_bytes": rec["size_bytes"],
                    "url": rec["url"],
                    "verified": True,
                }
            )
    mega_rec = vf.read_provenance(mgr.megabasins_path()) if mgr.megabasins_path().exists() else None
    if mega_rec:
        files.append(
            {
                "kind": "level2_index", "name": mega_rec["name"], "sha256": mega_rec["sha256"],
                "size_bytes": mega_rec["size_bytes"], "url": mega_rec["url"], "verified": True,
            }
        )
    downloaded: list[dict[str, Any]] = []
    for name in dict.fromkeys(downloaded_names):
        if name == "megabasins.db":
            path = mgr.megabasins_path()
        else:
            path = mgr._asset_path(next(k for k in vf.VALID_KINDS if name.startswith(k)), pfaf)
        rec = vf.read_provenance(path) if path.exists() else None
        if rec:
            downloaded.append(
                {"name": name, "bytes": rec["size_bytes"], "url": rec["url"], "sha256": rec["sha256"]}
            )
    if downloaded:
        _flag("MERIT_AUTO_STAGED")
    man = vf.load_manifest()
    data_provenance = {
        "files": files,
        "downloaded": downloaded,
        "acquisition_policy": stage_policy,
        "licence": MERIT_BASINS_LICENSE,
        "share_alike": True,
        "licence_detail": man["licence"],
        "pinned_manifest": {
            "package": man["source"]["package"],
            "package_version": man["source"]["package_version"],
            "data_version": man["source"]["data_version"],
        },
        "version_consistency": consistency,
        "refinement": {k: v for k, v in refine_info.items() if k != "bounds"} or None,
        "outlet_assembly": outlet_assembly,
        # the hybrid's own tier trail (the router keeps its own trail; this is the reason a refinement or
        # window fallback failed)
        "hybrid_fallback_history": [dict(row) for row in fallback_history],
    }

    cache_key = hashlib.sha256(
        f"hybrid:{pfaf}:{terminal_id}:{snap_reference.cache_key}:{terminal_refinement_used}:"
        f"{[f['sha256'] for f in files]}".encode("utf-8")
    ).hexdigest()
    raster_origin_local = mh.ORIGIN_LOCAL in (snap_origin, refine_origin)
    raster_ee = mh.ORIGIN_EE in (snap_origin, refine_origin)
    if raster_ee:
        raster_version = f"{MERIT_GEE_DATASET} (Earth Engine window)"
        raster_source = "Google Earth Engine"
    elif raster_origin_local:
        raster_version = MERIT_FLOWDIR_RASTER_VERSION.format(pfaf=pfaf)
        raster_source = MERIT_FLOWDIR_RASTER_SOURCE.format(pfaf=pfaf)
    else:
        raster_version = raster_source = None
    return MeritFlowdirResult(
        gdf=merged,
        area_km2=polygon_area,
        outlet_lat=snap.lat,
        outlet_lon=snap.lon,
        snap_distance_m=snap.distance_m,
        snapped_upa_km2=snap_reference.official_merit_upa_km2,
        snap_quality=snap.snap_quality,
        snap_validation=snap_validation,
        cache_key=cache_key,
        bbox=tuple(merged.total_bounds),
        scout_box_maxed=False,
        touches_edge=False,
        area_validation=area_validation,
        quality_flags=quality_flags,
        source="merit_basins_hybrid",
        routing_dataset="MERIT-Basins vector assembly + terminal raster refinement",
        routing_resolution_m=MERIT_SCALE_M,
        pfaf_region=pfaf,
        routing_data_source="pinned_mghydro_basins_db"
        + ("_and_regional_rasters" if raster_origin_local else "_and_gee_window" if raster_ee else ""),
        vector_dataset_version=MERIT_BASINS_VECTOR_VERSION,
        raster_dataset_version=raster_version,
        vector_data_source=MERIT_BASINS_VECTOR_SOURCE.format(pfaf=pfaf),
        raster_data_source=raster_source,
        snap_source=snap_reference.source,
        official_merit_upa_km2=snap_reference.official_merit_upa_km2,
        polygon_area_km2=polygon_area,
        validation_sources=validation_sources,
        regional_flowdir_cached=flowdir_path is not None,
        execution_mode="vector_topology_terminal_raster_refinement",
        peak_memory_mb=round(_peak_memory_mb(), 3),
        runtime_seconds=round(time.perf_counter() - started, 3),
        fallback_history=fallback_history,
        memory_telemetry={
            "rss_before_mb": round(rss_before, 3),
            "rss_after_vector_load_mb": round(rss_after_vector_load, 3),
            "rss_after_vector_assembly_mb": round(rss_after_vector_assembly, 3),
            "rss_after_terminal_refinement_mb": round(rss_after_terminal_refinement, 3),
            "rss_after_merge_mb": round(rss_after_merge, 3),
            "rss_peak_mb": round(rss_peak, 3),
            "rss_delta_mb": round(max(0.0, rss_peak - rss_before), 3),
        },
        safe_envelope_version=MERIT_SAFE_ENVELOPE_VERSION,
        terminal_catchment_id=terminal_id,
        upstream_catchment_count=len(upstream_ids),
        terminal_refinement_used=terminal_refinement_used,
        vector_assembly_area_km2=vector_area,
        refined_polygon_area_km2=refinement_area if refinement_area is not None else polygon_area,
        data_provenance=data_provenance,
    )


def merit_build_local_upstream_area_cache(pfaf_region: str) -> dict[str, Any]:
    """Build an optional local upstream-area cache for offline/benchmark workflows."""
    started = time.perf_counter()
    rss_before = _rss_memory_mb()
    import rasterio

    from aihydro_watershed.merit.merit_manager import MeritDataManager

    mgr = MeritDataManager()
    flowdir = mgr.flowdir_path(pfaf_region)
    if not flowdir:
        raise FileNotFoundError(f"Local MERIT flow direction raster missing for basin {pfaf_region}.")
    try:
        import pyflwdir
    except Exception as exc:
        raise RuntimeError("pyflwdir is required to build local upstream-area cache.") from exc
    with rasterio.open(flowdir) as src:
        cells = int(src.width * src.height)
        if cells > _MAX_LOCAL_FLOWDIR_CELLS:
            raise RuntimeError(
                "LOCAL_ROUTING_MEMORY_RISK: offline upstream-area cache build would "
                f"process {cells} cells, exceeding the safe limit {_MAX_LOCAL_FLOWDIR_CELLS}."
            )
        dir_arr = src.read(1)
        rss_after_flowdir_read = _rss_memory_mb()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
    d8 = merit_dir_to_pyflwdir(dir_arr)
    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        transform=transform,
        latlon=crs.upper() in ("EPSG:4326", "OGC:CRS84"),
    )
    rss_after_pyflwdir_build = _rss_memory_mb()
    uparea = flw.upstream_area(unit="km2").astype("float32")
    rss_after_upstream_area = _rss_memory_mb()
    dest = mgr._local_upstream_area_path(pfaf_region)  # internal path by design
    dest.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=-9999.0)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(uparea, 1)
    rss_after_write = _rss_memory_mb()
    rss_peak = max(
        rss_before,
        rss_after_flowdir_read,
        rss_after_pyflwdir_build,
        rss_after_upstream_area,
        rss_after_write,
    )
    return {
        "pfaf_region": pfaf_region.zfill(2),
        "local_upstream_area_path": str(dest),
        "build_runtime_seconds": round(time.perf_counter() - started, 3),
        "memory_telemetry": {
            "rss_before_mb": round(rss_before, 3),
            "rss_after_flowdir_read_mb": round(rss_after_flowdir_read, 3),
            "rss_after_pyflwdir_build_mb": round(rss_after_pyflwdir_build, 3),
            "rss_after_upstream_area_mb": round(rss_after_upstream_area, 3),
            "rss_after_write_mb": round(rss_after_write, 3),
            "rss_peak_mb": round(rss_peak, 3),
            "rss_delta_mb": round(max(0.0, rss_peak - rss_before), 3),
        },
    }


def merit_build_offline_snap_cache(
    pfaf_region: str,
    *,
    offline_snap_asset: str = "published_accum",
) -> dict[str, Any]:
    """
    Build explicit offline snap support for a staged flowdir region.

    The default expert/offline route uses a staged published accumulation
    product. Full-Pfaf local upstream-area derivation remains opt-in only.
    Online delineation does not require either cache.
    """
    from aihydro_watershed.merit.merit_manager import MeritDataManager

    mgr = MeritDataManager()
    if offline_snap_asset == "published_accum":
        status = mgr.routing_region_cache(pfaf_region)
        if not status.accum_ready:
            return {
                "pfaf_region": pfaf_region.zfill(2),
                "snap_cache_type": "published_accum",
                "ready": False,
                "quality_flags": ["REGIONAL_FLOWDIR_STAGING_REQUIRED"],
                "message": (
                    "Published MERIT accumulation is not staged. This expert/offline "
                    "snap cache is optional and is never downloaded for normal online routing."
                ),
            }
        return {
            "pfaf_region": pfaf_region.zfill(2),
            "snap_cache_type": "published_accum",
            "ready": True,
            "accum_path": status.accum_path,
            "usage": "offline-prepared mode only; online mode uses tiny GEE upa/wth.",
        }
    if offline_snap_asset != "local_upstream_area":
        raise ValueError("offline_snap_asset must be 'published_accum' or 'local_upstream_area'.")
    result = merit_build_local_upstream_area_cache(pfaf_region)
    return {
        **result,
        "snap_cache_type": "local_upstream_area",
        "usage": "offline-prepared mode only; online mode uses tiny GEE upa/wth.",
    }
