"""
Known-answer tests for the small-catchment (3DEP + embankment notch) tier.

The DEMs are synthetic so the answers can be counted by hand:

* A walled block ("mesa") standing above a plain. Inside the wall the surface
  is a funnel centred on one cell, drained south by a channel through a single
  gap in the wall (the spillway). Every interior cell can only leave through
  the spillway, so the catchment of the spillway cell is exactly the interior
  plus the spillway cell itself: 58 x 49 + 1 = 2843 cells.
* A road embankment 50 m high across the channel, plus a low saddle in the
  side wall upstream of it. Without the notch, the water ponded behind the
  road spills over the saddle and leaves the block; with the notch, it passes
  the culvert and the catchment matches the no-road case exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyflwdir")
pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import box  # noqa: E402

from aihydro_watershed.delineation import small_catchment as sc  # noqa: E402

N, M, DX = 100, 80, 10.0
R_TOP, R_BOT, C_LEFT, C_RIGHT = 10, 69, 15, 65
CENTRE = (40, 40)
CHANNEL_COL = 40
SPILLWAY = (R_BOT, CHANNEL_COL)
CULVERT = (55, CHANNEL_COL)
INTERIOR_CELLS = (R_BOT - R_TOP - 1) * (C_RIGHT - C_LEFT - 1)   # 58 * 49 = 2842

# Place the synthetic grid in EPSG:5070 over Indiana so lat/lon round-trips work.
X0, Y0 = 800_000.0, 1_900_000.0
TR = Affine(DX, 0, X0, 0, -DX, Y0)


def mesa_dem(road: bool = False, saddle: bool = False) -> np.ndarray:
    r, c = np.indices((N, M))
    z = 10 + 0.02 * (N - 1 - r) * DX                     # plain, draining south
    interior = (r > R_TOP) & (r < R_BOT) & (c > C_LEFT) & (c < C_RIGHT)
    wall = (r >= R_TOP) & (r <= R_BOT) & (c >= C_LEFT) & (c <= C_RIGHT) & ~interior
    z = np.where(wall, 300.0, z)
    z = np.where(interior, 100 + 0.1 * np.hypot(r - CENTRE[0], c - CENTRE[1]) * DX, z)
    chan = (c == CHANNEL_COL) & (r > CENTRE[0]) & (r <= R_BOT)   # cuts the wall at the spillway
    z = np.where(chan, 99.9 - 0.05 * (r - CENTRE[0]), z)
    if road:
        z = np.where(interior & (r >= CULVERT[0] - 1) & (r <= CULVERT[0] + 1), z + 50, z)
    if saddle:
        z[30, C_LEFT] = 140.0                           # below the road crest (~147 m)
    return z.astype("float32")


def cell_xy(rc):
    r, c = rc
    return X0 + (c + 0.5) * DX, Y0 - (r + 0.5) * DX


# ── delineate_on_dem ──────────────────────────────────────────────────────────

def test_spillway_catchment_is_exactly_the_walled_interior():
    out = sc.delineate_on_dem(mesa_dem(), TR, *cell_xy(SPILLWAY),
                              carve_notch=False, snap_radius_m=5.0)
    assert out["outlet_rc"] == SPILLWAY
    assert int(out["basin"].sum()) == INTERIOR_CELLS + 1
    assert out["area_m2"] == pytest.approx((INTERIOR_CELLS + 1) * DX * DX)
    assert not out["touches_edge"]


def test_notch_restores_catchment_behind_road_embankment():
    culvert = cell_xy(CULVERT)
    no_road = sc.delineate_on_dem(mesa_dem(), TR, *culvert, carve_notch=False)
    dam = mesa_dem(road=True, saddle=True)
    blocked = sc.delineate_on_dem(dam, TR, *culvert, carve_notch=False)
    notched = sc.delineate_on_dem(dam, TR, *culvert, carve_notch=True)

    # Without the notch most of the catchment spills over the saddle instead.
    assert blocked["basin"].sum() < 0.3 * no_road["basin"].sum()
    # With it, the catchment is the same cells as with no road at all.
    np.testing.assert_array_equal(notched["basin"], no_road["basin"])
    # Notch elevation = lowest DEM cell within 60 m of the culvert (channel, 6 rows down).
    assert notched["notch_elevation_m"] == pytest.approx(99.9 - 0.05 * (CULVERT[0] + 6 - CENTRE[0]),
                                                         abs=1e-4)


def test_snap_moves_offset_point_onto_channel():
    x, y = cell_xy(CULVERT)
    out = sc.delineate_on_dem(mesa_dem(), TR, x + 25.0, y, carve_notch=False)
    assert out["outlet_rc"][1] == CHANNEL_COL
    assert out["snap_distance_m"] <= sc.SNAP_RADIUS_M


def test_point_with_no_dem_nearby_raises():
    z = np.full((20, 20), np.nan, dtype="float32")
    with pytest.raises(RuntimeError):
        sc.delineate_on_dem(z, TR, *cell_xy((10, 10)))


# ── delineate_small_catchment (window loop, polygon, flags) ──────────────────

def _fake_fetcher(dem):
    calls = []

    def fetch(x, y, hw, res):
        calls.append(hw)
        r, c = np.indices(dem.shape)
        cx = X0 + (c + 0.5) * DX
        cy = Y0 - (r + 0.5) * DX
        keep = (np.abs(cx - x) <= hw) & (np.abs(cy - y) <= hw)
        rows = np.flatnonzero(keep.any(axis=1))
        cols = np.flatnonzero(keep.any(axis=0))
        sub = dem[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        tr = Affine(DX, 0, X0 + cols[0] * DX, 0, -DX, Y0 - rows[0] * DX)
        return sub, tr

    return fetch, calls


def _latlon(rc):
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(*cell_xy(rc))
    return lat, lon


def test_window_grows_until_basin_is_complete():
    fetch, calls = _fake_fetcher(mesa_dem())
    lat, lon = _latlon(SPILLWAY)
    res = sc.delineate_small_catchment(
        lat, lon, half_widths_m=(100.0, 1000.0), snap_radius_m=5.0,
        carve_notch=False, dem_fetcher=fetch,
    )
    assert calls == [100.0, 1000.0]
    assert res.status == "ok" and res.window_iterations == 2
    assert res.area_km2 == pytest.approx((INTERIOR_CELLS + 1) * DX * DX / 1e6)
    # Polygon area (equal-area CRS) agrees with the cell count.
    assert float(res.gdf.to_crs(5070).area.sum()) / 1e6 == pytest.approx(res.area_km2, rel=1e-6)
    assert res.gdf.crs.to_epsg() == 4326
    assert "BASIN_TOUCHES_MAX_WINDOW" not in res.quality_flags


def test_basin_truncated_at_last_window_is_flagged():
    fetch, _ = _fake_fetcher(mesa_dem())
    lat, lon = _latlon(SPILLWAY)
    res = sc.delineate_small_catchment(
        lat, lon, half_widths_m=(100.0,), snap_radius_m=5.0,
        carve_notch=False, dem_fetcher=fetch,
    )
    assert res.status == "touches_max_window"
    assert "BASIN_TOUCHES_MAX_WINDOW" in res.quality_flags
    assert "EMBANKMENT_NOTCH_APPLIED" not in res.quality_flags


def test_notch_flag_set_when_notch_carved():
    fetch, _ = _fake_fetcher(mesa_dem(road=True))
    lat, lon = _latlon(CULVERT)
    res = sc.delineate_small_catchment(lat, lon, half_widths_m=(1000.0,), dem_fetcher=fetch)
    assert "EMBANKMENT_NOTCH_APPLIED" in res.quality_flags
    assert res.notch_elevation_m is not None


def test_tiny_basin_is_flagged_as_likely_ditch_snap():
    # A cell near the wall drains only a handful of cells: far below 0.05 km2.
    fetch, _ = _fake_fetcher(mesa_dem())
    lat, lon = _latlon((12, 17))
    res = sc.delineate_small_catchment(
        lat, lon, half_widths_m=(1000.0,), snap_radius_m=5.0,
        carve_notch=False, dem_fetcher=fetch,
    )
    assert res.area_km2 < sc.DITCH_SNAP_KM2
    assert "LIKELY_DITCH_SNAP" in res.quality_flags


# ── Router integration ────────────────────────────────────────────────────────

INDIANA = (40.42, -86.93)


def _fake_result(area_km2=0.8, flags=None):
    import geopandas as gpd

    lat, lon = INDIANA
    poly = box(lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004)
    return sc.SmallCatchmentResult(
        gdf=gpd.GeoDataFrame(geometry=[poly], crs=4326),
        area_km2=area_km2, status="ok", snap_distance_m=12.0,
        outlet_lat=lat, outlet_lon=lon, window_half_width_m=2500.0,
        window_iterations=1, resolution_m=10.0, notch_elevation_m=200.0,
        quality_flags=list(flags or ["EMBANKMENT_NOTCH_APPLIED"]),
        window_bounds_5070=(0.0, 0.0, 1.0, 1.0),
    )


@pytest.fixture
def no_nldi(monkeypatch):
    from aihydro_watershed.delineation import nldi_point

    def fail(*a, **k):
        raise AssertionError("NLDI must not be called")

    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", fail)


def test_router_pinned_small_catchment_accepts_sub_km2_basin(monkeypatch, no_nldi):
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(sc, "delineate_small_catchment", lambda lat, lon: _fake_result(0.3))
    res = delineate_from_point(*INDIANA, method="small_catchment")
    d = res.data
    assert d["method_used"] == "small_catchment_3dep"
    assert d["area_km2"] == pytest.approx(0.3)
    assert d["routing_resolution_m"] == 10.0
    assert "EMBANKMENT_NOTCH_APPLIED" in d["quality_flags"]
    assert d["workflow_steps"][1]["step"] == "embankment_notch"
    assert any("3DEP" in s.name for s in res.meta.sources)


def test_router_3dep_alias(monkeypatch, no_nldi):
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(sc, "delineate_small_catchment", lambda lat, lon: _fake_result())
    assert delineate_from_point(*INDIANA, method="3dep").data["method_used"] == "small_catchment_3dep"


def test_router_auto_uses_small_tier_for_small_expected_area(monkeypatch, no_nldi):
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(sc, "delineate_small_catchment", lambda lat, lon: _fake_result(0.8))
    d = delineate_from_point(*INDIANA, expected_area_km2=1.0).data
    assert d["method_used"] == "small_catchment_3dep"
    assert d["area_validation"]["within_factor_2"] is True


def test_router_flags_area_far_from_expected(monkeypatch, no_nldi):
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(sc, "delineate_small_catchment", lambda lat, lon: _fake_result(0.1))
    d = delineate_from_point(*INDIANA, expected_area_km2=1.0).data
    assert "AREA_OUTSIDE_FACTOR_2_OF_EXPECTED" in d["quality_flags"]


def test_router_auto_falls_back_to_small_tier_when_nldi_basin_too_small(monkeypatch):
    from aihydro_core import HydroMeta, HydroResult

    from aihydro_watershed.delineation import nldi_point
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(
        nldi_point, "delineate_nldi_at_point",
        lambda lat, lon, **k: HydroResult(data={"area_km2": 0.4}, meta=HydroMeta(tool="t")),
    )
    monkeypatch.setattr(sc, "delineate_small_catchment", lambda lat, lon: _fake_result(0.6))
    d = delineate_from_point(*INDIANA).data
    assert d["method_used"] == "small_catchment_3dep"
    assert any(h["method"] == "small_catchment_3dep" and h["outcome"] == "succeeded"
               for h in d["fallback_history"])


def test_router_small_catchment_outside_conus_raises():
    from aihydro_core import ToolError

    from aihydro_watershed.delineation.router import delineate_from_point

    with pytest.raises(ToolError):
        delineate_from_point(50.9, 6.9, method="small_catchment")


def test_router_pinned_small_catchment_surfaces_failure(monkeypatch):
    from aihydro_core import ToolError

    from aihydro_watershed.delineation.router import delineate_from_point

    def boom(lat, lon):
        raise RuntimeError("3DEP unreachable")

    monkeypatch.setattr(sc, "delineate_small_catchment", boom)
    with pytest.raises(ToolError, match="3DEP unreachable"):
        delineate_from_point(*INDIANA, method="small_catchment")


# Areas from the INDOT SPR-4926 reference run (scripts/delineate_catchments.py,
# data/derived/catchment_delineation.csv), which the tier must reproduce.
INDOT_REFERENCE = [
    # (site, lat, lon, reference area km2, engineer memo area km2 or None)
    ("CV062-062-85.37", 38.22246, -86.69799, 0.3317, 0.3845),
    ("CV421-012-115.01", 40.24968, -86.38765, 1.1499, 1.2173),
    ("CV055-023-06.30", 40.21990, -87.17310, 0.0333, 0.6936),   # ditch snap
]


@pytest.mark.live
@pytest.mark.parametrize("site, lat, lon, ref_km2, memo_km2", INDOT_REFERENCE)
def test_live_parity_with_indot_reference(site, lat, lon, ref_km2, memo_km2):
    res = sc.delineate_small_catchment(lat, lon)
    assert res.resolution_m == pytest.approx(10.0)
    assert res.area_km2 == pytest.approx(ref_km2, abs=5e-4), site
    assert ("LIKELY_DITCH_SNAP" in res.quality_flags) == (ref_km2 < sc.DITCH_SNAP_KM2)
