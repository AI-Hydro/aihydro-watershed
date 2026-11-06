"""Offline tests: hybrid polygon agrees with its own snapped outlet (P1) and the area-gated snap (P2).

Uses the synthetic region from ``merit_fixture`` (terminal C3 with upstream C1, C2, C4 = 14.7 km2 of inflow) and a
stubbed Earth Engine, so nothing is downloaded.
"""

from __future__ import annotations

import geopandas as gpd
import merit_fixture as fx
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.windows import from_bounds

from aihydro_watershed.delineation import merit_flowdir_pipeline as pipe
from aihydro_watershed.delineation import merit_hybrid as mh
from aihydro_watershed.delineation.types import area_km2
from aihydro_watershed.merit.basins_store import BasinsStore

from test_merit_hybrid_default import _stage_all
from test_merit_hybrid_default import region  # noqa: F401  (pytest fixture, used by name)

C3 = area_km2(gpd.GeoDataFrame(geometry=[fx.box(0.04, 40.0, 0.06, 40.02)], crs=4326))
HALF_C3 = 0.5 * C3  # the refined terminal piece for the fixture outlet
INFLOW = 3 * C3  # C1 + C2 + C4 are the same size as C3; asserted against the store below


def _snap_ref(monkeypatch, *, upa: float, lat: float = fx.SNAPPED[0], lon: float = fx.SNAPPED[1]):
    def fake(lat0, lon0, **_):
        snap = pipe.MeritSnap(lat=lat, lon=lon, row=0, col=0, distance_m=250.0,
                              snapped_upa_km2=upa, snap_quality="nearest_merit_stream")
        return pipe.MeritSnapReference(snap=snap, source="gee", cache_key="stub", bbox=(lon0, lat0, lon0, lat0),
                                       official_merit_upa_km2=upa)

    monkeypatch.setattr(pipe, "merit_get_snap_reference", fake)


def _stub_fetch(monkeypatch, region):
    flowdir = region.files["flowdir73.tif"]
    calls: list = []

    def fake_fetch(bbox, *, bands, **_):
        calls.append((bbox, bands))
        with rasterio.open(flowdir) as src:
            w = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=w)
            tr = src.window_transform(w)
        return pipe.MeritRasterBundle(
            bands={"dir": arr}, transform=tr, crs="EPSG:4326", cache_key="stub", source="gee", bbox=bbox
        )

    monkeypatch.setattr(mh, "fetch_merit_gee_bbox", fake_fetch)
    return calls


@pytest.fixture()
def ee_region(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    calls = _stub_fetch(monkeypatch, region)
    return region, calls


# ------------------------------------------------------------------------------------------ P1


def test_fixture_inflow_is_what_the_tests_assume(region, tmp_path):
    root = tmp_path / "m"
    mgr = _stage_all(region, root)
    store = BasinsStore(mgr.basins_db_path(fx.PFAF), mgr.rivers_db_path(fx.PFAF))
    g = store.geometries(store.upstream_ids(fx.TERMINAL))
    assert g.loc[g.comid != fx.TERMINAL, "area_km2"].sum() == pytest.approx(INFLOW, rel=0.01)


def test_outlet_on_main_reach_still_unions_upstream(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=INFLOW + HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.area_km2 == pytest.approx(INFLOW + HALF_C3, rel=0.05)
    assert "OUTLET_OFF_TERMINAL_REACH" not in res.quality_flags and "HYBRID_LOCAL_ONLY" not in res.quality_flags
    assert res.area_validation["outlet_assembly"]["mode"] == "union_upstream"


def test_outlet_off_main_reach_returns_the_local_catchment_only(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=HALF_C3)  # a tributary outlet: drains the refined piece, not the inflow
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.upstream_catchment_count == 4  # the topology is unchanged, only the assembly differs
    assert res.area_km2 == pytest.approx(HALF_C3, rel=0.02)  # NOT the inflow larger
    assert res.area_km2 == pytest.approx(res.refined_polygon_area_km2, rel=1e-6)
    assert {"OUTLET_OFF_TERMINAL_REACH", "HYBRID_LOCAL_ONLY"} <= set(res.quality_flags)
    # the polygon now agrees with its own snapped outlet, so the existing mismatch flag stays quiet
    assert "HYBRID_AREA_VALIDATION_MISMATCH" not in res.quality_flags
    oa = res.area_validation["outlet_assembly"]
    assert oa["mode"] == "local_only" and oa["local_polygon"] == "refined"
    alt = oa["mainstem_alternative"]
    assert alt["area_km2"] == pytest.approx(INFLOW + C3, rel=0.01)  # rivers uparea at the terminal's downstream end
    assert alt["source"] == "merit_basins_rivers_uparea" and "tributary" in alt["note"]
    assert res.data_provenance["outlet_assembly"] == oa


def test_hybrid_area_validation_mismatch_is_kept_when_the_union_is_returned(ee_region, monkeypatch):
    # outlet off the reach, but the caller's area says the main stem is right: union kept, mismatch reported
    _snap_ref(monkeypatch, upa=HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, expected_area_km2=INFLOW + HALF_C3)
    assert res.area_km2 == pytest.approx(INFLOW + HALF_C3, rel=0.05)
    oa = res.area_validation["outlet_assembly"]
    assert oa["mode"] == "union_upstream_kept_by_expected_area" and "closer" in oa["tiebreak"]
    assert "OUTLET_OFF_TERMINAL_REACH" in res.quality_flags and "HYBRID_LOCAL_ONLY" not in res.quality_flags
    assert "HYBRID_AREA_VALIDATION_MISMATCH" in res.quality_flags


def test_expected_area_near_the_local_catchment_keeps_local_only(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, expected_area_km2=HALF_C3 * 1.05)
    assert res.area_validation["outlet_assembly"]["mode"] == "local_only"
    assert res.area_km2 == pytest.approx(HALF_C3, rel=0.02)


def test_refinement_failure_uses_an_outlet_centred_window_not_the_whole_terminal(ee_region, monkeypatch):
    _, calls = ee_region
    # a hillslope cell on the north side of the trunk: its raster basin is a few cells
    lat, lon = 40.015, 0.0525
    _snap_ref(monkeypatch, upa=0.05, lat=lat, lon=lon)
    monkeypatch.setattr(mh, "refine_ee", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("TERMINAL_REFINEMENT_WINDOW_TOO_LARGE: 99 km")))
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.area_km2 < 0.5  # a few 90 m cells, not the 4.9 km2 terminal catchment
    assert res.area_km2 < 0.2 * C3
    flags = set(res.quality_flags)
    assert {"TERMINAL_CATCHMENT_REFINEMENT_FAILED", "TERMINAL_CATCHMENT_LOCAL_WINDOW_USED",
            "OUTLET_OFF_TERMINAL_REACH", "HYBRID_LOCAL_ONLY"} <= flags
    assert res.terminal_refinement_used and res.refined_polygon_area_km2 == pytest.approx(res.area_km2, rel=1e-6)
    (bbox, bands), = calls
    assert bands == ("dir",)
    assert (bbox[0] + bbox[2]) / 2 == pytest.approx(lon, abs=1e-6) and (bbox[1] + bbox[3]) / 2 == pytest.approx(lat, abs=1e-6)
    steps = [(h["method"], h["outcome"]) for h in res.fallback_history]
    assert steps[:2] == [("terminal_gee_window_refinement", "failed"), ("outlet_centred_window", "succeeded")]


def test_failed_window_keeps_the_terminal_catchment_and_says_why(ee_region, monkeypatch):
    lat, lon = 40.015, 0.0525
    _snap_ref(monkeypatch, upa=0.05, lat=lat, lon=lon)
    monkeypatch.setattr(mh, "refine_ee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(mh, "delineate_on_local_window", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("LOCAL_WINDOW_TOUCHES_EDGE: x")))
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.area_km2 == pytest.approx(C3, rel=0.01)  # the whole terminal only, not terminal plus inflow
    assert "TERMINAL_CATCHMENT_LOCAL_WINDOW_USED" not in res.quality_flags
    assert res.area_validation["outlet_assembly"]["local_polygon"] == "whole_terminal_catchment_unrefined"
    failed = [h for h in res.fallback_history if h["method"] == "outlet_centred_window"]
    assert failed and failed[0]["outcome"] == "failed" and "LOCAL_WINDOW_TOUCHES_EDGE" in failed[0]["reason"]


def test_refinement_failure_on_the_main_reach_is_unchanged(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=INFLOW + HALF_C3)
    monkeypatch.setattr(mh, "refine_ee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.area_km2 == pytest.approx(res.vector_assembly_area_km2, rel=1e-6)
    assert "TERMINAL_CATCHMENT_LOCAL_WINDOW_USED" not in res.quality_flags


def test_internal_fallback_history_is_exposed_in_the_result(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    hist = res.data_provenance["hybrid_fallback_history"]
    assert hist == res.fallback_history and hist is not res.fallback_history
    assert [h["method"] for h in hist] == ["terminal_gee_window_refinement", "outlet_assembly"]
    assert hist[-1]["outcome"] == "local_only"


def test_local_window_size_rule():
    assert mh.local_window_half_side_km(0.5) == 3.0
    assert mh.local_window_half_side_km(100.0) == pytest.approx(20.0)
    assert mh.local_window_half_side_km(10_000.0) == 20.0
    assert mh.local_window_half_side_km(25.0) == pytest.approx(10.0)


# ------------------------------------------------------------------------------------------ P2


def _grid(shape=(61, 61)):
    """A 90 m-ish grid centred on lat 40, lon 0 with a west-east stream whose UPA grows eastwards."""
    px = 1.0 / 1200.0
    tr = Affine(px, 0, -(shape[1] // 2 + 0.5) * px, 0, -px, 40.0 + (shape[0] // 2 + 0.5) * px)
    upa = np.zeros(shape)
    mid = shape[0] // 2
    upa[mid, :] = np.linspace(1.0, 60.0, shape[1])  # the stream row
    upa[mid - 12, :] = 400.0  # a bigger river 12 rows (~1.1 km) north
    return upa, tr


def _cell_of(upa, tr, row, col):
    lon, lat = tr * (col + 0.5, row + 0.5)
    return lat, lon


def test_gated_snap_takes_the_nearest_cell_within_ten_percent():
    upa, tr = _grid()
    mid = upa.shape[0] // 2
    lat, lon = _cell_of(upa, tr, mid, 30)  # on the stream
    target = float(upa[mid, 36])  # six cells east
    snap = pipe.snap_outlet_area_gated(lat, lon, upa=upa, transform=tr, expected_area_km2=target)
    assert snap.snap_quality == "area_gated"
    assert abs(snap.snapped_upa_km2 - target) / target <= 0.10
    # of all cells within 10 % the nearest one wins (a lower-UPA neighbour may qualify first)
    qual = np.abs(upa[mid] - target) / target <= 0.10
    cols = np.nonzero(qual)[0]
    assert snap.col == int(cols[np.argmin(np.abs(cols - 30))])
    assert 0 < snap.distance_m <= pipe.AREA_GATE_RADIUS_M


def test_gated_snap_never_searches_beyond_the_gate_radius():
    upa, tr = _grid()
    mid = upa.shape[0] // 2
    lat, lon = _cell_of(upa, tr, mid, 5)
    far_target = float(upa[mid, 55])  # ~4.5 km east: outside the 1 km gate
    plain = pipe.snap_outlet_on_merit(lat, lon, upa=upa, transform=tr)
    snap = pipe.snap_outlet_area_gated(lat, lon, upa=upa, transform=tr, expected_area_km2=far_target)
    assert snap.snap_quality != "area_gated"
    assert (snap.row, snap.col, snap.snap_quality) == (plain.row, plain.col, plain.snap_quality)  # the plain snap, untouched
    assert snap.distance_m <= 3_000.0  # not the 30 km area scorer


def test_gated_snap_ignores_a_matching_cell_off_the_stream():
    upa, tr = _grid()
    mid = upa.shape[0] // 2
    upa[mid + 3, 30] = 0.2  # a sub-1 km2 hillslope cell that happens to match a tiny expected area
    lat, lon = _cell_of(upa, tr, mid + 3, 30)
    snap = pipe.snap_outlet_area_gated(lat, lon, upa=upa, transform=tr, expected_area_km2=0.2)
    assert snap.snap_quality != "area_gated"


def test_gated_snap_requires_an_area():
    upa, tr = _grid()
    with pytest.raises(ValueError):
        pipe.snap_outlet_area_gated(40.0, 0.0, upa=upa, transform=tr, expected_area_km2=0.0)


def test_snap_reference_with_area_gate_keeps_the_small_window(monkeypatch):
    upa, tr = _grid()
    seen = []

    def fake_fetch(bbox, *, bands, **_):
        seen.append(bbox)
        return pipe.MeritRasterBundle(bands={"upa": upa, "wth": np.ones_like(upa)}, transform=tr, crs="EPSG:4326",
                                      cache_key="k", source="gee_cache", bbox=bbox)

    monkeypatch.setattr(pipe, "fetch_merit_gee_bbox", fake_fetch)
    mid = upa.shape[0] // 2
    lat, lon = _cell_of(upa, tr, mid, 30)
    pipe.merit_get_snap_reference(lat, lon, expected_area_km2=float(upa[mid, 34]), area_gate=True)
    pipe.merit_get_snap_reference(lat, lon, expected_area_km2=1500.0, area_gate=True)
    pipe.merit_get_snap_reference(lat, lon, expected_area_km2=1500.0)  # the old scorer widens the window
    widths = [b[2] - b[0] for b in seen]
    assert widths[0] == pytest.approx(widths[1]) and widths[2] > 3 * widths[0]


def test_hybrid_flags_a_failed_gate_and_records_it(ee_region, monkeypatch):
    # the snapped cell (stub) is 2.4 km2; the caller says 40 km2: no cell qualifies, the plain snap is kept
    _snap_ref(monkeypatch, upa=HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, expected_area_km2=40.0)
    assert {"AREA_GATE_NO_MATCH", "SNAP_UPA_DIFFERS_FROM_EXPECTED"} <= set(res.quality_flags)
    gate = res.area_validation["outlet_assembly"]["area_gate"]
    assert gate["matched"] is False and gate["radius_m"] == 1000.0 and gate["tolerance"] == 0.10
    assert any(h["method"] == "area_gated_snap" and h["outcome"] == "no_match" for h in res.fallback_history)


def test_hybrid_records_a_gated_snap(ee_region, monkeypatch):
    def fake(lat0, lon0, *, expected_area_km2=None, area_gate=False, **_):
        assert area_gate is True and expected_area_km2 == 17.0  # the hybrid asks for the narrow gate
        snap = pipe.MeritSnap(lat=fx.SNAPPED[0], lon=fx.SNAPPED[1], row=0, col=0, distance_m=60.0,
                              snapped_upa_km2=INFLOW + HALF_C3, snap_quality="area_gated")
        return pipe.MeritSnapReference(snap=snap, source="gee", cache_key="s", bbox=(lon0, lat0, lon0, lat0),
                                       official_merit_upa_km2=INFLOW + HALF_C3)

    monkeypatch.setattr(pipe, "merit_get_snap_reference", fake)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, expected_area_km2=17.0)
    assert res.snap_quality == "area_gated"
    assert res.area_validation["outlet_assembly"]["area_gate"]["matched"] is True
    assert "AREA_GATE_NO_MATCH" not in res.quality_flags


def test_local_accum_snap_uses_the_gate_when_asked(region, tmp_path):
    mgr = _stage_all(region, tmp_path / "m")
    accum = mgr.raster_path("accum", fx.PFAF)
    with rasterio.open(accum) as src:
        arr = src.read(1).astype(float)
        tr = src.transform
    cell_km2 = mh._cell_area_km2(tr, 40.0)
    trunk = arr[14]  # cells; the trunk row grows eastwards
    target = float(trunk[30] * cell_km2)
    lat, lon = fx.SNAPPED[0], tr.c + 27.5 * tr.a  # a trunk cell a few cells west of the target
    gated = mh.snap_on_local_accum(accum, lat, lon, expected_area_km2=target, area_gate=True)
    assert gated.snap_quality == "area_gated"
    assert abs(gated.snapped_upa_km2 - target) / target <= 0.10
    # without the gate an expected area still selects the wide scorer (other callers are unchanged)
    wide = mh.snap_on_local_accum(accum, lat, lon, expected_area_km2=target)
    assert wide.snap_quality == "area_targeted"
