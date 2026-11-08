"""Offline tests: the flow-direction snap (P2) and the unsnapped-outlet reliability mark (P1).

Synthetic region from ``merit_fixture`` (terminal C3 with upstream C1, C2, C4 = 3 x C3 of inflow); nothing is
downloaded. Earth Engine is "unavailable" through ``force_offline`` and no accumulation raster is staged, so the only
snap possible is the one on the staged ``flowdir73.tif``.

The thresholds under test (1.25x and 300 m) are in-sample on CAMELS; these tests pin the mechanism, not the numbers.
"""

from __future__ import annotations

import shutil

import geopandas as gpd
import merit_fixture as fx
import pytest
from shapely.geometry import LineString

from aihydro_watershed.delineation import merit_flowdir_pipeline as pipe
from aihydro_watershed.delineation import merit_hybrid as mh
from aihydro_watershed.delineation import router
from aihydro_watershed.delineation.types import area_km2
from aihydro_watershed.merit import verified_fetch as vf
from aihydro_watershed.merit.basins_store import BasinsStore
from aihydro_watershed.merit.merit_manager import MeritDataManager

from test_merit_hybrid_default import _fake_result, _stage_all
from test_merit_hybrid_default import region  # noqa: F401  (pytest fixture, used by name)

C3 = area_km2(gpd.GeoDataFrame(geometry=[fx.box(0.04, 40.0, 0.06, 40.02)], crs=4326))
INFLOW = 3 * C3
HALF_C3 = 0.5 * C3


def _stage_flowdir_only(region, root):
    mgr = _stage_all(region, root, rasters=False)
    mgr._stage(vf.region_asset_spec("flowdir", fx.PFAF), mgr.raster_path("flowdir", fx.PFAF))
    return mgr


@pytest.fixture()
def with_flowdir(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_flowdir_only(region, root)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    return region


@pytest.fixture()
def vectors_only(region, tmp_path, monkeypatch):
    """Vectors staged, no flowdir, and auto-staging disabled (the offline user who cannot download)."""
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    monkeypatch.setenv("AIHYDRO_MERIT_AUTO_STAGE", "never")
    return region


def _offline(lat, lon, **kw):
    return pipe.merit_basins_hybrid_delineate(lat, lon, force_offline=True, allow_offline=True, **kw)


def _network(root):
    store = BasinsStore(root / "vector" / "basins73.db", root / "vector" / "rivers73.db")
    ids = store.upstream_ids(fx.TERMINAL)
    g = store.geometries(ids)
    return g[g.comid == fx.TERMINAL][["geometry"]], g[g.comid != fx.TERMINAL][["geometry"]]


# ------------------------------------------------------------------------------------------ reach geometry


def test_river_line_is_digitised_downstream_first(region, tmp_path):
    mgr = _stage_all(region, tmp_path / "m")
    store = BasinsStore(mgr.basins_db_path(fx.PFAF), mgr.rivers_db_path(fx.PFAF))
    line = store.river_line(fx.TERMINAL)
    assert isinstance(line, LineString)
    assert line.coords[0] == pytest.approx((0.06, fx.TRUNK_LAT))   # downstream end is the first vertex
    assert store.river_line(999) is None


def test_river_line_is_none_without_a_rivers_db(tmp_path):
    assert BasinsStore(tmp_path / "x.db", None).river_line(1) is None


# ------------------------------------------------------------------------------------------ P2: flowdir snap


def test_flowdir_snap_credits_the_inflow_to_the_main_reach(region, tmp_path):
    mgr = _stage_flowdir_only(region, tmp_path / "merit")
    terminal, nonterminal = _network(tmp_path / "merit")
    snap, info = mh.snap_on_terminal_flowdir(
        terminal, nonterminal, mgr.raster_path("flowdir", fx.PFAF), *fx.OUTLET_REQUEST, inflow_km2=INFLOW
    )
    assert (snap.lat, snap.lon) == pytest.approx(fx.SNAPPED)            # the trunk cell, one cell from the request
    assert snap.snap_quality == mh.OFFLINE_FLOWDIR_SNAP_QUALITY
    # local drainage of the trunk cell (half the terminal) plus the upstream catchments that drain through it
    assert snap.snapped_upa_km2 == pytest.approx(INFLOW + HALF_C3, rel=0.1)
    assert info["inflow_cells"] > 0 and info["origin"] == mh.ORIGIN_LOCAL


def test_flowdir_snap_never_leaves_the_home_catchment(region, tmp_path):
    mgr = _stage_flowdir_only(region, tmp_path / "merit")
    terminal, nonterminal = _network(tmp_path / "merit")
    # a point in C2, just west of the terminal's border: the trunk cell nearest it is in C2 and is not a candidate
    snap, _ = mh.snap_on_terminal_flowdir(
        terminal, nonterminal, mgr.raster_path("flowdir", fx.PFAF), fx.TRUNK_LAT, 0.0395, inflow_km2=INFLOW
    )
    assert snap.lon >= 0.04 - 1e-9


def test_flowdir_snap_with_expected_area_uses_the_narrow_gate(region, tmp_path):
    mgr = _stage_flowdir_only(region, tmp_path / "merit")
    terminal, nonterminal = _network(tmp_path / "merit")
    path = mgr.raster_path("flowdir", fx.PFAF)
    hit, info = mh.snap_on_terminal_flowdir(
        terminal, nonterminal, path, *fx.OUTLET_REQUEST, inflow_km2=INFLOW, expected_area_km2=INFLOW + HALF_C3
    )
    assert hit.snap_quality == "area_gated" and info["area_gated"]
    assert hit.snapped_upa_km2 == pytest.approx(INFLOW + HALF_C3, rel=0.1)
    miss, _ = mh.snap_on_terminal_flowdir(
        terminal, nonterminal, path, *fx.OUTLET_REQUEST, inflow_km2=INFLOW, expected_area_km2=10_000.0
    )
    assert miss.snap_quality == mh.OFFLINE_FLOWDIR_SNAP_QUALITY + pipe.AREA_GATE_NO_MATCH_SUFFIX


def test_hybrid_snaps_offline_on_the_staged_flowdir(with_flowdir):
    res = _offline(*fx.OUTLET_REQUEST)
    flags = set(res.quality_flags)
    assert res.snap_quality == mh.OFFLINE_FLOWDIR_SNAP_QUALITY and res.snap_source == "local_flowdir_home_catchment"
    assert {"OFFLINE_FLOWDIR_SNAP_USED", "GEE_VALIDATION_UNAVAILABLE"} <= flags
    # no longer an unsnapped outlet: refined, no reliability mark, no version mix, nothing from Earth Engine
    assert not ({"OUTLET_UNSNAPPED", "OUTLET_UNSNAPPED_UNRELIABLE",
                 "TERMINAL_CATCHMENT_REFINEMENT_SKIPPED_UNSNAPPED", "MERIT_VERSION_MIX"} & flags)
    # not an unsnapped outlet, but a snap on flow direction alone is not validated: provisional, with the reason
    assert res.reliability == "provisional" and "flow direction only" in res.reliability_reason
    assert "OFFLINE_FLOWDIR_SNAP_UNVALIDATED" in flags and res.data_provenance["outlet_reliability"] is None
    assert res.terminal_refinement_used and res.terminal_catchment_id == fx.TERMINAL
    assert (res.outlet_lat, res.outlet_lon) == pytest.approx(fx.SNAPPED)
    assert res.area_km2 == pytest.approx(INFLOW + HALF_C3, rel=0.05)    # the same polygon the accumulation snap gives
    assert "local_flowdir_upa" in res.validation_sources
    assert res.data_provenance["offline_flowdir_snap"]["inflow_cells"] > 0
    assert ("offline_flowdir_snap", "succeeded") in [(h["method"], h["outcome"]) for h in res.fallback_history]


def test_hybrid_flowdir_snap_with_expected_area_is_a_real_remedy(with_flowdir):
    res = _offline(*fx.OUTLET_REQUEST, expected_area_km2=INFLOW + HALF_C3)
    assert res.snap_quality == "area_gated"
    assert res.area_validation["outlet_assembly"]["area_gate"]["matched"] is True
    assert res.reliability is None and "OFFLINE_FLOWDIR_SNAP_UNVALIDATED" not in res.quality_flags
    # an expected area that matches no cell leaves the snap unvalidated
    miss = _offline(*fx.OUTLET_REQUEST, expected_area_km2=10_000.0)
    assert miss.reliability == "provisional" and "matched no cell" in miss.reliability_reason


def test_hybrid_stages_the_flowdir_under_the_tiered_policy(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    seen: list[str] = []

    def fake_stage(self, pfaf, progress_cb=None):
        seen.append(pfaf)
        dest = self.raster_path("flowdir", pfaf)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(region.files["flowdir73.tif"], dest)
        return self.routing_region_cache(pfaf)

    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir", fake_stage)
    res = _offline(*fx.OUTLET_REQUEST)
    assert seen == [fx.PFAF] and res.snap_quality == mh.OFFLINE_FLOWDIR_SNAP_QUALITY
    assert "MERIT_AUTO_STAGED" in res.quality_flags
    assert [d["name"] for d in res.data_provenance["downloaded"] if d["name"].startswith("flowdir")] == ["flowdir73.tif"]
    assert {"kind": "flowdir"} .items() <= next(f for f in res.data_provenance["files"] if f["kind"] == "flowdir").items()


def test_auto_stage_false_never_stages(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stage")))
    res = _offline(*fx.OUTLET_REQUEST, auto_stage=False)
    assert res.snap_quality == "offline_unsnapped" and "OFFLINE_FLOWDIR_SNAP_FAILED" in res.quality_flags


def test_flowdir_snap_failure_falls_back_to_unsnapped_and_says_why(with_flowdir, monkeypatch):
    monkeypatch.setattr(mh, "snap_on_terminal_flowdir",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OFFLINE_FLOWDIR_SNAP_FAILED: test")))
    res = _offline(fx.SNAPPED[0], fx.SNAPPED[1])
    assert res.snap_quality == "offline_unsnapped" and not res.terminal_refinement_used
    assert {"OFFLINE_FLOWDIR_SNAP_FAILED", "TERMINAL_CATCHMENT_REFINEMENT_SKIPPED_UNSNAPPED"} <= set(res.quality_flags)
    failed = next(h for h in res.fallback_history if h["method"] == "offline_flowdir_snap")
    assert failed["outcome"] == "failed" and "test" in failed["reason"]
    # the remedy does not tell the user to stage what is already staged
    assert "staged but the offline snap did not complete" in " ".join(res.remedy)


# ------------------------------------------------------------------------------------------ P1: reliability


def test_unsnapped_far_from_the_reach_is_unreliable_at_the_top_level(vectors_only):
    res = _offline(40.0185, 0.05)                        # ~830 m north of the terminal reach, same unit catchment
    assert res.snap_quality == "offline_unsnapped"
    assert res.reliability == "unreliable" and "OUTLET_UNSNAPPED_UNRELIABLE" in res.quality_flags
    assert "m from the terminal reach" in res.reliability_reason and "offline_unsnapped" in res.reliability_reason
    assert res.reliability_checks["distance_to_reach_m"] > 300
    assert res.reliability_checks["thresholds_are_in_sample"] is True
    # the along-reach estimate is recorded next to the polygon, which is unchanged (the whole assembly)
    assert res.estimated_area_km2_on_reach is not None
    assert res.area_km2 == pytest.approx(res.vector_assembly_area_km2, rel=1e-6)
    assert res.data_provenance["outlet_reliability"]["status"] == "unreliable"
    assert ("unsnapped_outlet_check", "unreliable") in [(h["method"], h["outcome"]) for h in res.fallback_history]


def test_unsnapped_near_the_upstream_end_fails_the_over_ratio(vectors_only):
    res = _offline(fx.TRUNK_LAT, 0.0405)                 # on the reach, but at its upstream end: ~all of C3 is below it
    assert res.reliability == "unreliable"
    chk = res.reliability_checks
    assert chk["distance_to_reach_m"] < 300 and chk["over_ratio"] > mh.UNSNAPPED_OVER_RATIO_MAX
    assert res.estimated_area_km2_on_reach == pytest.approx(INFLOW + 0.025 * C3, rel=0.02)
    assert "x the" in res.reliability_reason and "m from the terminal reach" not in res.reliability_reason


def test_unsnapped_at_the_reach_outlet_passes_and_is_left_alone(vectors_only):
    res = _offline(fx.TRUNK_LAT, 0.0595)                 # the terminal reach's downstream end: the polygon is right
    assert res.reliability == "provisional" and res.snap_quality == "offline_unsnapped"
    assert "OUTLET_UNSNAPPED" in res.quality_flags and "OUTLET_UNSNAPPED_UNRELIABLE" not in res.quality_flags
    assert res.reliability_checks["over_ratio"] == pytest.approx(1.0, abs=0.05)
    assert res.area_km2 == pytest.approx(INFLOW + C3, rel=0.01)       # same polygon as before this change
    assert "passed the screen" in res.reliability_reason


def test_unsnapped_mid_reach_is_provisional_with_a_modest_over_ratio(vectors_only):
    res = _offline(fx.TRUNK_LAT, 0.05)
    assert res.reliability == "provisional"
    assert res.reliability_checks["fraction_along_reach"] == pytest.approx(0.5, abs=0.05)
    assert res.estimated_area_km2_on_reach == pytest.approx(INFLOW + HALF_C3, rel=0.05)
    assert res.reliability_checks["over_ratio"] == pytest.approx((INFLOW + C3) / (INFLOW + HALF_C3), rel=0.05)


def test_missing_reach_line_cannot_pass_the_screen(vectors_only, monkeypatch):
    monkeypatch.setattr(BasinsStore, "river_line", lambda self, comid: None)
    res = _offline(fx.TRUNK_LAT, 0.0595)
    assert res.reliability == "unreliable" and res.estimated_area_km2_on_reach is None
    assert "could not be tested" in res.reliability_reason


def test_remedy_is_accurate_for_what_the_code_does(vectors_only):
    res = _offline(40.0185, 0.05, expected_area_km2=INFLOW + HALF_C3)
    text = " ".join(res.remedy)
    assert res.remedy[0].startswith("Sign in to Google Earth Engine")
    assert "merit_ensure_routing_region(pfaf_region='73', acquisition_policy='download')" in text
    assert "MB" in text and "non-commercial" in text
    assert "does not change the outlet while it is unsnapped" in text      # the area is a check, not a remedy
    # and that statement is true: the area changes nothing about the outlet or the polygon
    base = _offline(40.0185, 0.05)
    assert (res.outlet_lat, res.outlet_lon, res.area_km2) == (base.outlet_lat, base.outlet_lon, base.area_km2)
    assert "AREA_GATE_NO_MATCH" not in res.quality_flags
    # staging was refused by policy and the record says so
    assert any(h["method"] == "flowdir_staging" and h["outcome"] == "not_staged" for h in res.fallback_history)


def test_large_confirm_required_staging_is_recorded(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))

    def too_big(self, pfaf, progress_cb=None):
        st = self.routing_region_cache(pfaf)
        st.message = "Pfaf 73 flowdir is large (~3.0 GB > 2 GB threshold)."
        st.action_required, st.size_gb = "confirm_large_download", 3.0
        return st

    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir", too_big)
    res = _offline(40.0185, 0.05)
    row = next(h for h in res.fallback_history if h["method"] == "flowdir_staging")
    assert row["outcome"] == "confirm_required" and row["size_gb"] == 3.0
    assert "Automatic staging did not run" in " ".join(res.remedy)


def test_snapped_outlet_has_no_reliability_mark(with_flowdir):
    """None means no concern detected, not validated: only an area-gated match or an EE/accumulation snap gets it."""
    res = _offline(*fx.OUTLET_REQUEST, expected_area_km2=INFLOW + HALF_C3)
    assert res.reliability is None and res.reliability_reason is None and res.reliability_checks is None


def test_flowdir_only_tier_marks_an_unsnapped_result_unreliable(with_flowdir, monkeypatch):
    from types import SimpleNamespace

    mgr = MeritDataManager()
    monkeypatch.setattr(MeritDataManager, "ensure_basin",
                        lambda self, lat, lon, download=False: SimpleNamespace(pfaf_code=fx.PFAF))
    res = pipe.local_merit_flowdir_pyflwdir(*fx.OUTLET_REQUEST, allow_offline=True, force_offline=True)
    assert res.snap_quality == "offline_unsnapped" and mgr.verified_asset("flowdir", fx.PFAF) is not None
    assert res.reliability == "unreliable" and "OUTLET_UNSNAPPED_UNRELIABLE" in res.quality_flags
    assert "not screened" in res.reliability_reason and "does not change the outlet" in " ".join(res.remedy)


# ------------------------------------------------------------------------------------------ assess (pure)


class _Store:
    def __init__(self, line):
        self._line = line

    def river_line(self, comid):
        return self._line


def test_assess_over_ratio_arithmetic():
    # a 1 km reach at 40 N running east to west (downstream first); point 25% of the way up it
    line = LineString([(0.0100, 40.0), (0.0, 40.0)])
    x = 0.0100 * 0.75
    r = mh.assess_unsnapped_outlet(_Store(line), 1, 40.0, x, returned_area_km2=110.0, terminal_area_km2=10.0,
                                   inflow_km2=100.0)
    assert r["checks"]["fraction_along_reach"] == pytest.approx(0.25, abs=0.01)
    assert r["estimated_area_km2_on_reach"] == pytest.approx(100.0 + 0.75 * 10.0, rel=0.01)
    assert r["checks"]["over_ratio"] == pytest.approx(110.0 / 107.5, rel=0.01)
    assert r["status"] == "provisional" and r["reasons"] == []


@pytest.mark.parametrize("terminal, inflow, fired", [(100.0, 0.0, "over"), (1.0, 100.0, None)])
def test_assess_headwater_and_large_basin(terminal, inflow, fired):
    line = LineString([(0.0100, 40.0), (0.0, 40.0)])
    r = mh.assess_unsnapped_outlet(_Store(line), 1, 40.0, 0.0005, returned_area_km2=terminal + inflow,
                                   terminal_area_km2=terminal, inflow_km2=inflow)
    assert (r["status"] == "unreliable") is (fired == "over")


def test_assess_distance_rule():
    line = LineString([(0.0100, 40.0), (0.0, 40.0)])
    far = mh.assess_unsnapped_outlet(_Store(line), 1, 40.004, 0.0099, returned_area_km2=10.0,
                                     terminal_area_km2=10.0, inflow_km2=0.0)
    near = mh.assess_unsnapped_outlet(_Store(line), 1, 40.002, 0.0001, returned_area_km2=10.0,
                                      terminal_area_km2=10.0, inflow_km2=0.0)
    assert any("m from the terminal reach" in r for r in far["reasons"])   # ~445 m
    assert not any("m from the terminal reach" in r for r in near["reasons"])  # ~222 m


# ------------------------------------------------------------------------------------------ router


@pytest.fixture()
def hybrid_stub(monkeypatch):
    out: dict = {}
    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", lambda lat, lon, **kw: _fake_result("merit_basins_hybrid", **out))
    monkeypatch.setattr(pipe, "merit_get_snap_reference", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EE")))
    monkeypatch.setattr(pipe, "merit_resolve_pfaf_region", lambda *a, **k: "62")
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": False})
    return out


def test_router_auto_surfaces_the_reliability_at_the_top_level(hybrid_stub):
    hybrid_stub.update(
        reliability="unreliable", reliability_reason="outlet could not be snapped (offline_unsnapped); test",
        remedy=["step one", "step two"], estimated_area_km2_on_reach=41.5,
        reliability_checks={"over_ratio": 3.0}, snap_quality="offline_unsnapped",
    )
    d = router.delineate_from_point(-3.0, -60.0).data      # outside CONUS: the hybrid is the default global tier
    assert d["method_used"] == "merit_basins_hybrid"
    assert d["reliability"] == "unreliable" and d["reliability_reason"].endswith("test")
    assert d["remedy"] == ["step one", "step two"] and d["estimated_area_km2_on_reach"] == 41.5
    assert d["reliability_checks"] == {"over_ratio": 3.0}
    assert d["geometry_geojson"]["properties"]["reliability"] == "unreliable"   # visible in the map popup


def test_router_result_without_a_concern_has_null_reliability(hybrid_stub):
    d = router.delineate_from_point(-3.0, -60.0).data
    assert d["reliability"] is None and d["remedy"] is None and d["estimated_area_km2_on_reach"] is None
    assert "reliability" not in d["geometry_geojson"]["properties"]


def test_router_stages_flowdir_in_conus_too(hybrid_stub, monkeypatch):
    """The router's auto-stage branch used to skip CONUS points; it now runs under the same tiered policy."""
    staged: list[str] = []

    def record(self, pfaf, progress_cb=None):
        staged.append(pfaf)
        return self.routing_region_cache(pfaf)

    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir", record)
    monkeypatch.setattr(pipe, "merit_resolve_pfaf_region", lambda *a, **k: "78")
    monkeypatch.setattr(pipe, "delineate_merit_gee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EE")))
    try:
        router.delineate_from_point(43.1769, -109.2029, method="merit_gee")   # CONUS
    except Exception:
        pass  # nothing downstream is served in this stub; only the staging call is under test
    assert staged == ["78"]


def test_router_prefers_the_hybrid_flowdir_snap_over_an_unsnapped_local_tier(hybrid_stub, monkeypatch):
    calls: list[str] = []
    local = _fake_result("local_merit_flowdir", snap_quality="offline_unsnapped", reliability="unreliable",
                         reliability_reason="local")
    hyb = _fake_result("merit_basins_hybrid", snap_quality=mh.OFFLINE_FLOWDIR_SNAP_QUALITY, reliability="provisional",
                       reliability_reason="hybrid")
    monkeypatch.setattr(pipe, "local_merit_flowdir_pyflwdir", lambda *a, **k: (calls.append("local"), local)[1])
    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", lambda *a, **k: (calls.append("hybrid"), hyb)[1])
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": True})
    d = router.delineate_from_point(40.0, -100.0, method="local_merit").data
    assert calls == ["local", "hybrid"]
    assert d["method_used"] == "merit_basins_hybrid" and d["reliability_reason"] == "hybrid"
    # and when the hybrid cannot serve, the local result stays with its own mark
    calls.clear()
    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate",
                        lambda *a, **k: (calls.append("hybrid"), (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED")))[1])
    d = router.delineate_from_point(40.0, -100.0, method="local_merit").data
    assert calls == ["local", "hybrid"] and d["method_used"] == "local_merit_pyflwdir" and d["reliability"] == "unreliable"


# ------------------------------------------------------------------------------------------ unsnapped has no snap distance


def test_unsnapped_outlet_has_no_snap_distance_and_is_not_validated_ok(vectors_only):
    res = _offline(fx.TRUNK_LAT, 0.0595)
    assert res.snap_quality == "offline_unsnapped"
    assert res.snap_distance_m is None
    assert res.snap_validation["snap_distance_m"] is None and res.snap_validation["ok"] is None
    assert "not snapped" in res.snap_validation["reason"]
    assert "OUTLET_SNAP_FAR" not in res.quality_flags


def test_validate_snap_handles_none_distance_and_keeps_snapped_behaviour():
    uns = pipe.MeritSnap(lat=1, lon=1, row=-1, col=-1, distance_m=None, snapped_upa_km2=None, snap_quality="offline_unsnapped")
    v, flags = pipe.validate_snap(uns, expected_area_km2=100.0)
    assert v["ok"] is None and flags == []
    ok = pipe.MeritSnap(lat=1, lon=1, row=0, col=0, distance_m=10.0, snapped_upa_km2=100.0, snap_quality="nearest_merit_stream")
    assert pipe.validate_snap(ok, expected_area_km2=100.0)[0]["ok"] is True
    far = pipe.MeritSnap(lat=1, lon=1, row=0, col=0, distance_m=6000.0, snapped_upa_km2=100.0, snap_quality="nearest_merit_stream")
    assert pipe.validate_snap(far, expected_area_km2=None)[1] == ["OUTLET_SNAP_FAR"]


def test_router_passes_a_none_snap_distance_through(hybrid_stub):
    hybrid_stub.update(snap_quality="offline_unsnapped", snap_distance_m=None, snap_validation={"ok": None})
    d = router.delineate_from_point(-3.0, -60.0).data
    assert d["snap_distance_m"] is None and "snap_distance_m" not in d["geometry_geojson"]["properties"]
    assert d["snap_validation"]["ok"] is None
