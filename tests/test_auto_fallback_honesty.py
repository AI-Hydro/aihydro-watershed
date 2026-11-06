"""CONUS ``auto``: a failed NLDI must not silently hand the point to a less accurate tier.

Offline: NLDI, the MERIT hybrid and the 3DEP tier are stubbed. Trails mirror the ones recorded in the CAMELS
delineation evaluation (HTTP 502/429 at hydrolocation, ``Service returned no features`` and
``string indices must be integers`` from a throttled basin fetch).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import geopandas as gpd
import pytest
from aihydro_core import HydroMeta, HydroResult, ToolError
from shapely.geometry import box

from aihydro_watershed.delineation import merit_flowdir_pipeline as pipe
from aihydro_watershed.delineation import nldi_point, router, small_catchment
from aihydro_watershed.delineation.router import _nldi_failure_is_transient, delineate_from_point
from aihydro_watershed.merit.merit_manager import MeritDataManager

LAT, LON = 43.17690, -109.20291   # CONUS, no expected area


def _gdf():
    return gpd.GeoDataFrame(geometry=[box(-109.3, 43.1, -109.2, 43.2)], crs=4326)


def _hybrid_result(area=485.0, **kw):
    base = dict(
        gdf=_gdf(), area_km2=area, outlet_lat=LAT, outlet_lon=LON, snap_distance_m=30.0, snapped_upa_km2=None,
        snap_quality="x", snap_validation={}, cache_key="k", bbox=(0, 0, 1, 1), scout_box_maxed=False,
        touches_edge=False, area_validation={}, quality_flags=["HYBRID_ROUTING_USED"],
        source="merit_basins_hybrid", routing_dataset="d", routing_resolution_m=90.0, pfaf_region="78",
        terminal_catchment_id=1, data_provenance={"files": []}, fallback_history=[],
    )
    base.update(kw)
    return pipe.MeritFlowdirResult(**base)


def _sc(area=485.0, flags=()):
    return SimpleNamespace(
        gdf=_gdf(), area_km2=area, resolution_m=10.0, snap_distance_m=30.0, outlet_lon=LON, outlet_lat=LAT,
        window_iterations=3, window_bounds_5070=None, status="ok", quality_flags=list(flags), notch_elevation_m=1.0,
    )


def _nldi_ok(area):
    return HydroResult(
        data={"area_km2": area, "method_used": "nldi_comid", "comid": 1},
        meta=HydroMeta(tool="test", version="0", params={}, sources=[]),
    )


@pytest.fixture()
def tiers(monkeypatch):
    """Stub every tier; tests set ``calls`` entries to control behaviour and read what ran."""
    calls = {"hybrid": [], "sc": [], "gee": []}
    behaviour = {"hybrid": lambda: _hybrid_result(), "sc": lambda: _sc(), "gee": None}

    def hybrid(lat, lon, **kw):
        calls["hybrid"].append(kw)
        return behaviour["hybrid"]()

    def sc(lat, lon, **kw):
        calls["sc"].append(1)
        return behaviour["sc"]()

    def gee(lat, lon, **kw):
        calls["gee"].append(kw)
        if behaviour["gee"] is None:
            raise RuntimeError("GEE not available offline")
        return behaviour["gee"]()

    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", hybrid)
    monkeypatch.setattr(pipe, "delineate_merit_gee", gee)
    monkeypatch.setattr(small_catchment, "delineate_small_catchment", sc)
    monkeypatch.setattr(pipe, "merit_get_snap_reference", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EE")))
    monkeypatch.setattr(pipe, "merit_resolve_pfaf_region", lambda *a, **k: "78")
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": False})
    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir", lambda self, pfaf, **k: self.routing_region_cache(pfaf))
    calls["behaviour"] = behaviour
    return calls


def _nldi_raises(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", boom)


def _unavailable(status):
    return ToolError(
        code="NLDI_UNAVAILABLE",
        message=f"NLDI hydrolocation failed for ({LAT}, {LON}): HTTP {status} (host api.water.usgs.gov).",
        tool="t",
    )


def test_429_then_hybrid_served_and_flagged(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "merit_basins_hybrid"
    assert tiers["sc"] == []                      # the 3DEP tier never ran
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" in d["quality_flags"]
    assert "HYBRID_ROUTING_USED" in d["quality_flags"]   # tier flags survive the merge
    failed, served = d["fallback_history"]
    assert (failed["method"], failed["outcome"], failed["transient"]) == ("nldi_comid", "failed", True)
    assert "HTTP 429" in failed["reason"]
    assert served["method"] == "merit_basins_hybrid" and served["outcome"] == "succeeded"
    assert "NLDI transient failure" in served["reason"] and "HTTP 429" in served["reason"]


@pytest.mark.parametrize(
    "exc",
    [
        _unavailable(502),
        RuntimeError("Service returned no features."),                # pynhd on a throttled basin fetch
        TypeError("string indices must be integers, not 'str'"),      # pynhd parsing a 429 body
        RuntimeError("Cannot connect to host api.water.usgs.gov:443 ssl:default [Timeout while contacting DNS servers]"),
    ],
)
def test_5xx_and_throttle_shapes_fall_back_to_hybrid_not_3dep(monkeypatch, tiers, exc):
    _nldi_raises(monkeypatch, exc)
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "merit_basins_hybrid" and tiers["sc"] == []
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" in d["quality_flags"]


def test_expected_area_path_also_prefers_hybrid_after_transient_failure(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(503))
    d = delineate_from_point(LAT, LON, method="auto", expected_area_km2=480.0).data
    assert d["method_used"] == "merit_basins_hybrid" and tiers["gee"] == []   # hybrid ahead of merit_gee
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" in d["quality_flags"]


def test_hybrid_unavailable_falls_to_3dep_only_as_guarded_last_resort(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))

    # a clean 3DEP result is accepted, with the fallback flag and the reason on the trail
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "small_catchment_3dep"
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" in d["quality_flags"]
    assert "LAST_RESORT_UNVERIFIED" not in d["quality_flags"]
    served = d["fallback_history"][-1]
    assert served["outcome"] == "succeeded" and served["reason"].startswith("last resort after NLDI transient failure")


@pytest.mark.parametrize("flag", ["BASIN_TOUCHES_MAX_WINDOW", "LIKELY_DITCH_SNAP"])
def test_rejected_3dep_is_served_unverified_when_every_later_tier_fails(monkeypatch, tiers, flag):
    _nldi_raises(monkeypatch, _unavailable(429))
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))
    tiers["behaviour"]["sc"] = lambda: _sc(area=190.0, flags=[flag])
    monkeypatch.setattr(
        "aihydro_watershed.delineation.pysheds_pipeline.delineate_fast",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw DEM offline")),
    )
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "small_catchment_3dep"
    assert {flag, "LAST_RESORT_UNVERIFIED", "FALLBACK_AFTER_TRANSIENT_NLDI"} <= set(d["quality_flags"])
    assert len(tiers["sc"]) == 1                      # the rejected result is kept, not recomputed
    rows = d["fallback_history"]
    assert any(r["method"] == "small_catchment_3dep" and r["outcome"] == "rejected" and flag in r["reason"] for r in rows)
    assert rows[-1]["method"] == "small_catchment_3dep" and rows[-1]["outcome"] == "succeeded_unverified"


def test_rejected_3dep_gives_way_to_raw_dem_when_that_serves(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))
    tiers["behaviour"]["sc"] = lambda: _sc(area=190.0, flags=["BASIN_TOUCHES_MAX_WINDOW"])
    fast = SimpleNamespace(
        gdf=_gdf(), area_km2=480.0, pfaf_code="78", merit_snap_distance_m=10.0, snapped_outlet_lon=LON,
        snapped_outlet_lat=LAT, snap_grid_crs="EPSG:4326", snap_grid_resolution_m=90.0,
    )
    monkeypatch.setattr("aihydro_watershed.delineation.pysheds_pipeline.delineate_fast", lambda *a, **k: fast)
    monkeypatch.setattr(router, "_should_escalate", lambda *a, **k: None)
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "dem_raw_fallback"
    assert "LAST_RESORT_UNVERIFIED" not in d["quality_flags"]


def test_ditch_snap_with_every_tier_down_is_served_unverified(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))
    tiers["behaviour"]["sc"] = lambda: _sc(area=0.006, flags=["LIKELY_DITCH_SNAP"])
    monkeypatch.setattr(
        "aihydro_watershed.delineation.pysheds_pipeline.delineate_fast",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw DEM offline")),
    )
    d = delineate_from_point(LAT, LON, method="auto").data
    assert "LAST_RESORT_UNVERIFIED" in d["quality_flags"] and "LIKELY_DITCH_SNAP" in d["quality_flags"]


@pytest.mark.parametrize(
    "local_error", ["ADAPTIVE_WINDOW_LIMIT_REACHED", "LOCAL_MEMORY_THRESHOLD_EXCEEDED", "HYBRID_ROUTING_REQUIRED x"]
)
def test_failing_cached_local_tier_does_not_end_the_chain(monkeypatch, tiers, local_error):
    """flowdir_ready + a local-tier failure used to raise LARGE_BASIN_HYBRID_REQUIRED and stop the chain."""
    _nldi_raises(monkeypatch, _unavailable(429))
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": True})
    monkeypatch.setattr(
        pipe, "local_merit_flowdir_pyflwdir", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(local_error))
    )
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))

    # merit_gee serves
    tiers["behaviour"]["gee"] = lambda: SimpleNamespace(
        gdf=_gdf(), area_km2=480.0, snap_distance_m=1.0, outlet_lon=LON, outlet_lat=LAT, routing_resolution_m=90.0,
        snap_quality="x", snap_validation={}, snapped_upa_km2=None, official_merit_upa_km2=None,
        local_upstream_area_km2=None, polygon_area_km2=480.0, area_validation={}, quality_flags=[], cache_key="k",
        routing_data_source=None, snap_source="gee", validation_sources=[], regional_flowdir_cached=True,
    )
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "merit_gee_pyflwdir" and len(tiers["gee"]) == 1
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" in d["quality_flags"]
    assert any(r["method"] == "local_merit_flowdir_pyflwdir" and r["outcome"] == "failed" for r in d["fallback_history"])

    # merit_gee also down: the guarded 3DEP tier is still reached
    tiers["behaviour"]["gee"] = None
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "small_catchment_3dep" and len(tiers["sc"]) == 1


def test_failing_cached_local_tier_still_raises_for_forced_local_merit(monkeypatch, tiers):
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": True})
    monkeypatch.setattr(
        pipe, "local_merit_flowdir_pyflwdir",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ADAPTIVE_WINDOW_LIMIT_REACHED")),
    )
    with pytest.raises(ToolError) as caught:
        delineate_from_point(LAT, LON, method="local_merit")
    assert caught.value.code == "LARGE_BASIN_HYBRID_REQUIRED"


def test_source_disagreement_flagged_when_nldi_area_and_served_area_differ(monkeypatch, tiers):
    # NLDI answered 3000 km2 (outside the quick range, so it was skipped); the hybrid serves 485 km2.
    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_ok(20_000.0))
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "merit_basins_hybrid"
    assert "SOURCE_DISAGREEMENT" in d["quality_flags"]
    assert "FALLBACK_AFTER_NLDI_NOT_USABLE" in d["quality_flags"]
    assert "FALLBACK_AFTER_TRANSIENT_NLDI" not in d["quality_flags"]
    (sd,) = d["source_disagreement"]
    assert sd["source"] == "nldi_comid" and sd["served_method"] == "merit_basins_hybrid"
    assert sd["threshold"] == 0.25 and sd["relative_difference"] > 0.25


def test_no_source_disagreement_when_areas_agree(monkeypatch, tiers):
    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_ok(20_000.0))
    tiers["behaviour"]["hybrid"] = lambda: _hybrid_result(area=19_000.0)   # within 25 % of NLDI's 20 000
    d = delineate_from_point(LAT, LON, method="auto").data
    assert "SOURCE_DISAGREEMENT" not in d["quality_flags"] and d["source_disagreement"] is None


def test_plain_nldi_success_is_unchanged(monkeypatch, tiers):
    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_ok(420.0))
    res = delineate_from_point(40.71829, -96.41265, method="auto")
    assert res.data["fallback_history"] == [{"method": "nldi_comid", "outcome": "served", "reason": ""}]
    assert not res.data.get("quality_flags")
    assert tiers["hybrid"] == [] and tiers["sc"] == []


def test_nldi_tiny_reach_still_goes_to_3dep(monkeypatch, tiers):
    """NLDI answered with a sub-km2 reach: the culvert-scale case the 3DEP tier is for (unchanged)."""
    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_ok(0.3))
    tiers["behaviour"]["sc"] = lambda: _sc(area=0.31)
    d = delineate_from_point(LAT, LON, method="auto").data
    assert d["method_used"] == "small_catchment_3dep" and tiers["hybrid"] == []
    # serving the culvert-scale case is by design, not a fallback
    assert not set(d.get("quality_flags") or []) & {
        "FALLBACK_AFTER_NLDI_NOT_USABLE", "FALLBACK_AFTER_TRANSIENT_NLDI", "SOURCE_DISAGREEMENT",
    }


def test_forced_nldi_failure_still_raises(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    with pytest.raises(ToolError):
        delineate_from_point(LAT, LON, method="nldi")
    assert tiers["hybrid"] == [] and tiers["sc"] == []


def test_transient_classifier():
    assert _nldi_failure_is_transient(_unavailable(429))
    assert _nldi_failure_is_transient(RuntimeError("HTTP 502"))
    assert _nldi_failure_is_transient(RuntimeError("server replied status code: 503"))
    assert _nldi_failure_is_transient(RuntimeError("connection reset by peer"))
    # Real library text leads with the bare code and never says "HTTP".
    assert _nldi_failure_is_transient(RuntimeError("502, message='Bad Gateway', url='https://api.water.usgs.gov/nldi'"))
    assert _nldi_failure_is_transient(RuntimeError("500, message='Internal Server Error', url='https://x'"))
    assert _nldi_failure_is_transient(RuntimeError("502 Server Error: Bad Gateway for url: https://x"))
    status_exc = RuntimeError("")
    status_exc.status = 503
    assert _nldi_failure_is_transient(status_exc)
    assert not _nldi_failure_is_transient(ToolError(code="NLDI_NOT_FOUND", message="No NHD COMID indexed near", tool="t"))
    assert not _nldi_failure_is_transient(KeyError("comid"))


@pytest.mark.parametrize(
    "msg",
    [
        "bad geometry at (40.512, -100.5)",
        "No COMID within 500 m",
        "Invalid geometry area 530 km2",
        "client disconnected from the layer",
        "Unavailable layer name 'x'",
    ],
)
def test_transient_classifier_ignores_numbers_and_words_that_only_look_like_outages(msg):
    assert not _nldi_failure_is_transient(RuntimeError(msg))


@pytest.mark.parametrize("exc", [TimeoutError(), asyncio.TimeoutError(), ConnectionResetError(), ConnectionError("")])
def test_transient_classifier_uses_the_exception_type_when_the_message_is_empty(exc):
    assert _nldi_failure_is_transient(exc)


def test_transient_classifier_follows_the_cause_chain():
    try:
        try:
            raise TimeoutError()
        except TimeoutError as inner:
            raise RuntimeError("") from inner
    except RuntimeError as outer:
        assert _nldi_failure_is_transient(outer)
