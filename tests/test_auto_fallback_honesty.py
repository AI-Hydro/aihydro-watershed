"""CONUS ``auto``: a failed NLDI must not silently hand the point to a less accurate tier.

Offline: NLDI, the MERIT hybrid and the 3DEP tier are stubbed. Trails mirror the ones recorded in the CAMELS
delineation evaluation (HTTP 502/429 at hydrolocation, ``Service returned no features`` and
``string indices must be integers`` from a throttled basin fetch).
"""
from __future__ import annotations

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
    served = d["fallback_history"][-1]
    assert served["outcome"] == "succeeded" and served["reason"].startswith("last resort after NLDI transient failure")

    # the same tier with a known failure signature is rejected, not served
    tiers["behaviour"]["sc"] = lambda: _sc(area=190.0, flags=["BASIN_TOUCHES_MAX_WINDOW"])
    with pytest.raises(ToolError) as caught:
        delineate_from_point(LAT, LON, method="auto")
    rows = caught.value.details["fallback_history"]
    assert any(r["method"] == "small_catchment_3dep" and r["outcome"] == "rejected"
               and "BASIN_TOUCHES_MAX_WINDOW" in r["reason"] for r in rows)


def test_ditch_snap_is_rejected_as_a_stand_in(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    tiers["behaviour"]["hybrid"] = lambda: (_ for _ in ()).throw(RuntimeError("MERIT_BASINS_NOT_STAGED"))
    tiers["behaviour"]["sc"] = lambda: _sc(area=0.006, flags=["LIKELY_DITCH_SNAP"])
    with pytest.raises(ToolError) as caught:
        delineate_from_point(LAT, LON, method="auto")
    assert any(r["outcome"] == "rejected" and "LIKELY_DITCH_SNAP" in r["reason"]
               for r in caught.value.details["fallback_history"])


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


def test_forced_nldi_failure_still_raises(monkeypatch, tiers):
    _nldi_raises(monkeypatch, _unavailable(429))
    with pytest.raises(ToolError):
        delineate_from_point(LAT, LON, method="nldi")
    assert tiers["hybrid"] == [] and tiers["sc"] == []


def test_transient_classifier():
    assert _nldi_failure_is_transient(_unavailable(429))
    assert _nldi_failure_is_transient(RuntimeError("HTTP 502"))
    assert not _nldi_failure_is_transient(ToolError(code="NLDI_NOT_FOUND", message="No NHD COMID indexed near", tool="t"))
    assert not _nldi_failure_is_transient(KeyError("comid"))
