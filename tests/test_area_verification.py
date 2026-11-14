"""Offline tests: area verification against a reported drainage area, ambiguous outlet, reliability_state.

Audit 2026-11-14, P1 and P3. No network: the USGS lookup goes through an injected ``http_get`` (recorded body).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aihydro_watershed.delineation import area_verification as av
from aihydro_watershed.delineation import merit_flowdir_pipeline as pipe
from aihydro_watershed.delineation.router import delineate_from_point

from test_auto_fallback_honesty import LAT, LON, _hybrid_result, tiers  # noqa: F401  (fixture used by name)
from test_hybrid_outlet_consistency import HALF_C3, INFLOW, _snap_ref, ee_region  # noqa: F401
from test_merit_hybrid_default import region  # noqa: F401  (fixture used by name)
import merit_fixture as fx

FIX = Path(__file__).parent / "fixtures" / "usgs_waterdata" / "014_monitoring_locations_id_01013500.json"
SQMI = av.SQMI_TO_KM2


# --------------------------------------------------------------------------------- USGS provider (mocked)


def _resp(status=200, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body)


def _recorded_body():
    return json.loads(FIX.read_text())  # recorded: 01013500, drainage_area 870.0 (square miles)


def test_usgs_lookup_converts_square_miles_and_uses_the_documented_request():
    seen = {}

    def get(url, params=None, timeout=None, headers=None):
        seen.update(url=url, params=params, timeout=timeout, headers=headers)
        return _resp(200, _recorded_body())

    got = av.usgs_reported_area("01013500", http_get=get)
    assert got.area_km2 == pytest.approx(870.0 * SQMI) and got.reason is None
    assert seen["headers"]["User-Agent"].startswith("aihydro-watershed/")
    assert "contributing_drainage_area" in seen["params"]["properties"]
    assert got.contributing_km2 == pytest.approx(870.0 * SQMI)
    c = av.build_area_check(870.0 * SQMI, got)
    assert c["contributing_area_km2"] == pytest.approx(870.0 * SQMI, abs=1e-3) and c["status"] == "verified_by_area"
    assert seen["url"].startswith("https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations")
    assert seen["params"]["id"] == "USGS-01013500" and seen["timeout"] == av.USGS_LOOKUP_TIMEOUT_S


@pytest.mark.parametrize(
    "get, why",
    [
        (lambda *a, **k: (_ for _ in ()).throw(TimeoutError("read timed out")), "lookup failed: TimeoutError"),
        (lambda *a, **k: _resp(503, {}), "HTTP 503"),
        (lambda *a, **k: _resp(200, {"features": []}), "not a monitoring location"),
        (lambda *a, **k: _resp(200, {"features": [{"properties": {"drainage_area": None}}]}), "no drainage_area"),
        (lambda *a, **k: _resp(200, {"features": [{"properties": {"drainage_area": 0}}]}), "not a positive finite"),
        (lambda *a, **k: _resp(200, {"features": [{"properties": {"drainage_area": float("inf")}}]}), "not a positive finite"),
        (lambda *a, **k: _resp(200, {"features": [{"properties": {"drainage_area": float("nan")}}]}), "not a positive finite"),
        (lambda *a, **k: _resp(200, [1, 2]), "not a JSON object"),
        (lambda *a, **k: _resp(200, None), "not a JSON object"),
        (lambda *a, **k: _resp(200, {"features": [{"properties": {"drainage_area": "x"}}]}), "lookup failed"),
    ],
)
def test_usgs_lookup_failures_are_values_not_exceptions(get, why):
    got = av.usgs_reported_area("01013500", http_get=get)
    assert got.area_km2 is None and why in got.reason


@pytest.mark.parametrize("gid", ["GRDC-4115200", "NWIS-01013500", "usgs-01013500", "USGS-0101", "USGS-01013500x", ""])
def test_non_usgs_identifier_makes_no_request(gid):
    def get(*a, **k):
        raise AssertionError("no request expected")

    got = av.usgs_reported_area(gid, http_get=get)
    assert got.area_km2 is None and "not a USGS site number" in got.reason


# --------------------------------------------------------------------------------- resolution order


def test_caller_area_is_first_and_never_looks_up():
    def boom(_):
        raise AssertionError("provider must not run")

    r = av.resolve_reported_area(expected_area_km2=100.0, gauge_id="01013500", lookup=True, providers=[boom])
    assert (r.area_km2, r.source, r.independent) == (100.0, "caller_supplied", False)


def test_lookup_is_opt_in():
    def boom(_):
        raise AssertionError("provider must not run")

    r = av.resolve_reported_area(gauge_id="01013500", lookup=False, providers=[boom])
    assert r.area_km2 is None and "off" in r.reason


def test_lookup_without_gauge_id_and_raising_provider_are_unverified():
    assert "no gauge_id" in av.resolve_reported_area(lookup=True).reason

    def bad(_):
        raise RuntimeError("kaboom")

    r = av.resolve_reported_area(gauge_id="x", lookup=True, providers=[bad])
    assert r.area_km2 is None and "kaboom" in r.reason


def test_lookup_success_is_independent():
    ok = lambda g: av.ReportedArea(50.0, "stub")  # noqa: E731
    r = av.resolve_reported_area(gauge_id="x", lookup=True, providers=[ok])
    assert r.area_km2 == 50.0 and r.independent is True


# --------------------------------------------------------------------------------- check + state


@pytest.mark.parametrize(
    "poly, status",
    [(100.0, av.STATUS_VERIFIED), (110.0, av.STATUS_VERIFIED), (90.0, av.STATUS_VERIFIED),
     (120.0, av.STATUS_UNVERIFIED), (135.0, av.STATUS_UNVERIFIED), (136.0, av.STATUS_MISMATCH),
     (40.0, av.STATUS_MISMATCH)],
)
def test_thresholds(poly, status):
    c = av.build_area_check(poly, av.ReportedArea(100.0, "caller_supplied"))
    assert c["status"] == status and c["reported_area_km2"] == 100.0
    assert c["rel_err"] == pytest.approx((poly - 100.0) / 100.0)
    assert (c["tol_verify"], c["tol_flag"]) == (av.TOL_VERIFY, av.TOL_FLAG) == (0.10, 0.35)


def test_no_reported_area_is_unverified_with_reason():
    c = av.build_area_check(100.0, av.ReportedArea(None, "none", "nothing"))
    assert c["status"] == av.STATUS_UNVERIFIED and c["rel_err"] is None and c["reason"] == "nothing"


def test_reliability_state_three_way():
    v = {"status": av.STATUS_VERIFIED}
    m = {"status": av.STATUS_MISMATCH}
    u = {"status": av.STATUS_UNVERIFIED}
    assert av.reliability_state(v, False) == "verified"
    assert av.reliability_state(v, True) == "ambiguous"     # a passed check never clears an ambiguous outlet
    assert av.reliability_state(u, False, legacy_unreliable=True) == "ambiguous"
    assert av.reliability_state(m, False) == "ambiguous"
    assert av.reliability_state(u, True) == "ambiguous"
    assert av.reliability_state(u, False) == "unverified" and av.reliability_state(None, False) == "unverified"


# --------------------------------------------------------------------------------- router integration


def _serve(tiers, area=485.0, **kw):
    tiers["behaviour"]["hybrid"] = lambda: _hybrid_result(area=area, **kw)


def test_verified_by_caller_area_polygon_unchanged(tiers):
    _serve(tiers)
    d = delineate_from_point(LAT, LON, method="merit_basins", expected_area_km2=470.0).data
    assert d["area_check"]["status"] == "verified_by_area" and d["area_check"]["source"] == "caller_supplied"
    assert d["reliability_state"] == "verified" and d["reliability"] is None
    assert d["area_km2"] == 485.0
    assert "AREA_MISMATCH" not in d["quality_flags"]


def test_mismatch_is_unreliable_with_code_and_keeps_the_polygon(tiers):
    _serve(tiers, area=485.0)
    d = delineate_from_point(LAT, LON, method="merit_basins", expected_area_km2=1000.0).data
    assert d["area_check"]["status"] == "area_mismatch"
    assert d["reliability"] == "unreliable" and d["reliability_reason"].startswith("AREA_MISMATCH")
    assert "AREA_MISMATCH" in d["quality_flags"] and d["reliability_state"] == "ambiguous"
    assert d["area_km2"] == 485.0                           # never silently changed


def test_unverified_when_no_area(tiers):
    _serve(tiers)
    d = delineate_from_point(LAT, LON, method="merit_basins").data
    assert d["area_check"]["status"] == "area_unverified" and d["area_check"]["reported_area_km2"] is None
    assert d["reliability_state"] == "unverified" and d["reliability"] is None


def test_lookup_success_failure_and_default_off(tiers, monkeypatch):
    _serve(tiers, area=485.0)
    calls = []

    def fake_usgs(gauge_id, **k):
        calls.append(gauge_id)
        return av.ReportedArea(480.0, "usgs_waterdata_monitoring_locations")

    monkeypatch.setattr(av, "usgs_reported_area", fake_usgs)
    off = delineate_from_point(LAT, LON, method="merit_basins", gauge_id="01013500").data
    assert calls == [] and off["area_check"]["status"] == "area_unverified"
    on = delineate_from_point(LAT, LON, method="merit_basins", gauge_id="01013500", lookup_reported_area=True).data
    assert calls == ["01013500"] and on["area_check"]["status"] == "verified_by_area"
    assert on["area_check"]["independent_of_snap"] is True and on["reliability_state"] == "verified"

    monkeypatch.setattr(av, "usgs_reported_area", lambda g, **k: av.ReportedArea(None, "usgs", "lookup failed: TimeoutError"))
    fail = delineate_from_point(LAT, LON, method="merit_basins", gauge_id="01013500", lookup_reported_area=True).data
    assert fail["area_check"]["status"] == "area_unverified" and "TimeoutError" in fail["area_check"]["reason"]
    assert fail["reliability_state"] == "unverified"


def test_mismatch_keeps_an_earlier_reliability_reason(tiers):
    _serve(tiers, reliability="provisional", reliability_reason="flow direction only")
    d = delineate_from_point(LAT, LON, method="merit_basins", expected_area_km2=1000.0).data
    assert d["reliability"] == "unreliable" and "also: flow direction only" in d["reliability_reason"]


def test_legacy_provisional_is_unchanged_and_state_is_unverified(tiers):
    _serve(tiers, reliability="provisional", reliability_reason="snapped offline on flow direction only")
    d = delineate_from_point(LAT, LON, method="merit_basins").data
    assert d["reliability"] == "provisional" and d["reliability_state"] == "unverified"
    assert d["ambiguous_outlet"] is None


def test_ambiguous_outlet_without_area_is_ambiguous(tiers):
    amb = {"code": "AMBIGUOUS_OUTLET", "candidates": [{"id": "local_polygon"}, {"id": "mainstem_alternative"}]}
    kw = dict(reliability="unreliable", reliability_reason="AMBIGUOUS_OUTLET: x", ambiguous_outlet=amb,
              quality_flags=["HYBRID_ROUTING_USED", "AMBIGUOUS_OUTLET"])
    _serve(tiers, **kw)
    d = delineate_from_point(LAT, LON, method="merit_basins").data
    assert d["reliability"] == "unreliable" and d["reliability_state"] == "ambiguous"
    assert d["ambiguous_outlet"]["candidates"][1]["id"] == "mainstem_alternative"


def test_passing_area_does_not_clear_an_ambiguous_outlet(tiers, monkeypatch):
    amb = {"code": "AMBIGUOUS_OUTLET", "candidates": [{"id": "local_polygon"}]}
    _serve(tiers, reliability="unreliable", reliability_reason="AMBIGUOUS_OUTLET: x", ambiguous_outlet=amb,
           quality_flags=["HYBRID_ROUTING_USED", "AMBIGUOUS_OUTLET"])
    monkeypatch.setattr(av, "usgs_reported_area", lambda g, **k: av.ReportedArea(480.0, "stub"))
    d = delineate_from_point(LAT, LON, method="merit_basins", gauge_id="01013500", lookup_reported_area=True).data
    assert d["area_check"]["status"] == "verified_by_area"
    assert d["reliability_state"] == "ambiguous" and d["reliability"] == "unreliable"
    assert "AMBIGUOUS_OUTLET" in d["quality_flags"] and d["ambiguous_outlet"]["candidates"]
    assert d["ambiguous_outlet"]["area_check_status"] == "verified_by_area"
    assert d["reliability_state_basis"] == "independent_lookup"


def test_state_basis_does_not_overclaim_a_caller_area(tiers):
    _serve(tiers)
    d = delineate_from_point(LAT, LON, method="merit_basins", expected_area_km2=470.0).data
    assert d["reliability_state"] == "verified" and d["reliability_state_basis"] == "caller_area_also_steered_snap"
    assert "also steered the snap" in d["area_check"]["reason"] and "area-consistency" in d["area_check"]["reason"]
    none = delineate_from_point(LAT, LON, method="merit_basins").data
    assert none["reliability_state_basis"] == "none"


def test_unreliable_without_area_evidence_is_not_reported_unverified(tiers):
    _serve(tiers, reliability="unreliable", reliability_reason="outlet could not be snapped (offline_unsnapped)")
    d = delineate_from_point(LAT, LON, method="merit_basins").data
    assert d["reliability"] == "unreliable" and d["reliability_state"] == "ambiguous"


# --------------------------------------------------------------------------------- hybrid: ambiguous outlet


def test_off_reach_without_area_is_unreliable_with_both_candidates(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.reliability == "unreliable" and res.reliability_reason.startswith("AMBIGUOUS_OUTLET")
    assert "AMBIGUOUS_OUTLET" in res.quality_flags
    assert res.area_km2 == pytest.approx(HALF_C3, rel=0.02)           # polygon returned as before
    amb = res.ambiguous_outlet
    local, main = amb["candidates"]
    assert (local["id"], main["id"]) == ("local_polygon", "mainstem_alternative")
    assert local["area_km2"] == pytest.approx(res.area_km2, rel=1e-3)
    assert main["area_km2"] == pytest.approx(res.area_validation["outlet_assembly"]["mainstem_alternative"]["area_km2"])
    assert main["area_km2"] > local["area_km2"] and main["source"] == "merit_basins_rivers_uparea"
    assert main["outlet_lat"] is not None and main["outlet_lon"] is not None
    assert (local["outlet_lat"], local["outlet_lon"]) == (res.outlet_lat, res.outlet_lon)


def test_off_reach_with_expected_area_is_not_ambiguous(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=HALF_C3)
    for exp in (HALF_C3 * 1.05, INFLOW + HALF_C3):                    # local kept / main stem kept
        res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, expected_area_km2=exp)
        assert res.ambiguous_outlet is None and "AMBIGUOUS_OUTLET" not in res.quality_flags
        assert res.reliability != "unreliable"


def test_on_reach_outlet_is_not_ambiguous(ee_region, monkeypatch):
    _snap_ref(monkeypatch, upa=INFLOW + HALF_C3)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.ambiguous_outlet is None and res.reliability is None


def test_mainstem_candidate_outlet_is_the_downstream_end_of_the_reach(ee_region, monkeypatch):
    # river lines are digitised downstream -> upstream (basins_store.river_line): the outlet is coords[0]
    from shapely.geometry import LineString

    from aihydro_watershed.merit.basins_store import BasinsStore

    monkeypatch.setattr(BasinsStore, "river_line", lambda self, comid: LineString([(0.0123, 40.0456), (0.05, 40.03), (0.09, 40.01)]))
    _snap_ref(monkeypatch, upa=HALF_C3)
    main = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST).ambiguous_outlet["candidates"][1]
    assert (main["outlet_lon"], main["outlet_lat"]) == (0.0123, 40.0456)
    assert "first vertex" in main["note"]

