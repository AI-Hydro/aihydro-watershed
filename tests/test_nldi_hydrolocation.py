"""Offline tests: NLDI hydrolocation request form, retry, and the no-silent-downstream-jump rule.

Fixtures in ``tests/fixtures/nldi`` are recorded live responses (2026-11-05, api.water.usgs.gov):
the ``+`` form of the CAMELS 08155200 request (HTTP 502) and the ``%20`` form (HTTP 200), and the
``%20`` response for CAMELS 10336660. Basin areas in the 10336660 test (30 and 1414 km2) are the
pilot log's ("Quick COMID: start=8941693 (30 km2) -> selected=8934026 (1414 km2)"); they are
inputs to a fake NLDI, not live measurements.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from aihydro_core import ToolError
from aihydro_watershed.delineation import nldi_point as nl

FIX = Path(__file__).parent / "fixtures" / "nldi"


def _fx(name: str) -> str:
    return (FIX / name).read_text()


class Resp:
    def __init__(self, status, text="{}", headers=None):
        self.status_code = status
        self._text = text
        self.headers = headers or {}

    def json(self):
        return json.loads(self._text)


class Session:
    """Replays a script of responses (or exceptions) and records the URLs requested."""

    def __init__(self, script):
        self.script = list(script)
        self.urls = []

    def get(self, url, **kw):
        self.urls.append(url)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_url_is_percent_encoded_not_plus():
    url = nl.hydrolocation_url(-97.92557, 30.29632)
    assert url.endswith("hydrolocation?coords=POINT(-97.925570%2030.296320)")
    assert "+" not in url and " " not in url


def test_service_that_rejects_plus_form_is_served_by_our_form():
    """Mimics the recorded behaviour: '+' -> 502, '%20' -> 200 (08155200)."""

    class Service:
        def get(self, url, **kw):
            if "+" in url:
                return Resp(502, _fx("hydrolocation_plus_08155200.502.json"))
            return Resp(200, _fx("hydrolocation_pct20_08155200.200.json"))

    hit = nl.hydrolocation_comid(-97.92557, 30.29632, session=Service(), sleep=lambda s: None)
    assert hit.comid == 5781265
    assert hit.reachcode == "12090205000063"
    assert 30.0 < hit.lat < 30.6 and -98.1 < hit.lon < -97.8  # snapped point on the flowline


def test_recorded_502_body_names_the_failure():
    body = json.loads(_fx("hydrolocation_plus_08155200.502.json"))
    assert body["status"] == 502 and "Read timed out" in body["detail"]


def test_5xx_is_retried_with_backoff_then_succeeds():
    s = Session([Resp(502), Resp(503), Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))])
    waits = []
    hit = nl.hydrolocation_comid(-120.16214, 39.10741, session=s, sleep=waits.append)
    assert hit.comid == 8941693
    assert waits == [3.0, 8.0] and len(s.urls) == 3


def test_persistent_5xx_surfaces_http_status_not_a_geopandas_error():
    s = Session([Resp(502)] * 4)
    with pytest.raises(ToolError) as ei:
        nl.hydrolocation_comid(-97.92557, 30.29632, session=s, sleep=lambda x: None)
    assert ei.value.code == "NLDI_UNAVAILABLE"
    assert "HTTP 502" in str(ei.value) and "api.water.usgs.gov" in str(ei.value)
    assert len(s.urls) == 4


def test_retry_after_is_honoured():
    s = Session([Resp(429, headers={"Retry-After": "11"}), Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))])
    waits = []
    nl.hydrolocation_comid(-120.16214, 39.10741, session=s, sleep=waits.append)
    assert waits == [11.0]


def test_4xx_not_retried():
    s = Session([Resp(400)])
    with pytest.raises(ToolError) as ei:
        nl.hydrolocation_comid(-97.9, 30.3, session=s, sleep=lambda x: None)
    assert ei.value.code == "NLDI_UNAVAILABLE" and "HTTP 400" in str(ei.value) and len(s.urls) == 1


def test_connection_errors_retried():
    import requests

    s = Session([requests.ConnectionError("boom"), Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))])
    assert nl.hydrolocation_comid(-120.16214, 39.10741, session=s, sleep=lambda x: None).comid == 8941693


def test_nothing_indexed_is_not_found():
    only_provided = json.dumps({"features": [json.loads(_fx("hydrolocation_pct20_10336660.200.json"))["features"][1]]})
    with pytest.raises(ToolError) as ei:
        nl.hydrolocation_comid(-120.1, 39.1, session=Session([Resp(200, only_provided)]), sleep=lambda x: None)
    assert ei.value.code == "NLDI_NOT_FOUND"


# ---- 10336660: no silent downstream jump ---------------------------------
START, STEM = 8941693, 8934026
AREAS = {START: 30.0, STEM: 1414.0, 8934100: 2200.0}


class FakeNLDI:
    def get_basins(self, comid, fsource="comid"):
        c = int(comid)
        return gpd.GeoDataFrame({"cid": [c]}, geometry=[box(-120.3 - c % 7 * 0.01, 39.0, -120.1, 39.2)], crs=4326)

    def navigate_byid(self, *a, **k):
        import pandas as pd

        return pd.DataFrame({"nhdplus_comid": [START, STEM, 8934100]})


@pytest.fixture
def fake_nldi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynhd", types.SimpleNamespace(NLDI=FakeNLDI))
    monkeypatch.setattr(nl, "area_km2", lambda g: AREAS[int(g.cid.iloc[0])])
    monkeypatch.setattr(nl, "hydrolocation_comid", lambda lon, lat, **k: nl.HydrolocationHit(START, lon, lat))


def test_small_nearest_reach_is_kept_and_ambiguity_reported(fake_nldi):
    comid, area, amb = nl.quick_comid_resolution(39.10741, -120.16214)
    assert (comid, area) == (START, 30.0)
    assert amb["selected_comid"] == START and "outlet_moved_downstream" not in amb
    assert {c["comid"] for c in amb["candidates"]} == {START, STEM, 8934100}
    assert next(c for c in amb["candidates"] if c["comid"] == STEM)["area_km2"] == 1414.0
    assert "expected_area_km2" in amb["resolve_with"]


def test_explicit_consent_moves_downstream_and_says_so(fake_nldi):
    comid, area, amb = nl.quick_comid_resolution(39.10741, -120.16214, allow_downstream_expansion=True)
    assert comid == 8934100 or comid == STEM  # largest candidate within the 15,000 km2 cap
    assert amb["outlet_moved_downstream"] is True and amb["selected_comid"] == comid


def test_large_nearest_reach_has_no_ambiguity(fake_nldi, monkeypatch):
    monkeypatch.setitem(AREAS, START, 230.0)
    comid, area, amb = nl.quick_comid_resolution(44.36, -121.99)
    assert (comid, area, amb) == (START, 230.0, None)  # 14158500 shape: nothing to disambiguate


def test_delineate_result_carries_flag_and_nearest_reach(fake_nldi):
    res = nl.delineate_nldi_at_point(39.10741, -120.16214)
    assert res.data["comid"] == START and res.data["area_km2"] == 30.0
    assert res.data["quality_flags"] == ["NLDI_OUTLET_AMBIGUOUS"]
    assert res.data["outlet_ambiguity"]["selected_comid"] == START
    assert res.meta.params["allow_downstream_expansion"] is False


def test_expected_area_path_unchanged_and_not_flagged(fake_nldi, monkeypatch):
    monkeypatch.setattr(nl, "resolve_comid_for_area", lambda *a, **k: (STEM, 1414.0))
    res = nl.delineate_nldi_at_point(39.10741, -120.16214, expected_area_km2=1400.0)
    assert res.data["comid"] == STEM and "outlet_ambiguity" not in res.data


def test_router_forwards_consent_flag_default_false(monkeypatch):
    from aihydro_watershed.delineation import router

    seen = {}

    def fake(lat, lon, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(nl, "delineate_nldi_at_point", fake)
    with pytest.raises(ToolError):
        router.delineate_from_point(39.10741, -120.16214, method="nldi")
    assert seen == {"allow_downstream_expansion": False}
    seen.clear()
    with pytest.raises(ToolError):
        router.delineate_from_point(39.1, -120.1, method="nldi", allow_downstream_expansion=True)
    assert seen == {"allow_downstream_expansion": True}


def test_ambiguity_flag_is_in_the_nldi_trail_row(monkeypatch):
    from aihydro_watershed.delineation import router
    from aihydro_core import HydroMeta, HydroResult

    def fake(lat, lon, **kw):
        return HydroResult(
            data={"area_km2": 30.0, "quality_flags": ["NLDI_OUTLET_AMBIGUOUS"],
                  "method_used": "nldi_comid", "comid": 1},
            meta=HydroMeta(tool="t", version="0", params={}),
        )

    monkeypatch.setattr(nl, "delineate_nldi_at_point", fake)
    res = router.delineate_from_point(39.10741, -120.16214, method="nldi")
    row = [r for r in res.data["fallback_history"] if r["method"] == "nldi_comid"][-1]
    assert row["outcome"] == "served" and row["reason"] == "NLDI_OUTLET_AMBIGUOUS"


# ---- latency bound ---------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_memo(monkeypatch):
    monkeypatch.setattr(nl, "_unavailable_until", 0.0)
    monkeypatch.setattr(nl, "_exhausted_calls", 0)


def _exhaust(session=None):
    s = session or Session([Resp(502)] * 4)
    with pytest.raises(ToolError):
        nl.hydrolocation_comid(-97.9, 30.3, session=s, sleep=lambda x: None)
    return s


def test_one_exhausted_call_does_not_trip_the_memo_two_do(reset_memo):
    _exhaust()
    s_ok = Session([Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))])
    # still allowed to try: one exhausted call is not an outage; this 200 also resets the counter
    assert nl.hydrolocation_comid(-120.16214, 39.10741, session=s_ok, sleep=lambda x: None).comid == START
    _exhaust()
    _exhaust()  # two consecutive exhausted calls (the 200 reset the earlier one)
    s2 = Session([])
    with pytest.raises(ToolError) as ei:
        nl.hydrolocation_comid(-97.9, 30.3, session=s2, sleep=lambda x: None)
    assert ei.value.code == "NLDI_UNAVAILABLE" and "failing fast" in str(ei.value) and s2.urls == []


def test_200_resets_the_exhausted_counter(reset_memo):
    _exhaust()
    nl.hydrolocation_comid(
        -120.16214, 39.10741, session=Session([Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))]), sleep=lambda x: None
    )
    assert nl._exhausted_calls == 0
    _exhaust()
    assert nl._unavailable_until == 0.0  # one exhausted call after a 200: no trip


def test_404_does_not_set_the_memo(reset_memo):
    for _ in range(3):
        with pytest.raises(ToolError):
            nl.hydrolocation_comid(-97.9, 30.3, session=Session([Resp(404)]), sleep=lambda x: None)
    assert nl._unavailable_until == 0.0 and nl._exhausted_calls == 0


def test_deadline_spent_before_any_request_does_not_set_the_memo(reset_memo):
    s = Session([])
    for _ in range(3):
        with nl.nldi_deadline(-1.0):
            with pytest.raises(ToolError):
                nl.hydrolocation_comid(-97.9, 30.3, session=s, sleep=lambda x: None)
    assert s.urls == [] and nl._unavailable_until == 0.0 and nl._exhausted_calls == 0


def test_call_without_deadline_context_is_still_capped(reset_memo, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(nl.time, "monotonic", lambda: clock["t"])
    timeouts = []

    class S:
        def get(self, url, timeout=None, **kw):
            timeouts.append(timeout)
            clock["t"] += 80.0  # slow service
            return Resp(502)

    def sleep(w):
        clock["t"] += w

    with pytest.raises(ToolError):
        nl.hydrolocation_comid(-97.9, 30.3, session=S(), sleep=sleep)
    assert len(timeouts) == 2 and timeouts[0] == 30.0 and timeouts[1] == pytest.approx(7.0)  # 90 s cap, not 4 full tries


def test_slow_get_basins_is_not_waited_for(fake_nldi, monkeypatch):
    import time as _t

    class Slow(FakeNLDI):
        def get_basins(self, comid, fsource="comid"):
            if int(comid) != START:
                _t.sleep(5)
            return super().get_basins(comid, fsource)

    monkeypatch.setitem(sys.modules, "pynhd", types.SimpleNamespace(NLDI=Slow))
    t0 = _t.monotonic()
    with nl.nldi_deadline(0.5):
        comid, area, amb = nl.quick_comid_resolution(39.1, -120.1)
    assert _t.monotonic() - t0 < 3 and comid == START
    assert [c["comid"] for c in amb["candidates"]] == [START]


def test_memo_expires(reset_memo, monkeypatch):
    monkeypatch.setattr(nl, "_unavailable_until", nl.time.monotonic() - 1)
    s = Session([Resp(200, _fx("hydrolocation_pct20_10336660.200.json"))])
    assert nl.hydrolocation_comid(-120.16214, 39.10741, session=s, sleep=lambda x: None).comid == START


def test_deadline_stops_retries(reset_memo, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(nl.time, "monotonic", lambda: clock["t"])
    s = Session([Resp(502)] * 4)
    waits = []

    def sleep(w):
        waits.append(w)
        clock["t"] += w

    with nl.nldi_deadline(5.0):
        with pytest.raises(ToolError) as ei:
            nl.hydrolocation_comid(-97.9, 30.3, session=s, sleep=sleep)
    assert waits == [3.0] and len(s.urls) == 2  # the 8 s back-off would overrun the 5 s budget
    assert "deadline" in str(ei.value)


def test_expansion_requires_strictly_larger_basin(fake_nldi, monkeypatch):
    monkeypatch.setitem(AREAS, STEM, 30.0)
    monkeypatch.setitem(AREAS, 8934100, 30.0)
    comid, area, amb = nl.quick_comid_resolution(39.1, -120.1, allow_downstream_expansion=True)
    assert comid == START and "outlet_moved_downstream" not in amb


# ---- bounded worker pool, streak rules -----------------------------------
def test_pool_full_fails_fast_without_starting_a_thread(monkeypatch):
    import threading

    monkeypatch.setattr(nl, "_pool_slots", threading.BoundedSemaphore(2))
    release = threading.Event()
    started = []

    def hung():
        started.append(1)
        release.wait(10)

    try:
        for _ in range(2):
            with pytest.raises(ToolError):  # each hangs past its 0.2 s budget
                nl._bounded(hung, end=nl.time.monotonic() + 0.2)
        before = threading.active_count()
        with pytest.raises(ToolError) as ei:
            nl._bounded(hung, end=nl.time.monotonic() + 5)
        assert ei.value.code == "NLDI_UNAVAILABLE" and "busy" in str(ei.value)
        assert len(started) == 2 and threading.active_count() == before
    finally:
        release.set()


def test_slot_is_released_when_the_abandoned_call_finishes(monkeypatch):
    import threading
    import time as _t

    monkeypatch.setattr(nl, "_pool_slots", threading.BoundedSemaphore(1))
    with pytest.raises(ToolError):
        nl._bounded(lambda: _t.sleep(0.5), end=nl.time.monotonic() + 0.1)
    _t.sleep(0.8)
    assert nl._bounded(lambda: 7, end=nl.time.monotonic() + 2) == 7


def test_bounded_propagates_the_call_exception():
    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        nl._bounded(boom, end=nl.time.monotonic() + 2)


def test_first_get_basins_timeout_is_nldi_unavailable(fake_nldi, monkeypatch):
    import time as _t

    class Hung(FakeNLDI):
        def get_basins(self, comid, fsource="comid"):
            _t.sleep(3)
            return super().get_basins(comid, fsource)

    monkeypatch.setitem(sys.modules, "pynhd", types.SimpleNamespace(NLDI=Hung))
    with nl.nldi_deadline(0.3):
        with pytest.raises(ToolError) as ei:
            nl.quick_comid_resolution(39.1, -120.1)
    assert ei.value.code == "NLDI_UNAVAILABLE"


def test_final_get_basins_in_delineation_is_bounded(fake_nldi, monkeypatch):
    import time as _t

    class HangFinal(FakeNLDI):
        def get_basins(self, comid, fsource="comid"):
            _t.sleep(3)  # quick_comid_resolution is stubbed, so this is the final basin fetch
            return super().get_basins(comid, fsource)

    monkeypatch.setitem(sys.modules, "pynhd", types.SimpleNamespace(NLDI=HangFinal))
    monkeypatch.setattr(nl, "quick_comid_resolution", lambda *a, **k: (START, 30.0, None))
    t0 = _t.monotonic()
    with nl.nldi_deadline(0.3):
        with pytest.raises(ToolError) as ei:
            nl.delineate_nldi_at_point(39.1, -120.1)
    assert ei.value.code == "NLDI_UNAVAILABLE" and _t.monotonic() - t0 < 2.5


def test_5xx_then_4xx_is_not_an_exhausted_call(reset_memo):
    for _ in range(3):
        with pytest.raises(ToolError):
            nl.hydrolocation_comid(
                -97.9, 30.3, session=Session([Resp(502), Resp(404)]), sleep=lambda x: None
            )
    assert nl._exhausted_calls == 0 and nl._unavailable_until == 0.0


def test_4xx_between_exhausted_calls_resets_the_streak(reset_memo):
    _exhaust()
    assert nl._exhausted_calls == 1
    with pytest.raises(ToolError):
        nl.hydrolocation_comid(-97.9, 30.3, session=Session([Resp(404)]), sleep=lambda x: None)
    assert nl._exhausted_calls == 0
    _exhaust()
    assert nl._unavailable_until == 0.0  # not two in a row any more
