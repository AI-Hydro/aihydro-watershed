"""Wave U / U2: watershed consumes the USGS Water Data API through the aihydro-data adapter.

Fully offline: the adapter's single network seam (``usgs_waterdata._http_get``) replays the
recorded bodies in tests/fixtures/usgs_waterdata/ (copied from aihydro-data, not re-recorded).

Proof-site expectations below are the e2e-proof-1 / proof-recompute-pinned values for gauge
01013500, WY1990-2009 (docs/vision-2040/evidence/proof-recompute-pinned/results.json).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from _usgs_replay import EMPTY, Replay, doc, ok
from aihydro_data.sources import usgs_waterdata as uw
from aihydro_watershed.signatures import _streamflow_internal as si
from aihydro_watershed.signatures import signatures as sig

SITE = "01013500"
START, END = "1989-10-01", "2009-09-30"
AREA = 2258.5163954900077
POLY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}

# proof-1 recorded == pinned recompute (precipitation skipped)
PROOF1 = {"baseflow_index": 0.5853970180143581, "high_q_freq": 6.1, "q_mean": 1.6947786854267857,
          "high_q_dur": 8.714285714285714, "low_q_freq": 41.35, "hfd_mean": 207.25}
# proof-1's q came through aihydro-data's rounded constant 0.028316847; the adapter uses the
# exact 0.028316846592 -> every value scales by exactly this ratio (relative 1.44e-8).
CONST_RATIO = 0.028316846592 / 0.028316847


@pytest.fixture
def replay(monkeypatch):
    r = Replay()
    monkeypatch.setattr(uw, "_http_get", r)
    return r


def test_fetch_delegates_to_adapter_and_keeps_quality(replay):
    out = si._fetch_streamflow_internal(SITE, START, END)
    q = out["q_cms"]
    assert len(q) == 7305 and q.index.tz is None and str(q.index[0].date()) == START
    assert out["site"] == "USGS-01013500"
    # approval / qualifier survive (the old path discarded them)
    assert out["quality"]["approval_counts"] == {"Approved": 7305}
    assert out["quality"]["qualifier_counts"] == {"ESTIMATED": 2100}
    assert set(out["flags"].columns) == {"approval_code", "qualifier_code"}
    assert (out["flags"]["approval_code"] == "A").all()
    assert (out["flags"]["qualifier_code"] == "e").sum() == 2100
    assert len(out["flags"]) == len(q)
    m = out["meta"]
    assert m["gauge_name"] == "Fish River near Fort Kent, Maine"
    assert m["horizontal_datum"] == "NAD83" and "does not state" in m["coordinate_note"]
    assert m["units_native"] == "ft^3/s"
    assert out["upstream"] and si._fetch_diag() == {}
    # all traffic went to the seam, https api hosts only
    assert replay.calls and all(u.startswith("https://api.waterdata.usgs.gov/") for u, _ in replay.calls)


def test_proof1_recompute_bfi_and_high_q_freq_exact(replay):
    r = sig.extract_hydrological_signatures(
        gauge_id=SITE, watershed_geojson=POLY, area_km2=AREA,
        start_date=START, end_date=END, precipitation="skip")
    d = r.data
    # exact, not approximate: scale-free signatures are bit-identical to proof-1
    assert d["baseflow_index"] == PROOF1["baseflow_index"]
    assert d["high_q_freq"] == PROOF1["high_q_freq"]
    for k in ("high_q_dur", "low_q_freq", "hfd_mean"):
        assert d[k] == PROOF1[k], k
    # scale-bearing values differ only by the constant (1.44e-8 relative), nothing else
    assert d["q_mean"] == pytest.approx(PROOF1["q_mean"] * CONST_RATIO, rel=1e-12)
    assert abs(d["q_mean"] / PROOF1["q_mean"] - 1) < 1.5e-8
    s = d["_streamflow_source"]
    assert s["observation"] == "observed" and s["product"] == "NWIS_STREAMFLOW"
    assert s["site"] == "USGS-01013500"
    assert s["quality_summary"]["approval_counts"] == {"Approved": 7305}
    assert s["upstream"]
    assert "error" not in s


def test_no_discharge_constant_left_in_watershed():
    src = Path(sig.__file__).parent
    text = "".join(p.read_text() for p in src.glob("*.py"))
    assert "0.0283168" not in text


def test_terminal_refusal_is_reported_never_substituted(monkeypatch):
    # recorded: the 200-empty daily body for a nonexistent site; the follow-up series-metadata
    # lookup (the adapter's "does the site exist?" probe) is answered empty, as the server does.
    monkeypatch.setattr(uw, "_http_get", Replay(
        lambda u, p: ok(EMPTY) if "time-series-metadata" in u and "USGS-00000000" in str(p) else None))
    called = []
    monkeypatch.setattr(sig, "_fetch_global_streamflow", lambda *a, **k: called.append(1))
    r = sig.extract_hydrological_signatures(
        gauge_id="00000000", watershed_geojson=POLY, area_km2=AREA,
        start_date="2020-01-01", end_date="2020-12-31", precipitation="skip")
    s = r.data["_streamflow_source"]
    assert not called                                    # no modelled product served in its place
    assert s["observation"] == "none" and s["product"] is None
    assert s["error"]["terminal"] is True and s["error"]["code"] in uw.TERMINAL_CODES
    assert r.data["q_mean"] is None


def test_transient_failure_is_reported_as_non_terminal(monkeypatch):
    monkeypatch.setattr(uw, "_http_get", lambda *a, **k: (429, {"Retry-After": "30"}, b"{}"))
    assert si._fetch_streamflow_internal(SITE, START, END) is None
    d = si._fetch_diag()
    assert d["terminal"] is False and d["code"] == "USGS_RATE_LIMITED"


# --------------------------------------------------------------- watershed characterization
def _delineate(monkeypatch, replay):
    import geopandas as gpd
    from shapely.geometry import shape
    from aihydro_watershed.characterize import watershed as w

    basin = doc("r_comid_724696_basin")["features"][0]["geometry"]
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[shape(basin)], crs="EPSG:4326")

    class NLDI:
        def get_basins(self, ident, fsource=None):
            return gdf

    monkeypatch.setattr(w, "NLDI", NLDI, raising=False)
    monkeypatch.setattr(w, "_DEPS_OK", True)
    monkeypatch.setattr("aihydro_watershed.delineation.nldi_point._normalize_nldi_basins", lambda g: g)
    return w, w.delineate_watershed(SITE).data, basin


def test_delineate_uses_monitoring_location_not_pygeohydro(monkeypatch, replay):
    w, d, _ = _delineate(monkeypatch, replay)
    assert not hasattr(w, "NWIS")
    assert d["gauge_name"] == "Fish River near Fort Kent, Maine"
    assert d["gauge_lat"] == 47.2375 and d["gauge_lon"] == -68.5827777777778
    assert d["huc_02"] == "01" and d["gauge_hydrologic_unit_code"] == "010100030403"
    assert d["gauge_horizontal_datum"] == "NAD83"
    assert "does not state" in d["gauge_coordinate_note"]
    assert d["gauge_site_id"] == "USGS-01013500"
    assert any("monitoring-locations" in u for u, _ in replay.calls)


def test_basin_ref_anchor_id_stable_under_new_coordinates(monkeypatch, replay):
    """Legacy NWIS gave (-68.58277778, 47.2375); the API gives -68.5827777777778 (2e-9 deg apart)."""
    from aihydro_watershed import identity

    _, d, basin = _delineate(monkeypatch, replay)
    anchor, ok = identity.gauge_anchor(SITE)
    legacy = identity.mint_basin_ref(
        anchor=anchor, version_verified=ok, method="nldi_gauge_index", geometry_geojson=basin,
        outlet_lon=-68.58277778, outlet_lat=47.2375, area_km2=d["area_km2"], usgs_site=SITE).to_dict()
    new = d["basin_ref"]
    assert new["anchor"] == legacy["anchor"] == {
        "kind": "gauge_index", "network": "nhdplusv2", "network_version": "unversioned",
        "element": "usgs:01013500"}
    assert new["id"] == legacy["id"]                      # id depends on the anchor alone
    assert new["geometry_digest"] == legacy["geometry_digest"]
    assert new["outlet"]["lon"] != legacy["outlet"]["lon"]   # the outlet record does carry the new float


def test_gauge_not_found_is_terminal_tool_error(monkeypatch):
    from aihydro_core import ToolError
    from aihydro_watershed.characterize import watershed as w

    def missing(*a, **k):
        raise uw._err("USGS_SITE_NOT_FOUND", "no such site")

    monkeypatch.setattr(w._uw, "get_monitoring_locations", missing)
    with pytest.raises(ToolError) as e:
        w._gauge_location(SITE)
    assert e.value.code == "GAUGE_NOT_FOUND"


def test_unreachable_lookup_is_not_reported_as_not_found(monkeypatch):
    from aihydro_core import ToolError
    from aihydro_watershed.characterize import watershed as w

    def down(*a, **k):
        raise uw._err("USGS_UNREACHABLE", "cannot reach host")

    monkeypatch.setattr(w._uw, "get_monitoring_locations", down)
    with pytest.raises(ToolError) as e:
        w._gauge_location(SITE)
    assert e.value.code == "NETWORK_ERROR"


# --------------------------------------------------------------- source / hygiene gates
def test_citations_use_current_hosts():
    from aihydro_watershed.characterize import watershed as w

    urls = {s.name: s.url for s in w._SOURCES}
    assert urls["USGS NLDI"] == "https://api.water.usgs.gov/nldi/"
    assert urls["USGS NWIS"].startswith("https://api.waterdata.usgs.gov/ogcapi/v1")
    blob = " ".join(s.url + s.citation for s in list(w._SOURCES) + list(sig._SOURCES_NWIS))
    assert "labs.waterdata" not in blob and "waterservices" not in blob


def test_no_legacy_usgs_imports_in_package():
    root = Path(sig.__file__).resolve().parents[1]
    bad = []
    for p in root.rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names = [n.module or ""] + [f"{n.module}.{a.name}" for a in n.names]
            for m in names:
                if m.split(".")[0] in {"dataretrieval", "hydrofunctions"} or m.endswith("pygeohydro.NWIS"):
                    bad.append((str(p.relative_to(root)), m))
        txt = p.read_text()
        assert "waterservices.usgs.gov" not in txt or "# legacy-ok" in txt, p
    assert not bad, bad
