"""Slice-3 P3: BasinRef minting (offline). See aihydro_watershed.identity."""
from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import pytest

from aihydro_core.records.place import BasinRef, verify_basin_ref_dict
from aihydro_watershed import identity
from aihydro_watershed.delineation import router

SQUARE = {"type": "Polygon", "coordinates": [[[-90, 40], [-89, 40], [-89, 41], [-90, 41], [-90, 40]]]}
SQUARE2 = {"type": "Polygon", "coordinates": [[[-90, 40], [-88.9, 40], [-88.9, 41], [-90, 41], [-90, 40]]]}
SITE = "05520500"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("network access attempted while minting")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setenv("AIHYDRO_HOME", str(tmp_path / "home"))


def _nldi_data(comid=123, geom=SQUARE):
    return SimpleNamespace(data={
        "geometry_geojson": {"type": "Feature", "geometry": geom, "properties": {}},
        "area_km2": 9000.0, "outlet_lat": 40.5, "outlet_lon": -89.5,
        "method_used": "nldi_comid", "comid": comid,
    })


def _attached(**kw):
    return router._attach_workflow_steps(_nldi_data(**kw), method_used="nldi_comid").data


def test_nldi_run_twice_same_id_and_comid_propagated():
    a, b = _attached(), _attached(geom=SQUARE2)
    assert a["comid"] == 123
    assert a["basin_ref"]["id"] == b["basin_ref"]["id"]
    assert a["basin_ref"]["geometry_digest"] != b["basin_ref"]["geometry_digest"]
    assert verify_basin_ref_dict(a["basin_ref"])
    assert a["basin_ref"]["anchor"] == {"kind": "network_element", "network": "nhdplusv2",
                                        "network_version": "unversioned", "element": "123"}
    assert "network_version_unverified" in a["basin_ref"]["quality_flags"]
    json.dumps(a["basin_ref"])


def test_different_comid_different_id():
    assert _attached(comid=1)["basin_ref"]["id"] != _attached(comid=2)["basin_ref"]["id"]


def test_nldi_vs_merit_differ_but_share_usgs_alias():
    g = {"type": "Feature", "geometry": SQUARE}
    a1, o1 = identity.gauge_anchor(SITE)
    a2, o2 = identity.merit_basins_anchor("71000123", "v1.0")
    r1 = identity.mint_basin_ref(anchor=a1, version_verified=o1, method="nldi_gauge_index",
                                 geometry_geojson=g, usgs_site=SITE)
    r2 = identity.mint_basin_ref(anchor=a2, version_verified=o2, method="merit_basins_hybrid",
                                 geometry_geojson=g, usgs_site=SITE, merit_catchment="71000123")
    assert r1.id != r2.id
    key = lambda r: {(x.scheme, x.id) for x in r.aliases if x.scheme == "usgs"}
    assert key(r1) == key(r2) == {("usgs", SITE)}
    assert ("merit", "71000123") in {(x.scheme, x.id) for x in r2.aliases}
    assert "network_version_unverified" in r1.quality_flags
    assert "network_version_unverified" not in r2.quality_flags


def test_geoconnex_never_constructs_gage_pid_offline():
    a, ok = identity.gauge_anchor(SITE)
    r = identity.mint_basin_ref(anchor=a, version_verified=ok, method="m", geometry_geojson=SQUARE, usgs_site=SITE)
    ids = [x.id for x in r.aliases if x.scheme == "geoconnex"]
    assert ids == [f"https://geoconnex.us/usgs/monitoring-location/{SITE}"]
    assert all("/ref/gages/" not in i for i in ids)
    ml = [x for x in r.aliases if x.scheme == "geoconnex"][0]
    assert ml.verified is False
    assert "geoconnex_unresolved" in r.quality_flags
    assert identity.resolve_geoconnex_gage(SITE, timeout=0.1).status == "unresolved"


def test_geoconnex_gage_from_lookup_is_cached_and_used():
    calls = []

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"features": [{"properties": {"uri": "https://geoconnex.us/ref/gages/1234567",
                                                  "provider_id": f"USGS-{SITE}"}}]}

    def get(url, params, timeout):
        calls.append((url, params, timeout))
        return Resp()

    res = identity.resolve_geoconnex_gage(SITE, timeout=5, _get=get)
    assert res.status == "resolved" and res.uri.endswith("/ref/gages/1234567")
    assert calls[0][1] == {"provider_id": f"USGS-{SITE}", "f": "json"}
    assert identity.resolve_geoconnex_gage(SITE, _get=get).status == "cached" and len(calls) == 1
    a, ok = identity.gauge_anchor(SITE)
    r = identity.mint_basin_ref(anchor=a, version_verified=ok, method="m", geometry_geojson=SQUARE, usgs_site=SITE)
    gages = [x for x in r.aliases if x.scheme == "geoconnex" and "/ref/gages/" in x.id]
    assert len(gages) == 1 and gages[0].verified and "geoconnex_unresolved" not in r.quality_flags


def test_geoconnex_rejects_non_gage_payload():
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"features": [{"properties": {"uri": "https://evil.example/x"}}]}

    assert identity.resolve_geoconnex_gage(SITE, _get=lambda *a, **k: Resp()).status == "not_found"


def test_grid_cell_anchor_quantised():
    a1, _ = identity.grid_cell_anchor("merit-hydro", "v1", 90, -89.50001, 40.50001)
    a2, _ = identity.grid_cell_anchor("merit-hydro", "v1", 90, -89.50002, 40.50002)
    a3, _ = identity.grid_cell_anchor("merit-hydro", "v1", 90, -89.6, 40.6)
    assert a1 == a2 and a1 != a3 and a1.kind == "grid_cell"
    p1, _ = identity.grid_cell_anchor("3dep-dem", None, 10, -89.5, 40.5, crs="EPSG:5070")
    assert p1.network_version == "unversioned"


def test_invalid_usgs_site_not_aliased():
    a, ok = identity.comid_anchor(5)
    r = identity.mint_basin_ref(anchor=a, version_verified=ok, method="m", geometry_geojson=SQUARE, usgs_site="1234567")
    assert not any(x.scheme == "usgs" for x in r.aliases) and "usgs_site_invalid" in r.quality_flags


def test_compare_realisations_does_not_merge():
    c = identity.compare_realisations(SQUARE, SQUARE2, (-89.5, 40.5), (-89.4, 40.5))
    assert 0.85 < c["area_ratio"] < 0.95 and 0.85 < c["iou"] < 0.95
    assert 8000 < c["outlet_distance_m"] < 9000
    same = identity.compare_realisations(SQUARE, SQUARE)
    assert same["iou"] == pytest.approx(1.0) and same["outlet_distance_m"] is None


def test_mint_failure_yields_none_not_exception():
    d = _nldi_data(); d.data["geometry_geojson"] = {"type": "Polygon", "coordinates": []}
    out = router._attach_workflow_steps(d, method_used="nldi_comid").data
    assert out["basin_ref"] is None


def test_router_final_result_merit_hybrid_ref(monkeypatch):
    ref = router._mint_ref_dict(
        method_used="merit_basins_hybrid", geojson={"type": "Feature", "geometry": SQUARE}, area_km2=9000.0,
        lat=40.5, lon=-89.5, terminal_catchment_id=71000123, vector_dataset_version="v1.0",
    )
    assert ref["anchor"]["network"] == "merit-basins" and ref["anchor"]["element"] == "71000123"
    grid = router._mint_ref_dict(
        method_used="merit_gee_pyflwdir", geojson={"type": "Feature", "geometry": SQUARE}, area_km2=1.0,
        lat=40.5, lon=-89.5, raster_dataset_version=None, routing_resolution_m=90.0,
    )
    assert grid["anchor"]["kind"] == "grid_cell" and "grid_anchor_from_requested_outlet" in grid["quality_flags"]
    assert router._mint_ref_dict(method_used="weird", geojson=SQUARE, area_km2=1.0, lat=0, lon=0) is None


# ---- gauge path (fake NLDI / NWIS) ---------------------------------------
class _FakeGdf:
    pass


def _gauge_run(monkeypatch, index_fails):
    import geopandas as gpd
    from shapely.geometry import shape
    from aihydro_watershed.characterize import watershed as w

    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[shape(SQUARE)], crs="EPSG:4326")

    class NLDI:
        def get_basins(self, ident, fsource=None):
            if fsource is None and index_fails:
                raise RuntimeError("shape")
            return gdf
        def comid_byloc(self, xy):
            return SimpleNamespace(comid=SimpleNamespace(iloc=[777]))

    class NWIS:
        def get_info(self, q):
            import pandas as pd
            return pd.DataFrame({"site_no": [SITE], "dec_lat_va": [40.5], "dec_long_va": [-89.5],
                                 "station_nm": ["x"], "huc_cd": ["07120000"]})

    monkeypatch.setattr(w, "NLDI", NLDI, raising=False)
    monkeypatch.setattr(w, "NWIS", NWIS, raising=False)
    monkeypatch.setattr(w, "_DEPS_OK", True)
    monkeypatch.setattr("aihydro_watershed.delineation.nldi_point._normalize_nldi_basins", lambda g: g)
    return w.delineate_watershed(SITE).data


def test_gauge_index_path(monkeypatch):
    d = _gauge_run(monkeypatch, index_fails=False)
    r = d["basin_ref"]
    assert d["delineation_path"] == "gauge_index" and d["comid"] is None
    assert r["anchor"]["kind"] == "gauge_index" and r["anchor"]["element"] == f"usgs:{SITE}"
    assert "gauge_basin_comid_fallback" not in r["quality_flags"]
    again = _gauge_run(monkeypatch, index_fails=False)["basin_ref"]
    assert again["id"] == r["id"]


def test_gauge_comid_fallback_flagged(monkeypatch):
    d = _gauge_run(monkeypatch, index_fails=True)
    r = d["basin_ref"]
    assert d["delineation_path"] == "comid_fallback" and d["comid"] == 777
    assert r["anchor"] == {"kind": "network_element", "network": "nhdplusv2",
                           "network_version": "unversioned", "element": "777"}
    assert "gauge_basin_comid_fallback" in r["quality_flags"]
    usgs = [a for a in r["aliases"] if a["scheme"] == "usgs"]
    assert usgs[0]["relation"] == "fallback_of"
    idx = _gauge_run(monkeypatch, index_fails=False)["basin_ref"]
    assert idx["id"] != r["id"]
