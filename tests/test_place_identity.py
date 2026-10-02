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


def _mint(**kw):
    base = dict(geojson={"type": "Feature", "geometry": SQUARE}, area_km2=1.0, lat=40.5, lon=-89.5)
    base.update(kw)
    return router._mint_ref_dict(**base)


def test_merit_hybrid_with_terminal_id_is_network_element_anchor():
    ref, why = _mint(method_used="merit_basins_hybrid", area_km2=9000.0, terminal_catchment_id=71000123,
                     vector_dataset_version="v1.0")
    assert why is None
    assert ref["anchor"]["network"] == "merit-basins" and ref["anchor"]["element"] == "71000123"
    # and the same through the NLDI-style attach path (W1a)
    d = SimpleNamespace(data={"geometry_geojson": {"type": "Feature", "geometry": SQUARE}, "area_km2": 1.0,
                              "method_used": "merit_basins_hybrid", "terminal_catchment_id": 71000123,
                              "vector_dataset_version": "v1.0"})
    out = router._attach_workflow_steps(d, method_used="merit_basins_hybrid", lat=40.5, lon=-89.5).data
    assert out["basin_ref"]["anchor"] == ref["anchor"]


def test_grid_identity_follows_snapped_cell_not_request():
    kw = dict(method_used="merit_gee_pyflwdir", snap_grid_crs="EPSG:4326", snap_grid_resolution_m=90.0)
    # two different requests, same snapped cell -> same id
    a, _ = _mint(lat=40.5000, lon=-89.5000, snapped_outlet=(-89.50011, 40.50011), **kw)
    b, _ = _mint(lat=40.5003, lon=-89.5004, snapped_outlet=(-89.50012, 40.50012), **kw)
    assert a["id"] == b["id"]
    # same request, snapped to different cells (different streams) -> different ids
    c, _ = _mint(lat=40.5000, lon=-89.5000, snapped_outlet=(-89.50011, 40.50011), **kw)
    d, _ = _mint(lat=40.5000, lon=-89.5000, snapped_outlet=(-89.5101, 40.5101), **kw)
    assert c["id"] != d["id"]
    assert a["outlet"]["lon"] == pytest.approx(-89.50011)  # outlet recorded is the snapped point


def test_no_snapped_pour_point_means_no_ref():
    for method in ("merit_gee_pyflwdir", "local_merit_pyflwdir", "dem_raw_fallback", "small_catchment_3dep"):
        ref, why = _mint(method_used=method)
        assert ref is None and why == "no_snapped_pour_point"
    ref, why = _mint(method_used="weird")
    assert ref is None and why == "unsupported_method"


def test_attach_nldi_has_no_unavailable_reason():
    assert _attached()["basin_ref_unavailable"] is None


def test_pysheds_reports_snapped_pour_point_field():
    from aihydro_watershed.delineation.types import FastDelineationResult

    f = FastDelineationResult.__new__.__defaults__
    assert f[-4:] == (None, None, None, None)


def test_geoconnex_requires_exact_provider_id():
    class Resp:
        def __init__(self, props): self.props = props
        def raise_for_status(self): pass
        def json(self): return {"features": [{"properties": self.props}]}

    uri = "https://geoconnex.us/ref/gages/99"
    for props in ({"uri": uri}, {"uri": uri, "provider_id": ""}, {"uri": uri, "provider_id": "USGS-99999999"}):
        assert identity.resolve_geoconnex_gage(SITE, _get=lambda *a, **k: Resp(props)).status == "not_found"
    ok = identity.resolve_geoconnex_gage(SITE, _get=lambda *a, **k: Resp({"uri": uri, "provider_id": f"USGS-{SITE}"}))
    assert ok.status == "resolved"


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


# ---- regression: minting must never change tier selection -----------------
def test_attach_never_raises_on_minimal_result():
    r = SimpleNamespace(data={"area_km2": 420.0, "method_used": "nldi_comid", "comid": 1})
    out = router._attach_workflow_steps(r, method_used="nldi_comid").data
    assert out["basin_ref"] is None and out["basin_ref_unavailable"].startswith("mint_failed")  # no geometry: fail closed downstream, but no exception


def test_auto_conus_nldi_quick_selected_with_minimal_nldi_result(monkeypatch):
    """NLDI result without outlet_lat/lon (as mocked in aihydro-tools) must still win."""
    from shapely.geometry import box
    import geopandas as gpd

    geom = box(-96.5, 40.5, -96.2, 40.9)
    feat = json.loads(gpd.GeoDataFrame(geometry=[geom], crs=4326).to_json())["features"][0]
    res = SimpleNamespace(data={"geometry_geojson": feat, "area_km2": 420.0,
                                "method_used": "nldi_comid", "comid": 1})
    monkeypatch.setattr("aihydro_watershed.delineation.nldi_point.delineate_nldi_at_point", lambda *a, **k: res)

    def no_small(*a, **k):
        raise AssertionError("small-catchment tier must not run")

    monkeypatch.setattr("aihydro_watershed.delineation.small_catchment.delineate_small_catchment", no_small)
    out = router.delineate_from_point(40.71829, -96.41265, method="auto")
    assert out.data["method_used"] == "nldi_comid"
    assert out.data["basin_ref"]["anchor"]["element"] == "1"
    assert out.data["basin_ref"]["outlet"]["lat"] == 40.71829


def test_pysheds_snap_out_reports_snapped_cell_on_grid():
    pytest.importorskip("pysheds")
    import numpy as np
    import xarray as xr
    import rioxarray  # noqa: F401
    from aihydro_watershed.delineation.pysheds_pipeline import delineate_watershed_from_array

    n = 60
    ys = 4_500_000 - 30 * np.arange(n) - 15.0
    xs = 500_000 + 30 * np.arange(n) + 15.0
    X, Y = np.meshgrid(xs, ys)
    z = (np.abs(X - 500_900) * 0.5 + (Y - ys.min()) * 0.2 + 100).astype("float32")  # V valley, drains south
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": ys, "x": xs}).rio.write_crs("EPSG:32616")
    import pyproj
    lon, lat = pyproj.Transformer.from_crs(32616, 4326, always_xy=True).transform(500_930, 4_499_500)
    out: dict = {}
    delineate_watershed_from_array(da, lat, lon, snap_out=out)
    assert out["crs"] == "EPSG:32616" and out["resolution_m"] == pytest.approx(30.0)
    x, y = pyproj.Transformer.from_crs(4326, 32616, always_xy=True).transform(out["lon"], out["lat"])
    assert (x - 500_000) % 30 == pytest.approx(15, abs=1e-3) and (y % 30) == pytest.approx(15, abs=1e-3)
