"""Offline tests for the default global MERIT-Basins hybrid tier (synthetic region, no downloads)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import geopandas as gpd
import merit_fixture as fx
import pytest
import rasterio
from rasterio.windows import from_bounds

from aihydro_watershed.delineation import merit_flowdir_pipeline as pipe
from aihydro_watershed.delineation import merit_hybrid as mh
from aihydro_watershed.delineation import router
from aihydro_watershed.delineation.types import area_km2
from aihydro_watershed.merit import verified_fetch as vf
from aihydro_watershed.merit.basins_store import BasinsStore
from aihydro_watershed.merit.merit_manager import MeritDataManager


@pytest.fixture()
def region(tmp_path, monkeypatch):
    reg = fx.build_region(tmp_path)
    man = fx.patched_manifest(vf.load_manifest(), reg)
    monkeypatch.setattr(vf, "load_manifest", lambda: man)
    return reg


def _serve(region, tamper: str | None = None):
    """An opener that serves the fixture files by URL, optionally corrupting one."""
    by_url = {vf.asset_spec(n).url: n for n in region.files}

    def opener(url: str):
        name = by_url[url]
        data = region.files[name].read_bytes()
        if name == tamper:
            data = data[:-1] + bytes([data[-1] ^ 0xFF])
        return io.BytesIO(data)

    return opener


def _stage_all(region, root: Path, *, rasters: bool = True) -> MeritDataManager:
    mgr = MeritDataManager(root, opener=_serve(region))
    mgr.ensure_basins_region(fx.PFAF, acquisition_policy="download")
    if rasters:
        for kind in ("flowdir", "accum"):
            spec = vf.region_asset_spec(kind, fx.PFAF)
            mgr._stage(spec, mgr.raster_path(kind, fx.PFAF))
    return mgr


@pytest.fixture()
def staged(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    mgr = _stage_all(region, root)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    return mgr


# --------------------------------------------------------------------------- manifest / hashes


def test_real_manifest_pins_every_region_file():
    man = json.loads((Path(vf.__file__).parent / "delineator_data_manifest.json").read_text())
    assert len(man["files"]) == 240  # 60 megabasins x (basins, rivers, flowdir, accum)
    for name, entry in man["files"].items():
        assert len(entry["sha256"]) == 64 and int(entry["sha256"], 16) >= 0, name
        assert entry["size_bytes"] > 0, name
    assert man["source"]["package_version"] == "2.2.4"
    assert man["level2_index"]["sha256"] is None and len(man["level2_index"]["git_blob_sha1"]) == 40
    assert "non-commercial" in man["licence"]["summary"].lower()
    spec = vf.region_asset_spec("basins", "73")
    assert spec.url == "https://mghydro.com/watersheds/data/basins73.db" and spec.size_bytes == 237219840


def test_fetch_installs_only_matching_file(region, tmp_path):
    dest = tmp_path / "x" / "basins73.db"
    spec = vf.region_asset_spec("basins", fx.PFAF)
    vf.fetch_verified(spec, dest, opener=_serve(region))
    rec = vf.read_provenance(dest)
    assert rec["sha256"] == region.sha("basins73.db") and rec["data_version"] == "v3"
    assert "non-commercial" in rec["licence"].lower()


def test_hash_mismatch_fails_closed_and_installs_nothing(region, tmp_path):
    dest = tmp_path / "x" / "basins73.db"
    spec = vf.region_asset_spec("basins", fx.PFAF)
    with pytest.raises(vf.AssetIntegrityError, match="MERIT_ASSET_INTEGRITY_FAILED"):
        vf.fetch_verified(spec, dest, opener=_serve(region, tamper="basins73.db"))
    assert not dest.exists() and not dest.with_name(dest.name + ".part").exists()
    assert not vf.sidecar_path(dest).exists()


def test_tampered_existing_file_is_refused_and_sidecar_avoids_rehash(region, tmp_path, monkeypatch):
    mgr = _stage_all(region, tmp_path / "m", rasters=False)
    spec = vf.region_asset_spec("basins", fx.PFAF)
    path = mgr.basins_db_path(fx.PFAF)
    monkeypatch.setattr(vf, "sha256_file", lambda p: (_ for _ in ()).throw(AssertionError("re-hashed")))
    assert vf.verified_path(path, spec) == path  # sidecar hit: no re-hash
    monkeypatch.undo()
    monkeypatch.setattr(vf, "load_manifest", lambda: fx.patched_manifest(json.loads(
        (Path(vf.__file__).parent / "delineator_data_manifest.json").read_text()), region))
    with path.open("ab") as f:
        f.write(b"x")
    with pytest.raises(vf.AssetIntegrityError):
        vf.verified_path(path, spec)
    st = mgr.basins_region_cache(fx.PFAF)
    assert not st.catchments_ready and "MERIT_ASSET_INTEGRITY_FAILED" in (st.integrity_error or "")


def test_level2_pinned_by_blob_sha_then_sha256_enforced(region, tmp_path):
    mgr = MeritDataManager(tmp_path / "m", opener=_serve(region))
    assert mgr.ensure_level2_index(download=True) is not None
    rec = vf.read_provenance(mgr.megabasins_path())
    assert rec["sha256"] == region.sha("megabasins.db") and rec["pinned_sha256"] is None
    assert mgr.resolve_pfaf_region(40.011, 0.049) == fx.PFAF
    with mgr.megabasins_path().open("ab") as f:
        f.write(b"x")
    with pytest.raises(vf.AssetIntegrityError):
        mgr.verified_megabasins()


# ----------------------------------------------------------------------------- staging policy


def test_download_size_policy(region, tmp_path, monkeypatch):
    mgr = MeritDataManager(tmp_path / "m", opener=_serve(region))
    monkeypatch.setenv("AIHYDRO_MERIT_AUTO_STAGE", "never")
    st = mgr.auto_stage_vectors(fx.PFAF)
    assert st.acquisition_required and not mgr.basins_db_path(fx.PFAF).exists()
    monkeypatch.setenv("AIHYDRO_MERIT_AUTO_STAGE", "tiered")
    cfg = tmp_path / ".aihydro_cfg"
    monkeypatch.setenv("AIHYDRO_HOME", str(cfg))
    cfg.mkdir()
    (cfg / "config.json").write_text(json.dumps({"merit_auto_stage_max_gb": 1e-9}))
    st = mgr.auto_stage_vectors(fx.PFAF)
    assert getattr(st, "action_required", None) == "confirm_large_download"
    assert not mgr.basins_db_path(fx.PFAF).exists()
    (cfg / "config.json").write_text(json.dumps({"merit_auto_stage_max_gb": 2.0}))
    st = mgr.auto_stage_vectors(fx.PFAF)
    assert st.catchments_ready and st.rivers_ready and st.megabasins_ready


def test_staging_a_corrupt_download_installs_nothing(region, tmp_path):
    mgr = MeritDataManager(tmp_path / "m", opener=_serve(region, tamper="rivers73.db"))
    st = mgr.ensure_basins_region(fx.PFAF, acquisition_policy="download")
    assert "nothing unverified was installed" in st.message
    assert not mgr.rivers_db_path(fx.PFAF).exists() and not st.rivers_ready


def test_aihydro_home_is_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv("AIHYDRO_MERIT_DIR", raising=False)
    monkeypatch.delenv("MERIT_DATA_DIR", raising=False)
    monkeypatch.setenv("AIHYDRO_HOME", str(tmp_path / "home"))
    assert MeritDataManager().root == tmp_path / "home" / "merit"


# --------------------------------------------------------------------------------- assembly


def test_vector_assembly_traverses_upstream_only(staged):
    store = BasinsStore(staged.basins_db_path(fx.PFAF), staged.rivers_db_path(fx.PFAF))
    assert store.locate(fx.SNAPPED[1], fx.SNAPPED[0]) == fx.TERMINAL
    ids = store.upstream_ids(fx.TERMINAL)
    assert ids[0] == fx.TERMINAL and set(ids) == {73000001, 73000002, 73000003, 73000004}
    assert 73000006 not in ids  # downstream of the outlet
    geoms = store.geometries(ids)
    assert len(geoms) == 4 and geoms.crs.to_epsg() == 4326
    assert store.river_uparea_km2(fx.TERMINAL) == pytest.approx(geoms["area_km2"].sum(), rel=1e-6)
    assert set(store.upstream_ids(73000002)) == {73000002, 73000001, 73000004}


# --------------------------------------------------------------------------- hybrid, local rasters


def test_hybrid_local_rasters_refines_terminal_and_is_version_consistent(staged):
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST[:1], fx.OUTLET_REQUEST[1], allow_offline=False)
    assert res.source == "merit_basins_hybrid" and res.pfaf_region == fx.PFAF
    assert res.terminal_catchment_id == fx.TERMINAL and res.upstream_catchment_count == 4
    assert res.terminal_refinement_used and res.snap_source == "local_accum_mghydro"
    assert (res.outlet_lat, res.outlet_lon) == pytest.approx(fx.SNAPPED, abs=1e-6)
    # Upstream of the outlet = C1 + C2 + C4 whole, plus the western half of C3.
    full = res.vector_assembly_area_km2
    c3 = area_km2(gpd.GeoDataFrame(geometry=[fx.box(0.04, 40.0, 0.06, 40.02)], crs=4326))
    assert res.area_km2 == pytest.approx(full - 0.5 * c3, rel=0.01)
    assert res.refined_polygon_area_km2 == pytest.approx(0.5 * c3, rel=0.01)
    assert "MERIT_VERSION_MIX" not in res.quality_flags
    assert "VECTOR_ASSEMBLY_UPAREA_MISMATCH" not in res.quality_flags
    vc = res.data_provenance["version_consistency"]
    assert vc["mixed"] is False and vc["snap_dataset"] == vc["refinement_dataset"] == mh.ORIGIN_LOCAL
    names = {f["name"] for f in res.data_provenance["files"]}
    assert names == {"basins73.db", "rivers73.db", "flowdir73.tif", "accum73.tif", "megabasins.db"}
    assert all(len(f["sha256"]) == 64 and f["verified"] for f in res.data_provenance["files"])
    assert "non-commercial" in res.data_provenance["licence"].lower()
    assert not res.gdf.geometry.iloc[0].geom_type == "GeometryCollection"


def test_hybrid_refuses_a_file_that_fails_its_pin(staged, monkeypatch):
    with staged.basins_db_path(fx.PFAF).open("ab") as f:
        f.write(b"\0")
    monkeypatch.setattr(vf, "sha256_file", vf.sha256_file)  # real hashing
    with pytest.raises(RuntimeError, match="MERIT_ASSET_INTEGRITY_FAILED"):
        pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST, auto_stage=False)


def test_hybrid_not_staged_when_auto_stage_never(region, tmp_path, monkeypatch):
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AIHYDRO_MERIT_AUTO_STAGE", "never")
    mgr = MeritDataManager(opener=_serve(region))
    mgr.ensure_level2_index(download=True)
    with pytest.raises(RuntimeError, match="MERIT_BASINS_NOT_STAGED"):
        pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)


# ------------------------------------------------------------ hybrid, Earth Engine window (stubbed)


def _stub_ee(monkeypatch, region, calls: list):
    flowdir = region.files["flowdir73.tif"]

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

    def fake_snap_ref(lat, lon, **_):
        snap = pipe.MeritSnap(lat=fx.SNAPPED[0], lon=fx.SNAPPED[1], row=0, col=0, distance_m=250.0,
                              snapped_upa_km2=1.7, snap_quality="nearest_merit_stream")
        return pipe.MeritSnapReference(snap=snap, source="gee", cache_key="stub", bbox=(lon, lat, lon, lat),
                                       official_merit_upa_km2=1.7)

    monkeypatch.setattr(pipe, "merit_get_snap_reference", fake_snap_ref)


def test_hybrid_without_staged_rasters_uses_small_ee_window_and_flags_version_mix(
    region, tmp_path, monkeypatch
):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    calls: list = []
    _stub_ee(monkeypatch, region, calls)
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert res.terminal_refinement_used and res.terminal_catchment_id == fx.TERMINAL
    assert "MERIT_VERSION_MIX" in res.quality_flags
    vc = res.data_provenance["version_consistency"]
    assert vc["mixed"] is True and mh.ORIGIN_EE in (vc["snap_dataset"], vc["refinement_dataset"])
    assert res.raster_dataset_version.startswith("MERIT/Hydro/v1_0_1")
    assert len(calls) == 1 and calls[0][1] == ("dir",)
    (xmin, ymin, xmax, ymax), _ = calls[0]
    assert (xmax - xmin) < 0.2 and (ymax - ymin) < 0.2  # one small window, far inside the 40 km class
    assert res.refined_polygon_area_km2 > 0


def test_ee_window_too_large_is_refused_and_flagged(region, tmp_path, monkeypatch):
    big = gpd.GeoDataFrame(geometry=[fx.box(0, 40, 1.0, 41.0)], crs=4326)
    snap = pipe.MeritSnap(lat=40.5, lon=0.5, row=0, col=0, distance_m=0, snapped_upa_km2=None, snap_quality="x")
    with pytest.raises(RuntimeError, match="TERMINAL_REFINEMENT_WINDOW_TOO_LARGE"):
        mh.refine_ee(big, snap)
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    _stub_ee(monkeypatch, region, [])
    monkeypatch.setattr(mh, "refine_ee", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("TERMINAL_REFINEMENT_WINDOW_TOO_LARGE: 99 km")))
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert not res.terminal_refinement_used
    assert {"TERMINAL_CATCHMENT_REFINEMENT_FAILED", "TERMINAL_REFINEMENT_WINDOW_TOO_LARGE"} <= set(res.quality_flags)
    assert res.area_km2 == pytest.approx(res.vector_assembly_area_km2, rel=1e-6)  # whole terminal kept


def test_offline_unsnapped_skips_refinement(region, tmp_path, monkeypatch):
    root = tmp_path / "merit"
    _stage_all(region, root, rasters=False)
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    res = pipe.merit_basins_hybrid_delineate(fx.SNAPPED[0], fx.SNAPPED[1], force_offline=True, allow_offline=True)
    assert "TERMINAL_CATCHMENT_REFINEMENT_SKIPPED_UNSNAPPED" in res.quality_flags
    assert "GEE_VALIDATION_UNAVAILABLE" in res.quality_flags and not res.terminal_refinement_used
    assert "MERIT_VERSION_MIX" not in res.quality_flags


def test_version_consistency_record():
    ok = mh.version_consistency(vector_version="v", snap_origin=mh.ORIGIN_LOCAL, refine_origin=mh.ORIGIN_LOCAL)
    mix = mh.version_consistency(vector_version="v", snap_origin=mh.ORIGIN_EE, refine_origin=mh.ORIGIN_LOCAL)
    assert ok["mixed"] is False and mix["mixed"] is True and "poor results" in mix["note"]


# ---------------------------------------------------------------------------------- router


def _fake_result(source: str, **kw) -> pipe.MeritFlowdirResult:
    gdf = gpd.GeoDataFrame(geometry=[fx.box(10, 10, 10.1, 10.1)], crs=4326)
    base = dict(
        gdf=gdf, area_km2=123.0, outlet_lat=10.0, outlet_lon=10.0, snap_distance_m=10.0,
        snapped_upa_km2=None, snap_quality="x", snap_validation={}, cache_key="k", bbox=(10, 10, 10.1, 10.1),
        scout_box_maxed=False, touches_edge=False, area_validation={}, quality_flags=["HYBRID_ROUTING_USED"],
        source=source, routing_dataset="d", routing_resolution_m=90.0, pfaf_region="73",
        terminal_catchment_id=1, data_provenance={"files": [], "licence": "non-commercial"},
        fallback_history=[],
    )
    base.update(kw)
    return pipe.MeritFlowdirResult(**base)


@pytest.fixture()
def calls(monkeypatch):
    log: dict[str, list] = {"hybrid": [], "gee": []}

    def hybrid(lat, lon, **kw):
        log["hybrid"].append(kw)
        return _fake_result("merit_basins_hybrid")

    def gee(lat, lon, **kw):
        log["gee"].append(kw)
        return _fake_result("merit_gee")

    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", hybrid)
    monkeypatch.setattr(pipe, "delineate_merit_gee", gee)
    monkeypatch.setattr(pipe, "merit_get_snap_reference", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EE")))
    monkeypatch.setattr(pipe, "merit_resolve_pfaf_region", lambda *a, **k: "62")
    monkeypatch.setattr(MeritDataManager, "auto_stage_flowdir", lambda self, pfaf, **k: self.routing_region_cache(pfaf))
    monkeypatch.setattr(pipe, "merit_check_routing_region_cache", lambda *a, **k: {"flowdir_ready": False})
    return log


def test_router_global_auto_runs_hybrid_first_and_reads_merit_basins_hybrid(calls):
    res = router.delineate_from_point(-3.0, -60.0)  # Amazon: outside CONUS
    d = res.data
    assert d["method_used"] == "merit_basins_hybrid"
    assert len(calls["hybrid"]) == 1 and calls["gee"] == []
    assert calls["hybrid"][0]["snap_reference"] is None  # no up-front Earth Engine call
    assert d["fallback_history"][-1] == {
        "method": "merit_basins_hybrid", "outcome": "succeeded", "reason": "default_global_tier"
    }
    assert d["data_provenance"]["licence"] == "non-commercial"
    assert "non-commercial" in d["license"].lower() and "ODbL" in d["license"]


def test_router_forced_merit_hybrid_works_in_conus_and_alias(calls):
    for method in ("merit_hybrid", "merit_basins"):
        calls["hybrid"].clear()
        res = router.delineate_from_point(40.0, -100.0, method=method)
        assert res.data["method_used"] == "merit_basins_hybrid" and len(calls["hybrid"]) == 1
    assert calls["gee"] == []


def test_router_conus_auto_keeps_nldi_first(calls, monkeypatch):
    from aihydro_core import HydroMeta, HydroResult

    nldi = HydroResult(data={"area_km2": 500.0}, meta=HydroMeta(tool="nldi", version="0", params={}, sources=[]))
    import aihydro_watershed.delineation.nldi_point as np_mod

    monkeypatch.setattr(np_mod, "delineate_nldi_at_point", lambda *a, **k: nldi)
    res = router.delineate_from_point(40.0, -100.0)
    assert res.data["routing_dataset"] == "USGS NLDI / NHDPlus" and calls["hybrid"] == []


def test_router_falls_back_to_merit_gee_when_hybrid_cannot_serve(calls, monkeypatch):
    def hybrid(lat, lon, **kw):
        calls["hybrid"].append(kw)
        raise RuntimeError("MERIT_BASINS_NOT_STAGED HYBRID_ROUTING_REQUIRED: not staged")

    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", hybrid)
    res = router.delineate_from_point(-3.0, -60.0)
    d = res.data
    assert d["method_used"] == "merit_gee_pyflwdir" and len(calls["hybrid"]) == 1
    rows = [(h["method"], h["outcome"]) for h in d["fallback_history"]]
    assert rows[0] == ("merit_basins_hybrid", "failed")
    assert "not staged" in d["fallback_history"][0]["reason"]


def test_router_integrity_failure_is_flagged_and_hybrid_tried_once(calls, monkeypatch):
    def hybrid(lat, lon, **kw):
        calls["hybrid"].append(kw)
        raise RuntimeError("MERIT_ASSET_INTEGRITY_FAILED: basins73.db SHA-256 does not match")

    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", hybrid)
    monkeypatch.setattr(pipe, "delineate_merit_gee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gee down")))
    import aihydro_watershed.delineation.pysheds_pipeline as ps

    monkeypatch.setattr(ps, "delineate_fast", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dem down")))
    with pytest.raises(Exception) as ei:
        router.delineate_from_point(-3.0, -60.0)
    assert len(calls["hybrid"]) == 1  # later fallbacks do not retry it
    assert "DELINEATION_FAILED" in repr(ei.value) or "failed" in str(ei.value).lower()
