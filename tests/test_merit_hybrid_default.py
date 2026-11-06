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

    def opener(url: str, headers=None):
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
    assert not dest.exists() and not list(dest.parent.glob("*.part"))
    assert not vf.sidecar_path(dest).exists()
    marker = vf.active_integrity_marker(dest)
    assert marker["file"] == "basins73.db" and len(marker["observed_sha256"]) == 64
    assert marker["observed_sha256"] != spec.sha256


def test_sidecar_fast_path_for_big_files_and_always_rehash_for_small_dbs(region, tmp_path, monkeypatch):
    mgr = _stage_all(region, tmp_path / "m")
    real = vf.sha256_file
    hashed: list[str] = []
    monkeypatch.setattr(vf, "sha256_file", lambda p: (hashed.append(Path(p).name), real(p))[1])
    vf.verified_path(mgr.raster_path("flowdir", fx.PFAF), vf.region_asset_spec("flowdir", fx.PFAF))
    assert hashed == []  # raster: sidecar fast path, no re-hash
    vf.verified_path(mgr.basins_db_path(fx.PFAF), vf.region_asset_spec("basins", fx.PFAF))
    vf.verified_path(mgr.megabasins_path(), vf.asset_spec("megabasins.db"))
    assert hashed == ["basins73.db", "megabasins.db"]  # small .db and level-2 index: always re-hashed


def test_tamper_that_restores_mtime_is_caught(region, tmp_path, monkeypatch):
    import os

    mgr = _stage_all(region, tmp_path / "m")

    def tamper(path: Path):
        st = path.stat()
        with path.open("r+b") as f:
            first = f.read(1)
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF]))
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same size, same mtime, same inode

    db = mgr.basins_db_path(fx.PFAF)
    tamper(db)
    with pytest.raises(vf.AssetIntegrityError):
        vf.verified_path(db, vf.region_asset_spec("basins", fx.PFAF))  # small .db: always re-hashed
    monkeypatch.setattr(vf, "REHASH_BELOW_BYTES", 0)  # force the large-file fast path
    tif = mgr.raster_path("flowdir", fx.PFAF)
    tamper(tif)  # ctime cannot be restored by utime, so the sidecar check fails and the file is re-hashed
    with pytest.raises(vf.AssetIntegrityError):
        vf.verified_path(tif, vf.region_asset_spec("flowdir", fx.PFAF))


def test_tampered_existing_file_is_refused_and_reported(region, tmp_path):
    mgr = _stage_all(region, tmp_path / "m", rasters=False)
    with mgr.basins_db_path(fx.PFAF).open("ab") as f:
        f.write(b"x")
    with pytest.raises(vf.AssetIntegrityError):
        mgr.verified_asset("basins", fx.PFAF)
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
    assert "share-alike" in res.data_provenance["licence"].lower() and res.data_provenance["share_alike"] is True
    assert res.data_provenance["downloaded"] == [] and "MERIT_AUTO_STAGED" not in res.quality_flags
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
    assert "share-alike" in d["license"].lower()


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


def test_router_gee_window_limit_is_recorded_and_falls_through(calls, monkeypatch):
    """merit_gee over the window cap: a MERIT_GEE_WINDOW_LIMIT trail row, then the raw-DEM tier is tried."""
    def hybrid(lat, lon, **kw):
        calls["hybrid"].append(kw)
        raise RuntimeError("MERIT_BASINS_NOT_STAGED HYBRID_ROUTING_REQUIRED: not staged")

    def gee(lat, lon, **kw):
        calls["gee"].append(kw)
        raise pipe.MeritGeeWindowLimit("MERIT_GEE_WINDOW_LIMIT: basin needs 9.0M cells > cap 4.5M")

    dem = []

    def fast(*a, **k):
        dem.append(1)
        raise RuntimeError("dem down")

    import aihydro_watershed.delineation.pysheds_pipeline as ps

    monkeypatch.setattr(pipe, "merit_basins_hybrid_delineate", hybrid)
    monkeypatch.setattr(pipe, "delineate_merit_gee", gee)
    monkeypatch.setattr(ps, "delineate_fast", fast)
    with pytest.raises(Exception) as ei:
        router.delineate_from_point(-3.0, -60.0)
    rows = ei.value.details["fallback_history"]
    gee_rows = [r for r in rows if r["method"] == "merit_gee_pyflwdir"]
    assert gee_rows == [{"method": "merit_gee_pyflwdir", "outcome": "failed", "reason": "MERIT_GEE_WINDOW_LIMIT"}]
    assert len(calls["gee"]) == 1 and dem == [1]  # the raw-DEM tier ran after the cap


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


# ------------------------------------------------------------------ disclosure, policy, disk


def test_first_call_downloads_are_disclosed_and_flagged(region, tmp_path, monkeypatch):
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(tmp_path / "merit"))
    monkeypatch.delenv("AIHYDRO_MERIT_AUTO_STAGE", raising=False)
    monkeypatch.setattr(vf, "_default_opener", _serve(region))
    _stub_ee(monkeypatch, region, [])
    first = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    prov = first.data_provenance
    assert "MERIT_AUTO_STAGED" in first.quality_flags and prov["acquisition_policy"] == "tiered"
    got = {d["name"]: d for d in prov["downloaded"]}
    assert set(got) == {"megabasins.db", "basins73.db", "rivers73.db"}
    for name, d in got.items():
        assert d["bytes"] == region.files[name].stat().st_size and d["sha256"] == region.sha(name)
        assert d["url"].startswith("https://") or d["url"].startswith("http")
    again = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert again.data_provenance["downloaded"] == [] and "MERIT_AUTO_STAGED" not in again.quality_flags


def test_default_policy_is_one_constant(region, tmp_path, monkeypatch):
    import aihydro_watershed.merit.merit_manager as mm

    monkeypatch.delenv("AIHYDRO_MERIT_AUTO_STAGE", raising=False)
    assert mm.DEFAULT_AUTO_STAGE_POLICY == "tiered" and mm._merit_auto_stage_policy()[0] == "tiered"
    monkeypatch.setattr(mm, "DEFAULT_AUTO_STAGE_POLICY", "never")  # the one-line switch to confirm-first
    assert mm._merit_auto_stage_policy()[0] == "never"
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(tmp_path / "merit"))
    monkeypatch.setattr(vf, "_default_opener", _serve(region))
    mm.MeritDataManager(opener=_serve(region)).ensure_level2_index(download=True)
    with pytest.raises(RuntimeError, match="MERIT_BASINS_NOT_STAGED"):
        pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert not list((tmp_path / "merit" / "vector").glob("basins*"))


def test_disk_free_check_blocks_vector_staging(region, tmp_path, monkeypatch):
    import collections

    import aihydro_watershed.merit.merit_manager as mm

    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: Usage(10, 10, 1))
    opened: list = []
    mgr = MeritDataManager(tmp_path / "m", opener=lambda *a, **k: opened.append(a) or _serve(region)(*a))
    monkeypatch.delenv("AIHYDRO_MERIT_AUTO_STAGE", raising=False)
    st = mgr.auto_stage_vectors(fx.PFAF)
    assert "Not enough disk space" in st.message and opened == [] and not st.catchments_ready


# ------------------------------------------------------- negative cache, resume, locking


def test_integrity_failure_backs_off_for_24h_and_is_not_reported_as_not_staged(region, tmp_path, monkeypatch):
    import json as _json

    root = tmp_path / "merit"
    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(root))
    monkeypatch.delenv("AIHYDRO_MERIT_AUTO_STAGE", raising=False)
    MeritDataManager(opener=_serve(region)).ensure_level2_index(download=True)
    opens: list[str] = []
    bad = _serve(region, tamper="basins73.db")
    monkeypatch.setattr(vf, "_default_opener", lambda url, headers=None: opens.append(url) or bad(url))
    with pytest.raises(RuntimeError, match="MERIT_ASSET_INTEGRITY_FAILED"):
        pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    n = len(opens)
    assert n >= 1
    with pytest.raises(RuntimeError) as ei:  # second call: refused without touching the network
        pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert "MERIT_ASSET_INTEGRITY_FAILED" in str(ei.value) and "MERIT_BASINS_NOT_STAGED" not in str(ei.value)
    assert "automatic retry is refused" in str(ei.value) and len(opens) == n
    mgr = MeritDataManager(opener=_serve(region))
    marker = vf.integrity_marker_path(mgr.basins_db_path(fx.PFAF))
    # explicit forced retry works (and clears the marker)
    st = mgr.ensure_basins_region(fx.PFAF, acquisition_policy="download", force_retry=True)
    assert st.catchments_ready and not marker.exists()
    # an aged marker (> 24 h) no longer blocks
    (root / "vector" / "basins73.db").unlink()
    vf.sidecar_path(root / "vector" / "basins73.db").unlink()
    bad_again = MeritDataManager(opener=_serve(region, tamper="basins73.db"))
    bad_again.ensure_basins_region(fx.PFAF, acquisition_policy="download")
    rec = _json.loads(marker.read_text())
    rec["epoch"] -= 25 * 3600
    marker.write_text(_json.dumps(rec))
    assert vf.active_integrity_marker(mgr.basins_db_path(fx.PFAF)) is None
    st = MeritDataManager(opener=_serve(region)).ensure_basins_region(fx.PFAF, acquisition_policy="download")
    assert st.catchments_ready


class _Resp:
    def __init__(self, data: bytes, status: int = 200, fail_after: int | None = None):
        self._b, self.status, self._fail = io.BytesIO(data), status, fail_after
        self._read = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if self._fail is not None and self._read >= self._fail:
            raise ConnectionError("dropped")
        chunk = self._b.read(n if self._fail is None else min(n, self._fail - self._read))
        self._read += len(chunk)
        return chunk


def _range_server(data: bytes, *, supports_range: bool, drop_first_at: int | None):
    calls: list[dict] = []

    def opener(url, headers=None):
        calls.append(dict(headers or {}))
        rng = (headers or {}).get("Range")
        if rng and supports_range:
            start = int(rng.split("=")[1].split("-")[0])
            return _Resp(data[start:], status=206)
        return _Resp(data, status=200, fail_after=drop_first_at if not calls[1:] else None)

    return opener, calls


@pytest.mark.parametrize("supports_range", [True, False])
def test_interrupted_download_resumes_with_range_or_restarts(region, tmp_path, supports_range):
    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    opener, calls = _range_server(data, supports_range=supports_range, drop_first_at=len(data) // 2)
    dest = tmp_path / "d" / "rivers73.db"
    with pytest.raises(ConnectionError):
        vf.fetch_verified(spec, dest, opener=opener)
    parts = list(dest.parent.glob("rivers73.db.*.part"))
    assert len(parts) == 1 and 0 < parts[0].stat().st_size < len(data) and not dest.exists()
    assert vf.active_integrity_marker(dest) is None  # a dropped connection is not an integrity failure
    vf.fetch_verified(spec, dest, opener=opener)
    assert dest.read_bytes() == data and not list(dest.parent.glob("*.part")) and not dest.with_name("rivers73.db.lock").exists()
    assert len(calls) == 2
    assert calls[1]["Range"].startswith("bytes=") and calls[0] == {}  # resume asks for a range; a server that ignores it (200) restarts


def test_concurrent_first_downloads_do_not_interleave(region, tmp_path):
    import threading
    import time

    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    opens: list[int] = []

    def slow(url, headers=None):
        opens.append(1)
        time.sleep(0.4)
        return _Resp(data)

    dest = tmp_path / "d" / "rivers73.db"
    errs: list = []

    def run():
        try:
            vf.fetch_verified(spec, dest, opener=slow)
        except BaseException as exc:  # pragma: no cover
            errs.append(exc)

    ts = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(opens) == 1  # the others waited on the lock and found a verified file
    assert dest.read_bytes() == data and not list(dest.parent.glob("*.part"))


# ----------------------------------------------------------------------- store determinism


def test_locate_is_deterministic_among_overlapping_hits(tmp_path):
    import sqlite3

    rows = [
        {"comid": 73000010, "nextdown": 0, "geometry": fx.box(0, 40, 0.02, 40.02)},
        {"comid": 73000011, "nextdown": 0, "geometry": fx.box(0, 40, 0.02, 40.02)},
        {"comid": 73000012, "nextdown": 0, "geometry": fx.box(0, 40, 0.02, 40.02)},
    ]
    for order in (rows, rows[::-1]):
        db = tmp_path / f"b{order[0]['comid']}.db"
        g = gpd.GeoDataFrame(order, crs=4326)
        g["area_km2"] = 1.0
        g.to_file(db, layer="l0_basins", driver="SQLite", SPATIALITE="YES", engine="pyogrio")
        rdb = tmp_path / f"r{order[0]['comid']}.db"
        with sqlite3.connect(rdb) as con:
            con.execute("CREATE TABLE rivers (comid INTEGER, uparea REAL)")
            con.executemany("INSERT INTO rivers VALUES (?,?)", [(73000010, 5.0), (73000011, 9.0), (73000012, 9.0)])
        assert BasinsStore(db, rdb).locate(0.01, 40.01) == 73000011  # max uparea; tie -> lowest comid
        assert BasinsStore(db, None).locate(0.01, 40.01) == 73000010  # no rivers db: lowest comid


def test_missing_polygons_are_reported_and_flagged(staged, monkeypatch):
    store = BasinsStore(staged.basins_db_path(fx.PFAF), staged.rivers_db_path(fx.PFAF))
    real = gpd.read_file

    def drop(*a, **k):
        g = real(*a, **k)
        return g[g["comid"] != 73000004] if "comid" in g else g

    monkeypatch.setattr("aihydro_watershed.merit.basins_store.gpd.read_file", drop)
    geoms = store.geometries(store.upstream_ids(fx.TERMINAL))
    assert geoms.attrs["missing_comids"] == [73000004] and len(geoms) == 3
    res = pipe.merit_basins_hybrid_delineate(*fx.OUTLET_REQUEST)
    assert "VECTOR_ASSEMBLY_INCOMPLETE" in res.quality_flags
    assert res.area_validation["catchments_without_polygon"] == 1


# ------------------------------------------------------------------ legacy layout, shim, manifest


def test_legacy_shapefile_layout_is_noted_and_not_read(region, tmp_path, caplog):
    mgr = MeritDataManager(tmp_path / "m")
    legacy = mgr.catchment_shapefile_path(fx.PFAF)
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")
    with caplog.at_level("WARNING"):
        st = mgr.basins_region_cache(fx.PFAF)
    assert st.legacy_layout_found and not st.catchments_ready
    assert "not read" in st.message and "merit_ensure_basins_region" in st.message and "re-stage" in st.message.lower()
    assert any("Legacy shapefile layout" in r.message for r in caplog.records)


def test_deprecated_merit_download_shim_stages_verified_files(region, tmp_path, monkeypatch):
    from aihydro_watershed.merit import merit_download as md

    monkeypatch.setenv("AIHYDRO_MERIT_DIR", str(tmp_path / "m"))
    monkeypatch.setattr(vf, "_default_opener", _serve(region))
    with pytest.warns(DeprecationWarning):
        assert md.ensure_level2_index(tmp_path / "ignored") is True
    with pytest.warns(DeprecationWarning):
        assert md.download_river_shapefile(fx.PFAF, tmp_path / "ignored") is True
    with pytest.warns(DeprecationWarning):
        assert md.download_catchment_shapefile(fx.PFAF, tmp_path / "ignored") is True
    assert MeritDataManager().delineator_ready(fx.PFAF) is True
    assert not hasattr(MeritDataManager, "configure_delineator_env")


def test_manifest_matches_vendored_delineator_constants_and_generator_is_reproducible():
    import ast
    import hashlib
    import importlib.util
    import re

    fixture = Path(__file__).parent / "fixtures" / "delineator_v2.2.4" / "constants.py"
    raw = fixture.read_bytes()
    man = json.loads((Path(vf.__file__).parent / "delineator_data_manifest.json").read_text())
    assert (fixture.parent / "LICENSE").read_text().startswith("MIT License")  # delineator is MIT
    block = re.search(r"HASHES = (\{.*?\n\})", raw.decode(), re.S).group(1)
    pins = {k: v.replace("sha256:", "") for k, v in ast.literal_eval(block).items()}
    assert {k: v["sha256"] for k, v in man["files"].items()} == pins
    src = man["source"]
    assert src["constants_blob_sha1"] == hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert len(src["tag_commit"]) == 40 and src["tag_commit"] in src["hashes_file"]
    assert src["tag_commit"] in man["level2_index"]["url"] and src["tag"] == "v" + src["package_version"]
    script = Path(__file__).parents[1] / "scripts" / "gen_delineator_manifest.py"
    spec = importlib.util.spec_from_file_location("gen_manifest", script)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sizes = {k: v["size_bytes"] for k, v in man["files"].items()}
    rebuilt = gen.build(raw, src["tag_commit"], src["tag"], sizes, man["level2_index"], man)
    assert json.dumps(rebuilt, indent=1, sort_keys=True) + "\n" == (
        Path(vf.__file__).parent / "delineator_data_manifest.json"
    ).read_text()


# ---------------------------------------------------------------- resume edge cases, heartbeat


def _put_part(dest: Path, content: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.stale.part"
    part.write_bytes(content)
    return part


def test_failed_resume_retries_once_from_zero_before_any_marker(region, tmp_path):
    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    dest = tmp_path / "d" / "rivers73.db"
    _put_part(dest, b"\xff" * (len(data) // 2))  # corrupt partial: appending the real tail cannot verify
    opener, calls = _range_server(data, supports_range=True, drop_first_at=None)
    vf.fetch_verified(spec, dest, opener=opener)
    assert dest.read_bytes() == data and vf.active_integrity_marker(dest) is None
    assert [bool(c) for c in calls] == [True, False]  # Range attempt, then a clean full request
    assert not list(dest.parent.glob("*.part"))


def test_failed_resume_and_failed_retry_writes_the_marker_once(region, tmp_path):
    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    bad = data[:-1] + bytes([data[-1] ^ 0xFF])
    dest = tmp_path / "d" / "rivers73.db"
    _put_part(dest, data[: len(data) // 2])
    opener, calls = _range_server(bad, supports_range=True, drop_first_at=None)
    with pytest.raises(vf.AssetIntegrityError):
        vf.fetch_verified(spec, dest, opener=opener)
    assert len(calls) == 2 and vf.active_integrity_marker(dest) is not None
    assert not dest.exists() and not list(dest.parent.glob("*.part"))


@pytest.mark.parametrize("extra", [0, 7])
def test_full_size_or_oversize_partial_is_discarded_before_the_range_request(region, tmp_path, extra):
    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    dest = tmp_path / "d" / "rivers73.db"
    _put_part(dest, b"\x00" * (len(data) + extra))
    seen: list = []

    def strict(url, headers=None):
        seen.append(dict(headers or {}))
        if headers and headers.get("Range"):
            raise AssertionError("416 Range Not Satisfiable would follow")
        return _Resp(data)

    vf.fetch_verified(spec, dest, opener=strict)
    assert dest.read_bytes() == data and seen == [{}]


def test_lock_heartbeat_keeps_a_long_download_from_looking_stale(region, tmp_path, monkeypatch):
    import threading
    import time

    monkeypatch.setattr(vf, "_LOCK_STALE_SECONDS", 0.5)
    monkeypatch.setattr(vf, "_LOCK_HEARTBEAT_SECONDS", 0.1)
    monkeypatch.setattr(vf, "_LOCK_POLL_SECONDS", 0.05)
    spec = vf.region_asset_spec("rivers", fx.PFAF)
    data = region.files["rivers73.db"].read_bytes()
    opens: list[int] = []

    def slow(url, headers=None):
        opens.append(1)
        time.sleep(1.5)  # three times the stale limit
        return _Resp(data)

    dest = tmp_path / "d" / "rivers73.db"
    errs: list = []

    def run():
        try:
            vf.fetch_verified(spec, dest, opener=slow)
        except BaseException as exc:  # pragma: no cover
            errs.append(exc)

    ts = [threading.Thread(target=run), threading.Thread(target=run)]
    ts[0].start()
    time.sleep(0.2)
    ts[1].start()
    [t.join() for t in ts]
    assert not errs and len(opens) == 1  # without the heartbeat the second thread breaks the lock and downloads too
    assert dest.read_bytes() == data and not dest.with_name("rivers73.db.lock").exists()
