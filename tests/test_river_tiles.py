"""Offline tests for the MERIT river PMTiles builder, on a tiny SYNTHETIC rivers db (no MERIT data)."""
from __future__ import annotations

import json
import pathlib

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from aihydro_watershed.merit import verified_fetch as vf
from aihydro_watershed.tiles import AREA_BANDS, build_merit_river_tiles, tile_paths
from aihydro_watershed.tiles import pmtiles_merge as pm

# one reach per interesting uparea value; (comid, uparea, sorder)
_REACHES = [
    (72000001, 250_000.0, 6), (72000002, 10_000.0, 5), (72000003, 9_999.9, 5), (72000004, 2_500.0, 4),
    (72000005, 600.0, 3), (72000006, 500.0, 3), (72000007, 150.0, 2), (72000008, 100.0, 2),
    (72000009, 99.9, 1), (72000010, 25.0, 1), (72000011, None, 1),
]
_MAXZ = 11
_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)


def _make_synthetic_db(tmp_path):
    """SYNTHETIC spatialite-style rivers db: long parallel lines so none vanishes at low zoom."""
    geoms, rows = [], []
    for i, (comid, ua, so) in enumerate(_REACHES):
        y = 40.0 + i * 0.8
        geoms.append(LineString([(-100.0, y), (-92.0, y + 0.3)]))
        rows.append({"comid": comid, "lengthkm": 700.0 + i, "uparea": ua, "sorder": so})
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows).set_index("comid"), geometry=geoms, crs=4326)
    vec = tmp_path / "vector"
    vec.mkdir()
    db = vec / "rivers72.db"
    pyogrio.write_dataframe(gdf.reset_index(), db, layer="rivers", driver="SQLite", SPATIALITE="YES", FID="comid")
    (vec / "rivers72.db.provenance.json").write_text(json.dumps({
        "name": "rivers72.db", "sha256": "ab" * 32, "size_bytes": db.stat().st_size, "synthetic": True,
        "note": "SYNTHETIC TEST DATA - not MERIT",
    }))
    return db


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rt")
    db = _make_synthetic_db(tmp)
    out = tmp / "tiles"
    rec = build_merit_river_tiles(72, out, source_db=db, max_zoom=_MAXZ, verify_source=False)
    return tmp, db, out, rec


def _read(path, layer, zoom, **kw):
    return pyogrio.read_dataframe(path, layer=layer, ZOOM_LEVEL=zoom, **kw)


def test_layers_and_provenance(built):
    _tmp, db, out, rec = built
    pmt, side = tile_paths(72, out)
    assert pmt.exists() and side.exists() and rec["status"] == "built"
    assert [r[0] for r in pyogrio.list_layers(pmt)] == [b[0] for b in AREA_BANDS]
    assert rec["source"]["sha256"] == "ab" * 32  # taken from the staging sidecar, not recomputed
    assert rec["source"]["sha256_origin"] == "staging_provenance_sidecar_unverified"  # verify_source=False
    assert rec["source"]["source_pinned"] is False and rec["source"]["verified_at"] is None
    assert rec["source"]["synthetic"] is True
    assert "CC BY-NC-SA" in rec["licence"] and "unresolved" in rec["licence"]
    assert rec["citation"] == "Lin et al. 2019"
    assert "CC BY-NC-SA" in rec["attribution"] and "attributed per mghydro.com; exact variant unresolved" in rec["attribution"]
    assert "share-alike" in rec["licence_note"].lower() and "non-commercial" in rec["licence_note"].lower()
    assert rec["licence_note"].startswith(vf.licence_summary())
    meta = pm.PMTilesReader(pmt).metadata()
    assert meta["licence_note"] == rec["licence_note"] and meta["attribution"] == rec["attribution"]
    assert rec["reaches_absent_at_max_zoom"] == 0 and rec["reaches_absent_at_max_zoom_comids"] == []
    assert rec["tooling"]["pyogrio"] == pyogrio.__version__ and rec["tooling"]["gdal"]
    assert rec["output"]["size_bytes"] == pmt.stat().st_size and len(rec["output"]["sha256"]) == 64
    assert rec["reaches_per_layer"] == {
        "rivers_z0": 2, "rivers_z5": 2, "rivers_z7": 2, "rivers_z9": 2, "rivers_z11": 3}  # NaN uparea -> last band
    assert rec["reaches_without_uparea_placed_in_finest_band"] == 1
    assert json.loads(side.read_text())["output"]["sha256"] == rec["output"]["sha256"]


def test_attributes_equal_source_at_max_zoom(built):
    _tmp, db, out, _rec = built
    pmt, _ = tile_paths(72, out)
    src = pyogrio.read_dataframe(db, fid_as_index=True)
    for comid, ua, so in _REACHES:
        if ua is None:
            continue
        layer = next(n for n, _z, lo, hi in AREA_BANDS if lo <= ua < hi)
        x0, y0 = _TO_3857.transform(-100.0, src.loc[comid].geometry.bounds[1])
        x1, y1 = _TO_3857.transform(-92.0, src.loc[comid].geometry.bounds[3])
        d = _read(pmt, layer, _MAXZ, bbox=(x0, y0, x1, y1))
        m = d[d.comid == comid]
        assert len(m) and set(m.uparea) == {ua} and set(m.sorder) == {float(so)}, comid
        assert set(m.lengthkm) == {float(src.loc[comid].lengthkm)}


@pytest.mark.parametrize("zoom,threshold", [(0, 10_000), (4, 10_000), (5, 2_500), (6, 2_500),
                                            (7, 500), (8, 500), (9, 100), (10, 100)])
def test_z1_area_thresholds(built, zoom, threshold):
    _tmp, _db, out, _rec = built
    pmt, _ = tile_paths(72, out)
    d = pd.concat([_read(pmt, n, zoom) for n, *_ in AREA_BANDS])
    ids = set(d.comid.astype(int))
    assert d.uparea.min() >= threshold  # nothing below the band threshold
    expected = {c for c, ua, _ in _REACHES if ua is not None and ua >= threshold}
    assert ids == expected  # and every reach at/above it is there
    assert 72000011 not in ids  # unknown area never appears before the last band


def test_finest_zoom_has_every_non_degenerate_reach(built):
    """All reaches are at the finest zoom except degenerate sub-pixel ones, which the sidecar counts."""
    _tmp, _db, out, rec = built
    pmt, _ = tile_paths(72, out)
    d = pd.concat([_read(pmt, n, _MAXZ) for n, *_ in AREA_BANDS])
    absent = {c for c, *_ in _REACHES} - set(d.comid.astype(int))
    assert absent == set(rec["reaches_absent_at_max_zoom_comids"]) and len(absent) == rec["reaches_absent_at_max_zoom"]
    assert absent == set()  # the synthetic reaches are ~800 km long


def test_degenerate_reach_is_counted_not_hidden(tmp_path):
    db = _make_synthetic_db(tmp_path)
    gdf = pyogrio.read_dataframe(db, fid_as_index=True).rename_axis("comid").reset_index()
    tiny = gdf.iloc[[-1]].copy()
    tiny["comid"] = 72999999
    tiny["lengthkm"] = 0.0
    tiny["geometry"] = [LineString([(-96.0, 45.0), (-96.0, 45.00001)])]  # ~1 m
    both = gpd.GeoDataFrame(pd.concat([gdf, tiny], ignore_index=True), geometry="geometry", crs=4326)
    db2 = tmp_path / "vector" / "rivers73.db"
    pyogrio.write_dataframe(both, db2, layer="rivers", driver="SQLite", SPATIALITE="YES", FID="comid")
    rec = build_merit_river_tiles(73, tmp_path / "o", source_db=db2, max_zoom=_MAXZ, verify_source=False)
    assert rec["source"]["n_reaches"] == len(_REACHES) + 1
    assert rec["reaches_absent_at_max_zoom"] == len(rec["reaches_absent_at_max_zoom_comids"])
    assert set(rec["reaches_absent_at_max_zoom_comids"]) <= {72999999}


def test_idempotent_then_force_and_source_change(built):
    tmp, db, out, rec = built
    pmt, side = tile_paths(72, out)
    mtime = pmt.stat().st_mtime_ns
    again = build_merit_river_tiles(72, out, source_db=db, max_zoom=_MAXZ, verify_source=False)
    assert again["status"] == "skipped_up_to_date" and pmt.stat().st_mtime_ns == mtime
    # different build options -> rebuild
    other = build_merit_river_tiles(72, out, source_db=db, max_zoom=12, verify_source=False)
    assert other["status"] == "built"
    # source sha changes -> rebuild; force -> rebuild
    prov = db.with_name(db.name + ".provenance.json")
    j = json.loads(prov.read_text())
    j["sha256"] = "cd" * 32
    prov.write_text(json.dumps(j))
    changed = build_merit_river_tiles(72, out, source_db=db, max_zoom=12, verify_source=False)
    assert changed["status"] == "built" and changed["source"]["sha256"] == "cd" * 32
    assert build_merit_river_tiles(72, out, source_db=db, max_zoom=12, verify_source=False)["status"] == "skipped_up_to_date"
    assert build_merit_river_tiles(72, out, source_db=db, max_zoom=12, force=True, verify_source=False)["status"] == "built"
    # a damaged output is not trusted
    pmt.write_bytes(pmt.read_bytes()[:-10])
    assert build_merit_river_tiles(72, out, source_db=db, max_zoom=12, verify_source=False)["status"] == "built"


def test_pinned_flags_and_atomic_sidecar(tmp_path):
    db = _make_synthetic_db(tmp_path)
    prov = db.with_name(db.name + ".provenance.json")
    j = json.loads(prov.read_text())
    j.update(pinned_sha256=j["sha256"], verified_at="2026-11-08T00:00:00+00:00")
    prov.write_text(json.dumps(j))
    out = tmp_path / "o"
    rec = build_merit_river_tiles(72, out, source_db=db, max_zoom=_MAXZ, verify_source=False)
    assert rec["source"]["source_pinned"] is True and rec["source"]["verified_at"] == "2026-11-08T00:00:00+00:00"
    assert rec["source"]["sha256_origin"] == "staging_provenance_sidecar_unverified"
    assert not list(out.glob("*.tmp")) and not list(out.glob("*.building"))


def test_missing_pmtiles_driver_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pyogrio, "list_drivers", lambda *a, **k: {"MVT": "rw"})
    with pytest.raises(RuntimeError, match="PMTiles driver"):
        build_merit_river_tiles(72, tmp_path / "o", source_db=tmp_path / "x.db", verify_source=False)


def test_empty_z0_band_metadata_matches_header(tmp_path):
    """No reach >= 10,000 km2 -> no rivers_z0 archive; metadata bounds/centre must still be the union."""
    db = _make_synthetic_db(tmp_path)
    gdf = pyogrio.read_dataframe(db, fid_as_index=True).rename_axis("comid").reset_index()
    small = gdf[gdf.uparea < 10_000].copy()
    db2 = tmp_path / "vector" / "rivers74.db"
    pyogrio.write_dataframe(small, db2, layer="rivers", driver="SQLite", SPATIALITE="YES", FID="comid")
    rec = build_merit_river_tiles(74, tmp_path / "o", source_db=db2, max_zoom=_MAXZ, verify_source=False)
    assert rec["reaches_per_layer"]["rivers_z0"] == 0
    r = pm.PMTilesReader(tile_paths(74, tmp_path / "o")[0])
    _assert_meta_matches_header(r)


def _assert_meta_matches_header(r):
    meta = r.metadata()
    b = [float(x) for x in meta["bounds"].split(",")]
    assert b == pytest.approx([r.min_lon_e7 / 1e7, r.min_lat_e7 / 1e7, r.max_lon_e7 / 1e7, r.max_lat_e7 / 1e7])
    clon, clat, cz = (float(x) for x in meta["center"].split(","))
    assert (clon, clat, int(cz)) == pytest.approx((r.center_lon_e7 / 1e7, r.center_lat_e7 / 1e7, r.center_zoom))
    assert clon == pytest.approx((b[0] + b[2]) / 2, abs=1e-6) and clat == pytest.approx((b[1] + b[3]) / 2, abs=1e-6)
    assert r.min_zoom <= r.center_zoom <= r.max_zoom


def test_metadata_bounds_are_union_of_all_inputs(built):
    _tmp, _db, out, _rec = built
    pmt, _ = tile_paths(72, out)
    r = pm.PMTilesReader(pmt)
    _assert_meta_matches_header(r)
    # the union must be wider than the first (z0, a few reaches) band alone
    assert r.metadata()["bounds"] != "" and r.max_lat_e7 > r.min_lat_e7


def test_missing_sidecar_hashes_the_file(tmp_path):
    db = _make_synthetic_db(tmp_path)
    db.with_name(db.name + ".provenance.json").unlink()
    rec = build_merit_river_tiles(72, tmp_path / "o", source_db=db, max_zoom=11, verify_source=False)
    assert rec["source"]["sha256_origin"] == "computed_no_sidecar" and len(rec["source"]["sha256"]) == 64


def test_not_staged_and_bad_columns(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_merit_river_tiles(72, tmp_path / "o", source_db=tmp_path / "nope.db", verify_source=False)
    bad = tmp_path / "bad.gpkg"
    pyogrio.write_dataframe(
        gpd.GeoDataFrame({"x": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs=4326), bad, driver="GPKG")
    with pytest.raises(ValueError, match="missing column"):
        build_merit_river_tiles(72, tmp_path / "o", source_db=bad, verify_source=False)


def test_max_zoom_below_last_band_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_zoom"):
        build_merit_river_tiles(72, tmp_path / "o", source_db=tmp_path / "x.db", max_zoom=8, verify_source=False)


def test_default_paths_follow_aihydro_home(monkeypatch, tmp_path):
    monkeypatch.setenv("AIHYDRO_HOME", str(tmp_path))
    monkeypatch.delenv("AIHYDRO_MERIT_DIR", raising=False)
    monkeypatch.delenv("MERIT_DATA_DIR", raising=False)
    p, s = tile_paths("7")
    assert p == tmp_path / "merit" / "tiles" / "rivers07.pmtiles" and s.name == "rivers07.pmtiles.provenance.json"


def test_directory_codec_roundtrip():
    entries = [(1, 1, 100, 0), (2, 1, 50, 100), (7, 1, 10, 500), (9, 3, 20, 510)]  # contiguous + gap + run
    assert pm._decode_directory(pm._encode_directory(entries)) == entries


def test_merge_with_leaf_directories(built, tmp_path, monkeypatch):
    """Force the leaf-directory path (tiny root budget) and check the archive still reads the same."""
    _tmp, db, out, _rec = built
    pmt, _ = tile_paths(72, out)
    monkeypatch.setattr(pm, "_ROOT_BUDGET", 40)
    # re-merge two single-layer archives written by GDAL
    gdf = pyogrio.read_dataframe(db, fid_as_index=True).rename_axis("comid").reset_index()
    a, b = tmp_path / "a.pmtiles", tmp_path / "b.pmtiles"
    pyogrio.write_dataframe(gdf[gdf.uparea >= 1000], a, layer="la", driver="PMTiles", MINZOOM=0, MAXZOOM=8)
    pyogrio.write_dataframe(gdf[gdf.uparea < 1000], b, layer="lb", driver="PMTiles", MINZOOM=3, MAXZOOM=8)
    info = pm.merge_pmtiles([a, b], tmp_path / "m.pmtiles")
    assert info["leaf_directories"] is True and info["layers"] == ["la", "lb"]
    assert len(_read(tmp_path / "m.pmtiles", "la", 8)) == len(_read(a, "la", 8))
    assert len(_read(tmp_path / "m.pmtiles", "lb", 8)) == len(_read(b, "lb", 8))
    with pytest.raises(ValueError, match="duplicate layer"):
        pm.merge_pmtiles([a, a], tmp_path / "x.pmtiles")


def _two_archives(tmp_path):
    gdf = pyogrio.read_dataframe(_make_synthetic_db(tmp_path), fid_as_index=True).rename_axis("comid").reset_index()
    a, b = tmp_path / "a.pmtiles", tmp_path / "b.pmtiles"
    # first archive spans a small part of the data: the metadata must still be the union
    pyogrio.write_dataframe(gdf[gdf.uparea >= 100_000], a, layer="la", driver="PMTiles", MINZOOM=0, MAXZOOM=6)
    pyogrio.write_dataframe(gdf[gdf.uparea < 100_000], b, layer="lb", driver="PMTiles", MINZOOM=0, MAXZOOM=6)
    return a, b


def _as_uncompressed(src, dst):
    """Rewrite a GDAL archive with tile compression none (GDAL's PMTiles writer has no option for it)."""
    r = pm.PMTilesReader(src)
    data, entries = b"", []
    for tid, blob in r.tiles():
        entries.append((tid, 1, len(blob), len(data)))
        data += blob
    root = pm._compress(pm._encode_directory(entries), pm._GZIP)
    meta = pm._compress(json.dumps(r.metadata()).encode(), pm._GZIP)
    h = pm.HEADER_BYTES
    head = pm._HEADER.pack(
        b"PMTiles", 3, h, len(root), h + len(root), len(meta), h + len(root) + len(meta), 0,
        h + len(root) + len(meta), len(data), len(entries), len(entries), len(entries), 1, pm._GZIP, pm._NONE, 1,
        r.min_zoom, r.max_zoom, r.min_lon_e7, r.min_lat_e7, r.max_lon_e7, r.max_lat_e7,
        r.center_zoom, r.center_lon_e7, r.center_lat_e7)
    pathlib.Path(dst).write_bytes(head + root + meta + data)
    return dst


def test_merge_metadata_is_union_even_when_first_archive_is_narrow(tmp_path):
    a, b = _two_archives(tmp_path)
    pm.merge_pmtiles([a, b], tmp_path / "m.pmtiles")
    ra, rb, rm = pm.PMTilesReader(a), pm.PMTilesReader(b), pm.PMTilesReader(tmp_path / "m.pmtiles")
    assert rm.min_lat_e7 == min(ra.min_lat_e7, rb.min_lat_e7) and rm.max_lat_e7 == max(ra.max_lat_e7, rb.max_lat_e7)
    assert (ra.min_lat_e7, ra.max_lat_e7) != (rm.min_lat_e7, rm.max_lat_e7)  # a alone is narrower
    _assert_meta_matches_header(rm)


def test_mixed_tile_compression_raises(tmp_path):
    a, b = _two_archives(tmp_path)
    c = _as_uncompressed(b, tmp_path / "c.pmtiles")
    assert pm.PMTilesReader(c).tile_comp == 1 and pm.PMTilesReader(a).tile_comp == 2
    with pytest.raises(ValueError, match="different tile compression"):
        pm.merge_pmtiles([a, c], tmp_path / "x.pmtiles")


def test_uncompressed_tiles_round_trip(tmp_path):
    a, b = _two_archives(tmp_path)
    a0, b0 = _as_uncompressed(a, tmp_path / "a0.pmtiles"), _as_uncompressed(b, tmp_path / "b0.pmtiles")
    info = pm.merge_pmtiles([a0, b0], tmp_path / "m.pmtiles")
    m = pm.PMTilesReader(tmp_path / "m.pmtiles")
    assert m.tile_comp == 1 and info["layers"] == ["la", "lb"]
    assert dict(m.tiles()).keys() == (dict(pm.PMTilesReader(a).tiles()).keys() | dict(pm.PMTilesReader(b).tiles()).keys())
    assert len(_read(tmp_path / "m.pmtiles", "la", 6)) == len(_read(a, "la", 6))
    assert len(_read(tmp_path / "m.pmtiles", "lb", 6)) == len(_read(b, "lb", 6))


def test_missing_or_empty_vector_layers_raises(tmp_path, monkeypatch):
    a, b = _two_archives(tmp_path)
    for meta in ({"name": "x"}, {"vector_layers": []}):
        monkeypatch.setattr(pm.PMTilesReader, "metadata", lambda self, m=meta: dict(m))
        with pytest.raises(ValueError, match="no vector_layers"):
            pm.merge_pmtiles([a, b], tmp_path / "x.pmtiles")


def test_single_vs_multi_leaf_directory_boundary(tmp_path, monkeypatch):
    a, b = _two_archives(tmp_path)
    n = pm.merge_pmtiles([a, b], tmp_path / "one.pmtiles")["tiles"]
    assert n > 1
    ref = pm.PMTilesReader(tmp_path / "one.pmtiles")
    assert ref.leaf_len == 0
    root_len = ref.root_len
    # budget exactly at the root size: still a single directory; one byte less: leaves
    monkeypatch.setattr(pm, "_ROOT_BUDGET", root_len)
    assert pm.merge_pmtiles([a, b], tmp_path / "at.pmtiles")["leaf_directories"] is False
    monkeypatch.setattr(pm, "_ROOT_BUDGET", root_len - 1)
    info = pm.merge_pmtiles([a, b], tmp_path / "below.pmtiles")
    assert info["leaf_directories"] is True
    lv, one = pm.PMTilesReader(tmp_path / "below.pmtiles"), ref
    assert dict(lv.tiles()) == dict(one.tiles())  # same tiles through the leaf path
