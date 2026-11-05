"""Offline tests for the MERIT-via-Earth-Engine request/window logic and the area calculation.

``tests/fixtures/merit_gee_window_10336660.npz`` is a recorded 400 x 400 cell crop of the real
``MERIT/Hydro/v1_0_1`` ``dir``/``upa``/``wth`` bands around CAMELS gauge 10336660 (native dtypes) with
its global cell offsets. A fake ``ee`` + ``requests`` serve any ``getDownloadURL`` request from it
(cells outside the recording read as no-data), so no network or credentials are needed.
"""

from __future__ import annotations

import io
import math
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from pyproj import Geod

from aihydro_watershed.delineation import merit_flowdir_pipeline as m

FIXTURE = Path(__file__).parent / "fixtures" / "merit_gee_window_10336660.npz"
GAUGE = (39.10741, -120.16214)  # CAMELS 10336660


@pytest.fixture(scope="module")
def rec():
    z = np.load(FIXTURE)
    return {k: z[k] for k in ("dir", "upa", "wth")} | {"col0": int(z["col0"]), "row0": int(z["row0"])}


class FakeEE(types.ModuleType):
    """Records Initialize calls and getDownloadURL params; serves the recording via ``requests.get``."""

    def __init__(self):
        super().__init__("ee")
        self.init_calls = 0
        self.requests: list[dict] = []
        self.bands: list[str] = []
        self.tamper = None  # callable(transform_tuple, nrows, ncols) -> (transform_tuple, nrows, ncols)

    def Initialize(self, **kw):  # noqa: N802
        self.init_calls += 1

    def Image(self, _dataset):  # noqa: N802
        fake = self

        class _Img:
            def select(self, bands):
                fake.bands = list(bands)
                return self

            def getDownloadURL(self, params):  # noqa: N802
                fake.requests.append({**params, "bands": list(fake.bands)})
                return f"fake://{len(fake.requests) - 1}"

        return _Img()


@pytest.fixture
def fake_ee(rec, monkeypatch, tmp_path):
    import rasterio
    from rasterio.transform import Affine

    ee = FakeEE()
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setattr(m, "_EE_READY", None)
    monkeypatch.setenv("AIHYDRO_HOME", str(tmp_path / "aihome"))
    h, w = rec["dir"].shape

    def serve(url, timeout=None):
        req = ee.requests[int(url.split("//")[1])]
        a, _, c, _, e, f = req["crs_transform"]
        ncols, nrows = map(int, req["dimensions"].split("x"))
        col0 = round((c - m.MERIT_ORIGIN_X) / m.MERIT_NATIVE_PX_DEG)
        row0 = round((m.MERIT_ORIGIN_Y - f) / m.MERIT_NATIVE_PX_DEG)
        if ee.tamper:
            (a, _, c, _, e, f), nrows, ncols = ee.tamper((a, 0, c, 0, e, f), nrows, ncols)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for band in req["bands"]:
                out = np.full((nrows, ncols), 255 if band == "dir" else 0, dtype=np.float32)
                a0, a1 = max(col0, rec["col0"]), min(col0 + ncols, rec["col0"] + w)
                b0, b1 = max(row0, rec["row0"]), min(row0 + nrows, rec["row0"] + h)
                if a0 < a1 and b0 < b1:
                    out[b0 - row0 : b1 - row0, a0 - col0 : a1 - col0] = rec[band][
                        b0 - rec["row0"] : b1 - rec["row0"], a0 - rec["col0"] : a1 - rec["col0"]
                    ]
                mem = io.BytesIO()
                with rasterio.MemoryFile() as mf:
                    with mf.open(driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32",
                                 crs="EPSG:4326", transform=Affine(a, 0, c, 0, e, f)) as dst:
                        dst.write(out, 1)
                    mem.write(mf.read())
                zf.writestr(f"x.{band}.tif", mem.getvalue())
        return types.SimpleNamespace(ok=True, status_code=200, content=buf.getvalue(), text="")

    import requests

    monkeypatch.setattr(requests, "get", serve)
    return ee


# --- native grid --------------------------------------------------------------------------------


def test_native_window_is_cell_aligned_and_covers_bbox():
    bbox = m.bbox_from_point(*GAUGE, 20.0)
    tf, ncols, nrows = m.native_window(bbox)
    px = m.MERIT_NATIVE_PX_DEG
    xmin, ymax = tf.c, tf.f
    xmax, ymin = xmin + ncols * px, ymax - nrows * px
    assert xmin <= bbox[0] and ymin <= bbox[1] and xmax >= bbox[2] and ymax >= bbox[3]
    assert bbox[0] - xmin < px and xmax - bbox[2] < px and bbox[1] - ymin < px and ymax - bbox[3] < px
    for v, o in ((xmin, m.MERIT_ORIGIN_X), ((m.MERIT_ORIGIN_Y - ymax), 0.0)):
        assert v - o == pytest.approx(round((v - o) / px) * px, abs=1e-9)
    assert (tf.a, tf.e, tf.b, tf.d) == (px, -px, 0.0, 0.0)


def test_old_request_grid_was_not_native():
    """Regression: scale=92.77 m in EPSG:4326 is 0.004% off the native cell (a cell drifts every ~25000)."""
    old_px_deg = 92.77 / 111319.49
    assert 1000 * (old_px_deg - m.MERIT_NATIVE_PX_DEG) / m.MERIT_NATIVE_PX_DEG == pytest.approx(0.04, abs=0.01)


def test_request_uses_native_crs_transform_without_clip_or_region(rec, fake_ee):
    b = m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 5.0))
    (req,) = fake_ee.requests
    assert req["crs"] == "EPSG:4326" and "region" not in req and "scale" not in req
    tf, ncols, nrows = m.native_window(m.bbox_from_point(*GAUGE, 5.0))
    assert req["crs_transform"] == [tf.a, 0, tf.c, 0, tf.e, tf.f] and req["dimensions"] == f"{ncols}x{nrows}"
    assert req["bands"] == ["dir", "upa", "wth"]
    assert b.transform == tf or (abs(b.transform.c - tf.c) < 1e-12 and abs(b.transform.f - tf.f) < 1e-12)
    r0, c0 = round((m.MERIT_ORIGIN_Y - tf.f) / tf.a) - rec["row0"], round((tf.c - m.MERIT_ORIGIN_X) / tf.a) - rec["col0"]
    np.testing.assert_array_equal(b.bands["upa"], rec["upa"][r0 : r0 + nrows, c0 : c0 + ncols])


# --- initialise once ----------------------------------------------------------------------------


def test_earth_engine_is_initialised_once_per_process(fake_ee):
    for hs in (2.0, 3.0, 4.0):
        m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, hs))
    assert len(fake_ee.requests) == 3 and fake_ee.init_calls == 1


def test_reinitialises_for_a_different_project(fake_ee):
    m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 2.0), project_id="a")
    m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 3.0), project_id="b")
    assert fake_ee.init_calls == 2


# --- window growth and cap ----------------------------------------------------------------------


def test_adaptive_growth_gives_the_same_basin_as_one_big_window(rec, fake_ee):
    big = m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 12.0))
    truth = m.delineate_merit_bundle(big, *GAUGE)
    assert not truth.touches_edge and truth.area_km2 > 5
    n = len(fake_ee.requests)
    r = m.delineate_merit_gee(*GAUGE, initial_half_side_km=1.5)
    assert len(fake_ee.requests) - n >= 2  # grew at least once
    assert not r.touches_edge
    assert r.area_km2 == pytest.approx(truth.area_km2, rel=1e-12)
    assert r.gdf.geometry.iloc[0].symmetric_difference(truth.gdf.geometry.iloc[0]).area == pytest.approx(0.0, abs=1e-12)


def test_window_cap_raises_clear_error_before_any_request(fake_ee, monkeypatch):
    monkeypatch.setattr(m, "MERIT_GEE_MAX_WINDOW_CELLS", 4_000)
    with pytest.raises(m.MeritGeeWindowLimit, match="MERIT_GEE_WINDOW_LIMIT") as ei:
        m.delineate_merit_gee(*GAUGE, initial_half_side_km=1.5)
    assert "cells" in str(ei.value) and "MERIT-Basins hybrid" in str(ei.value)
    # Earlier, smaller windows were fetched; the oversized one never reached Earth Engine.
    for req in fake_ee.requests:
        w, h = map(int, req["dimensions"].split("x"))
        assert w * h <= 4_000


def test_cap_is_a_runtime_error_so_the_router_falls_through():
    assert issubclass(m.MeritGeeWindowLimit, RuntimeError)
    src = (Path(m.__file__).parent / "router.py").read_text()
    assert '"MERIT_GEE_WINDOW_LIMIT"' in src  # recorded as the fallback reason


# --- area ----------------------------------------------------------------------------------------


def test_polygon_area_equals_pixel_area_sum_and_geodesic_area(rec, fake_ee):
    """Reported area = true area of the basin-mask cells, well under 0.1%: no vectorisation or projection bias."""
    import pyflwdir

    from aihydro_watershed.delineation.types import area_km2

    b = m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 12.0))
    snap = m.snap_outlet_on_merit(*GAUGE, upa=b.bands["upa"], wth=b.bands["wth"], transform=b.transform)
    d8 = m.merit_dir_to_pyflwdir(b.bands["dir"])
    flw = pyflwdir.from_array(d8, ftype="d8", mask=np.isin(d8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
                              transform=b.transform, latlon=True)
    mask = flw.basins(xy=([snap.lon], [snap.lat]), ids=np.array([1], dtype=np.uint32)) == 1
    gdf = m._polygonize_mask(mask, b.transform, b.crs)
    poly_km2 = area_km2(gdf)

    geod = Geod(ellps="WGS84")
    px = m.MERIT_NATIVE_PX_DEG
    pixel_sum = 0.0
    for r in np.where(mask.any(axis=1))[0]:
        top = b.transform.f - r * px
        cell_m2 = abs(geod.polygon_area_perimeter([0, px, px, 0], [top, top, top - px, top - px])[0])
        pixel_sum += mask[r].sum() * cell_m2 / 1e6
    geo_km2 = sum(abs(geod.geometry_area_perimeter(g)[0]) for g in gdf.geometry.explode(index_parts=False)) / 1e6

    assert poly_km2 == pytest.approx(pixel_sum, rel=1e-3)
    assert poly_km2 == pytest.approx(geo_km2, rel=1e-3)
    assert poly_km2 == pytest.approx(snap.snapped_upa_km2, rel=0.02)  # MERIT's own UPA, same grid


def test_equal_area_crs_area_of_a_known_cell_block():
    import geopandas as gpd
    from shapely.geometry import box

    from aihydro_watershed.delineation.types import area_km2

    px, n = m.MERIT_NATIVE_PX_DEG, 1000
    g = gpd.GeoDataFrame(geometry=[box(-100.0, 45.0 - n * px, -100.0 + n * px, 45.0)], crs=4326)
    geo = abs(Geod(ellps="WGS84").geometry_area_perimeter(g.geometry.iloc[0])[0]) / 1e6
    assert area_km2(g) == pytest.approx(geo, rel=1e-4)


# --- snap does not depend on the window -----------------------------------------------------------


def test_snap_is_identical_at_two_window_sizes(rec, fake_ee):
    snaps = []
    for hs in (4.0, 12.0):
        b = m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, hs))
        s = m.snap_outlet_on_merit(*GAUGE, upa=b.bands["upa"], wth=b.bands["wth"], transform=b.transform)
        snaps.append((s.row + round((b.transform.f - m.MERIT_ORIGIN_Y) / -m.MERIT_NATIVE_PX_DEG),
                      s.col + round((b.transform.c - m.MERIT_ORIGIN_X) / m.MERIT_NATIVE_PX_DEG), s.snapped_upa_km2))
    assert snaps[0] == snaps[1]  # same global cell and UPA


def test_snap_ignores_a_large_river_outside_the_search_radius():
    """Synthetic flip case: normalising by the window maximum picked A when a big river was in
    the window, and B when it was not. B wins on the candidates inside the radius, either way."""
    n = 201
    px = m.MERIT_NATIVE_PX_DEG
    lat = 40.0
    tf = Affine(px, 0, -100.0, 0, -px, lat + (n // 2 + 0.5) * px)
    lon = tf.c + (n // 2 + 0.5) * px
    upa = np.zeros((n, n), dtype=np.float32)
    r0 = c0 = n // 2
    upa[r0, c0 + 7] = 2.0  # A: about 0.5 km away, tiny stream
    upa[r0, c0 - 13] = 100.0  # B: about 0.9 km away, real stream
    got = []
    for river in (False, True):
        u = upa.copy()
        if river:
            u[:, c0 + 90] = 1.0e6  # a continental river 6 km away, outside the 3 km radius
        s = m.snap_outlet_on_merit(lat, lon, upa=u, transform=tf)
        got.append((s.row, s.col))
    assert got[0] == got[1] == (r0, c0 - 13)


# --- grid edges ------------------------------------------------------------------------------------


def _check_window(bbox):
    tf, nc, nr = m.native_window(bbox)
    px = m.MERIT_NATIVE_PX_DEG
    xmin, ymax = tf.c, tf.f
    assert (xmin - m.MERIT_ORIGIN_X) / px == pytest.approx(round((xmin - m.MERIT_ORIGIN_X) / px), abs=1e-6)
    assert (m.MERIT_ORIGIN_Y - ymax) / px == pytest.approx(round((m.MERIT_ORIGIN_Y - ymax) / px), abs=1e-6)
    return xmin, ymax - nr * px, xmin + nc * px, ymax


def test_negative_longitude_window_covers_bbox_on_lattice():
    bbox = m.bbox_from_point(44.8774, -71.0575, 20.0)
    xmin, ymin, xmax, ymax = _check_window(bbox)
    assert xmin <= bbox[0] and xmax >= bbox[2] and ymin <= bbox[1] and ymax >= bbox[3]
    assert bbox[0] - xmin < m.MERIT_NATIVE_PX_DEG and xmax - bbox[2] < m.MERIT_NATIVE_PX_DEG


def test_high_latitude_window_is_clamped_at_the_top_of_the_asset():
    bbox = m.bbox_from_point(84.9, 10.0, 20.0)  # lon span is wide at 84.9N; top exceeds 85N
    assert bbox[3] > 85.0
    xmin, ymin, xmax, ymax = _check_window(bbox)
    assert ymax == pytest.approx(m.MERIT_ORIGIN_Y)  # clamped to row 0
    assert xmax - xmin > 3.0  # wide in degrees, not clamped
    assert m.native_window(bbox)[2] < m.MERIT_GRID_ROWS


def test_antimeridian_windows_clamp_at_both_ends_without_wrapping():
    east = _check_window(m.bbox_from_point(0.0, 179.99, 20.0))
    assert east[2] == pytest.approx(180.0 - 0.5 * m.MERIT_NATIVE_PX_DEG)
    west = _check_window(m.bbox_from_point(0.0, -179.99, 20.0))
    assert west[0] == pytest.approx(m.MERIT_ORIGIN_X)
    tf, nc, nr = m.native_window(m.bbox_from_point(0.0, 179.99, 20.0))
    assert tf.c + nc * tf.a <= 180.0


def test_bbox_outside_the_asset_raises():
    for bbox in ((10.0, 86.0, 11.0, 88.0), (10.0, -70.0, 11.0, -65.0)):
        with pytest.raises(ValueError, match="does not intersect"):
            m.native_window(bbox)


# --- returned grid must be the requested grid --------------------------------------------------------


def test_fetch_rejects_a_grid_that_is_not_the_requested_window(fake_ee):
    fake_ee.tamper = lambda tf, nr, nc: (tf, nr, nc - 1)
    with pytest.raises(RuntimeError, match="not the requested native window"):
        m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 3.0))
    fake_ee.tamper = lambda tf, nr, nc: ((tf[0], 0, tf[2] + 0.5 * m.MERIT_NATIVE_PX_DEG, 0, tf[4], tf[5]), nr, nc)
    with pytest.raises(RuntimeError, match="not the requested native window"):
        m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 3.0))


def test_cache_hit_is_validated_and_a_bad_cached_grid_is_refetched(fake_ee):
    import rasterio

    bbox = m.bbox_from_point(*GAUGE, 3.0)
    first = m.fetch_merit_gee_bbox(bbox)
    assert m.fetch_merit_gee_bbox(bbox).source == "gee_cache" and len(fake_ee.requests) == 1
    # Corrupt one cached band: same name, different shape.
    path = next(m._cache_dir().glob("*_upa.tif"))
    with rasterio.open(path) as src:
        prof = src.profile
    prof.update(width=prof["width"] - 1)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.zeros((prof["height"], prof["width"]), dtype="float32"), 1)
    again = m.fetch_merit_gee_bbox(bbox)
    assert again.source == "gee" and len(fake_ee.requests) == 2
    np.testing.assert_array_equal(again.bands["upa"], first.bands["upa"])


# --- cache key and deprecated scale_m ----------------------------------------------------------------


def test_cache_key_is_native_grid_versioned_and_has_no_scale():
    import inspect

    assert "scale_m" not in inspect.signature(m._cache_key).parameters
    k = m._cache_key(bbox=(0, 0, 1, 1), bands=("dir",), dataset=m.MERIT_GEE_DATASET)
    legacy = __import__("hashlib").sha256(
        __import__("json").dumps({"bbox": [0, 0, 1, 1], "bands": ["dir"], "scale_m": 92.77,
                                  "dataset": m.MERIT_GEE_DATASET}, sort_keys=True).encode()
    ).hexdigest()
    assert k != legacy


def test_scale_m_is_deprecated_and_ignored(fake_ee):
    with pytest.warns(DeprecationWarning, match="scale_m"):
        m.fetch_merit_gee_bbox(m.bbox_from_point(*GAUGE, 2.0), scale_m=92.77)
    assert "scale" not in fake_ee.requests[0]


# --- thread-safe, resettable init -----------------------------------------------------------------------


def test_concurrent_first_calls_initialise_once_and_reset_forces_reinit(fake_ee):
    import threading

    ts = [threading.Thread(target=m._ee_initialize, args=(fake_ee, None)) for _ in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert fake_ee.init_calls == 1
    m.reset_ee()
    m._ee_initialize(fake_ee, None)
    assert fake_ee.init_calls == 2


# --- AIHYDRO_HOME ---------------------------------------------------------------------------------


def test_cache_root_honours_aihydro_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AIHYDRO_HOME", str(tmp_path / "aihome"))
    assert m._cache_dir() == tmp_path / "aihome" / "cache" / "delineation" / "merit_gee"
    assert m._cache_dir().is_dir()


def test_cache_root_defaults_to_dot_aihydro(tmp_path, monkeypatch):
    monkeypatch.delenv("AIHYDRO_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert m._cache_dir() == tmp_path / "h" / ".aihydro" / "cache" / "delineation" / "merit_gee"


def test_import_does_not_create_cache_dirs(tmp_path):
    import os
    import subprocess

    env = dict(os.environ, HOME=str(tmp_path / "h2"))
    env.pop("AIHYDRO_HOME", None)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(m.__file__).parents[2]), *[p for p in sys.path if p]])
    subprocess.run([sys.executable, "-c", "import aihydro_watershed.delineation.merit_flowdir_pipeline"],
                   env=env, check=True, capture_output=True)
    assert not (tmp_path / "h2").exists()


def test_merit_manager_honours_aihydro_home(tmp_path, monkeypatch):
    from aihydro_watershed.merit import merit_manager as mm

    home = tmp_path / "ah"
    home.mkdir()
    (home / "config.json").write_text('{"merit_auto_stage": "never", "merit_auto_stage_max_gb": 7}')
    monkeypatch.setenv("AIHYDRO_HOME", str(home))
    monkeypatch.delenv("AIHYDRO_MERIT_DIR", raising=False)
    monkeypatch.delenv("MERIT_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    assert mm._merit_auto_stage_policy() == ("never", 7.0)
    assert mm.MeritDataManager().root == home / "merit"
    assert not (tmp_path / "elsewhere").exists()
