"""NumPy compat shim for pysheds' use of the removed ``np.in1d``."""

import numpy as np
import pytest

import aihydro_watershed
from aihydro_watershed import _numpy_compat

_MISSING = object()


@pytest.fixture
def restore_in1d():
    """Save and restore ``np.in1d`` so tests are independent of order/NumPy version."""
    saved = np.__dict__.get("in1d", _MISSING)
    yield
    if saved is _MISSING:
        np.__dict__.pop("in1d", None)
    else:
        np.in1d = saved


def test_public_alias_exported():
    assert aihydro_watershed.ensure_numpy_compat is _numpy_compat.ensure_in1d
    assert "ensure_numpy_compat" in aihydro_watershed.__all__
    assert hasattr(np, "in1d")  # applied at package import time


def test_in1d_shim_matches_legacy_semantics():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    out = _numpy_compat._in1d(a, [2, 5, 9])
    assert out.shape == (6,)  # flat, unlike np.isin
    assert out.tolist() == [False, True, False, False, True, False]
    assert _numpy_compat._in1d(a, [2, 5], invert=True).sum() == 4


def test_ensure_restores_missing_attribute(restore_in1d):
    np.__dict__.pop("in1d", None)  # simulate the newest NumPy
    _numpy_compat.ensure_in1d()
    assert np.in1d([1, 2], [2]).tolist() == [False, True]


def test_ensure_leaves_native_in1d_alone(restore_in1d):
    sentinel = object()
    np.in1d = sentinel
    _numpy_compat.ensure_in1d()
    assert np.in1d is sentinel


def test_pysheds_accumulation_without_native_in1d(restore_in1d, tmp_path):
    """Real pysheds Grid.accumulation (an in1d call site) after in1d is removed."""
    pytest.importorskip("pysheds")
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    from pysheds.grid import Grid

    n = 8
    dem = np.repeat(np.arange(n, 0, -1, dtype="float32")[:, None], n, axis=1)  # drains south
    path = tmp_path / "plane.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs="EPSG:32616", transform=Affine(30, 0, 500000, 0, -30, 4500000), nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)

    grid = Grid.from_raster(str(path))
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(grid.read_raster(str(path)), dirmap=dirmap)

    np.__dict__.pop("in1d", None)  # pysheds would raise AttributeError here without the shim
    _numpy_compat.ensure_in1d()
    acc = np.asarray(grid.accumulation(fdir, dirmap=dirmap))

    assert acc.shape == (n, n)
    assert (acc[-1, 1:-1] == n).all()  # every interior column collects its n cells
