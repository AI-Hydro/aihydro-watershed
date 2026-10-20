"""NumPy compat shim for pysheds' use of the removed ``np.in1d``."""

import numpy as np
import pytest

from aihydro_watershed import _numpy_compat


def test_in1d_shim_matches_legacy_semantics():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    out = _numpy_compat._in1d(a, [2, 5, 9])
    assert out.shape == (6,)  # flat, unlike np.isin
    assert out.tolist() == [False, True, False, False, True, False]
    assert _numpy_compat._in1d(a, [2, 5], invert=True).sum() == 4


def test_ensure_in1d_restores_missing_attribute(monkeypatch):
    if hasattr(np, "in1d"):
        monkeypatch.delattr(np, "in1d")  # simulate the newest NumPy
    _numpy_compat.ensure_in1d()
    assert np.in1d([1, 2], [2]).tolist() == [False, True]


def test_ensure_in1d_leaves_native_in1d_alone(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(np, "in1d", sentinel, raising=False)
    _numpy_compat.ensure_in1d()
    assert np.in1d is sentinel


def test_pysheds_accumulation_works_without_native_in1d(monkeypatch):
    pytest.importorskip("pysheds")
    if hasattr(np, "in1d"):
        monkeypatch.delattr(np, "in1d")
    import importlib

    import aihydro_watershed.delineation.pysheds_pipeline as pp

    importlib.reload(pp)  # import-time call applies the shim
    assert hasattr(np, "in1d")
