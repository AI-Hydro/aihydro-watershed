"""NumPy-2.x compatibility for third-party dependencies.

``pysheds`` (latest release 0.5) still calls ``numpy.in1d`` inside
``Grid.accumulation`` (``pysheds/sgrid.py``, ``_d8_accumulation``). ``np.in1d``
was deprecated in NumPy 2.0 in favour of ``np.isin`` and has been removed in the
newest NumPy, so delineation raised ``AttributeError`` there.

``ensure_in1d()`` restores the old name only when it is missing, so it is a
no-op on every NumPy that still ships ``in1d``. It must run before pysheds is
first used; call it next to ``from pysheds.grid import Grid``.
"""

from __future__ import annotations

import numpy as np


def _in1d(ar1, ar2, assume_unique: bool = False, invert: bool = False):
    """``np.in1d`` semantics: always returns a flat 1-D boolean array.

    ``np.isin`` preserves the shape of ``ar1``, so the result is raveled to
    match the removed function.
    """
    return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert).ravel()


def ensure_in1d() -> None:
    """Provide ``np.in1d`` when this NumPy no longer has it (idempotent)."""
    if not hasattr(np, "in1d"):
        np.in1d = _in1d  # type: ignore[attr-defined]
