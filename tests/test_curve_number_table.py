"""
Known-answer tests for the NRCS curve-number lookup table (TR-55, AMC II).

Reference: USDA-NRCS (1986) Urban Hydrology for Small Watersheds, TR-55,
Table 2-2b (cultivated agricultural lands) and 2-2c (other agricultural
lands). Values are for groups A, B, C, D.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

cn = pytest.importorskip("aihydro_watershed.terrain.curve_number")

# TR-55 Table 2-2c, "Pasture, grassland, or range", good condition.
TR55_PASTURE_GOOD = (39, 61, 74, 80)
# TR-55 Table 2-2b, row crops, straight row, good condition.
TR55_ROW_CROPS_SR_GOOD = (67, 78, 85, 89)


def _row(table, nlcd):
    return tuple(table[(nlcd, g)] for g in (1, 2, 3, 4))


def test_deps_import_cleanly():
    # A failed import inside curve_number silently disables the module
    # (_DEPS_AVAILABLE=False), so guard it explicitly.
    assert cn._DEPS_AVAILABLE


def test_pasture_hay_uses_tr55_pasture_not_row_crops():
    table = cn._create_cn_lookup_table()
    assert _row(table, 81) == TR55_PASTURE_GOOD
    assert _row(table, 81) != _row(table, 82)


def test_cultivated_crops_unchanged():
    table = cn._create_cn_lookup_table()
    assert _row(table, 82) == TR55_ROW_CROPS_SR_GOOD


def test_every_row_increases_from_a_to_d():
    table = cn._create_cn_lookup_table()
    classes = sorted({k[0] for k in table})
    for c in classes:
        r = _row(table, c)
        assert list(r) == sorted(r), f"NLCD {c}: CN must not decrease from A to D: {r}"


def test_vectorised_lookup_on_pasture_grid():
    """A 2x2 grid of pasture over groups A-D returns the TR-55 pasture row."""
    lookup = cn._build_joint_cn_lookup(cn._create_cn_lookup_table())
    lulc = np.full((2, 2), 81.0)
    groups = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = cn._vectorised_cn_lookup(lulc, groups, lookup)
    np.testing.assert_array_equal(out, np.array([[39, 61], [74, 80]], dtype=np.float32))


def test_missing_group_or_landcover_gives_nan():
    lookup = cn._build_joint_cn_lookup(cn._create_cn_lookup_table())
    lulc = np.array([[81.0, np.nan]])
    groups = np.array([[0, 2]], dtype=np.int32)
    out = cn._vectorised_cn_lookup(lulc, groups, lookup)
    assert np.isnan(out).all()


def test_default_nlcd_year_is_2021():
    from aihydro_watershed.terrain import _landcover

    assert _landcover.NLCD_LATEST_YEAR == 2021
    for fn in (
        _landcover.fetch_lulc_data,
        _landcover._fetch_nlcd_direct,
        cn.create_curve_number_grid,
        cn.create_curve_number_grid_from_geometry,
    ):
        assert inspect.signature(fn).parameters["year"].default == 2021, fn.__name__
