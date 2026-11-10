"""
Offline tests for the soil-source / hydrologic-group provenance and the
land-cover class fractions that ``create_cn_grid`` reports.

The gNATSGO route is made unavailable with a fake; nothing touches the network.
The record is additive: CN values come from the same lookup as before.
"""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")

from aihydro_watershed.terrain import _soil  # noqa: E402
from aihydro_watershed.terrain import curve_number as cn  # noqa: E402


def _grid(values):
    arr = np.asarray(values, dtype=float)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": np.arange(arr.shape[0])[::-1] * 30.0 + 1e6,
                "x": np.arange(arr.shape[1]) * 30.0 + 5e5},
    )
    return da.rio.write_crs("EPSG:5070")


# 2 x 4 land cover: open water x1, pasture x3, row crops x3, one NaN (nodata).
LULC = [[11, 81, 81, 81],
        [82, 82, 82, np.nan]]


def _lulc_ds():
    return _grid(LULC).rename("cover_2021").to_dataset()


def _polaris_ds():
    """POLARIS-style texture: sandy (group A) left half, clayey (group D) right half."""
    def tex(left, right):
        return _grid([[left, left, right, right]] * 2)
    ds = xr.Dataset({
        "sand_5": tex(80.0, 10.0), "silt_5": tex(15.0, 30.0), "clay_5": tex(5.0, 60.0),
    })
    ds.attrs.update(_adata_product="POLARIS", _adata_source="hydrology.cee.duke.edu")
    return ds


def _unavailable(*_a, **_k):
    raise OSError("blob host blocked: soils.blob.core.windows.net")


def test_gnatsgo_unavailable_fallback_is_recorded(monkeypatch):
    from aihydro_watershed.terrain import ssurgo

    monkeypatch.setattr(ssurgo, "fetch_soil_data_ssurgo", _unavailable)
    monkeypatch.setattr(_soil, "fetch_soil_data_polaris", lambda g, **k: _polaris_ds())
    monkeypatch.setattr(_soil, "_in_conus", lambda g: True)
    monkeypatch.setattr(_soil, "_to_single_geom", lambda g: g)

    soil = _soil.fetch_soil_data(object())          # auto route -> SSURGO fails -> POLARIS
    cn_grid, _, stats = cn._create_cn_grid_from_data(_lulc_ds(), soil, 2021, 30)
    prov = cn._soil_provenance(soil, stats)

    assert prov["soil_product_used"] == "POLARIS"
    assert prov["soil_source"] == "hydrology.cee.duke.edu"
    assert prov["fallback_used"] is True
    assert "blob host blocked" in prov["fallback_reason"]
    assert prov["hsg_method"] == "texture_thresholds"
    assert prov["hsg_inferred"] is True
    thr = prov["hsg_thresholds"]
    assert thr["A"] == cn.HSG_TEXTURE_THRESHOLDS["A"]
    assert thr["ksat_used"] is False                # the fake has no ksat variable
    # CN values are the lookup's, unchanged by the record.
    table = cn._create_cn_lookup_table()
    assert cn_grid.values[0, 1] == table[(81, 1)]   # pasture on sandy (A)
    assert cn_grid.values[1, 2] == table[(82, 4)]   # row crops on clayey (D)
    assert np.isnan(cn_grid.values[1, 3])


def test_recorded_lookup_is_not_marked_inferred():
    soil = xr.Dataset()
    soil.attrs.update(_adata_product="SSURGO", _adata_source="gNATSGO")
    prov = cn._soil_provenance(soil, {"hsg_method": "ssurgo_recorded"})
    assert prov["soil_product_used"] == "SSURGO"
    assert prov["hsg_inferred"] is False
    assert prov["hsg_thresholds"] is None
    assert prov["fallback_used"] is False and prov["fallback_reason"] is None


def test_recorded_thresholds_match_the_classifier():
    """The thresholds reported are the ones the classifier applies (edge cases)."""
    def group(sand, silt, clay):
        g, _ = cn._classify_soil_hydrologic_group(
            np.array([[sand]]), np.array([[silt]]), np.array([[clay]]))
        return int(g[0, 0])

    assert group(71, 20, 9) == 1          # A: sand > 70, clay < 10
    assert group(70, 20, 9) == 2          # sand not > 70; 50 <= sand <= 70, clay < 20 -> B
    assert group(40, 51, 26) == 2         # silt > 50, clay < 27 -> B
    assert group(30, 40, 30) == 3         # 20 <= clay < 40 -> C
    assert group(10, 30, 60) == 4         # default D
    assert cn.HSG_TEXTURE_THRESHOLDS["default_group"] == "D"


def test_class_fractions_sum_to_one_and_report_open_water():
    cover = np.array(LULC)
    classes, counts = np.unique(cover[~np.isnan(cover)], return_counts=True)
    out = cn._lulc_fraction_stats(classes, counts)
    fr = out["class_fractions"]
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert fr["11"] == pytest.approx(1 / 7)
    assert fr["81"] == pytest.approx(3 / 7) and fr["82"] == pytest.approx(3 / 7)
    assert out["open_water_fraction"] == pytest.approx(1 / 7)


def test_open_water_fraction_zero_when_absent_and_none_when_empty():
    out = cn._lulc_fraction_stats(np.array([81.0]), np.array([5]))
    assert out["open_water_fraction"] == 0.0
    assert abs(sum(out["class_fractions"].values()) - 1.0) < 1e-9
    empty = cn._lulc_fraction_stats(np.array([]), np.array([]))
    assert empty == {"class_fractions": {}, "open_water_fraction": None}
