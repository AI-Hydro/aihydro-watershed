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
    assert out["fraction_basis"] == "basin_polygon"


def test_open_water_fraction_zero_when_absent_and_none_when_empty():
    out = cn._lulc_fraction_stats(np.array([81.0]), np.array([5]))
    assert out["open_water_fraction"] == 0.0
    assert abs(sum(out["class_fractions"].values()) - 1.0) < 1e-9
    empty = cn._lulc_fraction_stats(np.array([]), np.array([]))
    assert empty == {"class_fractions": {}, "open_water_fraction": None,
                     "fraction_basis": "basin_polygon"}


def _wgs84_polygon_over_columns(x0, x1):
    """A polygon given in WGS84 that, in EPSG:5070, spans x0..x1 and both rows of the test grid."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoSeries([box(x0, 1e6 - 15, x1, 1e6 + 45)], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]


def test_fractions_exclude_open_water_outside_the_basin_polygon():
    # Column 0 holds the only open-water cell; the polygon covers columns 1-3 only.
    lulc = _grid(LULC)
    poly = _wgs84_polygon_over_columns(5e5 + 10, 5e5 + 100)
    classes, counts = cn._basin_lulc_counts(lulc, poly, "EPSG:4326")
    out = cn._lulc_fraction_stats(classes, counts)
    assert out["open_water_fraction"] == 0.0
    assert "11" not in out["class_fractions"]
    assert out["class_fractions"]["81"] == pytest.approx(3 / 5)
    assert out["class_fractions"]["82"] == pytest.approx(2 / 5)    # NaN cell not counted
    assert abs(sum(out["class_fractions"].values()) - 1.0) < 1e-9
    assert out["fraction_basis"] == "basin_polygon"
    # The raster-wide counts the old statistics use are unchanged and do include it.
    all_classes, all_counts = np.unique(lulc.values[np.isfinite(lulc.values)], return_counts=True)
    assert 11.0 in all_classes


def test_fractions_include_open_water_inside_the_basin_polygon():
    lulc = _grid(LULC)
    poly = _wgs84_polygon_over_columns(5e5 - 10, 5e5 + 100)         # columns 0-3
    classes, counts = cn._basin_lulc_counts(lulc, poly, "EPSG:4326")
    out = cn._lulc_fraction_stats(classes, counts)
    assert out["open_water_fraction"] == pytest.approx(1 / 7)


def test_hsg_missing_fraction_and_ksat_all_nan():
    soil = xr.Dataset()
    soil.attrs.update(_adata_product="SSURGO")
    prov = cn._soil_provenance(soil, {"hsg_method": "ssurgo_recorded", "hsg_coverage": 11 / 12})
    assert prov["hsg_missing_fraction"] == pytest.approx(1 / 12)

    ds = _polaris_ds()
    ds["ksat_5"] = _grid([[np.nan] * 4] * 2)
    _, _, stats = cn._create_cn_grid_from_data(_lulc_ds(), ds, 2021, 30)
    assert stats["hsg_ksat_used"] is False
    ds["ksat_5"] = _grid([[20.0] * 4] * 2)
    _, _, stats = cn._create_cn_grid_from_data(_lulc_ds(), ds, 2021, 30)
    assert stats["hsg_ksat_used"] is True
