"""
Offline regression tests for F11: the headline CN statistics (``cn_mean`` and the
rest of ``statistics``) must be taken over the basin polygon, and the basis must
be stated once (``statistics['stats_basis']``) with every other ``*stats_basis``
field agreeing with it.

Before the fix ``statistics`` covered every valid cell of the fetched raster
whenever the grid was not already NaN-masked (single-pass path, or a masked-path
fallback) yet the labels disagreed with how the numbers were computed. Synthetic
EPSG:5070 rasters only; nothing touches the network.
"""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

from shapely.geometry import box  # noqa: E402

from aihydro_watershed.terrain import curve_number as cn  # noqa: E402

X0, Y0, N, RES = 5e5, 1e6, 20, 30.0


def _raster(values):
    arr = np.asarray(values, dtype=float)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": Y0 - (np.arange(arr.shape[0]) + 0.5) * RES,
                "x": X0 + (np.arange(arr.shape[1]) + 0.5) * RES},
    )
    return da.rio.write_crs("EPSG:5070")


def _poly(col0, col1):
    """Lon/lat polygon over columns [col0, col1) of the grid, all rows."""
    return gpd.GeoSeries([box(X0 + col0 * RES, Y0 - N * RES, X0 + col1 * RES, Y0)],
                         crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]


def _lulc_ds():
    """Pasture (81) inside columns 5-14, row crops (82) outside."""
    v = np.full((N, N), 82.0)
    v[:, 5:15] = 81.0
    ds = _raster(v).rename("cover_2021").to_dataset()
    ds.attrs.update(_adata_product="NLCD", _adata_source="test", _adata_region="CONUS")
    return ds


def _soil_ds():
    """Sandy (group A) inside columns 5-14, clayey (group D) outside."""
    def tex(inside, outside):
        a = np.full((N, N), outside)
        a[:, 5:15] = inside
        return _raster(a)
    ds = xr.Dataset({"sand_5": tex(80.0, 10.0), "silt_5": tex(15.0, 30.0),
                     "clay_5": tex(5.0, 60.0)})
    ds.attrs.update(_adata_product="POLARIS", _adata_source="test")
    return ds


def _run(monkeypatch, geometry, *, soil=None):
    monkeypatch.setattr(cn, "fetch_lulc_data", lambda *a, **k: _lulc_ds())
    monkeypatch.setattr(cn, "fetch_soil_data", lambda *a, **k: soil or _soil_ds())
    return cn.create_curve_number_grid_from_geometry(
        geometry, save_outputs=False, create_visualizations=False)


def _table():
    t = cn._create_cn_lookup_table()
    return t[(81, 1)], t[(82, 4)]


def test_headline_statistics_follow_the_polygon_not_the_raster(monkeypatch):
    inside_cn, outside_cn = _table()
    assert inside_cn != outside_cn
    out = _run(monkeypatch, _poly(5, 15))     # small raster -> single-pass, unmasked grid
    st = out["statistics"]
    # The unmasked grid holds both values; the old code averaged them.
    assert np.nanmean(out["cn_grid"].values) == pytest.approx((inside_cn + outside_cn) / 2)
    assert st["cn_mean"] == pytest.approx(inside_cn)
    assert st["cn_median"] == inside_cn
    assert st["cn_min"] == st["cn_max"] == inside_cn
    assert st["cn_std"] == pytest.approx(0.0)
    assert out["cn_zones"]["percent_low_cn"] + out["cn_zones"]["percent_medium_cn"] \
        + out["cn_zones"]["percent_high_cn"] == pytest.approx(100.0)
    pct = {k: out["cn_zones"][k] for k in ("percent_low_cn", "percent_medium_cn", "percent_high_cn")}
    in_zone = ("percent_low_cn" if inside_cn < 70
               else "percent_medium_cn" if inside_cn <= 85 else "percent_high_cn")
    assert pct[in_zone] == pytest.approx(100.0)


def test_basis_is_stated_once_and_every_alias_agrees(monkeypatch):
    out = _run(monkeypatch, _poly(5, 15))
    basis = out["statistics"]["stats_basis"]
    assert basis == "basin_polygon"
    assert out["basin_statistics"]["stats_basis"] == basis
    assert out["data_provenance"]["cn_stats_basis"] == basis
    assert out["data_provenance"]["lulc_fraction_basis"] == basis
    # The *_basin aliases carry the same numbers as the unsuffixed keys.
    bs, st = out["basin_statistics"], out["statistics"]
    assert bs["cn_mean_basin"] == pytest.approx(st["cn_mean"])
    assert bs["cn_median_basin"] == st["cn_median"]
    assert bs["cn_min_basin"] == st["cn_min"] and bs["cn_max_basin"] == st["cn_max"]
    assert bs["cn_cells_basin"] == N * 10
    assert out["quality_flags"] == []


def test_chunked_and_single_pass_paths_give_the_same_statistics(monkeypatch):
    single = _run(monkeypatch, _poly(5, 15))["statistics"]
    monkeypatch.setattr(cn, "_CN_CHUNK_TRIGGER", 0)
    chunked = _run(monkeypatch, _poly(5, 15))["statistics"]
    assert chunked == single


def test_unavailable_mask_is_labelled_raster_extent_and_flagged(monkeypatch):
    inside_cn, outside_cn = _table()
    far = gpd.GeoSeries([box(0, 0, 0.001, 0.001)], crs="EPSG:4326").iloc[0]   # no overlap
    out = _run(monkeypatch, far)
    assert out["statistics"]["stats_basis"] == "raster_extent"
    assert out["statistics"]["cn_mean"] == pytest.approx((inside_cn + outside_cn) / 2)
    assert out["data_provenance"]["cn_stats_basis"] == "raster_extent"
    assert "basin_cn_stats_empty" in out["quality_flags"]


def test_dual_group_alternative_mean_is_over_the_polygon_too():
    v = np.full((N, N), 82.0)
    v[:, 5:15] = 81.0
    lulc = _raster(v)
    groups, alt = np.full((N, N), 4, dtype=np.int32), np.full((N, N), 2, dtype=np.int32)
    groups[:, 5:15], alt[:, 5:15] = 1, 2
    t = cn._create_cn_lookup_table()
    _, _, stats = cn._finish_cn_grid(lulc, groups, {"dual_hsg_condition": "drained"},
                                     2021, 30, _poly(5, 15), source="t",
                                     alt_groups=alt, geom_crs="EPSG:4326")
    assert stats["cn_mean_undrained"] == pytest.approx(t[(81, 2)])
    assert stats["cn_mean_basis"] == "basin_polygon"
