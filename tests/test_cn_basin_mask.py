"""
Offline tests for the basin-masked CN statistics, the chunked path's CRS
handling, and the loud (flagged) fallback when the masked path fails.

Synthetic EPSG:5070 rasters and lon/lat polygons only; nothing touches the network.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

from shapely.geometry import box  # noqa: E402

from aihydro_watershed.terrain import curve_number as cn  # noqa: E402

X0, Y0, N, RES = 5e5, 1e6, 20, 30.0     # 20 x 20 cells of 30 m, EPSG:5070


def _raster(values, crs="EPSG:5070"):
    arr = np.asarray(values, dtype=float)
    da = xr.DataArray(
        arr, dims=("y", "x"),
        coords={"y": Y0 - (np.arange(arr.shape[0]) + 0.5) * RES,
                "x": X0 + (np.arange(arr.shape[1]) + 0.5) * RES},
    )
    return da.rio.write_crs(crs)


def _lonlat_poly(col0, col1):
    """Lon/lat polygon covering columns [col0, col1) of the grid, all rows."""
    x0, x1 = X0 + col0 * RES, X0 + col1 * RES
    return gpd.GeoSeries([box(x0, Y0 - N * RES, x1, Y0)], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]


def _inputs():
    """Pasture (81) on group A inside columns 5-14, row crops (82) on D outside."""
    lulc = np.full((N, N), 82.0)
    lulc[:, 5:15] = 81.0
    groups = np.full((N, N), 4, dtype=np.int32)
    groups[:, 5:15] = 1
    return _raster(lulc), groups


def _chunked(monkeypatch):
    monkeypatch.setattr(cn, "_CN_CHUNK_TRIGGER", 0)     # force the chunked overlay


def test_projected_raster_with_lonlat_polygon_masks_correctly_not_all_nan(monkeypatch):
    _chunked(monkeypatch)
    lulc, groups = _inputs()
    poly = _lonlat_poly(5, 15)
    cn_grid, _, stats = cn._finish_cn_grid(
        lulc, groups, {}, 2021, 30, poly, source="t", geom_crs="EPSG:4326")

    table = cn._create_cn_lookup_table()
    inside = cn_grid.values[:, 5:15]
    assert np.isfinite(inside).all() and (inside == table[(81, 1)]).all()
    assert np.isnan(cn_grid.values[:, :5]).all() and np.isnan(cn_grid.values[:, 15:]).all()
    assert stats["quality_flags"] == []                  # no silent fallback was needed


def test_geom_crs_defaults_to_wgs84_like_the_lulc_counts(monkeypatch):
    _chunked(monkeypatch)
    lulc, groups = _inputs()
    cn_grid, _, _ = cn._finish_cn_grid(lulc, groups, {}, 2021, 30, _lonlat_poly(5, 15), source="t")
    assert int(np.isfinite(cn_grid.values).sum()) == N * 10


def test_polygon_already_in_raster_crs_is_not_reprojected(monkeypatch):
    _chunked(monkeypatch)
    lulc, groups = _inputs()
    poly = box(X0 + 5 * RES, Y0 - N * RES, X0 + 15 * RES, Y0)
    cn_grid, _, _ = cn._finish_cn_grid(
        lulc, groups, {}, 2021, 30, poly, source="t", geom_crs="EPSG:5070")
    assert int(np.isfinite(cn_grid.values).sum()) == N * 10


def test_unreprojected_chunked_call_is_all_nan_which_is_the_hazard():
    """What the unfixed code did: the sampler given a lon/lat polygon over EPSG:5070."""
    from aihydro_data.sampling import chunked_raster_apply
    lulc, _ = _inputs()
    out = chunked_raster_apply(lulc, _lonlat_poly(5, 15), lambda a, m: a,
                               auto_trigger_size=0, fill_value=np.nan)
    assert np.isnan(out.values).all()


def test_basin_mean_excludes_cells_outside_the_polygon():
    lulc, groups = _inputs()
    cn_grid, _, _ = cn._finish_cn_grid(lulc, groups, {}, 2021, 30, None, source="t")
    table = cn._create_cn_lookup_table()
    inside_cn, outside_cn = table[(81, 1)], table[(82, 4)]

    # Whole-grid (raster-extent) mean is dragged toward the outside cells ...
    assert np.nanmean(cn_grid.values) == pytest.approx((inside_cn + outside_cn) / 2)
    # ... the basin mean is not.
    out = cn._basin_cn_statistics(cn_grid, _lonlat_poly(5, 15), "EPSG:4326")
    assert out["stats_basis"] == "basin_polygon"
    assert out["cn_mean_basin"] == pytest.approx(inside_cn)
    assert out["cn_median_basin"] == inside_cn
    assert out["cn_min_basin"] == out["cn_max_basin"] == inside_cn
    assert out["cn_cells_basin"] == N * 10


def test_basin_stats_ignore_nan_cells_and_report_empty_polygon():
    lulc, groups = _inputs()
    cn_grid, _, _ = cn._finish_cn_grid(lulc, groups, {}, 2021, 30, None, source="t")
    cn_grid.values[0, 5:15] = np.nan
    out = cn._basin_cn_statistics(cn_grid, _lonlat_poly(5, 15), "EPSG:4326")
    assert out["cn_cells_basin"] == (N - 1) * 10

    far = gpd.GeoSeries([box(0, 0, 0.001, 0.001)], crs="EPSG:4326").iloc[0]
    empty = cn._basin_cn_statistics(cn_grid, far, "EPSG:4326")
    assert empty["cn_cells_basin"] == 0 and empty["cn_mean_basin"] is None
    _, flags, msgs = cn._basin_statistics_and_flags(cn_grid, far, "EPSG:4326", {})
    assert flags == ["basin_cn_stats_empty"] and msgs


def test_failed_masked_path_falls_back_loudly(monkeypatch):
    _chunked(monkeypatch)
    import aihydro_data.sampling as sampling

    def boom(*_a, **_k):
        raise ValueError("sampler exploded")

    monkeypatch.setattr(sampling, "chunked_raster_apply", boom)
    lulc, groups = _inputs()
    with pytest.warns(RuntimeWarning, match="fell back to the unmasked"):
        cn_grid, _, stats = cn._finish_cn_grid(
            lulc, groups, {}, 2021, 30, _lonlat_poly(5, 15), source="t", geom_crs="EPSG:4326")

    assert stats["quality_flags"] == ["cn_overlay_masked_path_failed"]
    assert "sampler exploded" in stats["warnings"][0]
    # The fallback really is unmasked ...
    assert int(np.isfinite(cn_grid.values).sum()) == N * N
    # ... and the basin-masked statistics are still correct and flow to the result flags.
    bstats, flags, msgs = cn._basin_statistics_and_flags(
        cn_grid, _lonlat_poly(5, 15), "EPSG:4326", stats)
    assert bstats["cn_cells_basin"] == N * 10
    assert flags == ["cn_overlay_masked_path_failed"] and len(msgs) == 1


def test_masked_path_yielding_no_valid_cell_is_flagged_not_silent(monkeypatch):
    _chunked(monkeypatch)
    lulc, groups = _inputs()
    far = gpd.GeoSeries([box(0, 0, 0.001, 0.001)], crs="EPSG:4326").iloc[0]   # no overlap
    with pytest.warns(RuntimeWarning):
        cn_grid, _, stats = cn._finish_cn_grid(
            lulc, groups, {}, 2021, 30, far, source="t", geom_crs="EPSG:4326")
    assert stats["quality_flags"] == ["cn_overlay_masked_path_empty"]
    assert int(np.isfinite(cn_grid.values).sum()) == N * N


def test_no_flags_and_no_warning_on_the_single_pass_path():
    lulc, groups = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, stats = cn._finish_cn_grid(lulc, groups, {}, 2021, 30, _lonlat_poly(5, 15),
                                         source="t", geom_crs="EPSG:4326")
    assert stats["quality_flags"] == [] and stats["warnings"] == []
