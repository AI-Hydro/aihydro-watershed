"""_normalize_dem refuses extra slices / missing CRS; elevation is warped bilinear. Offline."""
from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

import aihydro_watershed.characterize._dem as D
import aihydro_watershed.characterize.geomorphic as G


def _plane(shape=(20, 24), crs="EPSG:4326"):
    lon = 10 + np.arange(shape[1]) / 3600
    lat = (50 + np.arange(shape[0]) / 3600)[::-1]
    z = 500.0 + 3.0 * np.arange(shape[1])[None, :] + 2.0 * np.arange(shape[0])[:, None]
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": lat, "x": lon})
    return da if crs is None else da.rio.write_crs(crs)


def _stack(n):
    base = _plane()
    da = xr.concat([base + 100.0 * i for i in range(n)], dim="band").assign_coords(band=np.arange(1, n + 1))
    return da.rio.write_crs("EPSG:4326")


def test_singleton_band_is_squeezed():
    out = D._normalize_dem(_stack(1))
    assert out.dims == ("y", "x")
    np.testing.assert_array_equal(out.values, _plane().values)


def test_multi_slice_band_is_refused_not_sliced():
    with pytest.raises(D.DemInputError, match=r"more than one slice.*'band': 3"):
        D._normalize_dem(_stack(3))


def test_multi_slice_time_with_singleton_band_is_refused():
    da = xr.concat([_stack(1)] * 2, dim="time")
    with pytest.raises(D.DemInputError, match="time"):
        D._normalize_dem(da)


def test_missing_crs_is_refused_not_assumed():
    with pytest.raises(D.DemInputError, match="no CRS"):
        D._normalize_dem(_plane(crs=None))


def test_existing_crs_is_kept_unchanged():
    out = D._normalize_dem(_plane(crs="EPSG:32632"))
    assert out.rio.crs.to_epsg() == 32632


def test_lat_lon_names_are_still_renamed():
    da = _plane().rename({"y": "latitude", "x": "longitude"})
    out = D._normalize_dem(da)
    assert out.dims == ("y", "x")


def test_slope_from_dem_surfaces_it_as_slope_unavailable():
    with pytest.raises(D.SlopeUnavailableError, match="more than one slice"):
        D.slope_from_dem(_stack(2))
    with pytest.raises(D.SlopeUnavailableError, match="no CRS"):
        D.slope_from_dem(_plane(crs=None))


def test_fetch_dem_reports_why_instead_of_using_a_slice(monkeypatch):
    import aihydro_data
    monkeypatch.setattr(aihydro_data, "fetch",
                        lambda *a, **k: type("R", (), {"data": _stack(3), "product": "GLO30", "source": "gee"})())
    with pytest.raises(RuntimeError, match="more than one slice"):
        D.fetch_dem(box(10, 50, 10.005, 50.005))


def test_reproject_elevation_is_bilinear_not_nearest():
    lon = 10 + np.arange(200) / 3600
    lat = (50 + np.arange(200) / 3600)[::-1]
    LON, LAT = np.meshgrid(lon, lat)
    f = lambda x, y: 500.0 + 40.0 * (x - 10) * 3600 / 20 + 30.0 * (y - 50) * 3600 / 20  # noqa: E731
    src = xr.DataArray(f(LON, LAT), dims=("y", "x"), coords={"y": lat, "x": lon}).rio.write_crs("EPSG:4326")

    from rasterio.enums import Resampling
    bil = D.reproject_elevation(src, "EPSG:32632")
    nn = src.rio.reproject("EPSG:32632", resampling=Resampling.nearest)

    def rmse(da):
        from pyproj import Transformer
        X, Y = np.meshgrid(da.x.values, da.y.values)
        lo, la = Transformer.from_crs(32632, 4326, always_xy=True).transform(X, Y)
        ok = np.isfinite(da.values) & (da.values != da.rio.nodata)
        inner = ok & (lo > lon[3]) & (lo < lon[-4]) & (la > lat[-4]) & (la < lat[3])
        return float(np.sqrt(np.mean((da.values[inner] - f(lo, la)[inner]) ** 2)))

    assert rmse(bil) < 0.5 * rmse(nn)


def test_relief_and_twi_use_the_bilinear_helper(monkeypatch):
    calls = []
    real = D.reproject_elevation
    monkeypatch.setattr(D, "reproject_elevation", lambda dem, crs: (calls.append(crs), real(dem, crs))[1])
    monkeypatch.setattr(D, "fetch_dem", lambda g, resolution=30, prefer="auto": (_plane((300, 300)), "GLO30", "gee"))
    out = G._compute_relief_metrics(box(10.0, 50.0, 10.08, 50.08), 50.0, 10.0)
    assert out.get("flag") is None
    assert calls and calls[0].startswith("EPSG:326")
