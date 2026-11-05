"""slope_from_dem: metric spacing, or fail closed (audit C8 latent sibling).

Before: when the DEM was still in EPSG:4326 (the callers keep the native CRS if reprojection fails, with
a warning), ``rio.resolution()`` returned degrees, the gradient was metres per degree, and the slope went to
~90 degrees; if the resolution could not be read at all, spacing silently became 1.0. Offline; no network.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from aihydro_watershed.characterize._dem import SlopeUnavailableError, slope_from_dem


def _plane(t: float, cell: float, crs: str = "EPSG:5070", n: int = 80, az_deg: float = 25.0):
    x = 500_000.0 + cell * np.arange(n)
    y = 4_000_000.0 - cell * np.arange(n)
    X, Y = np.meshgrid(x, y)
    th = math.radians(az_deg)
    z = 50.0 + t * ((X - x[0]) * math.cos(th) + (Y - y[-1]) * math.sin(th))
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(crs)


def _m_per_km(slope_deg: xr.DataArray) -> float:
    v = np.tan(np.radians(slope_deg.values))[2:-2, 2:-2]
    return float(np.mean(v) * 1000.0)


@pytest.mark.parametrize("cell", [10.0, 30.0, 73.0, 90.0])
@pytest.mark.parametrize("t", [0.003, 0.0723, 0.25])
def test_plane_gives_1000t_regardless_of_pixel_size(t, cell):
    out = slope_from_dem(_plane(t, cell))
    assert _m_per_km(out) == pytest.approx(1000 * t, rel=0.01)
    assert out.attrs["units"] == "degrees"           # units metadata


def test_resampling_invariance_10_30_73():
    vals = [_m_per_km(slope_from_dem(_plane(0.0723, c, n=int(2400 / c)))) for c in (10.0, 30.0, 73.0)]
    assert max(vals) - min(vals) <= 0.01 * 72.3


def test_utm_grid_independent_of_latitude():
    # same plane on UTM grids at very different latitudes (northing differs): spacing is metres in both
    for crs, north in (("EPSG:32631", 10_000.0), ("EPSG:32631", 5_200_000.0)):
        da = _plane(0.0723, 30.0, crs=crs)
        da = da.assign_coords(y=da.y.values - 4_000_000.0 + north)
        assert _m_per_km(slope_from_dem(da)) == pytest.approx(72.3, rel=0.01)


def test_feet_crs_spacing_is_converted():
    t = 0.1
    cell_ft = 100.0
    n = 60
    x = 1_000_000.0 + cell_ft * np.arange(n)
    y = 500_000.0 - cell_ft * np.arange(n)
    X, _Y = np.meshgrid(x, y)
    z = t * (X - x[0]) * 0.3048006096012192
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:2272")
    assert _m_per_km(slope_from_dem(da)) == pytest.approx(100.0, rel=0.01)


def test_geographic_dem_fails_closed():
    """The pre-fix behaviour: ~90 degrees from a degrees-based spacing."""
    lon = -100.0 + np.arange(60) / 3600.0
    lat = (35.0 + np.arange(60) / 3600.0)[::-1]
    LON, LAT = np.meshgrid(lon, lat)
    z = 500.0 + 0.0723 * (LON - lon[0]) * 91_000.0   # 7 % slope in metres
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": lat, "x": lon}).rio.write_crs("EPSG:4326")
    with pytest.raises(SlopeUnavailableError, match="not projected"):
        slope_from_dem(da)


def test_unknown_crs_fails_closed():
    """No CRS: _normalize_dem assumes EPSG:4326, which is geographic, so it refuses."""
    da = xr.DataArray(np.random.rand(30, 30) * 100, dims=("y", "x"),
                      coords={"y": np.arange(30.0)[::-1], "x": np.arange(30.0)})
    with pytest.raises(SlopeUnavailableError):
        slope_from_dem(da)


def test_unreadable_resolution_fails_closed_not_unit_spacing(monkeypatch):
    da = _plane(0.1, 30.0)

    def _boom(self):
        raise RuntimeError("no transform")

    monkeypatch.setattr(type(da.rio), "resolution", _boom)
    with pytest.raises(SlopeUnavailableError, match="could not be determined"):
        slope_from_dem(da)


def test_error_is_a_value_error_so_existing_handlers_see_it():
    assert issubclass(SlopeUnavailableError, ValueError)
