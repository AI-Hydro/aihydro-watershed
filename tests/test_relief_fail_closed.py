"""_compute_relief_metrics must not compute on a native (degree) grid or default the cell size to 30 m.

Reviewer repro: UTM reprojection failure kept the native CRS and _cell_m fell back to 30.0, giving
mean_slope_pct = 564648.6 for a 7.2 % slope. Now NaN plus a flag. Offline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

import aihydro_watershed.characterize._dem as D
import aihydro_watershed.characterize.geomorphic as G


def _dem():
    lon = 10 + np.arange(300) / 3600
    lat = (50 + np.arange(300) / 3600)[::-1]
    Z = 500 + 0.0723 * (np.arange(300)[None, :] * 21.7) * np.ones((300, 1))   # 7.2 % slope eastwards, metres
    return xr.DataArray(Z, dims=("y", "x"), coords={"y": lat, "x": lon}).rio.write_crs("EPSG:4326")


GEOM = box(10.0, 50.0, 10.08, 50.08)


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(D, "fetch_dem", lambda g, resolution=30, prefer="auto": (_dem(), "GLO30", "gee"))


def test_failed_utm_reprojection_gives_nan_and_a_flag_not_a_huge_slope(served, monkeypatch):
    monkeypatch.setattr(type(_dem().rio), "reproject",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("proj db")))
    out = G._compute_relief_metrics(GEOM, 50.0, 10.0)
    assert math.isnan(out["mean_slope_pct"]) and math.isnan(out["Rh"]) and math.isnan(out["Rp"])
    assert "relief_unavailable:SlopeUnavailableError" in out["flag"]
    assert "native grid" in out["flag"]


def test_metric_path_still_gives_the_right_slope(served):
    out = G._compute_relief_metrics(GEOM, 50.0, 10.0)
    assert out.get("flag") is None
    assert out["mean_slope_pct"] == pytest.approx(7.88, rel=0.03)  # 1 arc-second of lon is 19.9 m at 50 N, not 21.7


def test_unreadable_resolution_no_longer_defaults_to_30_m(served, monkeypatch):
    def _res(self, *a, **k):
        raise RuntimeError("no transform")

    monkeypatch.setattr(type(_dem().rio), "resolution", _res)
    out = G._compute_relief_metrics(GEOM, 50.0, 10.0)
    assert math.isnan(out["mean_slope_pct"]) and out["flag"].startswith("relief_unavailable:")


@pytest.mark.parametrize("lon,epsg", [(179.9, "EPSG:32660"), (180.0, "EPSG:32601"), (-179.9, "EPSG:32601"), (10.0, "EPSG:32632")])
def test_utm_zone_number_wraps_at_60(lon, epsg):
    from shapely.geometry import Point
    assert G._metric_crs_for_geom(Point(lon, 10.0)) == epsg
