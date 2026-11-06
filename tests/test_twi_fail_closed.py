"""compute_twi must not run on a native (degree) grid when reprojection to a metric CRS fails."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

pytest.importorskip("pysheds")
pytest.importorskip("py3dep")
rioxarray = pytest.importorskip("rioxarray")

from aihydro_watershed.characterize import _dem, twi  # noqa: E402


def _geographic_dem():
    lon = -100.0 + np.arange(40) / 3600.0
    lat = (35.0 + np.arange(40) / 3600.0)[::-1]
    z = 500.0 + np.random.default_rng(0).random((40, 40))
    return xr.DataArray(z, dims=("y", "x"), coords={"y": lat, "x": lon}).rio.write_crs("EPSG:4326")


@pytest.mark.parametrize("geom", [box(-100.0, 35.0, -99.99, 35.01),      # CONUS: EPSG:5070 branch
                                  box(10.0, 50.0, 10.01, 50.01)])         # global: UTM branch
def test_failed_reprojection_raises_instead_of_computing_on_degrees(monkeypatch, geom):
    monkeypatch.setattr(_dem, "fetch_dem", lambda g, resolution=30, prefer="auto": (_geographic_dem(), "GLO30", "gee"))

    def _boom(self, *a, **k):
        raise RuntimeError("proj database unavailable")

    monkeypatch.setattr(type(_geographic_dem().rio), "reproject", _boom)
    with pytest.raises(ValueError, match="Failed to compute TWI.*native grid"):
        twi.compute_twi(geom, resolution=30)


def test_error_chain_is_slope_unavailable(monkeypatch):
    monkeypatch.setattr(_dem, "fetch_dem", lambda g, resolution=30, prefer="auto": (_geographic_dem(), "GLO30", "gee"))
    monkeypatch.setattr(type(_geographic_dem().rio), "reproject",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(ValueError) as ei:
        twi.compute_twi(box(-100.0, 35.0, -99.99, 35.01), resolution=30)
    assert isinstance(ei.value.__cause__, _dem.SlopeUnavailableError)
