from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import xarray as xr
from shapely.geometry import box


def _fake_dem() -> xr.DataArray:
    dem = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        dims=("latitude", "longitude"),
        coords={"latitude": [1.0, 0.0], "longitude": [0.0, 1.0]},
    )
    return dem


def test_fetch_dem_auto_routes_through_aihydro_data(monkeypatch):
    from aihydro_watershed.characterize._dem import fetch_dem

    fetch = Mock(return_value=SimpleNamespace(data=_fake_dem(), product="GLO30", source="gee"))
    fake_module = ModuleType("aihydro_data")
    setattr(fake_module, "fetch", fetch)
    monkeypatch.setitem(sys.modules, "aihydro_data", fake_module)

    geom = box(-90, 40, -89.9, 40.1)
    dem, product, source = fetch_dem(geom, prefer="auto")

    fetch.assert_called_once()
    args, kwargs = fetch.call_args
    assert args[0] == "dem"
    assert args[1].equals(geom)
    assert args[2:4] == ("", "")
    assert kwargs == {}
    assert dem.dims == ("y", "x")
    assert product == "GLO30"
    assert source == "gee"


def test_fetch_dem_py3dep_preference_pins_data_product(monkeypatch):
    from aihydro_watershed.characterize._dem import fetch_dem

    fetch = Mock(return_value=SimpleNamespace(data=_fake_dem(), product="DEM3DEP_10M", source="hyriver"))
    fake_module = ModuleType("aihydro_data")
    setattr(fake_module, "fetch", fetch)
    monkeypatch.setitem(sys.modules, "aihydro_data", fake_module)

    _, product, source = fetch_dem(box(-90, 40, -89.9, 40.1), prefer="py3dep")

    _, kwargs = fetch.call_args
    assert kwargs == {"mode": "manual", "product": "DEM3DEP_10M"}
    assert product == "DEM3DEP_10M"
    assert source == "hyriver"
