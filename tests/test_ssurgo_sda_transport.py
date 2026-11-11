"""SDA transport: requests only (no curl subprocess); failures reach the recorded soil fallback."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
import requests  # noqa: E402
from shapely.geometry import box  # noqa: E402

from aihydro_watershed.terrain import _soil, ssurgo as ss  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)


def _forbid_subprocess(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("subprocess must not be used for SDA")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)


def test_requests_failure_never_spawns_curl_and_raises(monkeypatch):
    _forbid_subprocess(monkeypatch)
    calls = []

    def _post(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("dns")

    monkeypatch.setattr(requests, "post", _post)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        ss.sda_query("SELECT 1", retries=3)
    assert len(calls) == 3  # bounded retries


def test_module_has_no_subprocess_or_curl():
    assert not hasattr(ss, "subprocess") and not hasattr(ss, "shutil")


def test_fallback_to_polaris_is_recorded_when_sda_fails(monkeypatch):
    _forbid_subprocess(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("dns")))

    mukey = xr.DataArray(
        np.array([[1001.0, 1001.0]]), dims=("y", "x"),
        coords={"y": [1.0], "x": [1.0, 2.0]}, name="mukey",
    )

    real = ss.fetch_soil_data_ssurgo

    def _ssurgo_with_real_sda(geometry):
        return real(geometry, mukey_fn=lambda g: mukey, query_fn=ss.sda_query)

    polaris = xr.Dataset(attrs={"_adata_product": "POLARIS"})
    monkeypatch.setattr(ss, "fetch_soil_data_ssurgo", _ssurgo_with_real_sda)
    monkeypatch.setattr(_soil, "fetch_soil_data_polaris", lambda g, **k: polaris)
    monkeypatch.setattr(_soil, "_in_conus", lambda g: True)
    monkeypatch.setattr(_soil, "_DEPS_AVAILABLE", True)

    ds = _soil.fetch_soil_data(box(-100, 35, -99.9, 35.1))
    assert ds.attrs["_adata_product"] == "POLARIS"
    assert "Soil Data Access query failed" in ds.attrs["_soil_fallback_reason"]


def test_success_records_requests_transport(monkeypatch):
    _forbid_subprocess(monkeypatch)

    class _Resp:
        text = '{"Table": [["mukey","cokey","comppct_r","hydgrp","kwfact","sandtotal_r","silttotal_r","claytotal_r"],' \
               '["1001","1","100","B",".32","20","60","20"]]}'

        def raise_for_status(self):
            pass

        def json(self):
            import json
            return json.loads(self.text)

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    mukey = xr.DataArray(
        np.array([[1001.0, 1001.0]]), dims=("y", "x"),
        coords={"y": [1.0], "x": [1.0, 2.0]}, name="mukey",
    ).rio.write_crs("EPSG:5070")
    ds = ss.fetch_soil_data_ssurgo(box(0, 0, 1, 1), mukey_fn=lambda g: mukey)
    assert ds.attrs["_adata_sda_transport"] == "requests"
    assert float(ds["kw"].values[0, 0]) == pytest.approx(0.32)


def test_injected_query_fn_is_labelled_injected():
    mukey = xr.DataArray(
        np.array([[1001.0]]), dims=("y", "x"), coords={"y": [1.0], "x": [1.0]}, name="mukey",
    ).rio.write_crs("EPSG:5070")
    rows = [{"mukey": "1001", "comppct_r": "100", "hydgrp": "B", "kwfact": ".3"}]
    ds = ss.fetch_soil_data_ssurgo(box(0, 0, 1, 1), mukey_fn=lambda g: mukey, query_fn=lambda sql: rows)
    assert ds.attrs["_adata_sda_transport"] == "injected"
