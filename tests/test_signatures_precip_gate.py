"""D4 regression: precipitation-dependent signatures are never computed from an
absent, placeholder or non-physical precipitation series, and the precipitation
product/digest (or its absence) is recorded in the result.

Offline: the precipitation fetch is replaced with a fake.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures import signatures as sig

SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2]]],
}
START, END = "1999-10-01", "2001-09-30"
IDX = pd.date_range(START, END, freq="D")


def _q_cms():
    rng = np.random.default_rng(1)
    return pd.Series(2.0 + np.sin(np.linspace(0, 12 * np.pi, len(IDX))) + rng.random(len(IDX)),
                     index=IDX)


def _precip(values, product="FAKE_PRECIP"):
    s = pd.Series(values, index=IDX, dtype=float, name="precip_mm")
    s.attrs["product"] = product
    return s


def _install(monkeypatch, series, reason=None, product=None):
    def fake(geom, start, end):
        diag = {}
        if reason:
            diag["reason"] = reason
        if product:
            diag["product"] = product
        sig._PRECIP_DIAG.d = diag
        return series
    monkeypatch.setattr(sig, "_fetch_precipitation_data_bygeom", fake)


def _run():
    return sig.extract_hydrological_signatures(
        None, SQUARE, 100.0, START, END, q_cms_series=_q_cms())


def test_no_source_returns_none_with_recorded_reason(monkeypatch):
    # Basin 1 of e2e proof 2: every backend refused.
    _install(monkeypatch, None, reason="no precipitation source returned data: Refused")
    d = _run().data
    assert d["runoff_ratio"] is None and d["stream_elas"] is None
    rec = d["_precipitation"]
    assert rec["status"] == "unavailable" and rec["digest"] is None and rec["product"] is None
    assert "Refused" in rec["reason"]
    assert rec["dependent_signatures"] == ["runoff_ratio", "stream_elas"]
    json.dumps(d)  # JSON-serialisable


@pytest.mark.parametrize("label,values", [
    ("fill_1e33", np.full(len(IDX), 4.3e32)),               # basin 2: ~1e33 netCDF fill
    ("fill_float32_max", np.full(len(IDX), 9.96921e36)),
    ("all_zero", np.zeros(len(IDX))),
    ("negative", np.full(len(IDX), -1.0)),
    ("denormal", np.full(len(IDX), 1e-310)),
])
def test_degenerate_series_is_rejected_not_computed(monkeypatch, label, values):
    _install(monkeypatch, _precip(values), product="CHIRPS_IRI")
    d = _run().data
    assert d["runoff_ratio"] is None, label
    assert d["stream_elas"] is None, label
    rec = d["_precipitation"]
    assert rec["status"] == "rejected" and rec["reason"]
    assert rec["product"] == "CHIRPS_IRI"
    assert rec["digest"].startswith("sha256:") and rec["n_days"] == len(IDX)


def test_nonfinite_values_rejected(monkeypatch):
    v = np.full(len(IDX), 2.0)
    v[10] = np.inf
    _install(monkeypatch, _precip(v))
    d = _run().data
    assert d["runoff_ratio"] is None and d["_precipitation"]["status"] == "rejected"


def test_valid_series_computes_and_records_product_and_digest(monkeypatch):
    rng = np.random.default_rng(2)
    p = _precip(rng.gamma(0.5, 6.0, len(IDX)), product="GRIDMET_PRECIP")
    _install(monkeypatch, p, product="GRIDMET_PRECIP")
    d = _run().data
    assert d["runoff_ratio"] is not None and 0 < d["runoff_ratio"] < 5
    rec = d["_precipitation"]
    assert rec["status"] == "used" and rec["reason"] is None
    assert rec["product"] == "GRIDMET_PRECIP"
    first = rec["digest"]
    # digest is a function of the data
    _install(monkeypatch, _precip(p.values * 1.01), product="GRIDMET_PRECIP")
    assert _run().data["_precipitation"]["digest"] != first
    # and deterministic
    _install(monkeypatch, p, product="GRIDMET_PRECIP")
    assert _run().data["_precipitation"]["digest"] == first


def test_compute_water_balance_gate_directly():
    q = _q_cms()
    out = sig.compute_water_balance_camels(q, _precip(np.full(len(IDX), 1e33)))
    assert np.isnan(out["runoff_ratio"]) and np.isnan(out["stream_elas"])
    out = sig.compute_water_balance_camels(q, None)
    assert np.isnan(out["runoff_ratio"])
    out = sig.compute_water_balance_camels(q, _precip(np.full(len(IDX), 2.0)))
    assert np.isfinite(out["runoff_ratio"])


def test_only_water_balance_signatures_depend_on_precipitation(monkeypatch):
    # Swapping a valid series for None changes exactly runoff_ratio/stream_elas.
    rng = np.random.default_rng(3)
    _install(monkeypatch, _precip(rng.gamma(0.5, 6.0, len(IDX))))
    with_p = _run().data
    _install(monkeypatch, None)
    without_p = _run().data
    changed = {k for k in with_p
               if not k.startswith("_") and with_p[k] != without_p[k]}
    # (stream_elas needs >= 3 hydro years; this 2-year fixture only moves runoff_ratio)
    assert "runoff_ratio" in changed and changed <= {"runoff_ratio", "stream_elas"}


def _shape():
    from shapely.geometry import shape
    return shape(SQUARE)


def test_real_fetch_path_with_fill_valued_frame_end_to_end(monkeypatch):
    """The un-patched fetch + gate, with aihydro_data.fetch faked to serve what
    an un-masked netCDF fill gives (CHIRPS_IRI clip(lower=0) keeps 1e33)."""
    from types import SimpleNamespace
    import aihydro_data

    frame = pd.DataFrame({"date": IDX, "precipitation": np.full(len(IDX), 4.3e32)})
    monkeypatch.setattr(aihydro_data, "fetch",
                        lambda **kw: SimpleNamespace(product="CHIRPS_IRI", data=frame))
    d = _run().data
    assert d["runoff_ratio"] is None and d["stream_elas"] is None
    assert d["_precipitation"]["status"] == "rejected"
    assert d["_precipitation"]["product"] == "CHIRPS_IRI"
    assert d["_precipitation"]["max_mm_day"] == pytest.approx(4.3e32)


def test_real_fetch_path_when_every_backend_fails(monkeypatch):
    import aihydro_data

    def boom(**kw):
        raise RuntimeError("all products failed")
    monkeypatch.setattr(aihydro_data, "fetch", boom)
    d = _run().data
    assert d["runoff_ratio"] is None
    rec = d["_precipitation"]
    assert rec["status"] == "unavailable" and "all products failed" in rec["reason"]
