"""Signature provenance records what was served, never what was requested.

Offline: aihydro-data and the NWIS helper are monkeypatched.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures import signatures as sig

SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2]]],
}


def _q_series(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(0)
    values = 5.0 + 3.0 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.random(n)
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=n, freq="D"), name="q_cms")


def _frame(n: int = 400) -> pd.DataFrame:
    q = _q_series(n)
    return pd.DataFrame({"date": q.index, "streamflow": q.values})


@pytest.fixture
def fake_fetch(monkeypatch):
    """Install a fake aihydro_data.fetch; returns the list of recorded calls."""
    import aihydro_data

    calls: list[dict] = []

    def install(behaviour):
        def fetch(**kwargs):
            calls.append(kwargs)
            return behaviour(kwargs)
        monkeypatch.setattr(aihydro_data, "fetch", fetch)
        return calls

    return install


@pytest.fixture(autouse=True)
def no_precip(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_precipitation_data_bygeom", lambda *a, **k: None)


def test_global_streamflow_pins_each_candidate_strictly(fake_fetch):
    def behaviour(kwargs):
        if kwargs["product"] == "GEOGLOWS_RETRO":
            raise RuntimeError("probe outage")
        return SimpleNamespace(product=kwargs["product"], data=_frame())

    calls = fake_fetch(behaviour)
    out = sig._fetch_global_streamflow(_shape(), "2000-01-01", "2001-02-03")

    assert [c["product"] for c in calls] == ["GEOGLOWS_RETRO", "OPENMETEO_FLOOD"]
    assert all(c["fallback"] == [] for c in calls), "every candidate must be a strict pin"
    assert out["_product"] == "OPENMETEO_FLOOD"


def test_global_streamflow_records_served_not_requested(fake_fetch):
    # Simulates a data layer that serves a different product than requested.
    fake_fetch(lambda kwargs: SimpleNamespace(product="GLOFAS_STREAMFLOW", data=_frame()))

    out = sig._fetch_global_streamflow(_shape(), "2000-01-01", "2001-02-03")

    assert out["_product"] == "GLOFAS_STREAMFLOW"


def _shape():
    from shapely.geometry import shape
    return shape(SQUARE)


def test_caller_supplied_series_is_labelled_unknown_origin():
    result = sig.extract_hydrological_signatures(
        gauge_id=None, watershed_geojson=SQUARE, area_km2=250.0,
        q_cms_series=_q_series().tolist(),
    )
    assert result.data["_streamflow_source"] == {"product": None, "observation": "caller_supplied"}


def test_gauge_path_is_labelled_observed(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: {"q_cms": _q_series()})

    result = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0,
    )

    assert result.data["_streamflow_source"] == {"product": "NWIS_STREAMFLOW", "observation": "observed"}
    assert np.isfinite(result.data["q_mean"])


def test_global_path_is_labelled_modelled_with_served_product_and_citation(monkeypatch):
    monkeypatch.setattr(
        sig, "_fetch_global_streamflow",
        lambda *a, **k: {"q_cms": _q_series(), "_product": "OPENMETEO_FLOOD"},
    )

    result = sig.extract_hydrological_signatures(
        gauge_id=None, watershed_geojson=SQUARE, area_km2=250.0,
    )

    assert result.data["_streamflow_source"] == {"product": "OPENMETEO_FLOOD", "observation": "modelled"}
    cited = {s.name for s in result.meta.sources}
    assert "Open-Meteo GloFAS v4" in cited
    assert "GEOGLOWS v2 (ECMWF)" not in cited
    assert "USGS NWIS" not in cited


def test_insufficient_streamflow_is_labelled_none(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: None)

    result = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0,
    )

    assert result.data["_streamflow_source"] == {"product": None, "observation": "none"}
    assert result.data["q_mean"] is None or not np.isfinite(result.data["q_mean"])
