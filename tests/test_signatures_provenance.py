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


# ---------------------------------------------------------------------------
# Version provenance: the running module, never stale dist-info (Wave U E7)
# ---------------------------------------------------------------------------

def test_meta_version_is_module_version_not_stale_dist_info(monkeypatch):
    import importlib.metadata as md
    import aihydro_watershed
    from aihydro_watershed import _version

    real = md.version
    monkeypatch.setattr(md, "version", lambda name: "0.0.1" if name.startswith("aihydro") else real(name))
    assert sig._get_version() == aihydro_watershed.__version__ != "0.0.1"
    p = _version.version_params()
    chk = p["version_check"]
    assert chk["version_mismatch"] is True
    assert chk["module_version"] == aihydro_watershed.__version__
    assert chk["distribution_version"] == "0.0.1"
    assert chk["version_source"] == "module"


def test_host_module_is_never_consulted(monkeypatch):
    """Watershed states its own identity even when an ai_hydro module is loaded."""
    import importlib.metadata as md
    import sys
    import types
    import aihydro_watershed
    from aihydro_watershed import _version

    host = types.ModuleType("ai_hydro")
    host.__version__ = "9.8.7"
    monkeypatch.setitem(sys.modules, "ai_hydro", host)
    monkeypatch.setattr(md, "version", lambda name: "2.3.7" if name == "aihydro-tools" else aihydro_watershed.__version__)
    assert sig._get_version() == aihydro_watershed.__version__
    assert _version.version_params() == {}          # aihydro-tools dist-info is not this package's business


def test_no_version_params_when_metadata_agrees(monkeypatch):
    import importlib.metadata as md
    import aihydro_watershed
    from aihydro_watershed import _version

    monkeypatch.setattr(md, "version", lambda name: aihydro_watershed.__version__)
    assert _version.version_params() == {}
    monkeypatch.setattr(md, "version", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)))
    assert _version.version_params() == {} and _version.tool_version() == aihydro_watershed.__version__


def test_no_package_reads_distribution_metadata_for_result_version():
    import pathlib
    root = pathlib.Path(sig.__file__).resolve().parents[1]
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                 if p.name != "_version.py" and "importlib.metadata" in p.read_text()]
    assert offenders == []


# ---------------------------------------------------------------------------
# Units and streamflow digest (session study S1 / M5)
# ---------------------------------------------------------------------------

def test_every_signature_carries_units(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: {"q_cms": _q_series(800)})
    result = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0,
    )
    units = result.data["units"]
    expected = set(sig._get_default_hydrology())
    assert expected <= set(units), f"signatures without units: {expected - set(units)}"
    # Flow statistics are depth rates, never discharge.
    for k in ("q_mean", "q_std", "q5", "q95", "q_median"):
        assert units[k] == "mm/day"
    assert units["runoff_ratio"] == "dimensionless"
    assert units["high_q_freq"] == "days/year" and units["high_q_dur"] == "days"
    assert units["zero_q_freq"] == "dimensionless"          # a fraction, not days/year
    assert units["hfd_mean"] == "day of water year"


def test_units_map_agrees_with_the_conversion():
    # q_mean is the mean of the mm/day series: m3/s * 86.4 / km2.
    q = _q_series(800)
    result = sig.extract_hydrological_signatures(
        gauge_id=None, watershed_geojson=SQUARE, area_km2=250.0, q_cms_series=q,
    )
    assert result.data["units"]["q_mean"] == "mm/day"
    assert result.data["q_mean"] == pytest.approx(float(q.mean()) * 86.4 / 250.0)


def test_units_present_when_streamflow_unavailable(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: None)
    result = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0,
    )
    assert result.data["units"]["q_mean"] == "mm/day"


def test_streamflow_series_is_digested(monkeypatch):
    q = _q_series(800)
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: {"q_cms": q})
    a = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0).data
    assert a["streamflow_digest"].startswith("sha256:")
    assert a["streamflow_n_days"] == 800
    assert a["streamflow_start"] == "2000-01-01"
    assert a["_streamflow"]["units"] == "m3/s"
    # Deterministic, and sensitive to a change in the data.
    b = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0).data
    assert a["streamflow_digest"] == b["streamflow_digest"]
    q2 = q.copy(); q2.iloc[10] += 1.0
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a_, **k: {"q_cms": q2})
    c = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0).data
    assert c["streamflow_digest"] != a["streamflow_digest"]


def test_no_digest_without_streamflow(monkeypatch):
    monkeypatch.setattr(sig, "_fetch_streamflow_internal", lambda *a, **k: None)
    d = sig.extract_hydrological_signatures(
        gauge_id="01013500", watershed_geojson=SQUARE, area_km2=250.0).data
    assert "streamflow_digest" not in d
