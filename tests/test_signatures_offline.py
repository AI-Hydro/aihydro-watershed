from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed.signatures.baseflow import compute_bfi, lyne_hollick
from aihydro_watershed.signatures.flow_duration import flow_duration_curve
from aihydro_watershed.signatures.signatures import (
    compute_event_stats_camels,
    extract_hydrological_signatures,
)


def test_missing_day_splits_high_flow_events():
    days = pd.date_range("2010-01-01", periods=30, freq="D")
    q = pd.Series(1.0, index=days)
    q.loc["2010-01-21"] = 100.0
    q.loc["2010-01-22"] = np.nan
    q.loc["2010-01-23"] = 100.0
    assert compute_event_stats_camels(q)["high_q_dur"] == 1.0


def test_supplied_series_preserves_dates(monkeypatch):
    from aihydro_watershed.signatures import signatures as sig

    observed = {}
    original = sig._to_mm_per_day
    def capture(q, area):
        observed["index"] = q.index.copy()
        return original(q, area)
    monkeypatch.setattr(sig, "_to_mm_per_day", capture)
    days = pd.date_range("2010-01-01", periods=400, freq="2D")
    q = pd.Series(2.0, index=days)
    square = {"type": "Polygon", "coordinates": [[
        [-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3],
        [-77.5, 39.3], [-77.5, 39.2],
    ]]}
    sig.extract_hydrological_signatures(
        gauge_id=None, watershed_geojson=square, area_km2=250.0,
        q_cms_series=q,
    )
    assert observed["index"].equals(days)


def test_flow_duration_curve_uses_hydrologic_exceedance_convention():
    q = np.arange(1, 101, dtype=float)

    fdc = flow_duration_curve(q)
    flows = fdc["percentile_flows"]

    assert flows["Q5"] > flows["Q50"] > flows["Q95"]
    assert fdc["n_days"] == 100


def test_flow_duration_curve_requires_two_valid_values():
    try:
        flow_duration_curve([math.nan, 1.0])
    except ValueError as exc:
        assert "at least 2 valid discharge values" in str(exc)
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("too-short FDC input should raise ValueError")


def test_lyne_hollick_baseflow_and_bfi_are_bounded():
    q = np.array([2.0, 3.0, 10.0, 8.0, 4.0, 3.0, 2.0])

    baseflow = lyne_hollick(q)
    bfi = compute_bfi(q, baseflow)

    assert np.all(baseflow >= 0)
    assert np.all(baseflow <= q)
    assert 0.0 <= bfi <= 1.0


def test_extract_signatures_rejects_invalid_area_before_fetching():
    try:
        extract_hydrological_signatures(
            gauge_id=None,
            watershed_geojson={"type": "Polygon", "coordinates": []},
            area_km2=0.0,
            q_cms_series=[1.0, 2.0, 3.0],
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "INVALID_AREA"
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("invalid area should raise INVALID_AREA")


def test_extract_signatures_records_baseflow_method_in_params():
    """HydroMeta.params must record which baseflow-separation method and
    parameters produced baseflow_index — the method shifts BFI by ~0.1-0.2
    for the same catchment, so it's part of the result's identity."""
    from aihydro_watershed.signatures.signatures import (
        BASEFLOW_SEPARATION_METHOD,
        BASEFLOW_SEPARATION_PARAMS,
    )

    rng = np.random.default_rng(42)
    q_series = (5.0 + 3.0 * np.sin(np.linspace(0, 8 * np.pi, 400)) + rng.random(400)).tolist()
    square = {
        "type": "Polygon",
        "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2]]],
    }

    result = extract_hydrological_signatures(
        gauge_id=None,
        watershed_geojson=square,
        area_km2=250.0,
        q_cms_series=q_series,
    )

    assert result.meta.params.get("baseflow_method") == BASEFLOW_SEPARATION_METHOD
    assert result.meta.params.get("baseflow_params") == BASEFLOW_SEPARATION_PARAMS
    assert "baseflow_reference" in result.meta.params
    assert np.isfinite(result.data["baseflow_index"])


def test_baseflow_index_has_block_bootstrap_ci_centred_on_reported_value():
    import numpy as np
    import pandas as pd

    from aihydro_watershed.signatures.signatures import (
        BFI_BOOTSTRAP_BLOCK_DAYS,
        _baseflow_index_uncertainty,
        compute_flow_stats_camels,
    )

    rng = np.random.default_rng(7)
    days = 3 * 365
    t = np.arange(days)
    seasonal = 2.0 + 1.5 * np.sin(2 * np.pi * t / 365.0)
    storms = rng.gamma(0.3, 4.0, size=days)
    q = seasonal + np.convolve(storms, np.exp(-np.arange(10) / 2.0), mode="full")[:days]
    q_mm_day = pd.Series(q, index=pd.date_range("2000-10-01", periods=days, freq="D"))

    point = compute_flow_stats_camels(q_mm_day)["baseflow_index"]
    u = _baseflow_index_uncertainty(q_mm_day.dropna().values, n=200)["baseflow_index"]

    assert u["value"] == pytest.approx(point, abs=1e-12)
    assert u["ci_low"] <= point <= u["ci_high"]
    assert 0.0 < u["ci_low"] < u["ci_high"] < 1.0
    assert u["method"] == "bootstrap_block" and u["ci_level"] == 0.90
    assert u["block_size"] == BFI_BOOTSTRAP_BLOCK_DAYS
    assert "excludes filter" in u["scope"]


def test_baseflow_index_ci_requires_two_blocks_of_data():
    import numpy as np

    from aihydro_watershed.signatures.signatures import _baseflow_index_uncertainty

    assert _baseflow_index_uncertainty(np.ones(400)) == {}
