from __future__ import annotations

import math

import numpy as np

from aihydro_watershed.signatures.baseflow import compute_bfi, lyne_hollick
from aihydro_watershed.signatures.flow_duration import flow_duration_curve
from aihydro_watershed.signatures.signatures import extract_hydrological_signatures


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
