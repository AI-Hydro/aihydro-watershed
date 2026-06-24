from __future__ import annotations

from types import SimpleNamespace

from aihydro_watershed.delineation.router import _should_escalate


def test_fast_result_escalates_tiny_basin():
    result = SimpleNamespace(
        area_km2=0.5,
        scout_box_maxed=False,
        merit_snap_distance_m=100.0,
        used_nldi_basin=False,
    )

    reason = _should_escalate(result, expected_area_km2=None)

    assert reason is not None
    assert "below minimum" in reason


def test_fast_result_accepts_nldi_basin_without_snap_distance():
    result = SimpleNamespace(
        area_km2=100.0,
        scout_box_maxed=False,
        merit_snap_distance_m=None,
        used_nldi_basin=True,
    )

    assert _should_escalate(result, expected_area_km2=None) is None


def test_fast_result_escalates_large_expected_area_mismatch():
    result = SimpleNamespace(
        area_km2=100.0,
        scout_box_maxed=False,
        merit_snap_distance_m=100.0,
        used_nldi_basin=False,
    )

    reason = _should_escalate(result, expected_area_km2=1_000.0)

    assert reason is not None
    assert "differs from expected" in reason
