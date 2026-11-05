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


def test_failed_delineation_error_carries_the_tiers_it_tried(monkeypatch):
    """A total failure keeps the attempt trail: ``ToolError.details['fallback_history']``."""
    import pytest
    from aihydro_core import ToolError

    from aihydro_watershed.delineation import small_catchment
    from aihydro_watershed.delineation.router import delineate_from_point

    def _boom(lat, lon, *a, **k):
        raise RuntimeError("3DEP service unavailable")

    monkeypatch.setattr(small_catchment, "delineate_small_catchment", _boom)
    with pytest.raises(ToolError) as caught:
        delineate_from_point(40.0, -100.0, method="small_catchment")
    exc = caught.value
    assert exc.code == "DELINEATION_FAILED"
    history = exc.details["fallback_history"]
    assert [(h["method"], h["outcome"]) for h in history] == [("small_catchment_3dep", "failed")]
    assert "3DEP service unavailable" in history[0]["reason"]
    assert exc.to_dict()["fallback_history"] == history      # reaches the envelope
