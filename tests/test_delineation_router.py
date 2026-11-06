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


def _nldi_result(area):
    from aihydro_core import HydroMeta, HydroResult

    return HydroResult(
        data={"area_km2": area, "method_used": "nldi_comid", "comid": 1},
        meta=HydroMeta(tool="test", version="0", params={}, sources=[]),
    )


def test_nldi_served_row_is_in_the_history(monkeypatch):
    from aihydro_watershed.delineation import nldi_point
    from aihydro_watershed.delineation.router import delineate_from_point

    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_result(420.0))
    result = delineate_from_point(40.71829, -96.41265, method="auto")
    assert result.data["fallback_history"] == [{"method": "nldi_comid", "outcome": "served", "reason": ""}]


def test_nldi_failed_and_skipped_rows_precede_the_tier_that_serves(monkeypatch):
    import pytest
    from aihydro_core import ToolError

    from aihydro_watershed.delineation import nldi_point, small_catchment
    from aihydro_watershed.delineation.router import delineate_from_point

    def _down(*a, **k):
        raise RuntimeError("NLDI 503")

    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", _down)
    monkeypatch.setattr(small_catchment, "delineate_small_catchment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("3DEP down")))
    with pytest.raises(ToolError) as caught:
        delineate_from_point(40.0, -100.0, method="nldi")
    rows = caught.value.details["fallback_history"]
    assert rows == [{"method": "nldi_comid", "outcome": "failed", "reason": "NLDI 503"}]

    monkeypatch.setattr(nldi_point, "delineate_nldi_at_point", lambda *a, **k: _nldi_result(0.2))
    with pytest.raises(ToolError) as caught:
        delineate_from_point(40.0, -100.0, method="nldi")
    assert caught.value.details["fallback_history"][0]["outcome"] == "skipped"


def test_non_toolerror_exception_carries_the_trail(monkeypatch):
    import pytest

    from aihydro_watershed.delineation import router, small_catchment

    monkeypatch.setattr(small_catchment, "delineate_small_catchment",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("3DEP down")))

    def _boom(*a, **k):
        raise KeyError("unexpected after a failed tier")

    monkeypatch.setattr(router, "_gdf_to_geojson", _boom, raising=False)
    # force an unexpected exception after a tier ran: the small-catchment tier succeeds in a stub
    from types import SimpleNamespace
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(geometry=[box(-100.1, 40.0, -100.0, 40.1)], crs=4326)
    monkeypatch.setattr(small_catchment, "delineate_small_catchment", lambda *a, **k: SimpleNamespace(
        gdf=gdf, area_km2=10.0, resolution_m=10.0, snap_distance_m=1.0, outlet_lon=-100.0, outlet_lat=40.0,
        window_iterations=1, window_bounds_5070=None, status="ok", quality_flags=[], notch_elevation_m=1.0))
    with pytest.raises(KeyError) as caught:
        router.delineate_from_point(40.0, -100.0, method="small_catchment")
    assert caught.value.fallback_history[0]["outcome"] == "succeeded"
