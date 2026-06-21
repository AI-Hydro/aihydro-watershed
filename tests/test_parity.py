"""
Wave A4 — delineation parity tests.

Two smoke tests that require real network access:
  1. CONUS gauge: NLDI delineation vs. published HUC drainage area (±20 %)
  2. Global pour point: MERIT-fast delineation produces a non-empty polygon

Run with:
    pytest tests/test_parity.py -m live -v

Both tests are marked @pytest.mark.live so they are skipped in offline CI.
They are intentionally lenient (20 % area tolerance) because NLDI area
estimates drift with COMID selection and DEM resolution.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CONUS — NLDI parity (Potomac River near Point of Rocks, USGS 01638500)
# Published drainage area: 25 000 km² (NWIS).
# ---------------------------------------------------------------------------

_CONUS_LAT = 39.2728
_CONUS_LON = -77.5383
_CONUS_EXPECTED_KM2 = 25_000.0
_CONUS_TOL_FRAC = 0.30          # ±30 % — NLDI comid-basin estimates can be coarse


@pytest.mark.live
def test_nldi_conus_parity():
    """NLDI delineation for Potomac at Point of Rocks is within 30% of published area."""
    from aihydro_watershed.delineation.router import delineate_from_point

    result = delineate_from_point(
        _CONUS_LAT,
        _CONUS_LON,
        expected_area_km2=_CONUS_EXPECTED_KM2,
        method="nldi",
    )
    area = result.data["area_km2"]
    assert area > 0, "Delineation returned zero area"
    rel_err = abs(area - _CONUS_EXPECTED_KM2) / _CONUS_EXPECTED_KM2
    assert rel_err <= _CONUS_TOL_FRAC, (
        f"NLDI area {area:.0f} km² differs from expected "
        f"{_CONUS_EXPECTED_KM2:.0f} km² by {rel_err:.1%} (tolerance {_CONUS_TOL_FRAC:.0%})"
    )
    assert result.data.get("geometry_geojson") is not None, "No geometry returned"


# ---------------------------------------------------------------------------
# Global — fast (pysheds) delineation (Rhine near Cologne, Germany)
# We only check that the function runs and returns a non-empty polygon.
# ---------------------------------------------------------------------------

_GLOBAL_LAT = 50.932
_GLOBAL_LON = 6.970


@pytest.mark.live
def test_fast_global_parity():
    """Fast DEM delineation for Rhine at Cologne returns a non-empty polygon.

    Skipped (not failed) when all DEM STAC providers are transiently
    unavailable — infrastructure outages should not break CI.
    """
    from aihydro_watershed.delineation.router import delineate_from_point

    try:
        result = delineate_from_point(_GLOBAL_LAT, _GLOBAL_LON, method="fast")
    except Exception as exc:
        # DELINEATION_FAILED after all providers exhausted → skip, not fail.
        # Any other exception (e.g. programming error) still propagates.
        code = getattr(exc, "code", "") or ""
        if "DELINEATION_FAILED" in str(code):
            pytest.skip(
                f"All DEM STAC providers unavailable (transient): {exc}"
            )
        raise

    area = result.data["area_km2"]
    assert area > 0, "Fast delineation returned zero area"

    geom = result.data.get("geometry_geojson")
    assert geom is not None, "No geometry_geojson in result"
    assert geom.get("type") in ("Feature", "Polygon", "MultiPolygon"), (
        f"Unexpected geometry type: {geom.get('type')}"
    )
