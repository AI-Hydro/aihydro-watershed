"""TWI must compute on one grid for any DEM product, and cite the DEM actually used."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

pytest.importorskip("pysheds")
py3dep = pytest.importorskip("py3dep")
pytest.importorskip("rioxarray")

from aihydro_watershed.characterize import _dem, twi  # noqa: E402
from aihydro_watershed.characterize._dem import reproject_elevation  # noqa: E402

N = 300


def _glo30_like():
    """1-arcsec geographic grid at 46.8N (Maine): native shape differs from the
    EPSG:5070 grid it is reprojected to, like GLO-30 in the 3DEP path."""
    lon = -68.6 + np.arange(N) / 3600.0
    lat = (46.8 + np.arange(N) / 3600.0)[::-1]
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = 300 + 0.05 * yy + 20 * np.exp(-((xx - 150) ** 2) / (2 * 40 ** 2)) \
        + np.random.default_rng(0).random((N, N)) * 0.3
    da = xr.DataArray(z.astype("float32"), dims=("y", "x"), coords={"y": lat, "x": lon})
    return da.rio.write_crs("EPSG:4326")


def _basin(dem):
    return box(float(dem.x[40]), float(dem.y[-40]), float(dem.x[-40]), float(dem.y[40]))


def _run(monkeypatch, tmp_path, product, source, save=True):
    dem = _glo30_like()
    monkeypatch.setattr(_dem, "fetch_dem", lambda g, resolution=30, prefer="auto": (dem, product, source))
    return dem, twi.compute_twi(
        _basin(dem), resolution=30, save_outputs=save, output_dir=str(tmp_path),
        create_visualizations=False,
    )


def test_glo30_dem_computes_full_twi_without_3dep_slope(monkeypatch, tmp_path):
    def _forbidden(*a, **k):
        raise AssertionError("3DEP slope must not be requested for a GLO30 DEM")

    monkeypatch.setattr(py3dep, "get_map", _forbidden)
    dem, out = _run(monkeypatch, tmp_path, "GLO30", "gee")
    reproj = reproject_elevation(dem, "EPSG:5070")
    assert out["twi_array"].shape == reproj.shape != dem.shape
    assert out["twi_method"] == "full" and out["twi_method_reason"] is None
    assert out["n_valid_cells"] > 0 and np.isfinite(out["twi_mean"])
    assert out["dem_product"] == "GLO30"
    assert "twi_raster_path" in out  # raster written on the DEM grid (previously raised)


def test_3dep_slope_on_a_different_grid_is_matched_not_trimmed(monkeypatch, tmp_path):
    dem = _glo30_like()

    def _offset_slope(*a, **k):
        s = reproject_elevation(dem, "EPSG:5070").isel(y=slice(10, -30), x=slice(5, -20))
        return s * 0 + 3.0

    monkeypatch.setattr(py3dep, "get_map", _offset_slope)
    _, out = _run(monkeypatch, tmp_path, "DEM3DEP_10M", "py3dep")
    assert out["twi_array"].shape == reproject_elevation(dem, "EPSG:5070").shape
    assert out["twi_method"] == "full" and np.isfinite(out["twi_mean"])


def test_grid_mismatch_raises_instead_of_trimming(monkeypatch, tmp_path):
    """If slope cannot be put on the accumulation grid, fail loudly (no zoom/trim)."""
    import aihydro_watershed.characterize._dem as d

    dem = _glo30_like()
    monkeypatch.setattr(d, "fetch_dem", lambda g, resolution=30, prefer="auto": (dem, "GLO30", "gee"))
    real = d.slope_from_dem
    monkeypatch.setattr(d, "slope_from_dem", lambda x: real(x).isel(y=slice(0, -7)))
    with pytest.raises(ValueError, match="grid mismatch"):
        twi.compute_twi(_basin(dem), save_outputs=False, create_visualizations=False)


@pytest.mark.parametrize("product", ["GLO30", "DEM3DEP_10M"])
def test_sources_follow_dem_product(product):
    pytest.importorskip("aihydro_data")
    srcs = twi._dem_sources(product)
    assert len(srcs) == 1
    assert product in srcs[0].name
    if product == "GLO30":
        assert "3DEP" not in srcs[0].name and "Copernicus" in (srcs[0].citation or "")


def test_unknown_product_is_named_not_mislabelled():
    srcs = twi._dem_sources("MYSTERY_DEM")
    assert srcs[0].name == "MYSTERY_DEM" and srcs[0].citation is None


def test_result_meta_sources_match_dem_product(monkeypatch):
    dem = _glo30_like()
    monkeypatch.setattr(_dem, "fetch_dem", lambda g, resolution=30, prefer="auto": (dem, "GLO30", "gee"))
    monkeypatch.setattr(py3dep, "get_map", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    from shapely.geometry import mapping

    res = twi.compute_twi_result(mapping(_basin(dem)))
    assert "GLO30" in res.meta.sources[0].name and "3DEP" not in res.meta.sources[0].name
    assert res.data["twi_method"] == "full"
