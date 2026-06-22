"""DEM fetch for the fast-tier delineation pipeline.

Delegates all DEM acquisition to aihydro_data, which manages the full
provider chain (GLO30 → SRTM → MERIT_DEM → GLO30_STAC → GLO30_ELEMENT84),
exponential-backoff retry, and disk cache.  This module's sole job is the
CRS bridge: convert the projected UTM bbox to WGS84 for aihydro_data, then
reproject the returned DataArray back to the caller's metric UTM CRS as
required by pysheds.

aihydro_data DEM routing (global fallback chain):
    GLO30        — Copernicus GLO-30 via GEE (30 m)
    SRTM         — NASA SRTM 30 m
    MERIT_DEM    — MERIT-Hydro conditioned DEM
    GLO30_STAC   — Planetary Computer (with Element84 automatic fallback)
    GLO30_ELEMENT84 — Element84 Earth Search (explicit routing backstop)
"""

from __future__ import annotations

import logging
from typing import Iterable

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401 — activates .rio accessor on xarray
import xarray as xr
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


class StacItemCache:
    """No-op stub — kept for backward compatibility with pysheds_pipeline.py.

    aihydro_data manages its own disk cache; the previous STAC item
    accumulation pattern is no longer needed here.
    """

    def __init__(self) -> None:
        pass

    def add(self, items: Iterable) -> None:
        pass

    @property
    def items(self) -> list:
        return []


def fetch_dem_bbox(
    bbox_proj: Polygon,
    crs_proj: str,
    resolution_m: float,
    item_cache: "StacItemCache | None" = None,
    collection: str = "copernicus",
    asset_key: "str | None" = None,
    verbose: bool = False,
) -> xr.DataArray:
    """Fetch DEM for a projected bbox via aihydro_data's routing chain.

    Parameters
    ----------
    bbox_proj:
        Shapely Polygon in the caller's projected (UTM) CRS.
    crs_proj:
        EPSG string for the UTM zone (e.g. ``"EPSG:32618"``).
    resolution_m:
        Target pixel size in metres after reprojection (e.g. 30, 90).
    item_cache, collection, asset_key:
        Accepted for backward compatibility with pysheds_pipeline.py but
        ignored — aihydro_data routing + disk cache supersede these.
    verbose:
        Emit INFO log lines when True.

    Returns
    -------
    xr.DataArray
        Single-band float32 DataArray in *crs_proj* at *resolution_m*.
    """
    import aihydro_data

    gdf = gpd.GeoDataFrame(geometry=[bbox_proj], crs=crs_proj).to_crs(4326)

    if verbose:
        log.info("Fetching DEM via aihydro_data @ %.0fm resolution", resolution_m)

    try:
        result = aihydro_data.fetch("dem", gdf, "", "", aggregation="raw_raster")
        dem_wgs84: xr.DataArray = result.data
    except Exception as exc:
        from aihydro_core.primitives import ToolError
        raise ToolError(
            code="DELINEATION_FAILED",
            message=(
                f"DEM fetch failed for bbox {gdf.total_bounds.tolist()}: {exc}"
            ),
            recovery=(
                "Check network connectivity and retry.  All aihydro_data DEM "
                "providers (GLO30/SRTM/GLO30_STAC/GLO30_ELEMENT84) are "
                "unavailable.  Alternatively use method='local_merit' with a "
                "pre-downloaded MERIT-Hydro tile."
            ),
        ) from exc

    if not isinstance(dem_wgs84, xr.DataArray):
        from aihydro_core.primitives import ToolError
        raise ToolError(
            code="DELINEATION_FAILED",
            message=(
                f"DEM fetch returned {type(dem_wgs84).__name__} instead of "
                "xr.DataArray — unexpected result from aihydro_data.fetch()."
            ),
            recovery="Report this as a bug to the aihydro-data maintainers.",
        )

    # Drop any degenerate leading dimensions (time, band) from static products.
    dem_wgs84 = dem_wgs84.squeeze()

    if dem_wgs84.rio.crs is None:
        dem_wgs84 = dem_wgs84.rio.write_crs("EPSG:4326")

    # Reproject to caller's metric UTM CRS at the requested pixel size.
    dem_utm = dem_wgs84.rio.reproject(
        crs_proj,
        resolution=(resolution_m, resolution_m),
    ).astype(np.float32)

    if verbose:
        log.info("DEM ready: shape=%s  CRS=%s", dem_utm.shape, crs_proj)

    return dem_utm
