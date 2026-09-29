"""
SSURGO soil attributes for CONUS: gNATSGO map units + USDA Soil Data Access.

SSURGO is the mapped soil survey of record for the United States. Each map
unit (``mukey``) is made of components, and each component records its
hydrologic soil group (``component.hydgrp``) and each horizon records its
erodibility (``chorizon.kwfact``) directly. This module reads those recorded
values instead of inferring them from texture.

Acquisition
-----------
1. Map units: the gNATSGO ``mukey`` raster (10 m, EPSG:5070) from Microsoft
   Planetary Computer. Only the catchment's bounding window is read.
   ``pygeohydro.soil_gnatsgo`` serves the same raster but reads whole
   16384 x 16384 tiles (1 to 2 GB each) before clipping, which exhausts memory
   on a laptop for small catchments, so a windowed read is used here.
2. Tables: USDA Soil Data Access (SDA), POST to ``SDA_URL`` with SQL over the
   ``component`` and ``chorizon`` tables. The surface horizon is the one with
   ``hzdept_r = 0``.

Per map unit
------------
* ``kw``, ``sand``, ``silt``, ``clay``: component-percentage-weighted mean of
  the surface horizon values (components with a null value are skipped).
* ``hydgrp``: hydrologic group of the dominant component (highest
  ``comppct_r`` with a non-empty group).
* Dual groups (``A/D``, ``B/D``, ``C/D``): NRCS NEH 630 Ch. 7 assigns the first
  letter to the drained condition and the second (always D) to the undrained
  condition. Both are returned, as ``hsg_drained`` and ``hsg_undrained``.

Units
-----
``kwfact`` is in US customary units (ton acre h / (hundreds acre ft tonf in)).
Multiply by ``KW_US_TO_SI`` (0.1317) for t ha h / (ha MJ mm), the SI unit used
by :mod:`aihydro_watershed.terrain.erosion`.

Everything that touches the network takes an injectable callable so the
parsing and aggregation logic is testable offline.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
PC_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
GNATSGO_COLLECTION = "gnatsgo-rasters"

KW_US_TO_SI = 0.1317
HSG_CODES = {"A": 1, "B": 2, "C": 3, "D": 4}
HSG_NAMES = {v: k for k, v in HSG_CODES.items()}

SSURGO_PRODUCT_ID = "SSURGO_GNATSGO"
SSURGO_SOURCE = "gNATSGO mukey raster (Planetary Computer) + USDA Soil Data Access"
SSURGO_CITATION = (
    "Soil Survey Staff. Gridded National Soil Survey Geographic (gNATSGO) "
    "Database for the Conterminous United States. USDA-NRCS. "
    "Tabular data: Soil Data Access, https://sdmdataaccess.nrcs.usda.gov/"
)

_SDA_CHUNK = 400


# ── Pure helpers ──────────────────────────────────────────────────────────────

def parse_hydgrp(value: Any) -> tuple[Optional[int], Optional[int]]:
    """Parse an SSURGO ``hydgrp`` string into (drained, undrained) group codes.

    Codes are 1=A, 2=B, 3=C, 4=D. A single group maps to itself in both
    conditions. A dual group such as ``"B/D"`` maps to (B, D). Empty or
    unrecognised values return (None, None).

    >>> parse_hydgrp("A/D")
    (1, 4)
    >>> parse_hydgrp("C")
    (3, 3)
    """
    if not isinstance(value, str):
        return None, None
    parts = [p.strip().upper() for p in value.split("/") if p.strip()]
    if not parts:
        return None, None
    drained = HSG_CODES.get(parts[0])
    undrained = HSG_CODES.get(parts[-1])
    if drained is None or undrained is None:
        return None, None
    return drained, undrained


def build_component_query(mukeys: Iterable[Any]) -> str:
    """SQL for SDA: every component of the given map units with its surface horizon."""
    keys = sorted({int(float(k)) for k in mukeys})
    if not keys:
        raise ValueError("build_component_query: no map-unit keys given")
    in_list = ",".join(f"'{k}'" for k in keys)
    return (
        "SELECT c.mukey, c.cokey, c.comppct_r, c.hydgrp, ch.kwfact, "
        "ch.sandtotal_r, ch.silttotal_r, ch.claytotal_r "
        "FROM component c "
        "LEFT JOIN chorizon ch ON ch.cokey = c.cokey AND ch.hzdept_r = 0 "
        f"WHERE c.mukey IN ({in_list})"
    )


def parse_sda_table(payload: dict) -> list[dict]:
    """Turn an SDA ``JSON+COLUMNNAME`` response into a list of row dicts."""
    table = (payload or {}).get("Table") or []
    if not table:
        return []
    header = table[0]
    return [dict(zip(header, row)) for row in table[1:]]


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def summarize_components(rows: Iterable[dict]) -> dict[int, dict]:
    """Aggregate SDA component rows to one record per map unit.

    Returns ``{mukey: {kw, sand, silt, clay, hydgrp, hsg_drained,
    hsg_undrained, is_dual}}``. See the module docstring for the rules.
    """
    by_mukey: dict[int, list[dict]] = {}
    for r in rows:
        try:
            mk = int(float(r.get("mukey")))
        except (TypeError, ValueError):
            continue
        by_mukey.setdefault(mk, []).append(r)

    out: dict[int, dict] = {}
    for mk, comps in by_mukey.items():
        pct = np.array([_to_float(c.get("comppct_r")) for c in comps])
        pct = np.where(np.isfinite(pct), pct, 0.0)

        def wavg(col: str) -> float:
            v = np.array([_to_float(c.get(col)) for c in comps])
            ok = np.isfinite(v) & (pct > 0)
            return float(np.average(v[ok], weights=pct[ok])) if ok.any() else float("nan")

        order = np.argsort(-pct, kind="stable")
        hydgrp = next(
            (comps[i].get("hydgrp").strip() for i in order
             if isinstance(comps[i].get("hydgrp"), str) and comps[i].get("hydgrp").strip()),
            None,
        )
        drained, undrained = parse_hydgrp(hydgrp)
        out[mk] = {
            "kw": wavg("kwfact"),
            "sand": wavg("sandtotal_r"),
            "silt": wavg("silttotal_r"),
            "clay": wavg("claytotal_r"),
            "hydgrp": hydgrp,
            "hsg_drained": drained,
            "hsg_undrained": undrained,
            "is_dual": bool(hydgrp and "/" in hydgrp),
        }
    return out


# ── Network: Soil Data Access ─────────────────────────────────────────────────

def sda_query(sql: str, *, timeout: float = 120.0, retries: int = 4) -> list[dict]:
    """Run one SQL query against USDA Soil Data Access and return row dicts.

    Python's resolver has been seen to fail intermittently on the USDA host
    while the system resolver works, so after ``retries`` failed attempts with
    ``requests`` the query is sent once more through ``curl`` if it is on PATH.
    """
    import requests

    body = json.dumps({"query": sql, "format": "JSON+COLUMNNAME"})
    headers = {"Content-Type": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(SDA_URL, data=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            # SDA returns an empty body (not {"Table": []}) when nothing matches.
            return parse_sda_table(resp.json()) if resp.text.strip() else []
        except Exception as exc:  # network, HTTP, or JSON error
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    curl = shutil.which("curl")
    if curl:
        proc = subprocess.run(
            [curl, "-s", "-m", str(int(timeout)), "-H", "Content-Type: application/json",
             "-d", body, SDA_URL],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            return parse_sda_table(json.loads(proc.stdout)) if proc.stdout.strip() else []
    raise RuntimeError(f"USDA Soil Data Access query failed after {retries} attempts: {last_exc}")


def fetch_mukey_properties(
    mukeys: Iterable[Any],
    *,
    query_fn: Callable[[str], list[dict]] = sda_query,
    chunk: int = _SDA_CHUNK,
) -> dict[int, dict]:
    """Per-map-unit SSURGO properties for ``mukeys`` (see :func:`summarize_components`)."""
    keys = sorted({int(float(k)) for k in mukeys if np.isfinite(_to_float(k)) and float(k) > 0})
    rows: list[dict] = []
    for i in range(0, len(keys), chunk):
        rows += query_fn(build_component_query(keys[i:i + chunk]))
    return summarize_components(rows)


# ── Network: gNATSGO map-unit raster ──────────────────────────────────────────

def _single_geom_wgs84(geometry):
    """Coerce GeoDataFrame / GeoSeries / shapely geometry to one WGS84 geometry."""
    import geopandas as gpd

    if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        g = geometry if geometry.crs is not None else geometry.set_crs(4326)
        g = g.to_crs(4326)
        gs = g.geometry if isinstance(g, gpd.GeoDataFrame) else g
        return gs.union_all() if hasattr(gs, "union_all") else gs.unary_union
    return geometry


def fetch_gnatsgo_mukey(geometry, *, pad_m: float = 90.0):
    """Windowed read of the gNATSGO ``mukey`` raster around ``geometry`` (WGS84).

    Returns an ``xarray.DataArray`` of map-unit keys (float, NaN where unmapped)
    in the raster's native CRS (EPSG:5070), merged across tiles when the window
    straddles more than one.
    """
    import geopandas as gpd
    import planetary_computer
    import rasterio
    import rasterio.merge
    import requests
    import rioxarray  # noqa: F401  (registers .rio)
    import xarray as xr

    geom = _single_geom_wgs84(geometry)
    resp = requests.post(
        PC_STAC_SEARCH,
        json={"collections": [GNATSGO_COLLECTION], "bbox": list(geom.bounds), "limit": 20},
        timeout=60,
    )
    resp.raise_for_status()
    items = resp.json().get("features", [])
    if not items:
        raise RuntimeError(
            "gNATSGO has no map-unit tiles for this geometry (outside CONUS?)."
        )
    hrefs = [planetary_computer.sign(f["assets"]["mukey"]["href"]) for f in items]
    srcs = [rasterio.open(h) for h in hrefs]
    try:
        crs = srcs[0].crs
        b = gpd.GeoSeries([geom], crs=4326).to_crs(crs).total_bounds
        bounds = (b[0] - pad_m, b[1] - pad_m, b[2] + pad_m, b[3] + pad_m)
        arr, tr = rasterio.merge.merge(srcs, bounds=bounds, nodata=0)
    finally:
        for s in srcs:
            s.close()
    ny, nx = arr.shape[1:]
    da = xr.DataArray(
        arr[0].astype("float64"),
        dims=("y", "x"),
        coords={
            "x": tr.c + (np.arange(nx) + 0.5) * tr.a,
            "y": tr.f + (np.arange(ny) + 0.5) * tr.e,
        },
        name="mukey",
    )
    da = da.where(da > 0)
    da = da.rio.write_crs(crs).rio.write_transform(tr)
    da.attrs["resolution_m"] = abs(tr.a)
    return da


# ── Assembly ──────────────────────────────────────────────────────────────────

def mukey_grid_to_dataset(mukey, props: dict[int, dict]):
    """Map per-map-unit properties onto a ``mukey`` grid.

    Returns an ``xarray.Dataset`` on the same grid with variables ``mukey``,
    ``hsg_drained``, ``hsg_undrained`` (1-4, NaN where unknown), ``kw``
    (US customary), ``kw_si``, and surface-horizon ``sand_surface``,
    ``silt_surface``, ``clay_surface`` (percent).
    """
    import xarray as xr

    vals = mukey.values
    finite = np.isfinite(vals)
    uniq, inv = np.unique(np.where(finite, vals, 0).astype(np.int64), return_inverse=True)
    inv = inv.reshape(vals.shape)

    def layer(key: str) -> np.ndarray:
        table = np.array(
            [np.nan if k == 0 else _nan_if_none(props.get(int(k), {}).get(key)) for k in uniq],
            dtype="float64",
        )
        out = table[inv]
        out[~finite] = np.nan
        return out

    data = {
        "mukey": mukey.values.astype("float64"),
        "hsg_drained": layer("hsg_drained"),
        "hsg_undrained": layer("hsg_undrained"),
        "kw": layer("kw"),
        "sand_surface": layer("sand"),
        "silt_surface": layer("silt"),
        "clay_surface": layer("clay"),
    }
    data["kw_si"] = data["kw"] * KW_US_TO_SI
    ds = xr.Dataset({k: (mukey.dims, v) for k, v in data.items()}, coords=mukey.coords)
    try:
        ds = ds.rio.write_crs(mukey.rio.crs)
    except Exception:
        pass
    ds.attrs.update({
        "_adata_product": SSURGO_PRODUCT_ID,
        "_adata_source": SSURGO_SOURCE,
        "_adata_citation": SSURGO_CITATION,
        "_adata_region": "CONUS",
        "_adata_resolution_m": mukey.attrs.get("resolution_m"),
        "kw_units": "US customary (ton acre h / (hundreds acre ft tonf in)); kw_si = kw * 0.1317",
        "hsg_codes": "1=A 2=B 3=C 4=D",
    })
    return ds


def _nan_if_none(v: Any) -> float:
    return float("nan") if v is None else _to_float(v)


def fetch_soil_data_ssurgo(
    geometry,
    *,
    mukey_fn: Callable[..., Any] = fetch_gnatsgo_mukey,
    query_fn: Callable[[str], list[dict]] = sda_query,
):
    """Fetch SSURGO soil attributes on the gNATSGO grid for a CONUS geometry.

    Returns the Dataset described in :func:`mukey_grid_to_dataset`, with
    provenance under the ``_adata_*`` attrs used by the other soil backends.
    """
    mukey = mukey_fn(geometry)
    keys = np.unique(mukey.values[np.isfinite(mukey.values)])
    if keys.size == 0:
        raise RuntimeError("gNATSGO returned no mapped soil units for this geometry.")
    props = fetch_mukey_properties(keys, query_fn=query_fn)
    return mukey_grid_to_dataset(mukey, props)


def summarize_ssurgo(ds, geometry=None) -> dict:
    """Area-weighted catchment summary of an SSURGO Dataset.

    ``geometry`` (WGS84) restricts the summary to cells inside the polygon;
    without it every cell of the window counts. Means are over mapped cells
    with a value. ``soil_coverage`` is the fraction of in-polygon cells with
    a map unit; ``hsg_coverage`` the fraction with a hydrologic group.
    """
    inside = np.ones(ds["mukey"].shape, dtype=bool)
    if geometry is not None:
        import geopandas as gpd
        import rasterio.features

        geom = _single_geom_wgs84(geometry)
        poly = gpd.GeoSeries([geom], crs=4326).to_crs(ds.rio.crs).iloc[0]
        inside = rasterio.features.geometry_mask(
            [poly], out_shape=inside.shape, transform=ds.rio.transform(), invert=True,
        )
    n_total = int(inside.sum())
    mu = ds["mukey"].values
    mapped = inside & np.isfinite(mu)

    def mean(var: str) -> Optional[float]:
        v = ds[var].values[mapped]
        v = v[np.isfinite(v)]
        return round(float(v.mean()), 4) if v.size else None

    def pct_groups(var: str) -> dict[str, float]:
        v = ds[var].values[mapped]
        v = v[np.isfinite(v)].astype(int)
        if not v.size:
            return {}
        return {HSG_NAMES[g]: round(100.0 * float(np.mean(v == g)), 2)
                for g in sorted(set(v.tolist())) if g in HSG_NAMES}

    hd = ds["hsg_drained"].values[mapped]
    hu = ds["hsg_undrained"].values[mapped]
    has_hsg = np.isfinite(hd)
    n_mapped = int(mapped.sum())
    return {
        "kw_mean": mean("kw"),
        "kw_si_mean": mean("kw_si"),
        "sand_pct_mean": mean("sand_surface"),
        "silt_pct_mean": mean("silt_surface"),
        "clay_pct_mean": mean("clay_surface"),
        "hsg_drained_pct": pct_groups("hsg_drained"),
        "hsg_undrained_pct": pct_groups("hsg_undrained"),
        "pct_dual_hsg": (
            round(100.0 * float(np.mean(hd[has_hsg] != hu[has_hsg])), 2)
            if has_hsg.any() else None
        ),
        "n_map_units": int(np.unique(mu[mapped]).size) if n_mapped else 0,
        "n_cells": n_total,
        "soil_coverage": round(n_mapped / n_total, 4) if n_total else 0.0,
        "hsg_coverage": round(int(has_hsg.sum()) / n_total, 4) if n_total else 0.0,
        "resolution_m": ds.attrs.get("_adata_resolution_m"),
        "product": ds.attrs.get("_adata_product"),
        "source": ds.attrs.get("_adata_source"),
        "kw_units": "US customary; kw_si in t ha h / (ha MJ mm)",
    }


def ssurgo_catchment_attributes(geometry, **fetch_kwargs) -> dict:
    """Fetch SSURGO for ``geometry`` and return ``{"summary": ..., "dataset": ...}``."""
    ds = fetch_soil_data_ssurgo(geometry, **fetch_kwargs)
    return {"summary": summarize_ssurgo(ds, geometry), "dataset": ds}
