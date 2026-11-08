"""
Read access to one megabasin of MERIT-Basins unit catchments (``basins##.db``).

The file is a SpatiaLite-format SQLite database published for the ``delineator`` package.
Only plain-SQL columns are read through :mod:`sqlite3` (topology), and geometries are read
through GDAL/pyogrio, so no SpatiaLite extension is required. Schema used (verified against
delineator 2.2.4 queries): ``l0_basins(comid, nextdown, area_km2, geometry)`` for unit catchments
and ``rivers(comid, uparea, ...)`` in ``rivers##.db``.

This module is an independent implementation of the vector-assembly step (upstream traversal of
``nextdown`` plus dissolve); it shares no code with ``delineator`` and does not use pysheds.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

_CHUNK_IDS = 2000
_NEAREST_DEG = 0.05  # ~5 km: accept the nearest catchment for a point just off the coast
_adjacency_cache: dict[tuple[str, int, int], "_Topology"] = {}


class _Topology:
    """comid -> row index, with a CSR upstream adjacency built from ``nextdown``."""

    def __init__(self, comids: np.ndarray, nextdown: np.ndarray):
        self.comids = comids
        self.index = {int(c): i for i, c in enumerate(comids)}
        n = len(comids)
        down_idx = np.full(n, -1, dtype=np.int64)
        for i, d in enumerate(nextdown):
            down_idx[i] = self.index.get(int(d), -1) if d else -1
        has_down = down_idx >= 0
        order = np.argsort(down_idx[has_down], kind="stable")
        self._children = np.nonzero(has_down)[0][order]
        counts = np.bincount(down_idx[has_down], minlength=n)
        self._offsets = np.concatenate([[0], np.cumsum(counts)])

    def upstream_of(self, comid: int) -> list[int]:
        start = self.index.get(int(comid))
        if start is None:
            return []
        seen = np.zeros(len(self.comids), dtype=bool)
        seen[start] = True
        out = [start]
        stack = [start]
        while stack:
            cur = stack.pop()
            for child in self._children[self._offsets[cur] : self._offsets[cur + 1]]:
                if not seen[child]:
                    seen[child] = True
                    out.append(int(child))
                    stack.append(int(child))
        return [int(self.comids[i]) for i in out]


class BasinsStore:
    def __init__(self, basins_db: Path, rivers_db: Path | None = None):
        self.basins_db = Path(basins_db)
        self.rivers_db = Path(rivers_db) if rivers_db else None

    def _topology(self) -> _Topology:
        st = self.basins_db.stat()
        key = (str(self.basins_db), st.st_size, st.st_mtime_ns)
        topo = _adjacency_cache.get(key)
        if topo is None:
            with closing(sqlite3.connect(f"file:{self.basins_db}?mode=ro", uri=True)) as con:
                rows = con.execute("SELECT comid, nextdown FROM l0_basins").fetchall()
            if not rows:
                raise ValueError(f"No unit catchments in {self.basins_db.name}.")
            arr = np.asarray(rows, dtype=np.int64)
            _adjacency_cache.clear()  # one megabasin at a time keeps memory bounded
            topo = _adjacency_cache[key] = _Topology(arr[:, 0], arr[:, 1])
        return topo

    def _rank(self, comids: list[int], dist: dict[int, float] | None = None) -> int:
        """Deterministic choice among several candidate unit catchments: nearest, then max ``uparea``, then lowest comid."""
        def key(c: int):
            up = self.river_uparea_km2(c)
            return ((dist or {}).get(c, 0.0), -(up if up is not None else -1.0), c)

        return min(comids, key=key)

    def locate(self, lon: float, lat: float) -> int:
        """comid of the unit catchment containing the point.

        Several hits (a point on a shared edge or overlapping slivers) resolve to the largest ``uparea``
        (rivers db), then the lowest comid. With no hit, the nearest catchment within ~5 km is used.
        """
        pt = Point(lon, lat)
        eps = _NEAREST_DEG
        cand = gpd.read_file(
            self.basins_db, layer="l0_basins", bbox=(lon - eps, lat - eps, lon + eps, lat + eps)
        )
        if cand.empty:
            raise ValueError("The point is not in any unit catchment (is it over land?).")
        for hit in (cand[cand.geometry.contains(pt)], cand[cand.geometry.intersects(pt)]):
            if not hit.empty:
                return self._rank([int(c) for c in hit["comid"]])
        d = cand.geometry.distance(pt)
        if float(d.min()) > eps:
            raise ValueError("The point is not in any unit catchment (is it over land?).")
        near = cand[d <= float(d.min()) + 1e-12]
        return self._rank([int(c) for c in near["comid"]])

    def upstream_ids(self, comid: int) -> list[int]:
        """``comid`` first, then every unit catchment draining to it."""
        ids = self._topology().upstream_of(int(comid))
        if not ids:
            raise ValueError(f"Catchment {comid} is not in {self.basins_db.name}.")
        return ids

    def geometries(self, ids: list[int]) -> gpd.GeoDataFrame:
        """Unit-catchment polygons (EPSG:4326) with ``comid`` and ``area_km2``.

        Every requested id should come back with a non-empty polygon. Ids that did not are listed in
        ``result.attrs["missing_comids"]`` so the caller can flag an incomplete assembly.
        """
        parts = []
        for i in range(0, len(ids), _CHUNK_IDS):
            chunk = ", ".join(str(int(c)) for c in ids[i : i + _CHUNK_IDS])
            parts.append(
                gpd.read_file(self.basins_db, layer="l0_basins", where=f"comid IN ({chunk})")
            )
        gdf = parts[0] if len(parts) == 1 else gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), crs=parts[0].crs
        )
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        has_poly = gdf.geometry.notna() & ~gdf.geometry.is_empty
        gdf = gdf[has_poly].copy()
        gdf.attrs["missing_comids"] = sorted(set(int(c) for c in ids) - set(int(c) for c in gdf["comid"]))
        return gdf

    def river_uparea_km2(self, comid: int) -> float | None:
        """MERIT-Basins cumulative drainage area at the downstream end of ``comid`` (km2)."""
        if self.rivers_db is None or not self.rivers_db.exists():
            return None
        with closing(sqlite3.connect(f"file:{self.rivers_db}?mode=ro", uri=True)) as con:
            row = con.execute("SELECT uparea FROM rivers WHERE comid = ?", (int(comid),)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def river_line(self, comid: int) -> LineString | None:
        """River-reach line of ``comid`` in EPSG:4326, digitised downstream to upstream, or ``None``.

        Read from the ``rivers`` layer of ``rivers##.db`` (the reach shares its ``comid`` with the unit
        catchment). ``None`` when the layer has no geometry for the id (or the file is absent), so a
        caller can say the reach could not be tested instead of guessing.
        """
        if self.rivers_db is None or not self.rivers_db.exists():
            return None
        try:
            gdf = gpd.read_file(self.rivers_db, layer="rivers", where=f"comid = {int(comid)}")
        except Exception:
            return None
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            return None
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        geom = gdf.geometry.iloc[0]
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        return geom if geom.geom_type == "LineString" else None
