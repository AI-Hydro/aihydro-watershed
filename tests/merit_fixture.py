"""
Tiny synthetic MERIT-Basins region (megabasin 73) built at test time; nothing is downloaded.

Geometry (degrees, ~2 km squares near 40 N). Unit catchments, flow west to east::

    lat 40.02-40.04              [C4 tributary, lon .02-.04] -> C2
    lat 40.00-40.02   [C1 0-.02] -> [C2 .02-.04] -> [C3 .04-.06] -> [C6 .06-.08] -> sea

Rasters are 0.002 deg cells (10 per catchment side). Hillslope cells drain to a trunk row; the
outlet cell sits in the middle of C3, so the refined terminal catchment is exactly half of C3.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyflwdir
import rasterio
from affine import Affine
from shapely.geometry import LineString, box

from aihydro_watershed.delineation.types import area_km2

PFAF = "73"
RES = 0.002
ORIGIN_LON, ORIGIN_LAT = 0.0, 40.04
WIDTH, HEIGHT = 40, 20
TRANSFORM = Affine(RES, 0, ORIGIN_LON, 0, -RES, ORIGIN_LAT)

# comid -> (west, south, east, north, nextdown)
CATCHMENTS = {
    73000001: (0.00, 40.00, 0.02, 40.02, 73000002),  # C1
    73000002: (0.02, 40.00, 0.04, 40.02, 73000003),  # C2
    73000003: (0.04, 40.00, 0.06, 40.02, 73000006),  # C3 (terminal for the test outlet)
    73000004: (0.02, 40.02, 0.04, 40.04, 73000002),  # C4 tributary
    73000006: (0.06, 40.00, 0.08, 40.02, 0),  # C6 downstream of the outlet
}
TERMINAL = 73000003
TRUNK_LAT = 40.011  # the trunk row of the rasters
# comid -> reach line (lon, lat) vertices, downstream end first
REACHES = {
    73000001: [(0.02, TRUNK_LAT), (0.00, TRUNK_LAT)],
    73000002: [(0.04, TRUNK_LAT), (0.02, TRUNK_LAT)],
    73000003: [(0.06, TRUNK_LAT), (0.04, TRUNK_LAT)],
    73000004: [(0.03, 40.02), (0.03, 40.04)],
    73000006: [(0.08, TRUNK_LAT), (0.06, TRUNK_LAT)],
}
OUTLET_REQUEST = (40.0131, 0.0485)  # lat, lon: one cell north of the trunk, in C3
SNAPPED = (40.011, 0.049)  # trunk cell centre
OUTSIDE_REQUEST = (50.0, 50.0)


def flowdir_array() -> np.ndarray:
    d = np.full((HEIGHT, WIDTH), 247, dtype=np.uint8)
    d[10:14, :] = 4  # south, towards the trunk
    d[14, :] = 1  # trunk flows east
    d[15:20, :] = 64  # north, towards the trunk
    d[0:10, 10:20] = 4  # tributary valley drains south into C2
    d[14, WIDTH - 1] = 0  # river mouth
    return d


def accum_array(d: np.ndarray) -> np.ndarray:
    flw = pyflwdir.from_array(
        d, ftype="d8", transform=TRANSFORM, latlon=True,
        mask=np.isin(d, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
    )
    return np.asarray(flw.upstream_area(unit="cell"), dtype="float32")


def _write_raster(path: Path, arr: np.ndarray, dtype: str, nodata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=1, dtype=dtype,
        crs="EPSG:4326", transform=TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


@dataclass
class Region:
    root: Path
    files: dict[str, Path]  # manifest name -> built file

    def sha(self, name: str) -> str:
        return hashlib.sha256(self.files[name].read_bytes()).hexdigest()

    def blob_sha1(self, name: str) -> str:
        data = self.files[name].read_bytes()
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def build_region(root: Path, *, with_rasters: bool = True) -> Region:
    src = root / "served"
    src.mkdir(parents=True, exist_ok=True)
    rows = [
        {"comid": c, "nextdown": nd, "geometry": box(w, s, e, n)}
        for c, (w, s, e, n, nd) in CATCHMENTS.items()
    ]
    basins = gpd.GeoDataFrame(rows, crs=4326)
    basins["area_km2"] = [area_km2(basins.iloc[[i]]) for i in range(len(basins))]
    db = src / f"basins{PFAF}.db"
    basins.to_file(db, layer="l0_basins", driver="SQLite", SPATIALITE="YES", engine="pyogrio")

    uparea = basins.set_index("comid")["area_km2"]
    cum = float(uparea.loc[[73000001, 73000002, 73000003, 73000004]].sum())
    ups = {73000001: float(uparea.loc[73000001]), TERMINAL: cum, 73000006: cum + float(uparea.loc[73000006])}
    # The rivers layer carries the reach lines, digitised downstream to upstream like MERIT-Basins' own.
    rivers = gpd.GeoDataFrame(
        {"comid": list(REACHES), "uparea": [ups.get(c, float(uparea.loc[c])) for c in REACHES]},
        geometry=[LineString(xy) for xy in REACHES.values()],
        crs=4326,
    )
    rdb = src / f"rivers{PFAF}.db"
    rivers.to_file(rdb, layer="rivers", driver="SQLite", SPATIALITE="YES", engine="pyogrio")

    mega = gpd.GeoDataFrame({"basin": [int(PFAF)]}, geometry=[box(-1, 39, 1, 41)], crs=4326)
    mdb = src / "megabasins.db"
    mega.to_file(mdb, layer="megabasins", driver="SQLite", SPATIALITE="YES", engine="pyogrio")

    files = {f"basins{PFAF}.db": db, f"rivers{PFAF}.db": rdb, "megabasins.db": mdb}
    if with_rasters:
        d = flowdir_array()
        _write_raster(src / f"flowdir{PFAF}.tif", d, "uint8", 247)
        _write_raster(src / f"accum{PFAF}.tif", accum_array(d), "float32", -1)
        files[f"flowdir{PFAF}.tif"] = src / f"flowdir{PFAF}.tif"
        files[f"accum{PFAF}.tif"] = src / f"accum{PFAF}.tif"
    return Region(root=root, files=files)


def patched_manifest(base: dict, region: Region) -> dict:
    """The real manifest with the fixture region's files pinned in place of megabasin 73's."""
    man = {**base, "files": dict(base["files"]), "level2_index": dict(base["level2_index"])}
    for name, path in region.files.items():
        data = path.read_bytes()
        if name == "megabasins.db":
            man["level2_index"].update(
                size_bytes=len(data), sha256=None, git_blob_sha1=region.blob_sha1(name)
            )
        else:
            man["files"][name] = {"sha256": region.sha(name), "size_bytes": len(data)}
    return man
