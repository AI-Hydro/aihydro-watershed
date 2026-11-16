"""Rebuild aihydro_watershed/delineation/data/conus_border_context.geojson from Natural Earth 1:50m.

Sources (public domain, from the local cartopy cache; no download):
  ne_50m_admin_1_states_provinces_lakes   (48 conterminous states plus DC dissolved = conus; Canadian provinces)
  ne_50m_admin_0_boundary_lines_land      (US-Mexico border line; Mexico is not in the admin-1 file)
The Natural Earth version is not readable from these files (no VERSION file ships in the cartopy cache).

Usage: python -s -E scripts/build_conus_border_context.py [--check]   (--check: compare, do not write)
Run with the repo on sys.path (done below). Output is deterministic given the same shapely/GEOS build.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.ops import unary_union

CULTURAL = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/cultural"
OUT = Path(__file__).resolve().parents[1] / "aihydro_watershed/delineation/data/conus_border_context.geojson"
SOURCE = (
    "Natural Earth 1:50m (public domain, naturalearthdata.com): ne_50m_admin_1_states_provinces_lakes (48 "
    "conterminous states plus DC dissolved = conus; Canadian provinces dissolved, clipped to lon -135..-50, lat "
    "40..56 = canada) and ne_50m_admin_0_boundary_lines_land (US-Mexico border line, clipped to lon -118.5..-96, "
    "lat 25..33.5 = mexico; Mexico is not in the admin-1 file); coordinates rounded to 0.001 deg; not simplified"
)


def _rnd(geom):
    m = mapping(geom)

    def r(c):
        if len(c) == 0:
            return []
        if isinstance(c[0], (int, float)):
            return [round(x, 3) for x in c]
        return [r(i) for i in c]

    return {"type": m["type"], "coordinates": r(m["coordinates"])}


def build() -> bytes:
    g = gpd.read_file(CULTURAL / "ne_50m_admin_1_states_provinces_lakes.shp")
    lines = gpd.read_file(CULTURAL / "ne_50m_admin_0_boundary_lines_land.shp")
    us = g[(g.iso_a2 == "US") & (~g.postal.isin(["AK", "HI"]))]
    assert len(us) == 49
    conus = unary_union(list(us.geometry)).buffer(0)
    ca = unary_union(list(g[g.iso_a2 == "CA"].geometry)).buffer(0).intersection(box(-135, 40, -50, 56))
    mx = unary_union(list(lines.geometry)).intersection(box(-118.5, 25, -96, 33.5))
    feats = [
        {"type": "Feature", "properties": {"id": k}, "geometry": _rnd(v)}
        for k, v in (("conus", conus), ("canada", ca), ("mexico", mx))
    ]
    doc = {"type": "FeatureCollection", "features": feats, "name": "conus_border_context",
           "properties": {"source": SOURCE}}
    return json.dumps(doc, separators=(",", ":")).encode()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = build()
    sha = hashlib.sha256(data).hexdigest()
    if a.check:
        print("identical" if OUT.read_bytes() == data else "DIFFERENT", sha)
    else:
        OUT.write_bytes(data)
        print("wrote", OUT, sha)
