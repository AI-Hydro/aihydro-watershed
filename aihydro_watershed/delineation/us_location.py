"""Offline, deterministic "is this outlet in the conterminous US?" test.

Used only to decide whether ``AMBIGUOUS_OUTLET`` may set ``reliability: "unreliable"``. The preregistered
held-out test ``delineation-holdout-2`` (offline arms, agent-adjudicated) found the flag unreliable as a
wrong-basin detector inside CONUS (3 of 32 flags were wrong basins; weighted precision 0.078 against a 0.5
gate) and supported outside the US (15 of 15, the preregistered minimum of 15 flags).

Data: ``data/conus_border_context.geojson``, derived from Natural Earth 1:110m admin-0 countries (public
domain): the conterminous-US polygon plus Canada and Mexico. No network access, no download.

Rule (``classify_outlet``):

* ``"conus"``: the point is inside the conterminous-US polygon and more than ``BORDER_BUFFER_DEG`` (0.25 deg,
  about 25 km) from Canadian or Mexican land.
* ``"conus_border_zone"``: inside the polygon but within that distance of Canada or Mexico. A basin there may
  cross the border, which the held-out test did not examine, so it is treated as not-CONUS (conservative).
* ``"non_conus"``: outside the polygon (includes Alaska and Hawaii, which are not conterminous and were not
  tested; and open water outside the 1:110m coastline).
* ``"undetermined"``: coordinates missing or non-finite, or the boundary file could not be read.

Limits: the boundary is 1:110m, so the coastline and border are generalised to a few tens of km; the buffer is
sized to absorb that near the land borders. Coastal points a little outside the generalised coastline are
``non_conus`` (conservative). Only the conterminous polygon is used, never a bounding box.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DATA = Path(__file__).parent / "data" / "conus_border_context.geojson"

#: Distance (degrees) from Canada/Mexico inside which an outlet is "border zone" (about 25 km).
BORDER_BUFFER_DEG = 0.25

CONUS = "conus"
CONUS_BORDER_ZONE = "conus_border_zone"
NON_CONUS = "non_conus"
UNDETERMINED = "undetermined"


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any] | None:
    """(conus polygon, canada+mexico border-zone polygon) or None if the file cannot be read."""
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        with open(_DATA, encoding="utf-8") as fh:
            feats = {f["properties"]["id"]: shape(f["geometry"]) for f in json.load(fh)["features"]}
        zone = unary_union([feats["canada"], feats["mexico"]]).buffer(BORDER_BUFFER_DEG)
        return feats["conus"], zone
    except Exception as exc:  # undeterminable -> the caller keeps the conservative behaviour
        log.warning("conterminous-US boundary unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def classify_outlet(lat: float | None, lon: float | None) -> str:
    """``conus`` | ``conus_border_zone`` | ``non_conus`` | ``undetermined`` (see the module docstring)."""
    try:
        if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
            return UNDETERMINED
        data = _load()
        if data is None:
            return UNDETERMINED
        from shapely.geometry import Point

        conus, zone = data
        pt = Point(float(lon), float(lat))
        if not conus.covers(pt):
            return NON_CONUS
        return CONUS_BORDER_ZONE if zone.covers(pt) else CONUS
    except Exception:
        return UNDETERMINED
