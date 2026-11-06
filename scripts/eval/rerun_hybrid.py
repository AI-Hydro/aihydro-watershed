#!/usr/bin/env python
"""Offline re-run of ``merit_basins_hybrid_delineate`` over the camels-delineation-1 gauges.

Reads only caches that the campaign already filled (MERIT-Basins vectors and Earth Engine windows under
``AIHYDRO_HOME``); every non-loopback socket is blocked, so a cache miss fails the gauge instead of
reaching the network. Nothing under ``AIHYDRO_HOME`` is written (``auto_stage=False``).

usage: rerun_hybrid.py --code <dir holding aihydro_watershed/> --reference <reference_671.csv>
           --out <dir> [--area none|gages2] [--first N] [--ids a,b,c]

Writes ``<out>/records/<gauge>.json`` and ``<out>/polys/<gauge>.geojson`` (resumable: an existing record is
kept). Score with ``score_rerun.py``.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from pathlib import Path

_real_connect = socket.socket.connect
_real_gai = socket.getaddrinfo
_LOOP = ("127.0.0.1", "::1", "localhost")


def _block_connect(self, addr, *a, **k):
    if isinstance(addr, tuple) and addr[0] not in _LOOP:
        raise OSError(f"network blocked in offline re-run: {addr}")
    return _real_connect(self, addr, *a, **k)


def _block_gai(host, *a, **k):
    if host not in _LOOP:
        raise OSError(f"network blocked in offline re-run (DNS): {host}")
    return _real_gai(host, *a, **k)


socket.socket.connect = _block_connect
socket.getaddrinfo = _block_gai


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--area", choices=("none", "gages2"), default="none")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import pandas as pd

    from aihydro_watershed.delineation import merit_flowdir_pipeline as m

    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "polys").mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(a.reference, dtype={"gauge_id": str})
    if a.ids:
        ref = ref[ref.gauge_id.isin(a.ids.split(","))]
    if a.first:
        ref = ref.head(a.first)
    (out / "run_info.json").write_text(
        json.dumps({"code": str(Path(m.__file__).resolve()), "area": a.area, "n": int(len(ref))}, indent=1)
    )
    for i, row in enumerate(ref.itertuples(index=False), 1):
        gid = row.gauge_id
        rec_path = out / "records" / f"{gid}.json"
        if rec_path.exists():
            continue
        exp = float(row.area_gages2) if a.area == "gages2" else None
        t0 = time.perf_counter()
        rec: dict = {"gauge_id": gid, "expected_area_km2": exp}
        try:
            r = m.merit_basins_hybrid_delineate(
                float(row.gauge_lat), float(row.gauge_lon), expected_area_km2=exp, auto_stage=False
            )
            rec.update(
                ok=True,
                polygon_area_km2=r.polygon_area_km2,
                snapped_upa_km2=r.snapped_upa_km2,
                snap_distance_m=r.snap_distance_m,
                snap_quality=r.snap_quality,
                terminal_catchment_id=r.terminal_catchment_id,
                upstream_catchment_count=r.upstream_catchment_count,
                terminal_refinement_used=r.terminal_refinement_used,
                refined_polygon_area_km2=r.refined_polygon_area_km2,
                vector_assembly_area_km2=r.vector_assembly_area_km2,
                quality_flags=r.quality_flags,
                fallback_history=r.fallback_history,
                outlet_assembly=(r.area_validation or {}).get("outlet_assembly"),
            )
            (out / "polys" / f"{gid}.geojson").write_text(r.gdf.to_crs(4326).to_json())
        except Exception as exc:  # recorded, not hidden: a cache miss is a failed gauge
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:600], tb=traceback.format_exc()[-800:])
        rec["secs"] = round(time.perf_counter() - t0, 2)
        rec_path.write_text(json.dumps(rec, indent=1, default=str))
        print(f"[{i}/{len(ref)}] {gid} ok={rec['ok']} {rec['secs']}s", flush=True)


if __name__ == "__main__":
    main()
