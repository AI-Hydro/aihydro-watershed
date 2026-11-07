#!/usr/bin/env python
"""Condition B' of the camels-delineation-1 controlled test: B (Earth Engine unavailable) plus the offline flow-direction snap.

B is the hybrid with no Earth Engine and no accumulation raster, where every outlet was ``offline_unsnapped``
(Amendment 4: 82 of 671 wrong basins, 45.8% of those under 100 km2). B' is the same call with ``flowdir##.tif``
staged: the outlet is snapped on it inside the home unit catchment (``offline_flowdir_home_catchment``), and a
gauge for which no snap is possible stays unsnapped and carries the top-level ``reliability`` mark.

Offline by construction: every non-loopback socket is blocked, ``auto_stage=False`` so nothing is downloaded,
and the MERIT-Basins vectors and ``flowdir##.tif`` rasters are read from ``AIHYDRO_MERIT_DIR`` (``vector/`` and
``raster/flowdir_basins/``). A region whose raster is missing is NOT staged here: its gauges run unsnapped and are
reported as such, so stage all seven CONUS regions first for the full evaluation (99-304 MB each,
``MeritDataManager().auto_stage_flowdir(pfaf)``).

usage: condition_bprime.py --code <dir holding aihydro_watershed/> --evidence <camels-delineation-1 dir>
           --out <dir> [--pfaf 74,77] [--cached-only] [--area none|gages2] [--first N] [--ids a,b,c]

Writes ``<out>/records/<gauge>.json`` and ``<out>/polys/<gauge>.geojson`` (resumable) and, when it finishes,
``<out>/summary.json`` and ``<out>/per_gauge.csv``. Scoring is the campaign's own (``score_rerun.load_arm``: IoU
against the CAMELS polygon, WRONG_BASIN when IoU < 0.5), compared with the A and B rows of
``controlled/results/table.csv`` on the SAME gauges. Bootstrap: 10,000 resamples, seed 20261105.

The pre-registered 671-gauge evaluation is a separate run with its own protocol; numbers from this script on a subset
are a smoke check, not that evaluation.
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
        raise OSError(f"network blocked in offline run: {addr}")
    return _real_connect(self, addr, *a, **k)


def _block_gai(host, *a, **k):
    if host not in _LOOP:
        raise OSError(f"network blocked in offline run (DNS): {host}")
    return _real_gai(host, *a, **k)


socket.socket.connect = _block_connect
socket.getaddrinfo = _block_gai

CONTROLLED = "release_2.8.9_ws0.4.2/controlled"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pfaf", default=None, help="comma list of Pfafstetter level-2 regions to run")
    ap.add_argument("--cached-only", action="store_true", help="skip gauges whose region has no staged flowdir")
    ap.add_argument("--area", choices=("none", "gages2"), default="none")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import score_rerun as sr
    from pyproj import Transformer

    from aihydro_watershed.delineation import merit_flowdir_pipeline as m
    from aihydro_watershed.merit.merit_manager import MeritDataManager

    ev = Path(a.evidence)
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "polys").mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(ev / "reference/reference_671.csv", dtype={"gauge_id": str})
    mgr = MeritDataManager()
    ref["pfaf"] = [mgr.resolve_pfaf_region(r.gauge_lat, r.gauge_lon) for r in ref.itertuples()]
    if a.pfaf:
        ref = ref[ref.pfaf.isin(a.pfaf.split(","))]
    if a.cached_only:
        ref = ref[[mgr.verified_asset("flowdir", p) is not None for p in ref.pfaf]]
    if a.ids:
        ref = ref[ref.gauge_id.isin(a.ids.split(","))]
    if a.first:
        ref = ref.head(a.first)
    (out / "run_info.json").write_text(
        json.dumps({"code": str(Path(m.__file__).resolve()), "area": a.area, "n": int(len(ref)),
                    "pfaf": sorted(ref.pfaf.unique().tolist())}, indent=1)
    )
    for i, row in enumerate(ref.itertuples(index=False), 1):
        rec_path = out / "records" / f"{row.gauge_id}.json"
        if rec_path.exists():
            continue
        exp = float(row.area_gages2) if a.area == "gages2" else None
        t0 = time.perf_counter()
        rec: dict = {"gauge_id": row.gauge_id, "expected_area_km2": exp, "pfaf": row.pfaf}
        try:
            r = m.merit_basins_hybrid_delineate(
                float(row.gauge_lat), float(row.gauge_lon), expected_area_km2=exp,
                force_offline=True, allow_offline=True, auto_stage=False,
            )
            rec.update(
                ok=True, polygon_area_km2=r.polygon_area_km2, snapped_upa_km2=r.snapped_upa_km2,
                snap_distance_m=r.snap_distance_m, snap_quality=r.snap_quality, snap_source=r.snap_source,
                terminal_catchment_id=r.terminal_catchment_id, upstream_catchment_count=r.upstream_catchment_count,
                terminal_refinement_used=r.terminal_refinement_used,
                refined_polygon_area_km2=r.refined_polygon_area_km2,
                vector_assembly_area_km2=r.vector_assembly_area_km2, quality_flags=r.quality_flags,
                fallback_history=r.fallback_history,
                outlet_assembly=(r.area_validation or {}).get("outlet_assembly"),
                reliability=r.reliability, reliability_reason=r.reliability_reason,
                estimated_area_km2_on_reach=r.estimated_area_km2_on_reach, reliability_checks=r.reliability_checks,
                runtime_seconds=r.runtime_seconds,
            )
            (out / "polys" / f"{row.gauge_id}.geojson").write_text(r.gdf.to_crs(4326).to_json())
        except Exception as exc:  # recorded, not hidden
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:600], tb=traceback.format_exc()[-800:])
        rec["secs"] = round(time.perf_counter() - t0, 2)
        rec_path.write_text(json.dumps(rec, indent=1, default=str))
        print(f"[{i}/{len(ref)}] {row.gauge_id} ok={rec['ok']} {rec.get('snap_quality')} {rec['secs']}s", flush=True)

    # -- score against the CAMELS polygon and compare with A and B on the same gauges --------------------------------
    full_ref = pd.read_csv(ev / "reference/reference_671.csv", dtype={"gauge_id": str})
    polys = gpd.read_file(ev / "reference/reference_polygons.gpkg").set_index("GAGEID")
    tf = Transformer.from_crs(4326, polys.crs, always_xy=True)
    sub = full_ref[full_ref.gauge_id.isin(ref.gauge_id)]
    bp = sr.load_arm("Bp", out, sub, polys, tf)
    tab = pd.read_csv(ev / CONTROLLED / "results/table.csv", dtype={"gauge_id": str})
    comp = {}
    for cond in ("A", "B"):
        t = tab[(tab.condition == cond) & tab.gauge_id.isin(sub.gauge_id)].rename(columns={"rel_err_poly": "rel_poly"})
        comp[cond] = t[["gauge_id", "class", "rel_poly", "IoU", "area_class"]].reset_index(drop=True)

    def wrong(d):
        return int((d["class"] == "WRONG_BASIN").sum())

    def within5(d):
        return int((d["rel_poly"].abs() <= 0.05).sum())

    recs = {p.stem: json.loads(p.read_text()) for p in (out / "records").glob("*.json")}
    bp["reliability"] = bp.gauge_id.map(lambda g: recs[g].get("reliability"))
    bp["snap_quality"] = bp.gauge_id.map(lambda g: recs[g].get("snap_quality"))
    bp["secs"] = bp.gauge_id.map(lambda g: recs[g].get("secs"))
    summary: dict = {"n": int(len(bp)), "seed": sr.SEED, "resamples": sr.B, "strata": {}}
    for label, mask in {
        "all": lambda d: d.area_class.notna(),
        "lt100": lambda d: d.area_class == "L1_lt100",
        "100_1000": lambda d: d.area_class == "L2_100-1000",
        "ge1000": lambda d: d.area_class == "L3_ge1000",
    }.items():
        sel = set(bp[mask(bp)].gauge_id)
        b_ = bp[bp.gauge_id.isin(sel)]
        entry = {
            "n": len(sel),
            "wrong_Bprime": wrong(b_),
            "wrong_B": wrong(comp["B"][comp["B"].gauge_id.isin(sel)]),
            "wrong_A": wrong(comp["A"][comp["A"].gauge_id.isin(sel)]),
            "within5_Bprime": within5(b_),
            "within5_B": within5(comp["B"][comp["B"].gauge_id.isin(sel)]),
            "within5_A": within5(comp["A"][comp["A"].gauge_id.isin(sel)]),
        }
        if len(sel) > 1:
            for name, other in (("B", comp["B"]), ("A", comp["A"])):
                o = other[other.gauge_id.isin(sel)]
                entry[f"paired_wrong_Bprime_minus_{name}"] = sr.paired(o, b_, "wrong")
                m_ = o.merge(b_, on="gauge_id", suffixes=("_o", "_b"))
                entry[f"discordant_wrong_Bprime_vs_{name}"] = [
                    int(((m_.class_b == "WRONG_BASIN") & (m_.class_o == "MATCHED")).sum()),  # B' worse
                    int(((m_.class_b == "MATCHED") & (m_.class_o == "WRONG_BASIN")).sum()),  # B' better
                ]
        summary["strata"][label] = entry
    summary["snap_quality_counts"] = {str(k): int(v) for k, v in bp.snap_quality.value_counts(dropna=False).items()}
    summary["reliability_counts"] = {str(k): int(v) for k, v in bp.reliability.value_counts(dropna=False).items()}
    uns = bp[bp.snap_quality == "offline_unsnapped"]
    summary["still_unsnapped"] = {
        "n": int(len(uns)),
        "wrong": wrong(uns),
        "flagged_unreliable": int((uns.reliability == "unreliable").sum()),
        "wrong_and_flagged": int(((uns["class"] == "WRONG_BASIN") & (uns.reliability == "unreliable")).sum()),
    }
    summary["median_secs_per_gauge"] = float(np.nanmedian(bp.secs.astype(float)))
    summary["max_secs_per_gauge"] = float(np.nanmax(bp.secs.astype(float)))
    summary["not_run_or_no_polygon"] = int((~bp["class"].isin(["MATCHED", "WRONG_BASIN"])).sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    bp.to_csv(out / "per_gauge.csv", index=False, float_format="%.6g")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
