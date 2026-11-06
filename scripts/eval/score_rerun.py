#!/usr/bin/env python
"""Score ``rerun_hybrid.py`` outputs against the CAMELS polygon and ``area_gages2``.

usage: score_rerun.py --evidence <camels-delineation-1 dir> --arm name=<rerun dir> [--arm ...]
           --baseline <name> --p1 <name> --p2 <name> --out <dir>

Classification is the campaign's own (analyze.py): IoU against the CAMELS reference polygon (both in the
reference CRS, ``make_valid``), MATCHED when IoU >= 0.5, otherwise WRONG_BASIN; a gauge with no polygon is
NO_POLYGON. ``area_gages2`` has no geometry, so against it we report relative area error only; it is not an
IoU. Bootstrap: 10,000 resamples of gauges with replacement, seed 20261105, percentile 2.5/97.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

SEED, B = 20261105, 10_000


def load_arm(name: str, d: Path, ref: pd.DataFrame, polys: gpd.GeoDataFrame, tf) -> pd.DataFrame:
    rows = []
    for r in ref.itertuples(index=False):
        gid = r.gauge_id
        row = {"gauge_id": gid, "arm": name, "class": "NOT_RUN"}
        f = d / "records" / f"{gid}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            row["class"] = "NO_POLYGON"
            row["error"] = rec.get("error")
            row["flags"] = ";".join(rec.get("quality_flags") or [])
            row["mode"] = (rec.get("outlet_assembly") or {}).get("mode")
            row["snapped_upa"] = rec.get("snapped_upa_km2")
            row["snap_quality"] = rec.get("snap_quality")
            row["refinement_used"] = rec.get("terminal_refinement_used")
            row["secs"] = rec.get("secs")
            pf = d / "polys" / f"{gid}.geojson"
            if rec.get("ok") and pf.exists():
                feats = json.loads(pf.read_text())
                geoms = [shape(x["geometry"]) for x in feats["features"]]
                g = make_valid(shp_transform(tf.transform, gpd.GeoSeries(geoms, crs=4326).union_all()))
                ref_g = make_valid(polys.loc[gid, "geometry"])
                if not g.is_empty and g.area > 0:
                    inter = g.intersection(ref_g).area
                    row["IoU"] = inter / g.union(ref_g).area
                    row["class"] = "MATCHED" if row["IoU"] >= 0.5 else "WRONG_BASIN"
                    row["A_pred"] = g.area / 1e6
                    row["rel_poly"] = row["A_pred"] / r.area_polygon_km2 - 1
                    row["rel_gages2"] = row["A_pred"] / r.area_gages2 - 1
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("IoU", "A_pred", "rel_poly", "rel_gages2", "snapped_upa", "error", "flags", "mode", "snap_quality", "refinement_used"):
        if c not in df:
            df[c] = np.nan
    return df.merge(ref[["gauge_id", "area_class", "area_gages2", "ref_internal_consistent_10pct"]], on="gauge_id")


def ci(vals: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    s = vals[idx].mean(axis=1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def frac(mask: np.ndarray) -> dict:
    m = mask.astype(float)
    return {"k": int(m.sum()), "n": int(len(m)), "est": round(float(m.mean()), 4), "ci95": ci(m)}


def block(df: pd.DataFrame) -> dict:
    surv = df["class"].isin(["MATCHED", "WRONG_BASIN"]).to_numpy()
    e_poly = np.where(surv, df["rel_poly"].abs().fillna(np.inf), np.inf)
    e_g2 = np.where(surv, df["rel_gages2"].abs().fillna(np.inf), np.inf)
    wrong = (df["class"] == "WRONG_BASIN").to_numpy()
    # reference defect: the two references disagree by >10% and the tool polygon sits within 15% of area_gages2
    defect = wrong & ~df["ref_internal_consistent_10pct"].astype(bool).to_numpy() & (e_g2 <= 0.15)
    return {
        "n": int(len(df)),
        "no_polygon_or_not_run": int((~surv).sum()),
        "wrong_basin_iou_lt_0.5": frac(wrong),
        "wrong_basin_excluding_reference_defects": int((wrong & ~defect).sum()),
        "reference_defect_wrong_basins": int(defect.sum()),
        "within_5pct_of_camels_polygon": frac(e_poly <= 0.05),
        "within_10pct_of_camels_polygon": frac(e_poly <= 0.10),
        "within_5pct_of_area_gages2": frac(e_g2 <= 0.05),
        "within_10pct_of_area_gages2": frac(e_g2 <= 0.10),
        "area_gt50pct_off_area_gages2": int(((e_g2 > 0.5) & surv).sum()),
    }


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    """b minus a for within-5%-of-polygon (points) or wrong-basin rate."""
    m = a.merge(b, on="gauge_id", suffixes=("_a", "_b"))
    if col == "within5":
        va = (m["rel_poly_a"].abs() <= 0.05).to_numpy(float)
        vb = (m["rel_poly_b"].abs() <= 0.05).to_numpy(float)
    else:
        va = (m["class_a"] == "WRONG_BASIN").to_numpy(float)
        vb = (m["class_b"] == "WRONG_BASIN").to_numpy(float)
    d = vb - va
    return {"n": int(len(d)), "mean_diff_pts": round(100 * float(d.mean()), 2), "ci95_pts": [round(100 * x, 2) for x in ci(d)]}


def transitions(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on="gauge_id", suffixes=("_a", "_b"))
    fixed = m[(m.class_a == "WRONG_BASIN") & (m.class_b == "MATCHED")]
    broke = m[(m.class_a == "MATCHED") & (m.class_b == "WRONG_BASIN")]
    lost = m[(m.class_a.isin(["MATCHED", "WRONG_BASIN"])) & (m.class_b == "NO_POLYGON")]
    return {
        "wrong_to_matched": int(len(fixed)),
        "matched_to_wrong": int(len(broke)),
        "matched_to_wrong_ids": sorted(broke.gauge_id),
        "polygon_lost": int(len(lost)),
        "matched_to_wrong_reference_consistent": int(broke.ref_internal_consistent_10pct_a.astype(bool).sum()),
        "matched_to_wrong_reference_consistent_ids": sorted(broke[broke.ref_internal_consistent_10pct_a.astype(bool)].gauge_id),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=dir")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = Path(a.evidence)
    ref = pd.read_csv(ev / "reference/reference_671.csv", dtype={"gauge_id": str})
    polys = gpd.read_file(ev / "reference/reference_polygons.gpkg").set_index("GAGEID")
    tf = Transformer.from_crs(4326, polys.crs, always_xy=True)
    arms = {}
    for spec in a.arm:
        name, d = spec.split("=", 1)
        arms[name] = load_arm(name, Path(d), ref, polys, tf)
    out: dict = {"seed": SEED, "resamples": B, "arms": {}, "paired": {}}
    strata = {
        "all_671": lambda d: d,
        "lt100": lambda d: d[d.area_class == "L1_lt100"],
        "lt100_reference_consistent": lambda d: d[(d.area_class == "L1_lt100") & d.ref_internal_consistent_10pct.astype(bool)],
        "100_1000": lambda d: d[d.area_class == "L2_100-1000"],
        "ge1000": lambda d: d[d.area_class == "L3_ge1000"],
    }
    for name, df in arms.items():
        out["arms"][name] = {s: block(f(df)) for s, f in strata.items()}
        # P1 checks: polygon much larger than the snapped cell's own drainage area although refinement worked
        ok = df[df["class"].isin(["MATCHED", "WRONG_BASIN"]) & df.snapped_upa.notna()]
        big = ok[(ok.A_pred / ok.snapped_upa > 1.25) & (ok.refinement_used == True)]  # noqa: E712
        out["arms"][name]["polygon_over_1.25x_snapped_upa_with_refinement"] = {"n": int(len(big)), "ids": sorted(big.gauge_id)}
        out["arms"][name]["flag_counts"] = {
            k: int(df["flags"].fillna("").str.contains(k).sum())
            for k in ("OUTLET_OFF_TERMINAL_REACH", "HYBRID_LOCAL_ONLY", "HYBRID_AREA_VALIDATION_MISMATCH",
                      "TERMINAL_CATCHMENT_REFINEMENT_FAILED", "TERMINAL_CATCHMENT_LOCAL_WINDOW_USED",
                      "AREA_GATE_NO_MATCH", "SNAP_UPA_DIFFERS_FROM_EXPECTED")
        }
        out["arms"][name]["snap_quality_counts"] = {str(k): int(v) for k, v in df.snap_quality.value_counts().items()}
        out["arms"][name]["error_counts"] = {str(k)[:120]: int(v) for k, v in df.error.dropna().value_counts().items()}
    for label, (x, y) in {"P1_vs_baseline": (a.baseline, a.p1), "P2_vs_baseline": (a.baseline, a.p2), "P2_vs_P1": (a.p1, a.p2)}.items():
        out["paired"][label] = {}
        for s, f in strata.items():
            if s not in ("all_671", "lt100", "lt100_reference_consistent"):
                continue
            ax, ay = f(arms[x]), f(arms[y])
            out["paired"][label][s] = {
                "within5_polygon_gain": paired(ax, ay, "within5"),
                "wrong_basin_rate_change": paired(ax, ay, "wrong"),
                "transitions": transitions(ax, ay),
            }
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "rescore.json").write_text(json.dumps(out, indent=1))
    pd.concat(arms.values()).to_csv(outd / "per_gauge.csv", index=False, float_format="%.6g")
    print(json.dumps(out["paired"], indent=1))


if __name__ == "__main__":
    main()
