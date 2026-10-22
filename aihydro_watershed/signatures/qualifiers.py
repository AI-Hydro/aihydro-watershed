"""Served-series retention and per-signature qualifier exposure (Science Wave 3, critique S3).

USGS daily values carry approval (``A`` approved / ``P`` provisional) and qualifier codes
(``e`` estimated; other tokens such as ``ICE`` are kept raw by aihydro-data). Whether a
signature is trustworthy depends on whether the days that DETERMINE it are estimated or
ice-affected, not on the share over the whole record: 2,100 of 7,305 days of 01013500 are
estimated, which matters far more for a low-flow quantile than for a mean.

``qualifier_exposure`` reports, per signature, the flagged share of its determining days;
the determining set is named in the output. The sets other than ``all_days`` and the exact
event masks are engineering definitions (bottom/top decile tails, days at or below the
median for baseflow) and are labelled as such.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TAIL_FRACTION = 0.10   # ENGINEERING CHOICE; SOURCE TO VERIFY: q5 / q95 determined by the 10% tail

#: signature -> (determining-day set, how it is defined)
DETERMINING_DAYS: Dict[str, tuple] = {
    "q_mean": ("all_days", "every valid day"),
    "q_std": ("all_days", "every valid day"),
    "q_median": ("all_days", "every valid day"),
    "flow_variability": ("all_days", "every valid day"),
    "runoff_ratio": ("all_days", "every valid streamflow day"),
    "stream_elas": ("all_days", "every valid streamflow day"),
    "slope_fdc": ("mid_range", "days between the 33rd and 66th percentile of q"),
    "slope_fdc_sawicz2011": ("mid_range", "days between the 33rd and 66th percentile of q"),
    "q5": ("low_tail", f"days with q <= the {TAIL_FRACTION:.0%} quantile (engineering definition)"),
    "q95": ("high_tail", f"days with q >= the {1 - TAIL_FRACTION:.0%} quantile (engineering definition)"),
    "low_q_freq": ("low_flow_events", "days with q <= 0.2 * mean(q) (the counted days)"),
    "low_q_dur": ("low_flow_events", "days with q <= 0.2 * mean(q) (the counted days)"),
    "high_q_freq": ("high_flow_events", "days with q > 9 * median(q) (the counted days)"),
    "high_q_dur": ("high_flow_events", "days with q > 9 * median(q) (the counted days)"),
    "zero_q_freq": ("zero_days", "days with q <= 0"),
    "baseflow_index": ("baseflow_days", "days with q <= median(q) (baseflow-dominated; engineering definition)"),
    "hfd_mean": ("all_days", "every valid day"),
    "half_flow_date_std": ("all_days", "every valid day"),
}


def _tokens(code) -> list:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return []
    return [t for t in str(code).split(",") if t]


def _masks(q: np.ndarray) -> Dict[str, np.ndarray]:
    mean_q, med = float(np.mean(q)), float(np.median(q))
    lo, hi = np.quantile(q, TAIL_FRACTION), np.quantile(q, 1 - TAIL_FRACTION)
    p33, p66 = np.quantile(q, 0.33), np.quantile(q, 0.66)
    return {
        "all_days": np.ones(q.size, bool),
        "mid_range": (q >= p33) & (q <= p66),
        "low_tail": q <= lo,
        "high_tail": q >= hi,
        "low_flow_events": q <= 0.2 * mean_q,
        "high_flow_events": q > 9.0 * med,
        "zero_days": q <= 0,
        "baseflow_days": q <= med,
    }


def qualifier_exposure(q_mm_day: pd.Series, flags: Optional[pd.DataFrame]) -> Optional[dict]:
    """Per-signature share of estimated / other-qualified / provisional days.

    ``flags`` is the frame the streamflow fetch returns (``approval_code``, ``qualifier_code``,
    indexed like the fetched series). ``None`` when there is nothing to report on (no flags:
    caller-supplied or modelled streamflow carries no USGS codes), recorded as such by the caller.
    """
    if flags is None or len(flags) == 0 or "qualifier_code" not in flags:
        return None
    qs = q_mm_day.dropna()
    fl = flags.copy()
    fl.index = pd.DatetimeIndex(fl.index).tz_localize(None) if getattr(fl.index, "tz", None) is not None else fl.index
    fl = fl[~fl.index.duplicated(keep="first")].reindex(qs.index)
    if fl["qualifier_code"].isna().all():
        return None
    q = qs.to_numpy(dtype=float)
    toks = [_tokens(c) for c in fl["qualifier_code"]]
    estimated = np.array(["e" in t for t in toks])
    other = np.array([any(x != "e" for x in t) for t in toks])
    any_flag = np.array([bool(t) for t in toks])
    prov = (fl["approval_code"].astype(str) == "P").to_numpy() if "approval_code" in fl else np.zeros(q.size, bool)
    masks = _masks(q)

    def _share(sel: np.ndarray, flag: np.ndarray) -> dict:
        n = int(sel.sum())
        k = int((flag & sel).sum())
        return {"n_days": n, "n_flagged": k, "share": (k / n) if n else None}

    by_set = {}
    for name, m in masks.items():
        by_set[name] = {"estimated": _share(m, estimated), "other_qualifier": _share(m, other),
                        "any_qualifier": _share(m, any_flag), "provisional": _share(m, prov)}
    counts: Dict[str, int] = {}
    for t in toks:
        for x in t:
            counts[x] = counts.get(x, 0) + 1
    per_signature = {}
    for sig, (setname, why) in DETERMINING_DAYS.items():
        per_signature[sig] = {"determining_days": setname, "definition": why, **by_set[setname]["any_qualifier"],
                              "estimated_share": by_set[setname]["estimated"]["share"],
                              "other_qualifier_share": by_set[setname]["other_qualifier"]["share"],
                              "provisional_share": by_set[setname]["provisional"]["share"]}
    return {
        "basis": "USGS approval and qualifier codes of the days the signatures were computed from",
        "n_days": int(q.size),
        "qualifier_token_counts": counts,
        "per_signature": per_signature,
        "sets": by_set,
        "note": "determining-day sets other than all_days / the counted event days are engineering "
                "definitions (source to verify); a high share on a determining set means that "
                "signature rests on estimated or ice-affected values",
    }


def served_series(q_cms: pd.Series, flags: Optional[pd.DataFrame]) -> dict:
    """The daily series as served (m3/s), with per-day qualifier and approval codes when known.

    Dates are ISO ``YYYY-MM-DD`` for daily data. ``qualifier_codes`` / ``approval_codes`` are
    ``None`` when the series carries none (caller-supplied or modelled). NaN becomes ``None``."""
    idx = pd.DatetimeIndex(q_cms.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    vals = q_cms.to_numpy(dtype=float)
    out = {
        "dates": [f"{d:%Y-%m-%d}" for d in idx],
        "q_cms": [float(v) if np.isfinite(v) else None for v in vals],
        "units": "m3/s",
        "qualifier_codes": None,
        "approval_codes": None,
    }
    if flags is not None and len(flags):
        fl = flags.copy()
        if getattr(fl.index, "tz", None) is not None:
            fl.index = pd.DatetimeIndex(fl.index).tz_localize(None)
        fl = fl[~fl.index.duplicated(keep="first")].reindex(idx)
        if "qualifier_code" in fl:
            out["qualifier_codes"] = ["" if (c is None or (isinstance(c, float) and np.isnan(c))) else str(c)
                                      for c in fl["qualifier_code"]]
        if "approval_code" in fl:
            out["approval_codes"] = ["" if (c is None or (isinstance(c, float) and np.isnan(c))) else str(c)
                                     for c in fl["approval_code"]]
    return out
