"""
Baseflow separation methods.

Provides two algorithms:
  - Lyne-Hollick recursive digital filter, in the Ladson et al. (2013) recipe
    (``lyne_hollick_ladson2013``; ``lyne_hollick`` is its array-in/array-out entry point)
  - UKIH (UK Institute of Hydrology five-day interval method, Gustard et al. 1992)

The MCP tool separate_baseflow writes results to the session's baseflow slot.
extract_hydrological_signatures computes its BFI scalar with the SAME kernel
(since 0.2.0), so the two values for one series agree.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

LH_ALPHA = 0.925
LH_PASSES = 3
LH_N_REFLECT = 30

#: Identity of the canonical kernel; surfaced in result params and quality flags.
#: Below this many days the filter's start-up transient is a large part of the record.
WARMUP_BIAS_DAYS = 365


class WarmupBiasWarning(UserWarning):
    """A Lyne-Hollick baseflow computed on a short series (31-364 days) is biased low by the
    filter's start-up transient (the initial quick flow equals the first flow)."""


def warmup_bias_disclosure(n_days: int) -> "dict | None":
    """``{"warmup_bias": True, ...}`` for a series of 31-364 days, else ``None``; for callers to record."""
    if LH_N_REFLECT < n_days < WARMUP_BIAS_DAYS:
        return {"warmup_bias": True, "n_days": int(n_days),
                "note": f"series shorter than {WARMUP_BIAS_DAYS} days: the baseflow index is biased low by "
                        "the filter start-up transient (the 30-day reflection only partly absorbs it)"}
    return None


LYNE_HOLLICK_LADSON2013_REFERENCE = (
    "Lyne & Hollick (1979); Nathan & McMahon (1990); Ladson et al. (2013) as implemented in "
    "TonyLadson/BaseflowSeparation_LyneHollick BFI.R and used by naddor/camels comp_i_bf_landson "
    "(alpha 0.925, 3 passes forward-backward-forward, 30-day reflection, quick flow subtracted only "
    "where positive, baseflow clipped at zero)"
)


def _ladson_segment(q: np.ndarray, alpha: float, passes: int, n_reflect: int) -> np.ndarray:
    """Baseflow of one gap-free segment, exactly as BFI.R ``BFI.calc``."""
    nr = n_reflect
    c = 0.5 * (1.0 + alpha)
    # Reflect nr days at each end (excluding the end point itself).
    x = np.concatenate([q[nr:0:-1], q, q[-2:-nr - 2:-1]])
    n = x.size

    def fwd_f(y: np.ndarray, f0: float) -> np.ndarray:
        f = np.empty(n)
        f[0] = f0
        for i in range(1, n):
            f[i] = alpha * f[i - 1] + c * (y[i] - y[i - 1])
        return f

    def bwd_f(y: np.ndarray) -> np.ndarray:
        f = np.empty(n)
        f[-1] = y[-1]
        for i in range(n - 2, -1, -1):
            f[i] = alpha * f[i + 1] + c * (y[i] - y[i + 1])
        return f

    # First pass: initial quick flow equals the flow itself (BFI.R: Qf1[1] <- Q[1]).
    f = fwd_f(x, x[0])
    b = np.where(f > 0, x - f, x)
    # Further passes alternate backward, forward, ...; the state is NEVER clamped, the
    # quick flow is subtracted only where positive (this is what distinguishes the
    # Ladson recipe from a per-step clamp-and-feed-back).
    for k in range(1, passes):
        f = bwd_f(b) if k % 2 == 1 else fwd_f(b, b[0])
        b = np.where(f > 0, b - f, b)
    b = b[nr:b.size - nr]
    b[b < 0] = 0.0
    return b


def lyne_hollick_ladson2013(
    streamflow: np.ndarray,
    alpha: float = LH_ALPHA,
    passes: int = LH_PASSES,
    n_reflect: int = LH_N_REFLECT,
) -> np.ndarray:
    """Canonical Lyne-Hollick baseflow filter (Ladson et al. 2013; CAMELS ``comp_i_bf_landson``).

    Reproduces the CAMELS v1.2 baseflow index to <= 2e-6 relative on 01013500 and 01031500
    (docs/vision-2040/evidence/signature-definitions-1).

    Recipe: alpha 0.925; ``passes`` alternating sweeps forward, backward, forward, ...; the
    series is reflected ``n_reflect`` (30) days at both ends; in each sweep the quick flow
    ``f(t) = alpha f(t-1) + (1+alpha)/2 (y(t) - y(t-1))`` is subtracted only where ``f > 0``
    and the (unclamped) result feeds the next sweep; the final baseflow is clipped at zero.

    A series of 31-364 days is computed but emits ``WarmupBiasWarning``; a series of 30 days or fewer
    raises ``ValueError``, as BFI.R does.

    NaN handling follows BFI.R: the filter runs on each run of more than ``n_reflect``
    consecutive finite values; days outside such runs return NaN. (BFI.R then averages segment
    BFIs weighted by length; here the caller sums baseflow and flow over the days that have
    a baseflow value, which is identical when there are no gaps.)

    ``passes`` is 3 in the recipe; BFI.R requires an odd count >= 3. Other positive counts
    are accepted here and simply continue the alternation (``passes=1`` is a single forward sweep).

    Returns a baseflow array the same length as ``streamflow``.
    """
    q = np.asarray(streamflow, dtype=float).ravel()
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if int(passes) < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    if q.size <= n_reflect:
        raise ValueError(
            f"lyne_hollick_ladson2013 needs more than n_reflect={n_reflect} days, got {q.size}")
    if q.size < WARMUP_BIAS_DAYS:
        warnings.warn(
            f"lyne_hollick_ladson2013 on {q.size} days (< {WARMUP_BIAS_DAYS}): the result carries a "
            "warm-up bias (BFI is biased low at short n); see warmup_bias_disclosure()",
            WarmupBiasWarning, stacklevel=2)
    out = np.full(q.size, np.nan)
    finite = np.isfinite(q)
    if finite.all():
        return _ladson_segment(q, alpha, int(passes), n_reflect)
    # Runs of consecutive finite values longer than n_reflect.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.view(np.int8), [0]])))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi - lo > n_reflect:
            out[lo:hi] = _ladson_segment(q[lo:hi], alpha, int(passes), n_reflect)
    return out


def lyne_hollick(
    streamflow: np.ndarray,
    alpha: float = LH_ALPHA,
    n_passes: int = LH_PASSES,
) -> np.ndarray:
    """
    Lyne-Hollick recursive digital filter for baseflow separation (Ladson et al. 2013 recipe).

    Since aihydro-watershed 0.2.0 this IS :func:`lyne_hollick_ladson2013`. Earlier versions
    clamped the baseflow at every step and fed the clamped value back into the recursion;
    that matches no reference implementation and read 5-19% low against CAMELS (BREAKING-VALUE,
    see CHANGELOG). The old behaviour is kept, deprecated, as :func:`lyne_hollick_legacy_clamp`.

    Parameters
    ----------
    streamflow : 1-D numpy array
        Daily streamflow series (any unit; negative values are set to zero).
    alpha : float
        Filter parameter (0.925 in the CAMELS recipe). Higher = more smoothing.
    n_passes : int
        Number of alternating sweeps (3 = forward-backward-forward, the recipe).

    Returns
    -------
    baseflow : 1-D numpy array, same length as streamflow.
    """
    q = np.asarray(streamflow, dtype=float)
    q = np.where(q < 0, 0.0, q)  # guard against negative values
    return lyne_hollick_ladson2013(q, alpha=alpha, passes=n_passes)


def lyne_hollick_legacy_clamp(
    streamflow: np.ndarray,
    alpha: float = 0.925,
    n_passes: int = 3,
) -> np.ndarray:
    """DEPRECATED pre-0.2.0 ``lyne_hollick``: baseflow-form recursion with ``b <= q`` clamped and fed back.

    Kept only so a caller can reproduce a pre-0.2.0 number. 5-19% below CAMELS on the verified
    basins; use :func:`lyne_hollick_ladson2013`.
    """
    warnings.warn(
        "lyne_hollick_legacy_clamp is deprecated and matches no reference implementation; "
        "use lyne_hollick_ladson2013", DeprecationWarning, stacklevel=2)
    q = np.asarray(streamflow, dtype=float)
    q = np.where(q < 0, 0.0, q)

    def _single_pass(q_in: np.ndarray, forward: bool) -> np.ndarray:
        data = q_in if forward else q_in[::-1]
        bf = np.empty_like(data)
        bf[0] = data[0]
        for t in range(1, len(data)):
            qf = alpha * bf[t - 1] + (1 - alpha) / 2 * (data[t] + data[t - 1])
            bf[t] = min(qf, data[t])
        return bf if forward else bf[::-1]

    bf = q.copy()
    for i in range(n_passes):
        bf = _single_pass(bf, i % 2 == 0)
    return np.clip(bf, 0.0, q)


def ukih(streamflow: np.ndarray) -> np.ndarray:
    """
    UKIH (UK Institute of Hydrology) five-day interval baseflow separation.

    Based on Gustard et al. (1992). Uses the minimum of each 5-day block
    as control points and connects them with straight lines.

    Parameters
    ----------
    streamflow : 1-D numpy array
        Daily streamflow series (any unit; must be non-negative).

    Returns
    -------
    baseflow : 1-D numpy array, same length as streamflow.
    """
    q = np.asarray(streamflow, dtype=float)
    q = np.where(q < 0, 0.0, q)
    n = len(q)
    block_size = 5
    n_blocks = n // block_size
    if n_blocks < 2:
        # Fallback: not enough data for UKIH; return zeros
        return np.zeros(n)

    # Find minimum in each block
    mins = []
    mins_idx = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        local_min_offset = int(np.argmin(q[start:end]))
        mins.append(q[start + local_min_offset])
        mins_idx.append(start + local_min_offset)

    # Keep only turning points (each min < 0.9 * adjacent mins)
    turning_q = [mins[0]]
    turning_idx = [mins_idx[0]]
    for i in range(1, len(mins) - 1):
        if 0.9 * mins[i] < mins[i - 1] and 0.9 * mins[i] < mins[i + 1]:
            turning_q.append(mins[i])
            turning_idx.append(mins_idx[i])
    turning_q.append(mins[-1])
    turning_idx.append(mins_idx[-1])

    # Interpolate between turning points
    bf = np.interp(np.arange(n), turning_idx, turning_q)
    bf = np.clip(bf, 0.0, q)
    return bf


def compute_bfi(streamflow: np.ndarray, baseflow: np.ndarray) -> float:
    """Baseflow Index = sum(baseflow) / sum(streamflow), clipped to [0, 1]."""
    total = np.nansum(streamflow)
    if total <= 0:
        return float("nan")
    return float(np.clip(np.nansum(baseflow) / total, 0.0, 1.0))
