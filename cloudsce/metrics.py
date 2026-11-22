from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def avg_util(util_series: Sequence[float]) -> float:
    return float(np.mean(util_series)) if len(util_series) else 0.0


def avg_imbalance(iblc_series: Sequence[float]) -> float:
    return float(np.mean(iblc_series)) if len(iblc_series) else 0.0


def asid(affi_pairs: int, affi_ok: int) -> float:
    if affi_pairs <= 0:
        return 0.0
    return float(1.0 - affi_ok / affi_pairs)


def aaid(anti_pairs: int, anti_viol: int) -> float:
    if anti_pairs <= 0:
        return 0.0
    return float(anti_viol / anti_pairs)


def convergence_speed(curve: Sequence[float], tol: float = 0.05) -> int:
    c = np.asarray(curve, dtype=float)
    if c.size < 5:
        return int(c.size)
    k = max(3, c.size // 20)
    smooth = np.convolve(c, np.ones(k) / k, mode="valid")
    final = float(np.mean(smooth[-max(3, len(smooth) // 10):]))
    band = tol * max(abs(final), 1e-9)
    out = np.where(np.abs(smooth - final) > band)[0]
    return int(out[-1] + 1) if out.size else 0


def mean_ci(x: Sequence[float], z: float = 1.96) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean()) if x.size else 0.0
    s = float(x.std(ddof=1)) if x.size > 1 else 0.0
    half = z * s / np.sqrt(max(x.size, 1))
    return {"mean": m, "std": s, "ci_low": m - half, "ci_high": m + half, "n": int(x.size)}


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    d = a[:n] - b[:n]
    md = float(d.mean()) if n else 0.0
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(max(n, 1))
    p = _welch_p(d)
    return {"md": md, "ci_low": md - half, "ci_high": md + half, "p": p, "n": n}


def _welch_p(d: np.ndarray) -> float:
    if d.size < 3 or np.allclose(d, d[0]):
        return 1.0 if np.allclose(d, 0) else 0.0
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))

    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def p_from_ci(md: float, half: float, z: float = 1.96) -> float:
    from math import erfc, sqrt
    if half <= 1e-12:
        return 0.0 if abs(md) > 0 else 1.0
    t = abs(md) / (half / z)
    return float(erfc(t / sqrt(2.0)))


def summarize(histories: List[Dict[str, float]]) -> Dict[str, float]:
    keys = histories[0].keys() if histories else []
    return {k: float(np.mean([h.get(k, 0.0) for h in histories])) for k in keys}
