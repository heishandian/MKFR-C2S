from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from cloudsce.env import CloudsceEnv, Observation


def shortlist(env: CloudsceEnv, obs: Observation, k: int = 48) -> np.ndarray:
    cl = env.cluster
    cand = np.where(obs.mask)[0]
    if cand.size <= k:
        return cand
    rem = np.maximum(cl.cap_mat[cand] - cl.use_mat[cand], 0.0)
    rate = np.where(cl.cap_mat[cand] > 0,
                    rem / np.maximum(cl.cap_mat[cand], 1e-9), 0.0)
    score = rate.mean(axis=1)
    w = obs.workload
    if w is not None:
        bonus = np.zeros(cand.size)
        for p in tuple(w.affinity) + tuple(w.deps):
            cnt = cl.wl_count.get(p)
            if cnt is not None:
                bonus += np.minimum(cnt[cand], 1.0)
        score = score + 0.35 * bonus / max(len(tuple(w.affinity) + tuple(w.deps)), 1)
    order = np.argsort(-score)[:k]
    return cand[order]


def demands_of(pending: Sequence[Tuple[object, object]]) -> np.ndarray:
    return np.stack([inst.demand for _, inst in pending]) if pending else np.zeros((0, 6))


def simulate(assign: np.ndarray, demands: np.ndarray, base_use: np.ndarray,
             cap: np.ndarray, nodes: np.ndarray) -> Tuple[float, float]:
    use = base_use.copy()
    np.add.at(use, nodes[assign], demands)
    mask = cap > 0
    frac = np.where(mask, use / np.maximum(cap, 1e-9), 0.0)
    frac = np.clip(frac, 0, 1.5)
    cnt = np.maximum(mask.sum(axis=1), 1)
    ur = (frac * mask).sum(axis=1) / cnt
    mu = ur
    var = (((frac - ur[:, None]) ** 2) * mask).sum(axis=1) / cnt
    cv = np.sqrt(np.maximum(var, 0)) / np.maximum(mu, 1e-9)
    blc = np.clip(1.0 - cv, 0.0, 1.0)
    active = (use.sum(axis=1) > 0) | (base_use.sum(axis=1) > 0)
    if not active.any():
        return 0.0, 0.0
    return float(ur[active].mean()), float(1.0 - blc[active].mean())


def objective(assign: np.ndarray, demands: np.ndarray, base_use: np.ndarray,
              cap: np.ndarray, nodes: np.ndarray, affi: np.ndarray,
              w=(0.45, 0.35, 0.20)) -> float:
    ur, iblc = simulate(assign, demands, base_use, cap, nodes)
    affi_pen = 1.0 - float(np.mean(affi[assign])) if len(assign) else 0.0
    return w[0] * (1.0 - ur) + w[1] * iblc + w[2] * affi_pen


def affinity_scores(env: CloudsceEnv, obs: Observation, nodes: np.ndarray) -> np.ndarray:
    w = obs.workload
    out = np.zeros(nodes.size)
    if w is None:
        return out
    partners = tuple(w.affinity) + tuple(w.deps)
    if not partners:
        return out
    for p in partners:
        cnt = env.cluster.wl_count.get(p)
        if cnt is not None:
            out += np.minimum(cnt[nodes], 1.0)
    return np.clip(out / max(len(partners), 1), 0, 1)
