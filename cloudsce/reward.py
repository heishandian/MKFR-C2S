from __future__ import annotations

from typing import Optional

import numpy as np

from .specs import ModelParams


def _node_terms(frac: np.ndarray, mask: np.ndarray, hosted: np.ndarray,
                qos) -> tuple:
    cnt = np.maximum(mask.sum(axis=1), 1)
    fc = np.clip(frac, 0.0, 1.5) * mask
    ur = (fc * mask).sum(axis=1) / cnt
    var = (((fc - ur[:, None]) ** 2) * mask).sum(axis=1) / cnt
    cv = np.sqrt(np.maximum(var, 0.0)) / np.maximum(ur, 1e-9)
    blc = np.clip(1.0 - cv, 0.0, 1.0)
    blc = np.where(ur <= 1e-9, 0.0, blc)
    sla = qos.node_sla_from(frac, hosted, np.ones(len(frac), dtype=bool))
    binding = np.where(mask, frac, 0.0).max(axis=1)
    return ur, blc, sla, binding


def _cv(v: np.ndarray) -> float:
    mu = float(v.mean())
    if mu <= 1e-9:
        return 0.0
    return float(v.std() / mu)


def placement_delta_all(cluster, qos, params: ModelParams, demand: np.ndarray,
                        n_batch: int) -> np.ndarray:
    cap, use = cluster.cap_mat, cluster.use_mat
    mask = cap > 0
    safe = np.where(mask, cap, 1.0)
    frac0 = np.where(mask, use / safe, 0.0)
    frac1 = np.where(mask, (use + demand[None, :]) / safe, 0.0)

    ur0, blc0, sla0, b0 = _node_terms(frac0, mask, cluster.hosted_n, qos)
    ur1, blc1, sla1, b1 = _node_terms(frac1, mask, cluster.hosted_n + 1.0, qos)

    d_sla = (cluster.hosted_n + 1.0) * sla1 - cluster.hosted_n * sla0

    m = max(int(cluster.n), 1)
    s1, q1 = float(b0.sum()), float((b0 ** 2).sum())
    new_s = s1 - b0 + b1
    new_q = q1 - b0 ** 2 + b1 ** 2
    mu = new_s / m
    var = np.maximum(new_q / m - mu ** 2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew1 = np.where(mu > 1e-9, np.sqrt(var) / np.maximum(mu, 1e-9), 0.0)
    d_skew = skew1 - _cv(b0)

    n = max(int(n_batch), 1)
    r = (params.alpha * ((ur1 - ur0) + (blc1 - blc0))
         - params.beta * d_sla
         - params.w_skew * d_skew
         + (params.w_task + params.w_idle) / n)
    return np.asarray(r, dtype=np.float64)


def placement_delta(cluster, qos, params: ModelParams, demand: np.ndarray,
                    node_idx: int, n_batch: int) -> float:
    return float(placement_delta_all(cluster, qos, params, demand, n_batch)[node_idx])


def rejection_delta(params: ModelParams, n_batch: int) -> float:
    n = max(int(n_batch), 1)
    return float(-(params.w_task + params.w_idle) / n
                 - params.beta * params.w_reject / n)


def shape(cluster, qos, params: ModelParams, demand: np.ndarray,
          n_batch: int) -> np.ndarray:
    return placement_delta_all(cluster, qos, params, demand, n_batch)


def placement_reward(cluster, qos, node_idx: int, alpha: float, beta: float,
                     n_batch: int, w_task: float = 0.0, w_bal: float = 0.0,
                     skew_before: Optional[float] = None) -> float:
    ur = cluster.utilization_vector()
    blc = cluster.balance_vector()
    sla = qos.node_sla(cluster)
    scale = cluster.n / max(n_batch, 1)
    r = float(scale * (alpha * (ur[node_idx] + blc[node_idx]) - beta * sla[node_idx]))
    if w_task:
        r += float(w_task / max(n_batch, 1))
    if w_bal and skew_before is not None:
        r += float(w_bal * (skew_before - cluster.load_skew()))
    return float(r)
