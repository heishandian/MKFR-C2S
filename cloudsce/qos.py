from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .cluster import Cluster
from .specs import ModelParams


BASE_SERVICE_MS = 7.0

QPS_PER_CORE = 0.55

RHO_KNEE = 0.82

RHO_FAIL = 0.90


@dataclass
class QoSample:
    rt_ms: float
    throughput_qps: float
    failure_rate: float


class QoSModel:
    def __init__(self, params: ModelParams, rt_threshold_ms: float = 300.0,
                 th_threshold_qps: float = 9.6):
        self.p = params
        self.rt_thr = rt_threshold_ms
        self.th_thr = th_threshold_qps

    def node_stats(self, cluster: Cluster) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = cluster.cap_mat > 0
        frac = np.where(mask, cluster.use_mat / np.where(mask, cluster.cap_mat, 1.0), 0.0)
        frac = np.clip(frac, 0.0, 1.2)
        rho = frac.max(axis=1) if frac.size else np.zeros(cluster.n)
        interference = 1.0 + 0.02 * cluster.hosted_n
        rho_eff = np.minimum(rho * interference, 0.975)
        svc = BASE_SERVICE_MS * interference
        rt = svc / np.maximum(1.0 - rho_eff, 0.025)
        cores = np.maximum(cluster.cap_mat[:, 0], 1.0)
        th = cores * QPS_PER_CORE * np.maximum(0.0, 1.0 - rho_eff ** 4) / interference
        fp = np.where(rho_eff < 0.88, 0.0, np.minimum((rho_eff - 0.88) * 2.5, 0.95))
        dead = ~cluster.healthy
        rt[dead] = 6.0 * self.rt_thr
        th[dead] = 0.0
        fp[dead] = 1.0
        return rt, th, fp

    def sample(self, cluster: Cluster) -> QoSample:
        rt, th, fp = self.node_stats(cluster)
        busy = cluster.hosted_n > 0
        if not busy.any():
            return QoSample(0.0, 0.0, 0.0)
        return QoSample(
            rt_ms=float(np.percentile(rt[busy], 95)),
            throughput_qps=float(np.mean(th[busy]) * cluster.n),
            failure_rate=float(np.mean(fp[busy])),
        )

    def node_sla_from(self, frac: np.ndarray, hosted: np.ndarray,
                      healthy: np.ndarray) -> np.ndarray:
        rho = np.clip(frac, 0.0, 1.2).max(axis=1) if len(frac) else np.zeros(0)
        interference = 1.0 + 0.02 * hosted
        tr = np.clip((rho - RHO_KNEE) / (1.0 - RHO_KNEE), 0.0, 1.0) * interference
        hr = np.clip((RHO_KNEE - rho) / RHO_KNEE, 0.0, 1.0)
        fr = np.clip((rho - RHO_FAIL) / (1.0 - RHO_FAIL), 0.0, 1.0)
        sla = self.p.lam1 * tr + self.p.lam2 * hr + self.p.lam3 * fr
        sla = np.where(healthy, sla, 1.0)
        return np.clip(sla, 0.0, 1.0)

    def node_sla(self, cluster: Cluster) -> np.ndarray:
        mask = cluster.cap_mat > 0
        frac = np.where(mask, cluster.use_mat / np.where(mask, cluster.cap_mat, 1.0), 0.0)
        return self.node_sla_from(frac, cluster.hosted_n, cluster.healthy)

    def sla(self, cluster: Cluster, reject_ratio: float = 0.0
            ) -> Tuple[float, float, float, float]:
        rt, th, fp = self.node_stats(cluster)
        busy = cluster.hosted_n > 0
        if not busy.any():
            return 0.0, 0.0, 0.0, 0.0
        rt_p95 = float(np.percentile(rt[busy], 95))
        th_mean = float(np.mean(th[busy]))
        node_fr = float(np.mean(fp[busy]))
        fr = reject_ratio + (1.0 - reject_ratio) * node_fr
        tr = max(0.0, (rt_p95 - self.rt_thr) / max(rt_p95, 1e-6))
        hr = max(0.0, (self.th_thr - th_mean) / self.th_thr)
        sla = self.p.lam1 * tr + self.p.lam2 * hr + self.p.lam3 * fr
        return float(sla), float(tr), float(hr), float(fr)
