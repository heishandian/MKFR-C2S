from __future__ import annotations

from typing import List

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from .base import BatchPlanner
from .swarm_utils import affinity_scores, demands_of, objective, shortlist

N_PART = 8
N_ITER = 5
W_INERTIA = 0.72
C1 = 1.49
C2 = 1.49
P_CROSS = 0.35
P_MUT = 0.08


class PSOPGA(BatchPlanner):
    name = "PSO-PGA"
    family = "swarm"
    trainable = False

    def make_plan(self, env: CloudsceEnv, obs: Observation) -> List[int]:
        cl = env.cluster
        pending = self.pending(env)
        if not pending:
            return []
        nodes = shortlist(env, obs, k=48)
        if nodes.size == 0:
            return [-1] * len(pending)
        dem = demands_of(pending)
        cap = cl.cap_mat
        base_use = cl.use_mat.copy()
        affi = affinity_scores(env, obs, nodes)
        n, m = len(dem), nodes.size

        seed = self._greedy(pending, dem, nodes, cap, base_use, affi)
        pos = np.stack([seed] + [self._greedy(pending, dem, nodes, cap, base_use, affi, noise=1.0)
                                 for _ in range(N_PART - 1)])
        vel = np.zeros((N_PART, n))
        fit = np.asarray([objective(p, dem, base_use, cap, nodes, affi) for p in pos])
        pbest, pbest_fit = pos.copy(), fit.copy()
        g = int(np.argmin(fit)); gbest, gbest_fit = pos[g].copy(), fit[g]

        for _ in range(N_ITER):
            r1, r2 = self.rng.random((N_PART, n)), self.rng.random((N_PART, n))
            vel = (W_INERTIA * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest[None, :] - pos))
            pos = np.clip(np.round(pos + vel), 0, m - 1).astype(np.int64)

            for i in range(N_PART):
                if self.rng.random() < P_CROSS:
                    cut = self.rng.integers(1, max(2, n - 1))
                    pos[i, cut:] = gbest[cut:]
                mut = self.rng.random(n) < P_MUT
                if mut.any():
                    pos[i, mut] = self.rng.integers(0, m, size=int(mut.sum()))
            fit = np.asarray([objective(p, dem, base_use, cap, nodes, affi) for p in pos])
            better = fit < pbest_fit
            pbest[better], pbest_fit[better] = pos[better], fit[better]
            j = int(np.argmin(pbest_fit))
            if pbest_fit[j] < gbest_fit:
                gbest, gbest_fit = pbest[j].copy(), pbest_fit[j]
        return [int(nodes[a]) for a in gbest]

    def _greedy(self, pending, dem, nodes, cap, base_use, affi, noise: float = 0.0) -> np.ndarray:
        rem = np.maximum(cap[nodes] - base_use[nodes], 0.0)
        assign = np.zeros(len(dem), dtype=np.int64)
        for i in range(len(dem)):
            feas = np.all(rem + 1e-9 >= dem[i][None, :], axis=1)
            feas &= np.all((dem[i][None, :] <= 1e-9) | (cap[nodes] > 1e-9), axis=1)
            if not feas.any():
                assign[i] = int(np.argmax(rem.mean(axis=1)))
                continue
            score = np.where(feas, rem.mean(axis=1) + 0.4 * affi, -1e9)
            if noise > 0:
                score = score + self.rng.normal(0, noise, size=score.shape)
            j = int(np.argmax(score))
            assign[i] = j
            rem[j] = np.maximum(rem[j] - dem[i], 0.0)
        return assign
