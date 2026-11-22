from __future__ import annotations

from typing import Dict, List

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from .base import BatchPlanner
from .swarm_utils import (affinity_scores, demands_of, objective, shortlist)

N_ANTS = 6
N_ITER = 4
ALPHA = 1.0
BETA = 2.2
RHO = 0.25
Q = 1.0


class MOTSACO(BatchPlanner):
    name = "MOTS-ACO"
    family = "swarm"
    trainable = False

    def __init__(self, seed: int = 0, **kw):
        super().__init__(seed, **kw)
        self.tau: Dict[int, np.ndarray] = {}

    def _tau(self, cid: int, n_types: int) -> np.ndarray:
        t = self.tau.get(cid)
        if t is None or t.size != n_types:
            t = np.ones(n_types)
            self.tau[cid] = t
        return t

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
        ntypes = int(cl.type_idx.max()) + 1
        types = cl.type_idx[nodes]

        best_assign, best_cost = None, np.inf
        for _ in range(N_ITER):
            for _ant in range(N_ANTS):
                assign = self._construct(pending, dem, nodes, cap, base_use, affi, types, ntypes)
                cost = objective(assign, dem, base_use, cap, nodes, affi)
                if cost < best_cost:
                    best_cost, best_assign = cost, assign.copy()
                    self._deposit(pending, nodes, assign, types, ntypes, Q / (cost + 1e-6))
            for cid, t in self.tau.items():
                t *= (1.0 - RHO)
                np.clip(t, 1e-3, 1e3, out=t)
        return [int(nodes[a]) for a in best_assign]

    def _construct(self, pending, dem, nodes, cap, base_use, affi, types, ntypes
                   ) -> np.ndarray:
        rem = np.maximum(cap[nodes] - base_use[nodes], 0.0)
        capn = np.maximum(cap[nodes], 1e-9)
        assign = np.zeros(len(dem), dtype=np.int64)
        for i, (w, _inst) in enumerate(pending):
            feas = np.all(rem + 1e-9 >= dem[i][None, :], axis=1)
            feas &= np.all((dem[i][None, :] <= 1e-9) | (cap[nodes] > 1e-9), axis=1)
            if not feas.any():
                assign[i] = int(np.argmax(rem.mean(axis=1)))
                continue
            room = np.where(feas, rem.mean(axis=1), 0.0)

            prof = np.where(feas, 1.0 - np.abs(rem / capn - dem[i] / capn).mean(axis=1), 0.0)
            eta = 0.55 * room + 0.30 * affi + 0.15 * prof
            tau = self._tau(w.cid, ntypes)[types]
            prob = np.power(np.maximum(tau, 1e-6), ALPHA) * np.power(np.maximum(eta, 1e-6), BETA)
            prob = np.where(feas, prob, 0.0)
            s = prob.sum()
            j = int(self.rng.choice(nodes.size, p=prob / s)) if s > 0 else int(np.argmax(np.where(feas, room, -1)))
            assign[i] = j
            rem[j] = np.maximum(rem[j] - dem[i], 0.0)
        return assign

    def _deposit(self, pending, nodes, assign, types, ntypes, delta: float) -> None:
        for (w, _), j in zip(pending, assign):
            self._tau(w.cid, ntypes)[types[j]] += delta
