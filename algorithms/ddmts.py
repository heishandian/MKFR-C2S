from __future__ import annotations

from typing import List

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from .base import BatchPlanner


class DDMTS(BatchPlanner):
    name = "DDMTS"
    family = "heuristic"
    trainable = False

    def make_plan(self, env: CloudsceEnv, obs: Observation) -> List[int]:
        cl = env.cluster
        rem = self.shadow(env)
        cap = cl.cap_mat
        healthy = cl.healthy
        used = np.maximum(cap - rem, 0.0)
        plan: List[int] = []

        pending = self.pending(env)

        def laxity(pair):
            w, inst = pair
            return w.deadline - w.duration - w.arrival
        pending.sort(key=laxity)

        for w, inst in pending:
            mask = self.feasible_shadow(rem, cap, inst.demand, healthy)
            cand = np.where(mask)[0]
            if cand.size == 0:
                plan.append(-1)
                continue

            load = np.where(cap[cand] > 0, used[cand] / np.maximum(cap[cand], 1e-9), 0.0)
            score = load.mean(axis=1) + 0.25 * load.max(axis=1)

            score = score + 5.0 * np.clip(score - 0.88, 0, None)
            node = int(cand[int(np.argmin(score))])
            rem[node] = np.maximum(rem[node] - inst.demand, 0.0)
            used[node] += inst.demand
            plan.append(node)
        return plan
