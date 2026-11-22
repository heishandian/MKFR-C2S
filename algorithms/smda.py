from __future__ import annotations

from typing import Dict, List

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from .base import BatchPlanner


class SMDA(BatchPlanner):
    name = "SMDA"
    family = "heuristic"
    trainable = False

    def make_plan(self, env: CloudsceEnv, obs: Observation) -> List[int]:
        cl = env.cluster
        rem = self.shadow(env)
        cap = cl.cap_mat
        healthy = cl.healthy
        plan: List[int] = []

        pending = self.pending(env)
        pending.sort(key=lambda pair: (-pair[0].duration * pair[0].n_instances, pair[1].iid))

        placement: Dict[int, List[int]] = {}
        for w, inst in pending:
            mask = self.feasible_shadow(rem, cap, inst.demand, healthy)
            cand = np.where(mask)[0]
            if cand.size == 0:
                plan.append(-1)
                continue

            load = rem[cand].sum(axis=1)
            finish = inst.demand.sum() / (load + 1e-6)
            node = int(cand[int(np.argmin(finish))])
            rem[node] = np.maximum(rem[node] - inst.demand, 0.0)
            plan.append(node)
            placement.setdefault(w.wid, []).append(node)
        return plan
