from __future__ import annotations

from typing import Dict, List

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from .base import Scheduler


class KubeScheduler(Scheduler):
    name = "Kube-scheduler"
    family = "production"

    def act(self, obs: Observation) -> int:
        if not obs.mask.any():
            return 0
        cl = self.env.cluster
        cand = np.where(obs.mask)[0]
        dem = obs.instance.demand
        cap = cl.cap_mat[cand]
        use = cl.use_mat[cand]
        after = (use + dem) / np.maximum(cap, 1e-9)

        balance = 1.0 - after.std(axis=1)
        score = 0.8 * (1.0 - after.mean(axis=1)) + 0.2 * balance
        return int(cand[int(np.argmax(score))])


class YuniKornScheduler(Scheduler):
    name = "YuniKorn"
    family = "production"

    def act(self, obs: Observation) -> int:
        if not obs.mask.any():
            return 0
        cl = self.env.cluster
        cand = np.where(obs.mask)[0]
        dem = obs.instance.demand
        cap = cl.cap_mat[cand]
        use = cl.use_mat[cand]
        ratio = (use + dem) / np.maximum(cap, 1e-9)
        ratio = ratio.max(axis=1)
        return int(cand[int(np.argmin(ratio))])


PRODUCTION_SCHEDULERS = {"Kube-scheduler": KubeScheduler, "YuniKorn": YuniKornScheduler}
