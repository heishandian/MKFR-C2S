from __future__ import annotations

from typing import Dict

import numpy as np

from models.encoders import NODE_DIM, VectorEncoder
from .drl_base import DRLScheduler


class TVWRL(DRLScheduler):
    name = "TVW-RL"
    family = "rl"

    def __init__(self, seed: int = 0, n_symbols: int = 512, **kw):
        enc = VectorEncoder(state_dim=128)
        super().__init__(enc, NODE_DIM, kind="rainbow", state_dim=128,
                         seed=seed, **kw)
        self.episode = 0
        self.alpha_t = 0.70
        self.beta_t = 0.30

    def reward_transform(self, reward: float, info: Dict) -> float:
        ur = info.get("ur_sum")
        sla = info.get("sla_sum")
        if ur is None or sla is None:
            return reward
        scale = self.env.cluster.n if self.env is not None else 1.0

        return float(self.alpha_t * ur / scale - self.beta_t * sla / scale)

    def end_episode(self) -> None:
        super().end_episode()
        self.episode += 1
        t = min(self.episode / 60.0, 1.0)
        self.alpha_t = 0.80 - 0.35 * t
        self.beta_t = 0.20 + 0.35 * t
