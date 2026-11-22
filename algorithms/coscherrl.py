from __future__ import annotations

from typing import Dict

from models.encoders import NODE_DIM, RelationalEncoder
from .drl_base import DRLScheduler


class CoScheRRL(DRLScheduler):
    name = "Co-ScheRRL"
    family = "rl"

    def __init__(self, seed: int = 0, **kw):
        enc = RelationalEncoder(state_dim=128, dim=96, heads=2)
        kw.setdefault("rollout", 64)
        super().__init__(enc, NODE_DIM, kind="a2c", state_dim=128, seed=seed, **kw)

    def reward_transform(self, reward: float, info: Dict) -> float:
        n = self.env.cluster.n if self.env is not None else 1
        return float(reward / max(n, 1))
