from __future__ import annotations

from typing import Dict

from models.encoders import BiGCNEncoder, NODE_DIM
from .drl_base import DRLScheduler


class HeHiFLDSA(DRLScheduler):
    name = "HeHiFLDSA"
    family = "rl"

    def __init__(self, seed: int = 0, n_symbols: int = 512, **kw):
        enc = BiGCNEncoder(state_dim=128, dim=64, n_symbols=n_symbols)
        kw.setdefault("rollout", 96)
        kw.setdefault("epochs", 4)
        super().__init__(enc, NODE_DIM, kind="ppo", state_dim=128,
                         seed=seed, **kw)

    def reward_transform(self, reward: float, info: Dict) -> float:
        n = self.env.cluster.n if self.env is not None else 1
        return float(reward / max(n, 1))
