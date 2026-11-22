from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from cloudsce.env import CloudsceEnv, Observation


class Scheduler:
    name: str = "base"
    family: str = "heuristic"
    trainable: bool = False

    def __init__(self, seed: int = 0, **kwargs):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.env: Optional[CloudsceEnv] = None

    def reset(self, env: CloudsceEnv) -> None:
        self.env = env

    def act(self, obs: Observation) -> int:
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, done, info) -> None:
        return None

    def end_episode(self) -> None:
        return None

    def on_round_end(self, info: Dict) -> None:
        return None

    @staticmethod
    def fallback(obs: Observation) -> int:
        cand = np.where(obs.mask)[0]
        return int(cand[0]) if cand.size else 0


class BatchPlanner(Scheduler):
    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(seed, **kwargs)
        self._plan: List[int] = []
        self._plan_key = None

    def act(self, obs: Observation) -> int:
        if not obs.mask.any():
            return 0
        key = (obs.meta.get("round"), len(self.env.batch))
        if self._plan_key != key or not self._plan or obs.meta.get("ptr", 0) >= len(self._plan):
            self._plan = self.make_plan(self.env, obs)
            self._plan_key = key
        ptr = int(obs.meta.get("ptr", 0))
        node = self._plan[ptr] if ptr < len(self._plan) else self.fallback(obs)
        if not obs.mask[node]:
            node = self.fallback(obs)
        return int(node)

    def make_plan(self, env: CloudsceEnv, obs: Observation) -> List[int]:
        raise NotImplementedError

    @staticmethod
    def shadow(env: CloudsceEnv) -> np.ndarray:
        return np.maximum(env.cluster.cap_mat - env.cluster.use_mat, 0.0)

    @staticmethod
    def pending(env: CloudsceEnv) -> List[tuple]:
        return list(env.batch[env.ptr:])

    @staticmethod
    def feasible_shadow(rem: np.ndarray, cap: np.ndarray, demand: np.ndarray,
                        healthy: np.ndarray) -> np.ndarray:
        m = np.all(rem + 1e-9 >= demand[None, :], axis=1)
        m &= np.all((demand[None, :] <= 1e-9) | (cap > 1e-9), axis=1)
        return m & healthy

    @staticmethod
    def affinity_bonus(env: CloudsceEnv, w, placement: Dict[int, List[int]]) -> np.ndarray:
        bonus = np.zeros(env.cluster.n)
        partners = tuple(w.affinity) + tuple(w.deps)
        if not partners:
            return bonus
        for p in partners:
            for node in placement.get(p, []):
                bonus[node] += 1.0
            cnt = env.cluster.wl_count.get(p)
            if cnt is not None:
                bonus += np.minimum(cnt, 1.0)
        return np.clip(bonus / max(len(partners), 1), 0, 1)
