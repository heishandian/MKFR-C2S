from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .specs import K, NODE_TYPE_INDEX, NODE_TYPES, RESOURCE_NAMES


@dataclass
class Node:
    nid: int
    type_name: str
    cap: np.ndarray
    usage: np.ndarray = field(default_factory=lambda: np.zeros(K))
    frac: np.ndarray = field(default_factory=lambda: np.zeros(K))
    hosted: List[int] = field(default_factory=list)
    n_by_class: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=np.int32))
    n_by_workload: dict = field(default_factory=dict)
    healthy: bool = True
    status: str = "ok"
    ready_time: float = 0.0

    @property
    def type_index(self) -> int:
        return NODE_TYPE_INDEX[self.type_name]

    @property
    def is_accelerator(self) -> bool:
        return bool(self.cap[1] > 0.0)

    def remaining(self) -> np.ndarray:
        return np.maximum(self.cap - self.usage, 0.0)

    def remaining_rate(self) -> np.ndarray:
        if not self.healthy:
            return np.zeros(K)
        safe = np.where(self.cap > 0, self.cap, 1.0)
        r = 1.0 - self.usage / safe
        r = np.clip(r, 0.0, 1.0)
        r[self.cap <= 0] = 0.0
        return r

    def utilization(self) -> float:
        mask = self.cap > 0
        if not mask.any():
            return 0.0
        return float(np.clip(self.frac[mask], 0.0, 1.5).max())

    def balance(self) -> float:
        mask = self.cap > 0
        if not mask.any():
            return 0.0
        f = np.clip(self.frac[mask], 0.0, 1.5)
        if f.size < 2:
            return 1.0 - min(abs(f[0] - 1.0), 1.0) * 0.0
        mu = float(f.mean())
        if mu <= 1e-9:
            return 0.0
        cv = float(f.std() / mu)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def can_fit(self, demand: np.ndarray, slack: float = 1.0) -> bool:
        if not self.healthy:
            return False
        need = demand * slack
        avail = self.remaining()

        if np.any((need > 1e-9) & (self.cap <= 1e-9)):
            return False
        return bool(np.all(avail + 1e-9 >= need))

    def allocate(self, demand: np.ndarray, inst_id: int, wl_id: int, cid: int) -> None:
        self.usage += demand
        safe = np.where(self.cap > 0, self.cap, 1.0)
        self.frac = np.where(self.cap > 0, self.usage / safe, 0.0)
        self.hosted.append(inst_id)
        self.n_by_class[cid] += 1
        self.n_by_workload[wl_id] = self.n_by_workload.get(wl_id, 0) + 1

    def release(self, demand: np.ndarray, inst_id: int, wl_id: int, cid: int) -> None:
        self.usage = np.maximum(self.usage - demand, 0.0)
        safe = np.where(self.cap > 0, self.cap, 1.0)
        self.frac = np.where(self.cap > 0, self.usage / safe, 0.0)
        if inst_id in self.hosted:
            self.hosted.remove(inst_id)
        self.n_by_class[cid] = max(0, self.n_by_class[cid] - 1)
        if wl_id in self.n_by_workload:
            self.n_by_workload[wl_id] -= 1
            if self.n_by_workload[wl_id] <= 0:
                del self.n_by_workload[wl_id]

    def feature(self, wl_id: Optional[int] = None, dep_wl_ids=()) -> np.ndarray:
        rem = self.remaining_rate()
        base = [
            *rem.tolist(),
            self.utilization(),
            self.balance(),
            1.0 if self.healthy else 0.0,
            1.0 if self.is_accelerator else 0.0,
        ]

        oh = np.zeros(len(NODE_TYPES))
        oh[self.type_index] = 1.0
        affinity = 0.0
        dep = 0.0
        if wl_id is not None:
            affinity = min(self.n_by_workload.get(wl_id, 0) / 4.0, 1.0)
        for d in dep_wl_ids:
            dep += min(self.n_by_workload.get(d, 0) / 2.0, 1.0)
        dep = min(dep / max(len(tuple(dep_wl_ids)), 1), 1.0)
        load_ratio = min(len(self.hosted) / 32.0, 1.0)
        return np.asarray(base + oh.tolist() + [affinity, dep, load_ratio], dtype=np.float32)


NODE_FEATURE_DIM = K + 4 + len(NODE_TYPES) + 3
