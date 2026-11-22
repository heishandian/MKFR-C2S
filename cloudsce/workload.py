from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .specs import (ARCHETYPE_BY_CLASS, ARCHETYPES, K, N_CLASSES, REF_CAP,
                    class_id)


@dataclass
class Instance:
    iid: int
    wl_id: int
    ridx: int
    demand: np.ndarray
    cid: int
    node: int = -1
    start: float = 0.0
    finish: float = 0.0
    waiting: float = 0.0
    state: str = "pending"

    @property
    def running(self) -> bool:
        return self.state == "running"


@dataclass
class Workload:
    wid: int
    exec_type: str
    dominant: str
    n_instances: int
    demand: np.ndarray
    duration: float
    deadline: float
    cid: int
    deps: Tuple[int, ...] = ()
    affinity: Tuple[int, ...] = ()
    anti_affinity: Tuple[int, ...] = ()
    latency: float = 0.3
    slack: float = 3.0
    arrival: float = 0.0
    priority_tier: int = 0
    instances: List[Instance] = field(default_factory=list)

    finish_round: int = -1
    deadline_round: int = -1
    violated: int = 0
    placed_instances: int = 0

    @property
    def total_demand(self) -> np.ndarray:
        return self.demand * self.n_instances

    def make_instances(self, next_iid: int) -> int:
        self.instances = []
        for r in range(self.n_instances):
            self.instances.append(
                Instance(iid=next_iid + r, wl_id=self.wid, ridx=r,
                         demand=self.demand.copy(), cid=self.cid,
                         start=self.arrival)
            )
        return next_iid + self.n_instances


def sample_workload(rng: np.random.Generator, wid: int, class_probs: np.ndarray,
                    arrival: float, scale: float = 1.0) -> Workload:
    cid = int(rng.choice(N_CLASSES, p=class_probs / class_probs.sum()))
    arch = ARCHETYPES[cid]
    frac = np.asarray(arch.frac, dtype=np.float64)

    noise = rng.lognormal(mean=0.0, sigma=0.28, size=K)
    demand = frac * noise * REF_CAP * scale

    didx = {"cpu": 0, "gpu": 1, "npu": 1, "mem": 3, "disk": 4, "bw": 5}.get(arch.dominant)
    if didx is not None and frac[didx] > 0:
        others = np.delete(demand, didx)
        demand[didx] = max(float(others.max()) * 1.15, demand[didx])
    demand = np.maximum(demand, 0.0)
    demand[demand < 1e-6] = 0.0

    n_inst = int(rng.integers(arch.instances[0], arch.instances[1] + 1))
    if arch.exec_type == "WF":
        n_inst = max(2, n_inst)
    dur = float(rng.uniform(*arch.dur))
    dl = dur * arch.deadline_slack * float(rng.uniform(0.85, 1.25))
    return Workload(
        wid=wid, exec_type=arch.exec_type, dominant=arch.dominant,
        n_instances=n_inst, demand=demand, duration=dur, deadline=dl,
        cid=cid, latency=arch.latency, slack=arch.deadline_slack,
        arrival=arrival,
    )


def build_dependencies(rng: np.random.Generator, wls: List[Workload],
                       p_dep: float = 0.70) -> None:
    by_exec: Dict[str, List[Workload]] = {"WF": [], "LRA": [], "BJ": []}
    for w in wls:
        by_exec[w.exec_type].append(w)

    for w in wls:
        if w.exec_type == "BJ":
            w.deps = ()
            continue
        pool = [x for x in by_exec[w.exec_type] if x.wid != w.wid]
        if not pool or rng.random() > p_dep:
            w.deps = ()
            continue
        pattern = rng.choice(["1-1", "1-n", "n-1"], p=[0.5, 0.3, 0.2])
        if pattern == "1-1":
            d = tuple(x.wid for x in rng.choice(pool, size=1, replace=False))
        elif pattern == "1-n":
            d = tuple(x.wid for x in rng.choice(pool, size=1, replace=False))
        else:
            k = min(len(pool), int(rng.integers(2, 4)))
            d = tuple(x.wid for x in rng.choice(pool, size=k, replace=False))
        w.deps = d

    for w in wls:
        if w.exec_type == "WF" and w.deps:
            w.affinity = w.deps
        elif w.exec_type == "LRA" and rng.random() < 0.45:
            same = [x.wid for x in by_exec["LRA"] if x.cid == w.cid and x.wid != w.wid]
            w.affinity = tuple(same[:2])

    for w in wls:
        same = [x.wid for x in wls if x.cid == w.cid and x.wid != w.wid]
        if same and rng.random() < 0.55:
            w.anti_affinity = tuple(same[-2:])


def class_distribution(kind: str = "alibaba") -> np.ndarray:
    p = np.zeros(N_CLASSES)
    base = {
        "alibaba": {"WF": 0.32, "LRA": 0.40, "BJ": 0.28},
        "google": {"WF": 0.22, "LRA": 0.33, "BJ": 0.45},
        "local": {"WF": 0.34, "LRA": 0.42, "BJ": 0.24},
    }[kind]
    dom = {"cpu": 0.30, "gpu": 0.06, "npu": 0.06, "mem": 0.22,
           "disk": 0.10, "bw": 0.10, "none": 0.16}
    for e, pe in base.items():
        for d, pd in dom.items():
            p[class_id(e, d)] = pe * pd
    return p / p.sum()
