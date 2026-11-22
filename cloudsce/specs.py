from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


RESOURCE_NAMES: Tuple[str, ...] = ("cpu", "npu", "vram", "mem", "disk", "bw")
K: int = len(RESOURCE_NAMES)
RES_INDEX: Dict[str, int] = {n: i for i, n in enumerate(RESOURCE_NAMES)}


EXEC_TYPES: Tuple[str, ...] = ("WF", "LRA", "BJ")
DOMINANT_TYPES: Tuple[str, ...] = ("cpu", "gpu", "npu", "mem", "disk", "bw", "none")
N_CLASSES: int = len(EXEC_TYPES) * len(DOMINANT_TYPES)


def class_id(exec_type: str, dominant: str) -> int:
    return EXEC_TYPES.index(exec_type) * len(DOMINANT_TYPES) + DOMINANT_TYPES.index(dominant)


def class_name(cid: int) -> str:
    e, d = divmod(cid, len(DOMINANT_TYPES))
    return f"{EXEC_TYPES[e]}-{DOMINANT_TYPES[d]}"


@dataclass(frozen=True)
class NodeType:
    name: str
    cap: Tuple[float, ...]
    weight: float
    is_accelerator: bool

    @property
    def cap_vec(self) -> np.ndarray:
        return np.asarray(self.cap, dtype=np.float64)


ASCEND_NODE_TYPES = (
    NodeType("Ascend-910A", (192.0, 2560.0, 256.0, 768.0, 4500.0, 90.0), 0.010, True),
    NodeType("Ascend-910B", (192.0, 2504.0, 512.0, 1000.0, 6400.0, 392.0), 0.030, True),
    NodeType("KunPeng-920", (12.0, 0.0, 0.0, 24.0, 90.0, 5.0), 0.960, False),
)
ASCEND_NODE_NAMES = tuple(t.name for t in ASCEND_NODE_TYPES)


CLOUD_NODE_TYPES = (
    NodeType("cloud-xl", (96.0, 0.0, 0.0, 384.0, 4096.0, 10.0), 0.07, False),
    NodeType("cloud-l", (64.0, 0.0, 0.0, 256.0, 2048.0, 10.0), 0.20, False),
    NodeType("cloud-m", (48.0, 0.0, 0.0, 128.0, 1024.0, 5.0), 0.33, False),
    NodeType("cloud-s", (32.0, 0.0, 0.0, 64.0, 500.0, 5.0), 0.35, False),
    NodeType("cloud-g", (64.0, 400.0, 64.0, 256.0, 2048.0, 25.0), 0.05, True),
)
CLOUD_NODE_NAMES = tuple(t.name for t in CLOUD_NODE_TYPES)


ASCEND5_NODE_TYPES = (
    NodeType("Ascend-T1-CPU", (192.0, 0.0, 0.0, 96.0, 1200.0, 25.0), 0.30, False),
    NodeType("Ascend-T2-MEM", (96.0, 0.0, 0.0, 768.0, 800.0, 25.0), 0.20, False),
    NodeType("Ascend-T3-DISK", (64.0, 0.0, 0.0, 128.0, 12000.0, 25.0), 0.15, False),
    NodeType("Ascend-T4-BW", (64.0, 0.0, 0.0, 128.0, 2000.0, 200.0), 0.15, False),
    NodeType("Ascend-T5-HYB", (128.0, 1280.0, 256.0, 512.0, 4000.0, 100.0), 0.20, True),
)
ASCEND5_NODE_NAMES = tuple(t.name for t in ASCEND5_NODE_TYPES)

NODE_TYPES = {t.name: t
              for t in (ASCEND_NODE_TYPES + CLOUD_NODE_TYPES + ASCEND5_NODE_TYPES)}
N_NODE_TYPES = len(NODE_TYPES)
NODE_TYPE_INDEX = {name: i for i, name in enumerate(NODE_TYPES)}


REF_CAP = np.asarray([12.0, 2504.0, 512.0, 24.0, 90.0, 5.0], dtype=np.float64)


@dataclass(frozen=True)
class Archetype:
    exec_type: str
    dominant: str
    frac: Tuple[float, ...]
    instances: Tuple[int, int]
    dur: Tuple[float, float]
    latency: float
    deadline_slack: float


def _mk(exec_type, dominant, frac, instances, dur, latency, slack):
    return Archetype(exec_type, dominant, tuple(frac), tuple(instances),
                     tuple(dur), float(latency), float(slack))


ARCHETYPES = (
    _mk("WF", "cpu",  (0.42, 0.00, 0.00, 0.18, 0.05, 0.02), (2, 6), (2.0, 5.0), 0.35, 1.25),
    _mk("WF", "gpu",  (0.18, 0.10, 0.12, 0.22, 0.06, 0.03), (2, 5), (3.0, 6.0), 0.55, 1.20),
    _mk("WF", "npu",  (0.22, 0.30, 0.28, 0.30, 0.08, 0.04), (2, 4), (3.0, 7.0), 0.60, 1.20),
    _mk("WF", "mem",  (0.20, 0.00, 0.00, 0.62, 0.10, 0.02), (2, 6), (2.0, 5.0), 0.40, 1.25),
    _mk("WF", "disk", (0.16, 0.00, 0.00, 0.22, 0.55, 0.04), (2, 5), (2.0, 5.0), 0.30, 1.30),
    _mk("WF", "bw",   (0.18, 0.00, 0.00, 0.20, 0.06, 0.58), (2, 6), (2.0, 4.0), 0.65, 1.20),
    _mk("WF", "none", (0.22, 0.00, 0.00, 0.24, 0.10, 0.05), (2, 6), (2.0, 5.0), 0.30, 1.35),
    _mk("LRA", "cpu",  (0.30, 0.00, 0.00, 0.16, 0.03, 0.04), (2, 8), (8.0, 20.0), 0.95, 1.10),
    _mk("LRA", "gpu",  (0.14, 0.08, 0.10, 0.20, 0.04, 0.05), (2, 6), (8.0, 20.0), 0.90, 1.10),
    _mk("LRA", "npu",  (0.18, 0.26, 0.24, 0.26, 0.05, 0.06), (2, 5), (8.0, 18.0), 0.92, 1.10),
    _mk("LRA", "mem",  (0.16, 0.00, 0.00, 0.55, 0.06, 0.03), (2, 8), (8.0, 20.0), 0.85, 1.10),
    _mk("LRA", "disk", (0.12, 0.00, 0.00, 0.18, 0.48, 0.04), (2, 6), (8.0, 18.0), 0.75, 1.15),
    _mk("LRA", "bw",   (0.14, 0.00, 0.00, 0.18, 0.05, 0.52), (2, 8), (8.0, 20.0), 0.98, 1.05),
    _mk("LRA", "none", (0.18, 0.00, 0.00, 0.20, 0.08, 0.06), (2, 8), (8.0, 20.0), 0.70, 1.20),
    _mk("BJ", "cpu",  (0.55, 0.00, 0.00, 0.14, 0.04, 0.01), (2, 8), (1.0, 3.0), 0.10, 2.0),
    _mk("BJ", "gpu",  (0.20, 0.14, 0.14, 0.18, 0.05, 0.02), (2, 6), (1.0, 3.0), 0.15, 1.6),
    _mk("BJ", "npu",  (0.26, 0.34, 0.30, 0.24, 0.06, 0.02), (2, 5), (1.0, 4.0), 0.20, 1.6),
    _mk("BJ", "mem",  (0.24, 0.00, 0.00, 0.70, 0.08, 0.01), (2, 8), (1.0, 3.0), 0.10, 2.0),
    _mk("BJ", "disk", (0.14, 0.00, 0.00, 0.16, 0.62, 0.02), (2, 6), (1.0, 3.0), 0.08, 2.2),
    _mk("BJ", "bw",   (0.16, 0.00, 0.00, 0.16, 0.04, 0.62), (2, 8), (1.0, 3.0), 0.20, 1.6),
    _mk("BJ", "none", (0.26, 0.00, 0.00, 0.18, 0.06, 0.03), (2, 8), (1.0, 3.0), 0.10, 2.2),
)

assert len(ARCHETYPES) == N_CLASSES
ARCHETYPE_BY_CLASS = {(a.exec_type, a.dominant): a for a in ARCHETYPES}


@dataclass
class ModelParams:
    lam_irc: float = 0.25
    lam_iwc: float = 0.25
    lam_iac: float = 0.30
    lam_iaac: float = 0.20

    lam_m: float = 0.45

    alpha: float = 0.55
    beta: float = 0.45

    lam1: float = 0.35
    lam2: float = 0.35
    lam3: float = 0.30

    w_reject: float = 2.5

    w_aff: float = 0.6

    w_task: float = 0.35

    w_idle: float = 0.25

    w_skew: float = 0.30

    w_bal: float = 0.60

    w_trunc: float = 0.50

    n_intervals: int = 100
    minibatch: int = 128
    lr: float = 0.006
    gamma: float = 0.90
    temperature: float = 0.07


DEFAULT_PARAMS = ModelParams()
