from __future__ import annotations

from typing import Dict, List, Type

from knowledge.kg_builder import SymbolVocab
from .base import BatchPlanner, Scheduler
from .coscherrl import CoScheRRL
from .ddmts import DDMTS
from .decima import Decima
from .hehifldsa import HeHiFLDSA
from .mkfr_c2s import MKFRC2S
from .mots_aco import MOTSACO
from .pso_pga import PSOPGA
from .smda import SMDA
from .tvw_rl import TVWRL


ALGORITHM_ORDER: List[str] = [
    "SMDA", "DDMTS", "PSO-PGA", "MOTS-ACO",
    "Co-ScheRRL", "TVW-RL", "Decima", "HeHiFLDSA", "MKFR-C2S",
]

_CLASSES: Dict[str, Type] = {
    "SMDA": SMDA,
    "DDMTS": DDMTS,
    "MOTS-ACO": MOTSACO,
    "PSO-PGA": PSOPGA,
    "Co-ScheRRL": CoScheRRL,
    "TVW-RL": TVWRL,
    "Decima": Decima,
    "HeHiFLDSA": HeHiFLDSA,
    "MKFR-C2S": MKFRC2S,
}


COLORS = {
    "SMDA": "#7ED0D4",
    "DDMTS": "#E7E93B",
    "PSO-PGA": "#7F7F7F",
    "MOTS-ACO": "#9C3B24",
    "Co-ScheRRL": "#3A5BA0",
    "TVW-RL": "#F5B7CC",
    "Decima": "#000000",
    "HeHiFLDSA": "#2F7D33",
    "MKFR-C2S": "#F58220",
}
MARKERS = {"SMDA": "v", "DDMTS": "<", "PSO-PGA": ">", "MOTS-ACO": "^",
           "Co-ScheRRL": "s", "TVW-RL": "D", "Decima": "P", "HeHiFLDSA": "X",
           "MKFR-C2S": "o"}


def build(name: str, seed: int = 0, **kwargs) -> Scheduler:
    if name not in _CLASSES:
        raise KeyError(f"unknown algorithm {name!r}; available: {list(_CLASSES)}")
    kw = dict(kwargs)
    if name in ("Decima", "HeHiFLDSA", "MKFR-C2S"):
        kw.setdefault("n_symbols", len(SymbolVocab()))
    if name == "MKFR-C2S":
        kw.setdefault("lam_m", 0.45)
        kw.setdefault("weights", (0.25, 0.25, 0.30, 0.20))
    return _CLASSES[name](seed=seed, **kw)


def is_trainable(name: str) -> bool:
    return getattr(_CLASSES[name], "trainable", False)


__all__ = ["ALGORITHM_ORDER", "COLORS", "MARKERS", "build", "is_trainable",
           "Scheduler", "BatchPlanner"]
