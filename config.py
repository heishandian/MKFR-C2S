from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    torch.set_num_threads(int(os.environ.get("MKFR_THREADS", "1")))
    torch.set_num_interop_threads(1)
except Exception:
    torch = None

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "results", "figures")

SEED = 2026


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
