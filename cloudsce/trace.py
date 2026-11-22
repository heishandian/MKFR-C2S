from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .specs import (ARCHETYPES, DOMINANT_TYPES, N_CLASSES, REF_CAP,
                    class_id)
from .workload import Workload

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SUBSET_DIR = os.path.join(DATA_DIR, "subset")


@dataclass
class TraceRecord:
    cid: int
    frac: np.ndarray
    n_inst: int
    dur: float
    slack: float
    latency: float


class TraceSubset:
    def __init__(self, name: str, kind: str, records: List[TraceRecord]):
        self.name = name
        self.kind = kind
        self.records = records
        self.by_class: Dict[int, List[TraceRecord]] = {}
        for r in records:
            self.by_class.setdefault(r.cid, []).append(r)

    def __len__(self) -> int:
        return len(self.records)

    def sample_workload(self, rng: np.random.Generator, wid: int,
                        load_factor: float = 1.0) -> Workload:
        rec = self.records[int(rng.integers(0, len(self.records)))]
        frac = rec.frac * rng.lognormal(0.0, 0.14, size=len(rec.frac))
        demand = np.maximum(frac, 0.0) * REF_CAP * load_factor
        demand[demand < 1e-6] = 0.0
        arch = ARCHETYPES[rec.cid]
        dur = float(max(0.6, rec.dur * rng.uniform(0.85, 1.2)))
        return Workload(
            wid=wid, exec_type=arch.exec_type, dominant=arch.dominant,
            n_instances=int(max(1, round(rec.n_inst * rng.uniform(0.8, 1.25)))),
            demand=demand, duration=dur,
            deadline=dur * rec.slack * float(rng.uniform(0.9, 1.15)),
            cid=rec.cid, latency=rec.latency, slack=rec.slack,
        )


def synthesize_subset(name: str, kind: str, n_records: int = 4096,
                     seed: int = 2018) -> TraceSubset:
    rng = np.random.default_rng(seed)
    from .workload import class_distribution
    p = class_distribution(kind)
    recs: List[TraceRecord] = []
    for _ in range(n_records):
        cid = int(rng.choice(N_CLASSES, p=p))
        a = ARCHETYPES[cid]
        frac = np.asarray(a.frac) * rng.lognormal(0.0, 0.30, size=6)

        frac = np.clip(frac, 0.0, 0.95)
        recs.append(TraceRecord(
            cid=cid, frac=frac,
            n_inst=int(rng.integers(a.instances[0], a.instances[1] + 1)),
            dur=float(rng.uniform(*a.dur)),
            slack=float(a.deadline_slack * rng.uniform(0.9, 1.1)),
            latency=float(a.latency),
        ))
    return TraceSubset(name, kind, recs)


def load_subset(name: str = "alibaba", kind: str = "alibaba",
                n_records: int = 4096) -> TraceSubset:
    path = os.path.join(SUBSET_DIR, f"{name}_records.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        recs = [TraceRecord(cid=r["cid"], frac=np.asarray(r["frac"], dtype=float),
                            n_inst=r["n_inst"], dur=r["dur"], slack=r["slack"],
                            latency=r["latency"]) for r in raw["records"]]
        return TraceSubset(raw.get("name", name), raw.get("kind", kind), recs)
    return synthesize_subset(name, kind, n_records)


def train_record_from_csv(batch_task_csv: str, sample_frac: float = 0.02,
                          seed: int = 2018) -> List[TraceRecord]:
    import pandas as pd

    df = pd.read_csv(batch_task_csv)
    need = {"job_name", "task_name", "instance_num", "plan_cpu", "plan_mem", "start_time"}
    if not need.issubset(df.columns):
        raise ValueError(f"batch_task.csv is missing columns: {need - set(df.columns)}")
    df = df.sample(frac=sample_frac, random_state=seed) if sample_frac < 1 else df
    cpu_cap = max(float(df["plan_cpu"].quantile(0.995)), 1.0)
    mem_cap = max(float(df["plan_mem"].quantile(0.995)), 1.0)
    rng = np.random.default_rng(seed)
    recs: List[TraceRecord] = []
    for _, row in df.iterrows():
        cpu_cores = float(row["plan_cpu"]) / 100.0
        mem_norm = float(row["plan_mem"]) / 100.0
        dom = "cpu" if cpu_cores >= mem_norm else "mem"
        cid = class_id("BJ", dom)
        frac = np.zeros(6)
        frac[0] = cpu_cores / 12.0
        frac[3] = mem_norm * 24.0 / 24.0
        frac[4] = 0.02
        frac[5] = 0.02
        recs.append(TraceRecord(cid=cid, frac=frac,
                                n_inst=int(max(1, row["instance_num"])),
                                dur=float(rng.uniform(1.0, 3.0)),
                                slack=5.0, latency=0.15))
    return recs
