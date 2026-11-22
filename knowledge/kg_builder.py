from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from typing import TYPE_CHECKING

from .ctssm import PROMPT_TEMPLATE, RELATIONS, VIEWS, Triple, route

if TYPE_CHECKING:
    from cloudsce.cluster import Cluster
    from cloudsce.workload import Workload

DOMINANT_TYPES = ("cpu", "gpu", "npu", "mem", "disk", "bw", "none")
EXEC_TYPES = ("WF", "LRA", "BJ")
RESOURCE_NAMES = ("cpu", "npu", "vram", "mem", "disk", "bw")


MAX_TRIPLES_PER_VIEW = 12
TOKENS_PER_VIEW = 24

_NODE_BUCKETS = 64
_TASK_BUCKETS = 64
_VALUE_BUCKETS = 8


def node_symbol(nid: int) -> str:
    return f"N{int(nid) % _NODE_BUCKETS}"


def task_symbol(wid: int) -> str:
    return f"T{int(wid) % _TASK_BUCKETS}"


def value_symbol(name: str, value: float) -> str:
    lvl = int(np.clip(round(value * (_VALUE_BUCKETS - 1)), 0, _VALUE_BUCKETS - 1))
    return f"{name}:L{lvl}"


@dataclass
class ViewGraph:
    triples: Dict[str, List[Triple]]

    def flat(self) -> List[Triple]:
        out: List[Triple] = []
        for v in VIEWS:
            out.extend(self.triples.get(v, []))
        return out

    def text(self, view: str) -> str:
        return " ; ".join(str(t) for t in self.triples.get(view, []))


class KGuanBuilder:
    def __init__(self, max_per_view: int = MAX_TRIPLES_PER_VIEW):
        self.max_per_view = max_per_view

    def build(self, cluster: Cluster, w: Workload, inst=None) -> ViewGraph:
        rv = self._resource_view(cluster)
        wv = self._workload_view(cluster, w, inst)
        av = self._affinity_view(cluster, w)
        aav = self._anti_affinity_view(cluster, w)
        return ViewGraph({"RV": rv, "WV": wv, "AV": av, "AAV": aav})

    def _resource_view(self, cluster: Cluster) -> List[Triple]:
        rem = np.clip(1.0 - cluster.use_mat / np.maximum(cluster.cap_mat, 1e-9), 0, 1)
        mask = cluster.cap_mat > 0
        score = np.where(mask, rem, np.nan)
        mean_room = np.nanmean(score, axis=1) if cluster.n else np.zeros(0)
        order = np.argsort(-mean_room)
        out: List[Triple] = []
        for idx in order[:4]:
            nid = node_symbol(idx)
            out.append(route(nid, "hasResourceID", f"N{idx % _NODE_BUCKETS}"))
            rtype = "AcceleratorNode" if cluster.cap_mat[idx, 1] > 0 else "ComputeNode"
            out.append(route(nid, "hasResourceType", rtype))
            status = "Normal" if cluster.healthy[idx] else "Faulty"
            out.append(route(nid, "hasResourceStatus", status))
            dom = int(np.argmax(cluster.cap_mat[idx]))
            out.append(route(nid, "hasResourceQuota", f"{RESOURCE_NAMES[dom]}:L{int(np.clip(cluster.cap_mat[idx, dom] / 400, 0, 1) * 7)}"))
            out.append(route(nid, "hasRemainingResource", value_symbol("rem", float(mean_room[idx]))))
        return out[: self.max_per_view]

    def _workload_view(self, cluster: Cluster, w: Workload, inst) -> List[Triple]:
        tid = task_symbol(w.wid)
        out = [
            route(tid, "hasTaskType", f"{w.exec_type}-{w.dominant}"),
            route(tid, "hasTaskStatus", "Waiting"),
            route(tid, "hasInstanceNum", f"n{Lmin(w.n_instances)}"),
        ]
        demand = inst.demand if inst is not None else w.demand
        for k, name in enumerate(RESOURCE_NAMES):
            if demand[k] > 1e-6:
                out.append(route(tid, "requiresResource",
                                 value_symbol(name, float(demand[k] / max(np.max(cluster.cap_mat[:, k]), 1e-9)))))
        return out[: self.max_per_view]

    def _affinity_view(self, cluster: Cluster, w: Workload) -> List[Triple]:
        tid = task_symbol(w.wid)
        out: List[Triple] = []
        for p in w.deps:
            out.append(route(tid, "dependsOn", task_symbol(p)))
        for p in w.affinity:
            out.append(route(tid, "hasPreferenceFor", task_symbol(p)))
            node = cluster._last_node.get(p, -1)
            if node >= 0:
                out.append(route(task_symbol(p), "isLocatedAt", node_symbol(node)))
        if not out and w.exec_type == "LRA":
            node = int(np.argmax(cluster.balance_vector())) if cluster.n else 0
            out.append(route(tid, "hasPreferenceFor", node_symbol(node)))
        return out[: self.max_per_view]

    def _anti_affinity_view(self, cluster: Cluster, w: Workload) -> List[Triple]:
        tid = task_symbol(w.wid)
        out: List[Triple] = []
        for a in w.anti_affinity:
            out.append(route(tid, "hasSimilarityWith", task_symbol(a)))
            node = cluster._last_node.get(a, -1)
            if node >= 0:
                out.append(route(tid, "hasAversionTo", node_symbol(node)))
        if w.n_instances > 1:
            out.append(route(tid, "isTwinOf", tid))
        return out[: self.max_per_view]


def Lmin(n: int) -> str:
    return f"L{int(np.clip(n // 2, 0, 7))}"


class LLMCondenser:
    def __init__(self, model: str = "deepseek-v2-33b"):
        self.model = model

    def prompt(self, input_text: str, instructions: Dict[str, Sequence[str]]) -> str:
        return PROMPT_TEMPLATE.format(
            input_text=input_text,
            instruction_rv=instructions["RV"][0],
            instruction_wv=instructions["WV"][0],
            instruction_av=instructions["AV"][0],
            instruction_aav=instructions["AAV"][0],
        )

    @staticmethod
    def parse(answer: str) -> List[Triple]:
        out: List[Triple] = []
        for line in answer.replace("。", "\n").splitlines():
            line = line.strip().strip("<>").strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r not in {rel for rs in RELATIONS.values() for rel in rs}:
                continue
            out.append(route(h, r, t))
        return out


def view_token_ids(kg: ViewGraph, vocab) -> np.ndarray:
    out = np.zeros((len(VIEWS), TOKENS_PER_VIEW), dtype=np.int64)
    for vi, v in enumerate(VIEWS):
        toks: List[int] = []
        for t in kg.triples.get(v, []):
            for sym in (t.head, t.relation, t.tail):
                toks.append(vocab.id(sym))
                if len(toks) >= TOKENS_PER_VIEW:
                    break
            if len(toks) >= TOKENS_PER_VIEW:
                break
        for i, tok in enumerate(toks):
            out[vi, i] = tok
    return out


class SymbolVocab:
    PAD, UNK = 0, 1

    def __init__(self, size: int = 512):
        self.size = size
        self._map: Dict[str, int] = {}
        self._free = 2
        for v in VIEWS:
            self._map[v] = self._next()
        for exec_t in EXEC_TYPES:
            for dom in DOMINANT_TYPES:
                self._map[f"{exec_t}-{dom}"] = self._next()
        for name in RESOURCE_NAMES:
            for lvl in range(_VALUE_BUCKETS):
                self._map[f"{name}:L{lvl}"] = self._next()
        for lvl in range(_VALUE_BUCKETS):
            self._map[f"rem:L{lvl}"] = self._next()
        for rels in RELATIONS.values():
            for r in rels:
                self._map[r] = self._next()
        for i in range(_NODE_BUCKETS):
            self._map[f"N{i}"] = self._next()
        for i in range(_TASK_BUCKETS):
            self._map[f"T{i}"] = self._next()
        for name in ("ComputeNode", "AcceleratorNode", "Normal", "Faulty",
                     "Waiting", "Running"):
            self._map[name] = self._next()

    def _next(self) -> int:
        i = self._free
        self._free += 1
        return min(i, self.size - 1)

    def id(self, sym: str) -> int:
        i = self._map.get(sym)
        if i is not None:
            return i
        h = int(hashlib.md5(sym.encode("utf-8")).hexdigest()[:8], 16)
        return 2 + h % (self.size - 2)

    def __len__(self) -> int:
        return self.size
