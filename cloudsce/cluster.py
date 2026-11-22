from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .node import NODE_FEATURE_DIM, Node
from .specs import (ASCEND5_NODE_TYPES, ASCEND_NODE_TYPES, CLOUD_NODE_TYPES,
                    K, NODE_TYPE_INDEX, NODE_TYPES)
from .workload import Instance, Workload


class Cluster:
    def __init__(self, nodes: Sequence[Node]):
        self.nodes: List[Node] = list(nodes)
        self.index: Dict[int, int] = {n.nid: i for i, n in enumerate(self.nodes)}
        self.workloads: Dict[int, Workload] = {}
        self.instances: Dict[int, Instance] = {}
        self.time: float = 0.0
        self.total_rejected: int = 0
        self.total_placed: int = 0
        self.affi_pairs: int = 0
        self.affi_ok: int = 0
        self.anti_pairs: int = 0
        self.anti_viol: int = 0
        self._last_node: Dict[int, int] = {}
        self.wl_finish: Dict[int, int] = {}
        self.wl_violation: Dict[int, int] = {}

        self.cap_mat = np.stack([n.cap for n in self.nodes]).astype(np.float64)
        self.use_mat = np.zeros_like(self.cap_mat)
        self.healthy = np.ones(self.n, dtype=bool)
        self.type_idx = np.asarray([n.type_index for n in self.nodes], dtype=np.int64)
        self.n_class = np.zeros((self.n, 21), dtype=np.float32)
        self.hosted_n = np.zeros(self.n, dtype=np.float32)
        self.wl_count: Dict[int, np.ndarray] = {}

    @property
    def n(self) -> int:
        return len(self.nodes)

    def utilization_vector(self) -> np.ndarray:
        mask = self.cap_mat > 0
        frac = np.where(mask, self.use_mat / np.where(mask, self.cap_mat, 1.0), 0.0)
        frac = np.clip(frac, 0.0, 1.5) * mask
        cnt = np.maximum(mask.sum(axis=1), 1)
        return (frac * mask).sum(axis=1) / cnt

    def balance_vector(self) -> np.ndarray:
        mask = self.cap_mat > 0
        frac = np.where(mask, self.use_mat / np.where(mask, self.cap_mat, 1.0), 0.0)
        frac = np.clip(frac, 0.0, 1.5)
        cnt = np.maximum(mask.sum(axis=1), 1)
        mu = (frac * mask).sum(axis=1) / cnt
        var = (((frac - mu[:, None]) ** 2) * mask).sum(axis=1) / cnt
        cv = np.sqrt(np.maximum(var, 0)) / np.maximum(mu, 1e-9)
        blc = np.clip(1.0 - cv, 0.0, 1.0)
        blc[mu <= 1e-9] = 0.0
        return blc

    def avg_util(self) -> float:
        return float(np.mean(self.utilization_vector()))

    def avg_imbalance(self) -> float:
        return float(np.mean(1.0 - self.balance_vector()))

    def binding_utilization(self) -> np.ndarray:
        mask = self.cap_mat > 0
        frac = np.where(mask, self.use_mat / np.where(mask, self.cap_mat, 1.0), 0.0)
        return frac.max(axis=1)

    def load_skew(self) -> float:
        u = self.binding_utilization()
        mu = float(u.mean())
        if mu <= 1e-9:
            return 0.0
        return float(np.std(u) / mu)

    def total_usage(self) -> np.ndarray:
        return self.use_mat.sum(axis=0)

    def total_capacity(self) -> np.ndarray:
        return self.cap_mat.sum(axis=0)

    def global_util_by_resource(self) -> np.ndarray:
        cap = self.total_capacity()
        return np.where(cap > 0, np.clip(self.total_usage() / np.where(cap > 0, cap, 1), 0, 1.5), 0.0)

    def register_workload(self, w: Workload) -> None:
        self.workloads[w.wid] = w
        self.wl_count[w.wid] = np.zeros(self.n, dtype=np.float32)

    def feasible(self, inst: Instance, w: Workload) -> np.ndarray:
        need = inst.demand[None, :]
        ok_cap = np.all(self.use_mat + need <= self.cap_mat + 1e-9, axis=1)
        ok_zero = np.all((need <= 1e-9) | (self.cap_mat > 1e-9), axis=1)
        mask = ok_cap & ok_zero & self.healthy
        same = self.wl_count.get(w.wid)
        if same is not None:
            mask &= same < 0.5
        if not mask.any():
            mask = ok_cap & ok_zero & self.healthy
        return mask

    def affinity_mask(self, inst: Instance, w: Workload) -> np.ndarray:
        mask = np.zeros(self.n, dtype=bool)
        partners = tuple(w.affinity) + tuple(w.deps)
        if not partners:
            return mask
        ok_cap = np.all(self.use_mat + inst.demand[None, :] <= self.cap_mat + 1e-9, axis=1) & self.healthy
        hit = np.zeros(self.n, dtype=bool)
        for p in partners:
            c = self.wl_count.get(p)
            if c is not None:
                hit |= c > 0.5
        return ok_cap & hit

    def place(self, inst: Instance, w: Workload, node_idx: int,
              count_affinity: bool = True) -> np.ndarray:
        nd = self.nodes[node_idx]
        nd.allocate(inst.demand, inst.iid, w.wid, w.cid)
        self.use_mat[node_idx] += inst.demand
        self.hosted_n[node_idx] += 1
        self.n_class[node_idx, w.cid] += 1
        self.wl_count.setdefault(w.wid, np.zeros(self.n, dtype=np.float32))[node_idx] += 1
        inst.node = nd.nid
        inst.state = "running"
        inst.start = self.time
        inst.finish = self.time + w.duration
        self.instances[inst.iid] = inst
        self.total_placed += 1
        if count_affinity:
            self._update_affinity(w, node_idx)
        self._last_node[w.wid] = node_idx
        return self.use_mat[node_idx]

    def _update_affinity(self, w: Workload, node_idx: int) -> None:
        for p in w.affinity:
            if p not in self.workloads:
                continue
            self.affi_pairs += 1
            c = self.wl_count.get(p)
            if (c is not None and c[node_idx] > 0.5) or self._last_node.get(p, -1) == node_idx:
                self.affi_ok += 1
        for a in w.anti_affinity:
            self.anti_pairs += 1
            c = self.wl_count.get(a)
            if c is not None and c[node_idx] > 0.5:
                self.anti_viol += 1

    def complete_workload(self, w: Workload, round_now: int) -> bool:
        same_class = 0
        for inst in w.instances:
            if inst.node >= 0:
                same_class = max(same_class, int(self.n_class[inst.node, w.cid]))
        interference = 1.0 + 0.08 * max(same_class - 1, 0)
        dep_delay = 0.0
        for p in w.deps:
            f = self.wl_finish.get(p)
            if f is None:
                continue
            dep_delay = max(dep_delay, float(f) - round_now)
            pf = self._last_node.get(p, -1)
            if pf >= 0 and any(i.node >= 0 and i.node != pf for i in w.instances):
                dep_delay += 0.5
        finish = int(np.ceil(round_now + dep_delay + max(1.0, w.duration * interference)))
        self.wl_finish[w.wid] = finish
        if w.deadline_round < 0:
            w.deadline_round = int(np.ceil(w.arrival + w.deadline))
        w.finish_round = finish
        w.violated = int(finish > w.deadline_round)
        self.wl_violation[w.wid] = w.violated
        return bool(w.violated)

    def reject(self, inst: Instance) -> None:
        inst.state = "rejected"
        self.total_rejected += 1

    def release_finished(self) -> int:
        freed = 0
        for nd in self.nodes:
            for iid in list(nd.hosted):
                inst = self.instances.get(iid)
                if inst is None:
                    continue
                if inst.finish <= self.time + 1e-9:
                    self.release(nd, inst)
                    freed += 1
        return freed

    def release(self, nd: Node, inst: Instance) -> None:
        idx = nd.nid
        nd.release(inst.demand, inst.iid, inst.wl_id, inst.cid)
        self.use_mat[idx] = np.maximum(self.use_mat[idx] - inst.demand, 0.0)
        self.hosted_n[idx] = max(0.0, self.hosted_n[idx] - 1)
        self.n_class[idx, inst.cid] = max(0.0, self.n_class[idx, inst.cid] - 1)
        c = self.wl_count.get(inst.wl_id)
        if c is not None:
            c[idx] = max(0.0, c[idx] - 1)
        inst.state = "done"

    def affinity_stats(self) -> Tuple[int, int, int, int]:
        nodes: Dict[int, set] = {}
        for wid, w in self.workloads.items():
            ns = {i.node for i in w.instances if i.node >= 0}
            if ns:
                nodes[wid] = ns
        ap = ao = aa = av = 0
        for wid, w in self.workloads.items():
            nw = nodes.get(wid)
            if nw is None:
                continue
            for p in w.affinity:
                np_ = nodes.get(p)
                if np_ is None:
                    continue
                ap += 1
                if nw & np_:
                    ao += 1
            for p in w.anti_affinity:
                np_ = nodes.get(p)
                if np_ is None:
                    continue
                aa += 1
                if nw & np_:
                    av += 1
        return ap, ao, aa, av

    def occupancy_by_class(self) -> np.ndarray:
        tot = self.n_class.sum(axis=1, keepdims=True)
        return np.where(tot > 0, self.n_class / np.maximum(tot, 1e-9), 0.0)

    def node_features(self, w: Optional[Workload] = None) -> np.ndarray:
        mask = self.cap_mat > 0
        rem = np.where(mask, 1.0 - self.use_mat / np.where(mask, self.cap_mat, 1.0), 0.0)
        rem = np.clip(rem, 0.0, 1.0) * mask
        ur = self.utilization_vector()
        blc = self.balance_vector()
        oh = np.zeros((self.n, len(NODE_TYPES)), dtype=np.float32)
        oh[np.arange(self.n), self.type_idx] = 1.0
        isacc = (self.cap_mat[:, 1] > 0).astype(np.float32)
        aff = np.zeros(self.n, dtype=np.float32)
        dep = np.zeros(self.n, dtype=np.float32)
        if w is not None:
            c = self.wl_count.get(w.wid)
            if c is not None:
                aff = np.minimum(c / 4.0, 1.0)
            partners = tuple(w.deps) if w.deps else ()
            for p in partners:
                cp = self.wl_count.get(p)
                if cp is not None:
                    dep += np.minimum(cp, 1.0)
            dep = np.minimum(dep / max(len(partners), 1), 1.0)
        load_ratio = np.minimum(self.hosted_n / 32.0, 1.0)
        feats = np.concatenate([
            rem.astype(np.float32), ur[:, None].astype(np.float32),
            blc[:, None].astype(np.float32), self.healthy[:, None].astype(np.float32),
            isacc[:, None], oh,
            aff[:, None], dep[:, None], load_ratio[:, None],
        ], axis=1)
        return feats.astype(np.float32)

    def inject_background(self, rng: np.random.Generator, target_util: float,
                          n_rounds_hint: int = 400) -> None:
        for idx, nd in enumerate(self.nodes):
            if not nd.healthy:
                continue

            if rng.random() < 0.30:
                level = target_util * rng.uniform(0.45, 0.75)
            else:
                level = target_util * rng.uniform(1.15, 1.45)
            frac = np.clip(level * rng.uniform(0.90, 1.10, size=K), 0.0, 0.96)
            frac[nd.cap <= 0] = 0.0
            k = int(rng.integers(0, K))
            if nd.cap[k] > 0:
                frac[k] = min(frac[k] * 1.12, 0.97)
            demand = frac * nd.cap
            fake = Instance(iid=-1 - nd.nid, wl_id=-1 - nd.nid, ridx=0,
                            demand=demand, cid=int(rng.integers(0, 21)),
                            node=nd.nid, state="running",
                            start=-1.0, finish=10 * n_rounds_hint)
            nd.allocate(demand, fake.iid, fake.wl_id, fake.cid)
            self.use_mat[idx] = demand
            self.hosted_n[idx] = 1
            self.instances[fake.iid] = fake

    def mark_failed(self, node_idx: int, status: str = "crash") -> List[Instance]:
        nd = self.nodes[node_idx]
        nd.healthy = False
        self.healthy[node_idx] = False
        lost = []
        for iid in list(nd.hosted):
            inst = self.instances.get(iid)
            if inst is not None:
                self.release(nd, inst)
                inst.state = "rejected"
                self.total_rejected += 1
                lost.append(inst)
        self.use_mat[node_idx] = 0.0
        self.hosted_n[node_idx] = 0.0
        self.n_class[node_idx] = 0.0
        nd.status = status
        return lost


def build_cloudsce_nodes(rng: np.random.Generator, n_nodes: int,
                         flavour: str = "cloud") -> List[Node]:
    types = CLOUD_NODE_TYPES if flavour == "cloud" else ASCEND_NODE_TYPES
    w = np.asarray([t.weight for t in types], dtype=np.float64)
    w = w / w.sum()
    picks = rng.choice(len(types), size=n_nodes, p=w)
    return [Node(nid=i, type_name=types[int(picks[i])].name,
                 cap=types[int(picks[i])].cap_vec.copy()) for i in range(n_nodes)]


def build_ascend_nodes(rng: np.random.Generator, n_nodes: int) -> List[Node]:
    return build_cloudsce_nodes(rng, n_nodes, flavour="ascend")


def build_ascend5_nodes(rng: np.random.Generator, n_nodes: int) -> List[Node]:
    types = ASCEND5_NODE_TYPES
    w = np.asarray([t.weight for t in types], dtype=np.float64)
    w = w / w.sum()
    picks = rng.choice(len(types), size=n_nodes, p=w)
    return [Node(nid=i, type_name=types[int(picks[i])].name,
                 cap=types[int(picks[i])].cap_vec.copy()) for i in range(n_nodes)]


T_SUPPLY_INDEX = {
    "Ascend-T1-CPU": 0, "Ascend-T2-MEM": 1, "Ascend-T3-DISK": 2,
    "Ascend-T4-BW": 3, "Ascend-T5-HYB": 4,
    "KunPeng-920": 0, "Ascend-910A": 4, "Ascend-910B": 4,
}
