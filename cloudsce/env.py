from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cluster import (Cluster, build_ascend5_nodes, build_ascend_nodes,
                      build_cloudsce_nodes)
from .heatmap import H, TS_C2EHM, W
from .node import NODE_FEATURE_DIM
from .qos import QoSModel
from .reward import placement_delta_all, rejection_delta
from .specs import K, ModelParams, N_CLASSES, REF_CAP
from .workload import (Workload, build_dependencies, class_distribution,
                       sample_workload)
from knowledge.kg_builder import KGuanBuilder, SymbolVocab, view_token_ids

INST_DIM = 6 + 6 + N_CLASSES + 6


@dataclass
class Observation:
    img: np.ndarray
    delta: np.ndarray
    node_feat: np.ndarray
    inst_feat: np.ndarray
    mask: np.ndarray
    tokens: Optional[np.ndarray] = None
    prior: Optional[np.ndarray] = None
    workload: Optional[Workload] = None
    instance: object = None
    meta: Dict = field(default_factory=dict)

    def as_state(self) -> Dict[str, np.ndarray]:
        return {"img": self.img, "node": self.node_feat,
                "inst": self.inst_feat, "mask": self.mask,
                "prior": (self.prior if self.prior is not None
                          else np.zeros(len(self.mask), np.float32)),
                "tokens": self.tokens if self.tokens is not None else np.zeros((4, 24), np.int64)}


@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: Dict


@dataclass
class EnvConfig:
    n_nodes: int = 50
    flavour: str = "cloud"
    rounds: int = 14
    batch_ratio: float = 0.15
    bg_util: float = 0.45
    load_factor: float = 4.0
    dt: float = 1.0
    faults: bool = False
    fault_rate: float = 0.01
    qos_rt_ms: float = 300.0
    qos_th_qps: float = 9.6
    heatmap_window: int = 6
    affinity_aware: bool = True

    stop_on_infeasible: bool = False

    skew_limit: float = 0.0
    skew_rounds: int = 3

    stop_when_drained: bool = False
    affinity_density: float = 0.35
    reject_streak: int = 0


class CloudsceEnv:
    def __init__(self, cfg: EnvConfig, params: ModelParams = ModelParams(),
                 trace=None, seed: int = 0):
        self.cfg = cfg
        self.p = params
        self.trace = trace
        kind = getattr(trace, "kind", "alibaba") if trace is not None else "alibaba"
        self.class_p = class_distribution(kind)
        self.qos = QoSModel(params, cfg.qos_rt_ms, cfg.qos_th_qps)
        self.vocab = SymbolVocab()
        self.kg_builder = KGuanBuilder()
        self._rv_tokens = None
        self.hm = TS_C2EHM(cfg.heatmap_window)
        self.cluster: Optional[Cluster] = None
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        if cfg.flavour == "ascend":
            nodes = build_ascend_nodes(self.rng, cfg.n_nodes)
        elif cfg.flavour == "ascend5":
            nodes = build_ascend5_nodes(self.rng, cfg.n_nodes)
        else:
            nodes = build_cloudsce_nodes(self.rng, cfg.n_nodes)
        self.cluster = Cluster(nodes)
        self.cluster.inject_background(self.rng, cfg.bg_util, cfg.rounds)
        self.hm = TS_C2EHM(cfg.heatmap_window)
        self.round = 0
        self.ptr = 0
        self.batch: List[Tuple[Workload, object]] = []
        self.cum_reward = 0.0
        self.round_reward = 0.0
        self.history: List[Dict] = []
        self._next_wid = 0
        self._next_iid = 0
        self._rejected_round = 0
        self._placed_round = 0
        self._reject_streak = 0
        self._streak_limit = cfg.reject_streak or max(3, int(0.03 * cfg.n_nodes))
        self._skew_streak = 0
        self.ended_by = "horizon"
        self._deadline_viol_round = 0
        self._affi_prev = (0, 0, 0, 0)
        self.jct: List[float] = []
        self.abnormal = {"failed": 0, "timeout": 0, "long_wait": 0,
                         "interrupted": 0, "preempted": 0}
        self._backlog: List[Workload] = []

        self._n_batch = max(1, int(round(cfg.batch_ratio * cfg.n_nodes)))
        self._img_cache = None
        self._mask_cache = None
        self._mask_key = None
        self._delta_key = None
        self._delta_val = None
        self._rv_cache = None
        self.hm.update(self.cluster)
        self._start_round()
        return self._observation()

    def _sample(self) -> Workload:
        if self.trace is not None:
            w = self.trace.sample_workload(self.rng, self._next_wid, self.cfg.load_factor)
        else:
            w = sample_workload(self.rng, self._next_wid, self.class_p,
                                arrival=self.round * self.cfg.dt, scale=self.cfg.load_factor)
        self._next_wid += 1
        self._next_iid = w.make_instances(self._next_iid)
        self.cluster.register_workload(w)
        return w

    def _start_round(self) -> None:
        cfg = self.cfg

        n_jobs = int(np.clip(self.rng.poisson(max(1.0, self._n_batch / 4.0)),
                             1, 4 * self._n_batch))
        new = [self._sample() for _ in range(n_jobs)]
        build_dependencies(self.rng, new)
        if cfg.affinity_density > 0:
            self._attach_affinity(new, cfg.affinity_density)
        self._backlog.extend(new)

        self.batch = []
        still: List[Workload] = []
        for w in self._backlog:
            remaining = [i for i in w.instances if i.state == "pending"]
            if not remaining:
                continue
            room = self._n_batch - len(self.batch)
            if room <= 0:
                still.append(w)
                continue
            self.batch.extend((w, i) for i in remaining[:room])
            if len(remaining) > room:
                still.append(w)
        self._backlog = still
        self.ptr = 0
        self._placed_round = 0
        self._rejected_round = 0
        self._img_cache = None
        self._mask_cache = None
        self._mask_key = None
        self._delta_key = None
        self._delta_val = None

    def _attach_affinity(self, new: List[Workload], density: float) -> None:
        pool = [w.wid for w in self.cluster.workloads.values()]
        for w in new:
            others = [x for x in pool if x != w.wid]
            if not others:
                continue
            if not w.affinity and self.rng.random() < density:
                k = min(len(others), int(self.rng.integers(1, 3)))
                w.affinity = tuple(int(x) for x in self.rng.choice(others, size=k, replace=False))
            if not w.anti_affinity and self.rng.random() < density * 0.8:
                k = min(len(others), int(self.rng.integers(1, 3)))
                w.anti_affinity = tuple(int(x) for x in self.rng.choice(others, size=k, replace=False))

    def _current(self) -> Optional[Tuple[Workload, object]]:
        return self.batch[self.ptr] if self.ptr < len(self.batch) else None

    def _inst_feature(self, w: Workload, inst) -> np.ndarray:
        dem = inst.demand / REF_CAP
        ratio = inst.demand / np.maximum(self.cluster.cap_mat[0], 1e-6)
        oh = np.zeros(N_CLASSES); oh[w.cid] = 1.0
        scal = np.asarray([
            min(w.n_instances / 8.0, 1.0),
            inst.ridx / max(w.n_instances - 1, 1),
            w.latency,
            min(w.slack / 7.0, 1.0),
            min(len(w.deps) / 3.0, 1.0),
            min(len(w.affinity) / 3.0, 1.0),
        ])
        return np.concatenate([np.clip(dem, 0, 2.0), np.clip(ratio, 0, 2.0), oh, scal]).astype(np.float32)

    def _observation(self) -> Observation:
        cur = self._current()
        if self._img_cache is None:
            self._img_cache = (self.hm.observation(), self.hm.delta())
        if cur is None:
            mask = np.zeros(self.cluster.n, dtype=bool)
            feat = np.zeros(INST_DIM, dtype=np.float32)
            nodef = self.cluster.node_features(None)
            w = inst = None
            prior = np.zeros(self.cluster.n, dtype=np.float32)
        else:
            w, inst = cur
            key = (self.round, self.ptr)
            if self._mask_key != key:
                self._mask_cache = self.cluster.feasible(inst, w)
                self._mask_key = key
            mask = self._mask_cache
            feat = self._inst_feature(w, inst)
            nodef = self.cluster.node_features(w)
            prior = self._placement_prior(inst, mask, key)
        img, delta = self._img_cache
        if cur is not None:
            kg = self.kg_builder.build(self.cluster, w, inst)
            tokens = view_token_ids(kg, self.vocab)
        else:
            tokens = np.zeros((4, 24), dtype=np.int64)
        return Observation(img=img, delta=delta, node_feat=nodef,
                           inst_feat=feat, mask=mask, tokens=tokens,
                           prior=prior,
                           workload=w, instance=inst,
                           meta={"round": self.round, "ptr": self.ptr,
                                 "batch": len(self.batch)})

    def _placement_prior(self, inst, mask: np.ndarray,
                         key: Tuple[int, int]) -> np.ndarray:
        if self._delta_key != key:
            d = placement_delta_all(self.cluster, self.qos, self.p,
                                    inst.demand, max(len(self.batch), 1))
            self._delta_key, self._delta_val = key, d
        d = self._delta_val
        z = np.zeros(self.cluster.n, dtype=np.float32)
        v = d[mask]
        if v.size > 1:
            sd = float(v.std())
            if sd > 1e-12:
                z[mask] = (v - v.mean()) / sd
        return z

    def step(self, action: int) -> StepResult:
        cur = self._current()
        if cur is None:
            res = self._finish_round(done=self.round + 1 >= self.cfg.rounds)
            res.reward = float(res.info.get("r_trunc", 0.0))
            return res
        w, inst = cur
        mask = self.cluster.feasible(inst, w)
        info: Dict = {"violation": False, "rejected": False, "round_end": False,
                      "round_reward": 0.0, "action": int(action)}
        if mask.any():
            a = int(action)
            if a < 0 or a >= self.cluster.n or not mask[a]:
                info["violation"] = True
                cand = np.where(mask)[0]
                load = self.cluster.utilization_vector()[cand]
                a = int(cand[int(np.argmin(load))])
                info["action"] = a
            self.cluster.place(inst, w, a, self.cfg.affinity_aware)
            self._placed_round += 1
            self._reject_streak = 0
            w.placed_instances += 1
            if w.placed_instances >= w.n_instances:
                self.cluster.complete_workload(w, self.round)

            key = (self.round, self.ptr)
            if self._delta_key != key:
                self._delta_key = key
                self._delta_val = placement_delta_all(
                    self.cluster, self.qos, self.p, inst.demand,
                    max(len(self.batch), 1))
            info["r_shaped"] = float(self._delta_val[a])
        else:
            self.cluster.reject(inst)
            info["rejected"] = True
            self._rejected_round += 1
            self._reject_streak += 1
            info["r_shaped"] = rejection_delta(self.p, max(len(self.batch), 1))
        self.ptr += 1
        exhausted = self.cfg.stop_on_infeasible and self._reject_streak >= self._streak_limit
        if self.ptr >= len(self.batch) or exhausted:
            if exhausted:
                self.ended_by = "saturation"
            res = self._finish_round(done=(self.round + 1 >= self.cfg.rounds) or exhausted,
                                     info=info)

            res.reward = float(info.get("r_shaped", 0.0)) + float(res.info.get("r_trunc", 0.0))
            res.info["r_shaped"] = info.get("r_shaped", 0.0)
            return res
        return StepResult(self._observation(), float(info.get("r_shaped", 0.0)), False, info)

    def _finish_round(self, done: bool, info: Optional[Dict] = None) -> StepResult:
        info = dict(info or {})
        cl = self.cluster
        cl.time += self.cfg.dt

        done_now = [w for w in cl.workloads.values() if w.finish_round == self.round]
        self._deadline_viol_round = float(sum(w.violated for w in done_now))
        for w in done_now:
            self.jct.append(max(0.0, (w.finish_round - w.arrival)) * self.cfg.dt * 60.0)
            if w.violated:
                self.abnormal["timeout"] += 1
        self.abnormal["failed"] = int(cl.total_rejected)
        self.abnormal["long_wait"] = int(sum(
            1 for w in cl.workloads.values()
            if w.duration and any(i.state == "pending" for i in w.instances)
            and (self.round - w.arrival) > w.deadline))
        cl.release_finished()
        if self.cfg.faults:
            self._inject_faults()
        self.hm.update(cl)
        self._img_cache = None
        self._mask_cache = None
        self._mask_key = None
        self._delta_key = None
        self._delta_val = None

        ur = cl.utilization_vector()
        blc = cl.balance_vector()
        sla, tr, hr, fr = self.qos.sla(cl)

        node_sla = self.qos.node_sla(cl)
        sla_sum = float((cl.hosted_n * node_sla).sum())
        sla_sum += float(self._deadline_viol_round)
        sla_sum += float(self.p.w_reject * self._rejected_round)

        ap, ao, aa, av = cl.affinity_stats()
        d_unsat = (ap - ao) - self._affi_prev[0] + self._affi_prev[1]
        d_anti = av - self._affi_prev[3]
        self._affi_prev = (ap, ao, aa, av)
        sla_sum += float(self.p.w_aff * (max(d_unsat, 0) + max(d_anti, 0)))
        r = (self.p.alpha * float(np.sum(ur + blc))
             - self.p.beta * sla_sum)

        skew = float(cl.load_skew())
        n_disp = max(len(self.batch), 1)
        placed_frac = self._placed_round / n_disp
        r += self.p.w_task * placed_frac
        r -= self.p.w_idle * (1.0 - placed_frac)
        r -= self.p.w_skew * skew

        nxt = self.round + 1
        blocked = False
        if self.cfg.skew_limit > 0 and skew > self.cfg.skew_limit:
            blocked = self._admissions_blocked()
            self._skew_streak = self._skew_streak + 1 if blocked else 0
        else:
            self._skew_streak = 0
        if not done:
            if self.cfg.skew_limit > 0 and self._skew_streak >= self.cfg.skew_rounds:
                done = True
                self.ended_by = "skew"

            elif self.cfg.stop_on_infeasible and self._backlog_infeasible():
                done = True
                self.ended_by = "infeasible"
            elif (self.cfg.stop_when_drained and not self._backlog
                  and self.ptr >= len(self.batch)):
                done = True
                self.ended_by = "drained"

        r_trunc = 0.0
        if done and nxt < self.cfg.rounds and self.p.w_trunc:
            unused = float(self.cfg.rounds - nxt)
            r_trunc = self.p.w_trunc * unused * self.p.alpha * cl.n
            r -= r_trunc
        self.round_reward = float(r)
        self.cum_reward += float(r)
        q = self.qos.sample(cl)
        ap, ao, aa, av = cl.affinity_stats()
        self.history.append({
            "round": self.round,
            "avg_util": cl.avg_util(),
            "avg_imblc": cl.avg_imbalance(),
            "asid": 0.0 if ap == 0 else 1.0 - ao / ap,
            "aaid": 0.0 if aa == 0 else av / aa,
            "rt": q.rt_ms, "th": q.throughput_qps, "fp": q.failure_rate,
            "sla": sla,
            "placed": self._placed_round,
            "rejected": self._rejected_round,
            "skew": skew,
            "reward": float(r),
            "ended_by": self.ended_by,
        })
        info.update({"round_end": True, "round_reward": float(r),
                     "r_trunc": float(r_trunc),
                     "ur_sum": float(np.sum(ur)), "blc_sum": float(np.sum(blc)),
                     "sla_sum": float(sla_sum), "load_skew": skew,
                     "avg_util": cl.avg_util(), "avg_imblc": cl.avg_imbalance(),
                     "ended_by": self.ended_by})
        self.round += 1
        self._deadline_viol_round = 0
        self._affi_prev = (0, 0, 0, 0)
        if not done:
            self._start_round()
            return StepResult(self._observation(), self.round_reward, False, info)
        return StepResult(self._observation(), self.round_reward, True, info)

    def _admissions_blocked(self) -> bool:
        cap_max = self.cluster.cap_mat.max(axis=0)
        for w, inst in self.batch[self.ptr:]:
            if inst.state != "pending" or np.any(inst.demand > cap_max):
                continue
            if not self.cluster.feasible(inst, w).any():
                return True
        for w in self._backlog:
            for inst in w.instances:
                if np.any(inst.demand > cap_max) or inst.state != "pending":
                    continue
                if not self.cluster.feasible(inst, w).any():
                    return True
        return False

    def _backlog_infeasible(self) -> bool:
        if not self._backlog:
            return False
        cap_max = self.cluster.cap_mat.max(axis=0)
        placeable = False
        for w in self._backlog:
            for inst in w.instances:
                if inst.state != "pending":
                    continue
                if np.any(inst.demand > cap_max):
                    continue
                placeable = True
                if self.cluster.feasible(inst, w).any():
                    return False
        return placeable

    def _inject_faults(self) -> None:
        for idx in range(self.cluster.n):
            if self.cluster.healthy[idx] and self.rng.random() < self.cfg.fault_rate:
                self.cluster.mark_failed(idx, str(self.rng.choice(["throttle", "exhausted", "crash"])))

    def asid(self) -> float:
        ap, ao, _aa, _av = self.cluster.affinity_stats()
        return 0.0 if ap == 0 else float(1.0 - ao / ap)

    def aaid(self) -> float:
        _ap, _ao, aa, av = self.cluster.affinity_stats()
        return 0.0 if aa == 0 else float(av / aa)

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        h = self.history
        tail = h[-max(1, len(h) // 3):]
        return {
            "cum_reward": self.cum_reward,
            "avg_util": float(np.mean([x["avg_util"] for x in h])),
            "final_util": float(np.mean([x["avg_util"] for x in tail])),
            "avg_imblc": float(np.mean([x["avg_imblc"] for x in h])),
            "final_imblc": float(np.mean([x["avg_imblc"] for x in tail])),
            "asid": float(np.mean([x["asid"] for x in h[len(h) // 2:]])),
            "aaid": float(np.mean([x["aaid"] for x in h[len(h) // 2:]])),
            "rt": float(np.mean([x["rt"] for x in h])),
            "th": float(np.mean([x["th"] for x in h])),
            "fp": float(np.mean([x["fp"] for x in h])),
            "placed": int(self.cluster.total_placed),
            "rejected": int(self.cluster.total_rejected),
            "rounds_used": len(h),
            "unused_frac": float(max(0.0, (self.cfg.rounds - len(h))
                                     / max(self.cfg.rounds, 1))),
            "load_skew": float(np.mean([x.get("skew", 0.0) for x in h])),
            "ended_by": self.ended_by,
        }


def make_env(n_nodes: int, rounds: Optional[int] = None, flavour: str = "cloud",
             trace=None, seed: int = 0, bg_util: float = 0.45,
             load_factor: float = 4.0, faults: bool = False,
             params: ModelParams = ModelParams(), **kw) -> CloudsceEnv:
    if rounds is None:
        rounds = {50: 12, 500: 10, 1000: 10}.get(int(n_nodes),
                                                 max(10, int(round(1.2 * np.sqrt(n_nodes)))))
    cfg = EnvConfig(n_nodes=n_nodes, flavour=flavour, rounds=rounds,
                    bg_util=bg_util, load_factor=load_factor, faults=faults, **kw)
    return CloudsceEnv(cfg, params=params, trace=trace, seed=seed)
