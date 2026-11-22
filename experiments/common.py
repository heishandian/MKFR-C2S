from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import config
from algorithms import ALGORITHM_ORDER, COLORS, MARKERS, build, is_trainable
from cloudsce import make_env
from cloudsce.metrics import convergence_speed, mean_ci, paired_test
from cloudsce.trace import TraceSubset, load_subset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

_TRACE_CACHE: Dict[str, TraceSubset] = {}


def get_trace(name: str = "alibaba") -> TraceSubset:
    if name not in _TRACE_CACHE:
        _TRACE_CACHE[name] = load_subset(name, name)
    return _TRACE_CACHE[name]


def make_result_dir(exp: str) -> str:
    d = os.path.join(RESULTS_DIR, exp)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  -> wrote {path}")


@dataclass
class EpisodeResult:
    cum_reward: float
    summary: Dict[str, float]
    history: List[Dict]
    seconds: float


def run_episode(env, sched, train: bool = True, record: bool = True,
                seed: Optional[int] = None) -> EpisodeResult:
    t0 = time.time()

    _skew_limit = env.cfg.skew_limit
    if not train and _skew_limit:
        env.cfg.skew_limit = 0.0
    obs = env.reset(seed)
    sched.reset(env)

    if hasattr(sched, "_eval_mode"):
        sched._eval_mode = not train
    done = False
    k = 0
    while not done:
        a = sched.act(obs)
        res = env.step(a)
        if train:
            sched.observe(obs, a, res.reward, res.obs, res.done, res.info)
        if res.info.get("round_end"):
            sched.on_round_end(res.info)
        obs = res.obs
        done = res.done
        k += 1
        if k > 400000:
            break
    if train:
        sched.end_episode()
    out = EpisodeResult(env.cum_reward, env.summary(),
                        env.history if record else [], time.time() - t0)
    env.cfg.skew_limit = _skew_limit
    return out


def collect_pairs(n_nodes: int, trace_name: str, seed: int, episodes: int = 2,
                  max_steps: int = 220, batch_ratio: float = 0.1
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    trace = get_trace(trace_name)
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for e in range(episodes):
        env = make_env(n_nodes, trace=trace, seed=seed + 977 * e,
                       rounds=6, batch_ratio=batch_ratio)
        obs = env.reset(seed + 977 * e)
        done = False; k = 0
        rng = np.random.default_rng(seed + e)
        while not done and k < max_steps:
            pairs.append((obs.img.copy(), obs.tokens.copy()))
            cand = np.where(obs.mask)[0]
            a = int(cand[rng.integers(len(cand))]) if cand.size else 0
            res = env.step(a)
            obs = res.obs; done = res.done; k += 1
    return pairs


def prepare_scheduler(name: str, seed: int, pairs: Optional[list] = None,
                      pretrain_epochs: int = 6, verbose: bool = False,
                      **kw):
    sched = build(name, seed=seed, **kw)
    if getattr(sched, "pretrain", None) is not None and pairs:
        sched.pretrain(pairs, epochs=pretrain_epochs, verbose=verbose)
    return sched


@dataclass
class TrainResult:
    name: str
    curve: List[float]
    eval_metrics: Dict[str, float]
    episode_metrics: List[Dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0
    eval_curve: List[float] = field(default_factory=list)
    eval_at: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"name": self.name,
                "curve": [float(x) for x in self.curve],
                "eval_curve": [float(x) for x in self.eval_curve],
                "eval_at": [int(x) for x in self.eval_at],
                "eval": {k: float(v) for k, v in self.eval_metrics.items()},
                "convergence": convergence_speed(self.curve),
                "seconds": self.seconds}


def train_one(name: str, n_nodes: int, trace_name: str, episodes: int, seed: int,
              eval_episodes: int = 2, pairs: Optional[list] = None,
              batch_ratio: Optional[float] = None, flavour: str = "cloud",
              params=None, verbose: bool = True, rounds: Optional[int] = None,
              episode_seeds: Optional[Sequence[int]] = None,
              load_factor: Optional[float] = None, eval_every: int = 0,
              eval_seeds: Optional[Sequence[int]] = None,
              env_kwargs: Optional[Dict] = None, show_elapsed: bool = True,
              **kw) -> TrainResult:
    trace = get_trace(trace_name)
    br = batch_ratio if batch_ratio is not None else default_batch_ratio(n_nodes)
    env_kw = {"params": params} if params is not None else {}
    if rounds is not None:
        env_kw["rounds"] = rounds
    if load_factor is not None:
        env_kw["load_factor"] = load_factor
    if env_kwargs:
        env_kw.update(env_kwargs)
    env = make_env(n_nodes, trace=trace, seed=seed, flavour=flavour,
                   batch_ratio=br, **env_kw)
    t0 = time.time()
    sched = prepare_scheduler(name, seed, pairs, **kw)
    trainable = is_trainable(name)
    curve: List[float] = []
    emetrics: List[Dict[str, float]] = []
    eval_curve: List[float] = []
    eval_at: List[int] = []
    ev_seeds = list(eval_seeds) if eval_seeds else list(episode_seeds or [7])

    def _eval(ep: int) -> None:
        vals = [run_episode(env, sched, train=False, record=False, seed=int(s0)).cum_reward
                for s0 in ev_seeds]
        eval_curve.append(float(np.mean(vals)))
        eval_at.append(int(ep))

        if verbose:
            elapsed = f"  ({time.time() - t0:5.0f}s)" if show_elapsed else ""
            print(f"    {name:11s} ep {ep:4d}/{episodes}  "
                  f"eval={eval_curve[-1]:8.1f}{elapsed}", flush=True)

    if verbose:
        print(f"    {name:11s} start ({episodes} episodes, "
              f"{n_nodes} nodes, eval every {eval_every or '-'})", flush=True)
    if eval_every:
        _eval(0)
    for ep in range(episodes):
        es = None if not episode_seeds else int(episode_seeds[ep % len(episode_seeds)])
        res = run_episode(env, sched, train=trainable, seed=es)
        curve.append(res.cum_reward)
        emetrics.append(res.summary)
        if eval_every and trainable and (ep + 1) % eval_every == 0:
            _eval(ep + 1)
        if verbose and not eval_every and (ep + 1) % max(1, episodes // 5) == 0:
            ma = float(np.mean(curve[-min(10, len(curve)):]))
            elapsed = f"  ({time.time() - t0:5.0f}s)" if show_elapsed else ""
            print(f"    {name:11s} ep {ep + 1:4d}/{episodes}  CR(ma10)={ma:9.1f} "
                  f"util={res.summary['avg_util']:.3f}{elapsed}", flush=True)

    evals = []
    if eval_seeds:
        final_seeds = [int(s) for s in ev_seeds]
    else:
        final_seeds = [int(ev_seeds[k % len(ev_seeds)])
                       for k in range(max(1, eval_episodes))]
    for s0 in final_seeds:
        res = run_episode(env, sched, train=False, seed=s0)
        evals.append(res.summary)
    keys = [k for k in evals[0] if isinstance(evals[0][k], (int, float, np.floating, np.integer))]
    eval_metrics = {k: float(np.mean([e[k] for e in evals])) for k in keys}
    if not eval_curve:
        eval_curve = [float(np.mean(curve))] * 2
        eval_at = [0, episodes]
    return TrainResult(name, curve, eval_metrics, emetrics, time.time() - t0,
                       eval_curve, eval_at)


def default_batch_ratio(n_nodes: int) -> float:
    return 0.15 if n_nodes <= 50 else 0.10


def aggregate(results: Sequence[TrainResult], n_nodes: int) -> Dict:
    out = {"n_nodes": n_nodes, "algorithms": {}}
    for r in results:
        out["algorithms"][r.name] = r.to_dict()
    return out


def improvement(better: float, worse: float) -> float:
    if abs(worse) < 1e-12:
        return 0.0
    return (better - worse) / abs(worse) * 100.0


def order(names: Iterable[str]) -> List[str]:
    return [n for n in ALGORITHM_ORDER if n in set(names)]


VIEW_COMBOS: List[Tuple[bool, bool, bool, bool]] = [
    (False, False, False, False),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, True),
    (True, False, False, True),
    (False, True, False, True),
    (False, False, True, True),
    (True, True, False, True),
    (True, False, True, True),
    (False, True, True, True),
    (True, True, True, True),
]


def combo_label(combo: Sequence[bool]) -> str:
    names = ("RV", "WV", "AV", "AAV")
    return "+".join(n for n, on in zip(names, combo) if on) or "none"


def bootstrap_ci(x: np.ndarray, n: int = 500, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    if len(x) < 3:
        return float(np.mean(x)), float(np.mean(x))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(x, len(x), replace=True))) for _ in range(n)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 160,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans",
    })
    return plt


def smooth(y: np.ndarray, k: int = 9) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k - 1 - k // 2, y[-1])])
    return np.convolve(pad, ker, mode="valid")


def episode_ticks(last: int, n: int = 5) -> List[int]:
    last = int(last)
    if last <= 0:
        return [0]
    step = max(1, int(round(last / max(1, n - 1) / 10.0)) * 10) if last >= 10 else 1
    ticks = list(range(0, last + 1, step))
    if ticks[-1] != last:
        ticks.append(last)
    return ticks


MODES = {
    "quick": (("alibaba",), (50,), {50: 16}, 1),
    "standard": (("alibaba", "google"), (50, 500, 1000),
                 {50: 40, 500: 10, 1000: 8}, 2),
    "full": (("alibaba", "google"), (50, 500, 1000),
             {50: 240, 500: 60, 1000: 40}, 5),
}


def mode_cfg(mode: str = "quick", **over):
    traces, scales, eps, seeds = MODES[mode]
    cfg = {"traces": list(traces), "scales": list(scales),
           "episodes": dict(eps), "seeds": int(seeds)}
    cfg.update(over)
    return cfg


def describe_mode(mode: str) -> str:
    traces, scales, eps, seeds = MODES[mode]
    return (f"mode={mode}  traces={list(traces)}  scales={list(scales)}  "
            f"episodes={eps}  seeds={seeds}")


def save_figure(fig, name: str, exp: str) -> str:
    import matplotlib.pyplot as plt
    d = make_result_dir(exp)
    path = os.path.join(d, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {path}")
    return path


def cli(default_mode: str = "quick"):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=default_mode, choices=list(MODES))
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the number of training episodes at 50 nodes")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--only", default=None,
                    help="run only a part of an experiment (comma separated)")
    ap.add_argument("--scales", default=None,
                    help="override the cluster sizes, e.g. --scales 5 or 5,50")
    ap.add_argument("--quiet", action="store_true",
                    help="only one line per algorithm (no per-evaluation progress)")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="greedy-evaluation period in episodes (1 = every "
                         "episode; overrides the per-scale protocol)")
    return ap.parse_args()
