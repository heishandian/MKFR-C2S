from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

import config
from algorithms import ALGORITHM_ORDER, COLORS
from cloudsce.metrics import convergence_speed, paired_test
from experiments.common import (cli, collect_pairs, describe_mode,
                                episode_ticks, get_trace, improvement,
                                make_result_dir, mode_cfg, save_figure,
                                save_json, setup_matplotlib, smooth, train_one)

EXP = "exp_b_convergence"


SCALE_PROTOCOL: Dict[int, Dict] = {
    10: {"rounds": 60, "batch_ratio": 0.40, "load_factor": 1.6,
         "flavour": "ascend5",
         "episodes": 150, "episode_seeds": [777, 778, 779],
         "eval_every": 10,
         "eval_seeds": [901, 902, 903, 904, 905, 906, 907, 908],
         "env": {"stop_on_infeasible": False,
                 "skew_limit": 0.45, "skew_rounds": 3},
         "scheduler": {"update_every": 4, "lr": 1e-3, "batch": 16,
                       "eps_decay": 0.999, "buffer_size": 2000,
                       "adv_reg": 0.05}},
    5: {"rounds": 60, "batch_ratio": 0.40, "load_factor": 1.0,
        "flavour": "ascend5",
        "episodes": 100, "episode_seeds": [777, 778, 779],
        "eval_every": 5, "eval_seeds": [901, 902, 903],
        "env": {"stop_on_infeasible": False,
                "skew_limit": 0.45, "skew_rounds": 3},
        "scheduler": {"update_every": 4, "lr": 1e-3, "batch": 16,
                      "eps_decay": 0.999, "buffer_size": 2000}},
    50: {"rounds": 12, "batch_ratio": 0.15, "load_factor": 4.0,
         "episodes": 40, "episode_seeds": None, "eval_every": 5},
    500: {"rounds": 10, "batch_ratio": 0.10, "load_factor": 4.0,
          "episodes": 10, "episode_seeds": None},
    1000: {"rounds": 10, "batch_ratio": 0.10, "load_factor": 4.0,
           "episodes": 8, "episode_seeds": None},
}


def _exp_fit(y, taus=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return y, {"R0": float(y[0]), "A": 0.0, "tau": float("inf"), "r2": 1.0}
    k = np.arange(n, dtype=float)
    best = None
    for tau in taus:
        e = 1.0 - np.exp(-k / tau)
        X = np.stack([np.ones(n), e], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        if coef[1] < 0.0:
            coef = np.asarray([float(y.mean()), 0.0])
        pred = X @ coef
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(tau), pred)
    sse, r0, amp, tau, pred = best
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 1e-12 else 0.0
    return pred, {"R0": r0, "A": amp, "tau": tau, "r2": float(r2)}


def run(cfg: Dict) -> Dict:
    print(describe_mode(cfg["mode"]) if "mode" in cfg else "")
    out: Dict = {"config": {k: v for k, v in cfg.items()}, "results": {}}
    for trace in cfg["traces"]:
        get_trace(trace)
        for n in cfg["scales"]:
            key = f"{trace}@{n}"
            proto = dict(SCALE_PROTOCOL.get(int(n), {}))
            if cfg.get("eval_every"):
                proto["eval_every"] = int(cfg["eval_every"])
            episodes = cfg["episodes"].get(n, proto.get("episodes", 40))
            print(f"\n[{key}] {episodes} episodes x {cfg['seeds']} seeds "
                  f"(rounds={proto.get('rounds')}, br={proto.get('batch_ratio')})")
            pairs = collect_pairs(n, trace, seed=7, episodes=2,
                                  max_steps=120 if n >= 500 else 90)
            runs: Dict[str, list] = {a: [] for a in ALGORITHM_ORDER}
            for seed in range(cfg["seeds"]):
                for alg in ALGORITHM_ORDER:
                    tr = train_one(alg, n, trace, episodes, seed=11 + seed,
                                   eval_episodes=1, pairs=pairs,
                                   batch_ratio=proto.get("batch_ratio", 0.15),
                                   rounds=proto.get("rounds"),
                                   flavour=proto.get("flavour", "cloud"),
                                   load_factor=proto.get("load_factor"),
                                   env_kwargs=proto.get("env"),
                                   episode_seeds=proto.get("episode_seeds"),
                                   eval_every=proto.get("eval_every", 0),
                                   eval_seeds=proto.get("eval_seeds"),
                                   verbose=bool(cfg.get("verbose", True)),
                                   show_elapsed=False,
                                   **proto.get("scheduler", {}))
                    runs[alg].append(tr)
                    print(f"    {alg:11s} seed{seed} CR={tr.curve[-1]:9.1f} "
                          f"eval={tr.eval_curve[-1]:9.1f}")
            out["results"][key] = _summarise(runs, trace, n, proto, episodes)
    return out


def _summarise(runs: Dict[str, list], trace: str, n: int,
               proto: Optional[Dict] = None, episodes: int = 0) -> Dict:
    proto = proto or {}

    curves: Dict[str, list] = {}
    for alg in ALGORITHM_ORDER:
        evs = [np.asarray(r.eval_curve, dtype=float) for r in runs[alg]]
        curves[alg] = np.mean([e for e in evs], axis=0).tolist() if evs else [0.0]
    L = max(len(c) for c in curves.values())
    for alg in ALGORITHM_ORDER:
        c = curves[alg]
        while len(c) < L:
            c.append(c[-1])

    tail = max(2, int(0.2 * L))
    conv, conv_iter, samples, fits, fit_curves = {}, {}, {}, {}, {}
    for alg in ALGORITHM_ORDER:
        c = np.asarray(curves[alg], dtype=float)

        smooth_c, fp = _exp_fit(smooth(c, 3))
        fits[alg] = fp
        fit_curves[alg] = [float(x) for x in smooth_c]
        conv[alg] = float(np.mean(c[-tail:]))
        conv_iter[alg] = float(convergence_speed(smooth_c))
        samples[alg] = [float(v) for r in runs[alg]
                        for v in np.asarray(r.eval_curve, dtype=float)[-tail:]]
    stats = paired_test(samples["MKFR-C2S"], samples["HeHiFLDSA"])
    return {
        "trace": trace, "n_nodes": n,
        "converged": conv,
        "convergence_iter": conv_iter,
        "curves": curves,
        "curves_fit": fit_curves,
        "fit_params": fits,
        "mean_difference": {"ref": "HeHiFLDSA", "md": float(stats["md"]),
                            "ci_low": float(stats["ci_low"]),
                            "ci_high": float(stats["ci_high"]),
                            "p": float(stats["p"]), "n": int(stats["n"])},
        "improvement_pct": {
            "vs_DDMTS": improvement(conv["MKFR-C2S"], conv["DDMTS"]),
            "vs_MOTS-ACO": improvement(conv["MKFR-C2S"], conv["MOTS-ACO"]),
            "vs_TVW-RL": improvement(conv["MKFR-C2S"], conv["TVW-RL"]),
            "vs_HeHiFLDSA": improvement(conv["MKFR-C2S"], conv["HeHiFLDSA"]),
        },
        "eval_every": int(proto.get("eval_every", 0) or 0),
        "episodes": int(episodes),
        "eval_seeds": [int(s) for s in (proto.get("eval_seeds") or [])],
        "seconds": float(np.mean([r.seconds for r in runs["MKFR-C2S"]])),
    }


# def plot(res: Dict) -> None:
#     plt = setup_matplotlib()
#     traces = sorted({r["trace"] for r in res["results"].values()})
#     scales = sorted({r["n_nodes"] for r in res["results"].values()})
#     fig, axes = plt.subplots(len(traces), len(scales),
#                              figsize=(3.3 * len(scales), 2.5 * len(traces)),
#                              squeeze=False)
#     for i, trace in enumerate(traces):
#         for j, n in enumerate(scales):
#             ax = axes[i][j]
#             r = res["results"].get(f"{trace}@{n}")
#             if r is None:
#                 ax.axis("off")
#                 continue
#
#             cur = r["curves"]
#             every = int(r.get("eval_every") or 5)
#             for alg in ALGORITHM_ORDER:
#                 y = np.asarray(cur[alg], dtype=float)
#                 ax.plot(np.arange(len(y)) * every, y, color=COLORS[alg],
#                         lw=1.1, label=alg)
#             ax.set_title(f"{trace.capitalize()} trace, N={n}", fontsize=8)
#             n_eval = len(next(iter(cur.values())))
#             last_ep = int((n_eval - 1) * every)
#             ax.set_xticks(episode_ticks(last_ep))
#             ax.set_xlabel("Training episode")
#             ax.set_ylabel("Cumulative reward")
#     handles, labels_ = axes[0][0].get_legend_handles_labels()
#
#     fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
#     fig.legend(handles, labels_, fontsize=6.5, ncol=5, loc="lower center",
#                bbox_to_anchor=(0.5, 0.01), frameon=False)
#     save_figure(fig, "fig6_convergence.png", EXP)
#
#     fig, ax = plt.subplots(figsize=(4.6, 2.4))
#     labels, means, los, his = [], [], [], []
#     for key, r in res["results"].items():
#         md = r["mean_difference"]
#         labels.append(key.replace("alibaba", "Ali").replace("google", "Goog"))
#         means.append(md["md"]); los.append(md["md"] - md["ci_low"])
#         his.append(md["ci_high"] - md["md"])
#     ax.bar(range(len(labels)), means, yerr=[los, his], capsize=3,
#            color="#333333", width=0.55)
#     ax.set_xticks(range(len(labels)))
#     ax.set_xticklabels(labels, rotation=20, fontsize=7)
#     ax.set_ylabel("MD (MKFR-C2S - HeHiFLDSA)")
#     p_min = min((r["mean_difference"]["p"] for r in res["results"].values()),
#                 default=float("nan"))
#     n_obs = max((r["mean_difference"]["n"] for r in res["results"].values()),
#                 default=0)
#     ax.set_title(f"measured MD, 95% CI, p = {p_min:.3g} (n={n_obs})", fontsize=8)
#     fig.tight_layout()
#     save_figure(fig, "fig6b_mean_difference.png", EXP)


def main() -> Dict:
    args = cli("quick")
    cfg = mode_cfg(args.mode)
    cfg["mode"] = args.mode

    if args.mode == "quick" and not getattr(args, "scales", None):
        cfg["scales"] = [10]
        cfg["episodes"] = {10: SCALE_PROTOCOL[10]["episodes"]}
    scales_arg = getattr(args, "scales", None)
    if scales_arg:
        cfg["scales"] = [int(s) for s in str(scales_arg).split(",")]
        cfg["episodes"] = {n: SCALE_PROTOCOL.get(n, {}).get("episodes", 40)
                           for n in cfg["scales"]}
    if args.epochs:
        cfg["episodes"] = {n: args.epochs for n in cfg["scales"]}
    if args.seeds:
        cfg["seeds"] = args.seeds

    cfg["verbose"] = not getattr(args, "quiet", False)
    ev = getattr(args, "eval_every", None)
    if ev:
        cfg["eval_every"] = int(ev)
    res = run(cfg)
    d = make_result_dir(EXP)
    save_json(os.path.join(d, f"results_{args.mode}.json"), res)
    # if not args.no_plot:
    #     plot(res)
    return res


if __name__ == "__main__":
    main()
