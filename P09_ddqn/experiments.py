#!/usr/bin/env python3
"""P09 Part B — maximization bias, three times over.

Panel 1: the bias with no reinforcement learning in it. Ten unbiased estimates
of the same number; their maximum is not unbiased. Split the samples in two and
it is.

Panel 2: the same bias inside Q-learning, on Sutton & Barto's two-state MDP.
The agent prefers a strictly worse action for hundreds of episodes.

Panel 3: the same bias inside a neural network. DQN and Double DQN on a corridor
whose q* we can compute exactly, so the overestimation is measured against the
truth rather than against a proxy for it.

    python experiments.py            # ~50 s, writes maxbias.png
    python experiments.py --quick    # tabular panels only, ~10 s
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.maxbias import (           # noqa: E402
    double_estimate, max_of_estimates, overestimation, run_bias_mdp,
)

TRUE_Q_B = -0.1
N_B_ACTIONS = 10
N_SAMPLES = 20
N_REPS = 20_000
EPISODES = 300
SEEDS = 100
DEEP_SEEDS = 4
GAMMA = 0.95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--deep-seeds", type=int, default=DEEP_SEEDS)
    ap.add_argument("--quick", action="store_true",
                    help="skip the neural half")
    ap.add_argument("--out", default="maxbias.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    # ------------------------------------------------ 1. the estimator alone
    rng = np.random.default_rng(0)
    print(f"\nA maximum over {N_B_ACTIONS} unbiased estimates of "
          f"{TRUE_Q_B}, {N_REPS} repetitions\n")
    print(f"  {'samples per action':>19} {'max of estimates':>18} "
          f"{'double estimate':>17}")
    print("  " + "-" * 58)
    for n in (5, 10, 20, 40, 80, 160):
        m = max_of_estimates(rng, N_B_ACTIONS, TRUE_Q_B, 1.0, n, N_REPS).mean()
        d = double_estimate(rng, N_B_ACTIONS, TRUE_Q_B, 1.0, n, N_REPS).mean()
        print(f"  {n:19d} {m:18.4f} {d:17.4f}")
    print(f"\n  the truth is {TRUE_Q_B}. Neither column contains a single "
          f"biased estimate:\n  every action's value is estimated by an "
          f"unbiased sample mean. The bias is\n  created by the max.\n")

    dist_max = max_of_estimates(rng, N_B_ACTIONS, TRUE_Q_B, 1.0, N_SAMPLES,
                                N_REPS)
    dist_dbl = double_estimate(rng, N_B_ACTIONS, TRUE_Q_B, 1.0, N_SAMPLES,
                               N_REPS)

    # ------------------------------------------------ 2. inside Q-learning
    print(f"Sutton & Barto's two-state MDP, {args.seeds} seeds, "
          f"{EPISODES} episodes\n")
    curves, visits, final_q = {}, {}, {}
    for double in (False, True):
        runs = [run_bias_mdp(EPISODES, N_B_ACTIONS, double=double, seed=s)
                for s in range(args.seeds)]
        left = np.array([r[0] for r in runs])
        maxq = np.array([r[1] for r in runs])
        name = "Double Q-learning" if double else "Q-learning"
        curves[name] = left.mean(axis=0)
        visits[name] = left.sum(axis=1).mean()
        final_q[name] = overestimation(maxq[:, -1], TRUE_Q_B).mean() + TRUE_Q_B
    print(f"  {'algorithm':>18} {'left, first 50 ep':>18} "
          f"{'left, last 50 ep':>17} {'visits to B':>12}")
    print("  " + "-" * 70)
    for name, c in curves.items():
        print(f"  {name:>18} {100 * c[:50].mean():17.1f}% "
              f"{100 * c[-50:].mean():16.1f}% {visits[name]:12.1f}")
    print(f"\n  an epsilon-greedy agent with the correct values takes left "
          f"5.0% of the time.\n")
    print(f"  max_a Q(B,a) at the end, against a true value of {TRUE_Q_B}: "
          + ", ".join(f"{k} {v:+.4f}" for k, v in final_q.items()))
    print(f"  read that line together with the visit counts before drawing a "
          f"conclusion\n  from it. It is question 2 of the report.\n")

    # ------------------------------------------------ 3. inside a network
    deep = None
    if not args.quick:
        import torch                                        # noqa: F401
        from rl_lab.deep import REWARD_NOISE, collect, exact_q, fit_offline

        centres, Q_star = exact_q(GAMMA)
        probe = np.linspace(0.02, 0.85, 40)
        idx = np.clip((probe * len(centres)).astype(int), 0, len(centres) - 1)
        truth = float(Q_star[idx].max(axis=1).mean())
        print(f"Corridor with a known q*, reward noise sd = {REWARD_NOISE}, "
              f"{args.deep_seeds} seeds\n")
        print(f"  mean over 40 probe states of max_a q*(s,a) = {truth:.4f}")
        print(f"  both algorithms are fitted on the same buffer, seed by "
              f"seed\n")
        deep = {"DQN": [], "Double DQN": []}
        print(f"  {'seed':>5} {'DQN':>10} {'vs q*':>9} {'Double DQN':>12} "
              f"{'vs q*':>9} {'DQN - DDQN':>12}")
        print("  " + "-" * 62)
        for s in range(args.deep_seeds):
            buffer = collect(seed=s)              # one buffer, both algorithms
            a = fit_offline(buffer, double=False, seed=s, probe=probe)[0]
            b = fit_offline(buffer, double=True, seed=s, probe=probe)[0]
            deep["DQN"].append(a)
            deep["Double DQN"].append(b)
            print(f"  {s:5d} {a[-1]:10.4f} {a[-1] - truth:+9.4f} "
                  f"{b[-1]:12.4f} {b[-1] - truth:+9.4f} "
                  f"{a[-1] - b[-1]:+12.4f}")
        deep = {k: np.array(v) if k != "truth" else v for k, v in deep.items()}
        ends_a, ends_b = deep["DQN"][:, -1], deep["Double DQN"][:, -1]
        diff = ends_a - ends_b
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print(f"\n  paired difference DQN - Double DQN: {diff.mean():+.4f} "
              f"+- {se:.4f} (standard error),\n  positive on "
              f"{int((diff > 0).sum())} of {len(diff)} seeds.")
        print(f"  DQN alone is above q* on {int((ends_a > truth).sum())} of "
              f"{len(ends_a)} seeds, with a spread of {ends_a.std(ddof=1):.4f} "
              f"across seeds.")
        print(f"  Those two sentences do not say the same thing. Report "
              f"question 3.\n")
        deep["truth"] = truth

    # ----------------------------------------------------------------- figure
    n_panels = 2 if args.quick else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.5))
    ax = axes[0]
    bins = np.linspace(-1.0, 1.2, 90)
    ax.hist(dist_max, bins=bins, alpha=0.6, label="max of 10 estimates",
            density=True)
    ax.hist(dist_dbl, bins=bins, alpha=0.6, label="double estimate",
            density=True)
    ax.axvline(TRUE_Q_B, color="k", ls="--", lw=1.6,
               label=f"truth = {TRUE_Q_B}")
    ax.set_xlabel("estimate"); ax.set_ylabel("density")
    ax.set_title("no environment, no bootstrapping, still biased")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    bx = axes[1]
    for name, c in curves.items():
        bx.plot(100 * c, lw=2, label=name)
    bx.axhline(5.0, color="k", ls="--", lw=1.4, label="optimal (5%)")
    bx.set_xlabel("episode"); bx.set_ylabel("% left from A")
    bx.set_title("the same bias, choosing the worse action")
    bx.legend(fontsize=9); bx.grid(alpha=0.3)

    if not args.quick:
        cx = axes[2]
        for name in ("DQN", "Double DQN"):
            runs = deep[name]
            x = np.arange(runs.shape[1])
            med = np.median(runs, axis=0)
            lo, hi = np.percentile(runs, 25, axis=0), np.percentile(runs, 75, axis=0)
            cx.plot(x, med, lw=2, label=name)
            cx.fill_between(x, lo, hi, alpha=0.18)
        cx.axhline(deep["truth"], color="k", ls="--", lw=1.6,
                   label=r"$\max_a q^{*}$, exact")
        cx.set_xlabel("probe (every 100 gradient steps)")
        cx.set_ylabel(r"mean $\max_a Q_w(s,a)$")
        cx.set_title("the same bias, in a neural network")
        cx.legend(fontsize=9); cx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
