#!/usr/bin/env python3
"""
Part B: run the agents, and put the theory on the same axes.

    python experiments.py                 # the main figure
    python experiments.py --broken        # Part C: break UCB on purpose

Runtime: about 40 seconds on a laptop CPU for the default settings. No GPU, no
environment library: a Bernoulli bandit is 15 lines of numpy, which is exactly
why this is the first lab.

What you should see, and how it relates to what the lecture proved:
  * greedy (one pull per arm, then commit) -> a straight line: slope 1 in the
    log-log panel, the linear regret of the theorem, visible inside one run.
  * explore-then-commit -> slope 1 while exploring, then flat in the median (the
    runs that committed to the best arm pay nothing more). Its T^(2/3) is how
    the FINAL regret grows when T grows and m is re-tuned to T: one run of fixed
    length cannot show it. Report question 2 is about that experiment.
  * UCB, Thompson -> curves that keep bending down. sqrt(T) is the worst case
    over all instances; on this instance, at this T, the slope is still on its
    way down to the log T of the gap-dependent bound.
  * "UCB" is yours, the anytime version (delta_t = delta / t^3, no T needed);
    "UCB, T known (lecture)" is the lecture's, given in select_ucb_known_horizon.
    Same rate, different constants: compare the two final numbers.
The dotted lines in the right panel are slope guides (1, 2/3, 1/2) to read the
curves against; the dashed line in the left panel is the UCB bound.

The figure draws the median over seeds; the printout gives the mean as well.
When the two are far apart, a few seeds went very differently from the others:
that is worth a sentence in the report.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_lab.bandits import pseudo_regret, run_bandit

MU = np.array([0.50, 0.55, 0.45, 0.40, 0.30])   # gaps: 0.05 is the hard one
HORIZON = 4000
SEEDS = 30
ALGOS = [("greedy", "greedy (no exploration)"), ("etc", "explore-then-commit"),
         ("eps", "eps-greedy (0.1)"), ("ucb", "UCB"),
         ("ucb_T", "UCB, T known (lecture)"), ("thompson", "Thompson")]


def regret_curve(algorithm: str, seed: int, horizon: int, **kw) -> np.ndarray:
    """Cumulative pseudo-regret as a function of t, for one run."""
    pulls = run_bandit(MU, horizon, algorithm, seed, **kw)
    gaps = MU.max() - MU[pulls]
    return np.cumsum(gaps)


def sweep(horizon: int = HORIZON, seeds: int = SEEDS, **kw) -> dict:
    out = {}
    for algo, label in ALGOS:
        runs = np.stack([regret_curve(algo, s, horizon, **kw) for s in range(seeds)])
        out[label] = runs
    return out


def theory_curves(horizon: int) -> dict:
    t = np.arange(1, horizon + 1)
    anchor = horizon // 2
    gaps = np.array([MU.max() - m for m in MU if m < MU.max()])
    # gap-dependent UCB bound, shape only: sum_a 8 log(t) / Delta_a
    ucb = 8 * np.log(np.maximum(t, 2)) * (1 / gaps).sum()
    return {
        r"$\propto T$ (linear)": t / t[anchor],
        r"$\propto T^{2/3}$": t ** (2 / 3) / t[anchor] ** (2 / 3),
        r"UCB bound $\sum_a \frac{8\log T}{\Delta_a}$": ucb,
    }


def plot(curves: dict, horizon: int, path: str, title: str) -> None:
    """Two panels. Left: what the regret looks like. Right: the same in log-log,
    where a growth rate is a *slope* and the theorem becomes visible."""
    t = np.arange(1, horizon + 1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))

    for label, runs in curves.items():
        med = np.median(runs, 0)
        lo, hi = np.percentile(runs, [25, 75], axis=0)
        ax.plot(t, med, label=label, lw=1.8)
        ax.fill_between(t, lo, hi, alpha=0.12, lw=0)
        bx.loglog(t[9:], med[9:], lw=1.8)

    gaps = np.array([MU.max() - m for m in MU if m < MU.max()])
    bound = 8 * np.log(np.maximum(t, 2)) * (1 / gaps).sum()
    ax.plot(t, bound, "--", lw=1.1, color="0.3",
            label=r"UCB bound $\sum_a 8\log T/\Delta_a$")

    # slope guides in log-log, anchored at t = horizon//10
    a = horizon // 10
    for expo, lbl in [(1.0, r"slope 1  ($\propto T$)"),
                      (2 / 3, r"slope 2/3  ($\propto T^{2/3}$)"),
                      (0.5, r"slope 1/2  ($\propto \sqrt{T}$)")]:
        ref = np.median(curves["greedy (no exploration)"], 0)[a] * 0.9
        bx.loglog(t[9:], ref * (t[9:] / t[a]) ** expo, ":", lw=1.0, color="0.35")
        bx.annotate(lbl, (t[-1], ref * (t[-1] / t[a]) ** expo), fontsize=7,
                    ha="right", va="bottom", color="0.3")

    ax.set_xlabel("t"); ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title(title); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    top = max(np.median(r, 0)[-1] for r in curves.values()) * 1.35
    ax.set_ylim(0, top)   # the bound leaves the top of the frame: it is loose,
    # and how loose is the point of report question 2
    bx.set_xlabel("t (log)"); bx.set_ylabel("regret (log)")
    bx.set_title("same data, log-log: the growth rate is the slope")
    bx.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--broken", action="store_true",
                    help="Part C: UCB with a confidence level that does not shrink")
    ap.add_argument("--out", default="regret.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()

    print(f"\n{len(ALGOS)} algorithms x {args.seeds} seeds x T={args.horizon}")
    curves = sweep(args.horizon, args.seeds)
    print("\n  final regret (mean +- std over seeds; the median is the curve in the figure):")
    for label, runs in curves.items():
        fin = runs[:, -1]
        print(f"    {label:26} {fin.mean():7.1f} +- {fin.std():5.1f}"
              f"   median {np.median(fin):7.1f}")
    gaps = np.array([MU.max() - m for m in MU if m < MU.max()])
    bound = 8 * np.log(args.horizon) * (1 / gaps).sum()
    ucb_emp = curves["UCB"][:, -1].mean()
    print(f"\n  UCB: empirical {ucb_emp:.0f}  vs  gap-dependent bound {bound:.0f}"
          f"  ->  the bound is {bound/ucb_emp:.1f}x loose (see report.md Q2)")
    plot(curves, args.horizon, args.out,
         f"Bernoulli bandit, K={len(MU)}, {args.seeds} seeds")

    if args.broken:
        print("\n  Part C: same UCB, but delta_t is constant (see report.md Q3)")
        print("  Edit select_ucb to use `delta_t = delta` and re-run this flag.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
