#!/usr/bin/env python3
"""P02 Part B — the theorem made visible: the posterior variance *is* the
exploration, and it shows up in the tail, not in the average.

One environment, same seeds, two agents. `thompson` draws from the posterior;
`posterior-greedy` is the same agent with the variance thrown away — it acts on
the posterior mean.

The result is more interesting than "greedy is worse", and it is the reason this
figure has two panels. On the *typical* run posterior-greedy is the better agent:
its median regret is lower than Thompson's. But on roughly a quarter of the seeds
it commits to the wrong course in one context and pays a fixed gap for the rest
of the horizon. Thompson never does. Look at the average alone and you see
nothing; look at the spread and the failure is obvious.

    python experiments.py            # ~5 s, writes regret.png

This is a VERIFICA session (labs/PATTERN.md): there is no training to tune and
no report to write. Predict which agent has the lower *median* before you run it,
and then predict which one you would deploy.
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from rl_lab.contextual import contextual_regret, run_contextual  # noqa: E402

# The practical's course recommender.
CTR = np.array([
    [0.30, 0.20, 0.03],   # AI engineer
    [0.08, 0.30, 0.05],   # management engineer
    [0.05, 0.10, 0.10],   # high school kid
])
CONTEXTS = ["AI engineer", "management engineer", "high-school kid"]
COURSES = ["RL course", "optimisation course", "trading course"]
HORIZON = 3000
# 60 e non 30: con 30 seed la mediana e' gia' stabile ma la coda no, e qui
# la coda e' il punto — il fallimento di posterior-greedy sta li'.
SEEDS = 60
ALGOS = {
    "Thompson (samples the posterior)": "thompson",
    "posterior-greedy (mean only)": "posterior-greedy",
    "uniform": "uniform",
}


def regret_curve(algorithm: str, seed: int, horizon: int) -> np.ndarray:
    states, actions = run_contextual(CTR, horizon, algorithm, seed)
    step = CTR.max(axis=1)[states] - CTR[states, actions]
    return np.cumsum(step)


def sweep(horizon: int, seeds: int) -> dict[str, np.ndarray]:
    return {label: np.stack([regret_curve(a, s, horizon) for s in range(seeds)])
            for label, a in ALGOS.items()}


def plot(curves: dict[str, np.ndarray], horizon: int, seeds: int, path: str) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    t = np.arange(1, horizon + 1)

    # Left: median and IQR, plus every single run of posterior-greedy in faint
    # lines. The band alone would hide the bifurcation, because fewer than half
    # the runs fail: the individual traces are the point.
    colours = {}
    for (label, runs), col in zip(curves.items(), ("C0", "C3", "C7")):
        colours[label] = col
        med = np.median(runs, axis=0)
        lo, hi = np.percentile(runs, [25, 75], axis=0)
        ax.plot(t, med, col, label=f"{label} — median")
        ax.fill_between(t, lo, hi, color=col, alpha=0.18)
    for run in curves["posterior-greedy (mean only)"]:
        ax.plot(t, run, "C3", alpha=0.13, lw=0.7)

    # The theory on the same axes: an agent stuck on the wrong course in one of
    # the three contexts pays 0.10 every third step, i.e. t/30 exactly.
    ax.plot(t, t / 30.0, "k--", lw=1.2, label="one wrong context for ever: $t/30$")
    ax.set_xlabel("t"); ax.set_ylabel("cumulative regret")
    ax.set_title(f"{len(CTR)} contexts, {seeds} seeds — faint red = single runs")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(0, max(np.percentile(r, 97, axis=0)[-1] for r in curves.values()) * 1.05)

    # Right: the distribution of final regret, sorted. A curve that stays flat
    # and then turns up is an agent that is fine most of the time and
    # catastrophic sometimes — which is exactly what an average hides.
    for label, runs in curves.items():
        bx.plot(np.sort(runs[:, -1]), np.linspace(0, 100, len(runs)),
                colours[label], marker=".", ms=4, label=label)
    bx.axvline(horizon / 30.0, color="k", ls="--", lw=1.2)
    bx.text(horizon / 30.0, 8, "  one wrong\n  context", fontsize=8)
    bx.set_xlabel("final regret"); bx.set_ylabel("% of seeds below")
    bx.set_title("the same data as a distribution: the tail is the story")
    bx.legend(fontsize=8); bx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="regret.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\n{len(ALGOS)} agents x {args.seeds} seeds x T={args.horizon}")
    curves = sweep(args.horizon, args.seeds)

    print("\n  final regret (mean +- std over seeds):")
    for label, runs in curves.items():
        print(f"    {label:34} {runs[:, -1].mean():7.1f} +- {runs[:, -1].std():5.1f}")

    # The number that makes the theorem quantitative: regret per step over the
    # second half of the run. Sublinear means it keeps falling; linear means it
    # settles on the gap.
    # The mean is the wrong summary here, and saying so is half the lesson.
    threshold = args.horizon / 30.0
    print(f"\n  median, 90th percentile, and how often the run ends above "
          f"{threshold:.0f}\n  (which is the regret of being wrong in one context "
          f"for the whole horizon):")
    for label, runs in curves.items():
        f = runs[:, -1]
        bad = int((f > threshold).sum())
        print(f"    {label:34} median {np.median(f):6.1f}   p90 {np.percentile(f, 90):6.1f}"
              f"   above: {bad:2d}/{len(f)}")

    print("\n  regret per step, second half of the run:")
    for label, runs in curves.items():
        half = args.horizon // 2
        rate = (runs[:, -1] - runs[:, half]).mean() / (args.horizon - half)
        print(f"    {label:34} {rate:.4f}"
              + ("   <- flat: this agent has stopped learning" if rate > 0.02 else ""))

    plot(curves, args.horizon, args.seeds, args.out)
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
