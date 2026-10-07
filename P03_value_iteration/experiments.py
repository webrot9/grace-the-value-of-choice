#!/usr/bin/env python3
"""P03 Part B — the theorem made visible: value iteration contracts by at least
gamma, every sweep.

Left panel: the error ||V_k - V*||_inf on a log axis. The theory says it is
bounded by gamma^k times the initial error, so on a log axis it must be a
straight line of slope log(gamma). The dashed lines are that theory; the solid
ones are the measurement.

Right panel: the same statement as a number, the ratio of consecutive errors,
which the theorem bounds by gamma. On this deterministic grid the bound is
attained until the values become exact; with --slip 0.1 the ratio stays well
below gamma, which the theorem allows just as well.

    python experiments.py            # ~2 s, writes contraction.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.dp import (                # noqa: E402
    error_ratios, gridworld, greedy_policy, policy_evaluation_exact,
    sup_norm, value_iteration,
)

GAMMAS = (0.5, 0.8, 0.9, 0.99)
ITERS = 250
GRID = (5, 5)


def run(gamma: float, iters: int, slip: float = 0.0):
    P, R = gridworld(*GRID, slip=slip, seed=1)
    V, history = value_iteration(P, R, gamma, iters=iters)
    errs = np.array([sup_norm(v, V) for v in history])
    return P, R, V, history, errs


def plot(data: dict, path: str) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for gamma, (errs, ratios) in data.items():
        k = np.arange(len(errs))
        line, = ax.semilogy(k, np.maximum(errs, 1e-17), label=f"$\\gamma$ = {gamma}")
        # The theory on the same axes: gamma^k * ||V_0 - V*||.
        ax.semilogy(k, errs[0] * gamma ** k, "--", color=line.get_color(), lw=1,
                    alpha=0.8)
        bx.plot(np.arange(len(ratios)), ratios, color=line.get_color(),
                label=f"$\\gamma$ = {gamma}")
        bx.axhline(gamma, color=line.get_color(), ls="--", lw=1, alpha=0.6)

    ax.set_ylim(1e-16, None)
    ax.set_xlabel("sweep $k$")
    ax.set_ylabel(r"$\|V_k - V^{*}\|_\infty$  (log)")
    ax.set_title("solid: measured. dashed: $\\gamma^k\\,\\|V_0-V^{*}\\|$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    bx.set_xlabel("sweep $k$")
    bx.set_ylabel(r"$\|V_{k+1}-V^{*}\|_\infty \,/\, \|V_k-V^{*}\|_\infty$")
    bx.set_title("at most $\\gamma$; on a deterministic grid, exactly $\\gamma$")
    bx.set_ylim(0, 1.05); bx.legend(fontsize=8); bx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"  figure -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--slip", type=float, default=0.0,
                    help="probability of slipping sideways (0 = deterministic)")
    ap.add_argument("--out", default="contraction.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()

    t0 = time.time()
    print(f"\ngridworld {GRID[0]}x{GRID[1]}, slip={args.slip}, "
          f"{args.iters} sweeps\n")
    data = {}
    print(f"  {'gamma':>6} {'sweeps to 1e-6':>15} {'1/(1-gamma)':>13} "
          f"{'final ratio':>13} {'|ratio - gamma|':>16}")
    for gamma in GAMMAS:
        P, R, V, history, errs = run(gamma, args.iters, args.slip)
        ratios = error_ratios(history, V)
        reached = np.argmax(errs < 1e-6) if (errs < 1e-6).any() else -1
        gap = abs(ratios[-10:] - gamma).max()
        print(f"  {gamma:6.2f} {reached:15d} {1/(1-gamma):13.1f} "
              f"{ratios[-1]:13.9f} {gap:16.2e}")
        data[gamma] = (errs, ratios)

    # How many sweeps the bound predicts, against how many it takes. The bound
    # is a worst case over *all* MDPs; this one has an absorbing goal and a
    # bounded diameter, and that turns out to matter more than gamma does.
    print(f"\n  sweeps to reach 1e-6, measured against the bound "
          f"log(1/eps)/(1-gamma):")
    print(f"  {'slip':>5} {'gamma':>6} {'measured':>10} {'bound':>8} "
          f"{'exact zero at':>14}")
    for slip in (0.0, 0.1, 0.3):
        for gamma in (0.9, 0.99):
            _, _, V, history, errs = run(gamma, 3000, slip)
            eps = int(np.argmax(errs < 1e-6)) if (errs < 1e-6).any() else -1
            zero = int(np.argmax(errs == 0)) if (errs == 0).any() else -1
            bound = np.log(1 / 1e-6) / (1 - gamma)
            print(f"  {slip:5.1f} {gamma:6.2f} {eps:10d} {bound:8.0f} "
                  f"{zero:14d}")

    # The greedy policy read off the converged values is optimal (theorem 2):
    # evaluating it exactly must return V* itself. The comparison is with the V
    # this run converged to, which is V* only if value_iteration computes it.
    gamma = 0.9
    P, R, V, _, _ = run(gamma, args.iters, args.slip)
    pi = greedy_policy(P, R, V, gamma)
    back = policy_evaluation_exact(P, R, pi, gamma)
    print(f"\n  greedy policy evaluated exactly vs the V of this run: "
          f"max difference {np.abs(back - V).max():.2e}")

    plot(data, args.out)
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
