#!/usr/bin/env python3
"""P05 Part B — the theorem made visible: certainty equivalence.

Add Gaussian noise to a linear system and the optimal gain does not change. Not
approximately: the Riccati recursion takes Fx, Fu, Cx, Cu and the horizon, and
sigma is not one of its arguments. What the noise changes is the cost you pay,
by an additive amount that has nothing to do with where you are.

Left panel: the gains K_t for four noise levels, plotted on top of each other.
Right panel: the cost, which does move, against the theoretical offset.

    python experiments.py            # ~10 s, writes certainty.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.control import (           # noqa: E402
    cost_to_go, ilqr, linearize, lqr_backward, rollout,
)

DT = 0.1
HORIZON = 60
SIGMAS = (0.0, 0.05, 0.2, 0.5)
# 2000 e non 200: con 200 lo scarto fra costo misurato e teoria sembra
# sistematico (-4% a ogni sigma) perche' i seed sono gli stessi per tutti i
# sigma, quindi gli errori sono correlati. E' la domanda 2 del report.
SEEDS = 2000
X0 = np.array([1.0, 0.0])


def system():
    Fx = np.array([[1.0, DT], [0.0, 1.0]])
    Fu = np.array([[0.0], [DT]])
    Cx = np.eye(2)
    Cu = np.array([[0.5]])
    return Fx, Fu, Cx, Cu


def unicycle(x, u, dt=DT):
    """A genuinely non-linear system, for the iLQR half: (px, py, theta) with
    forward speed and turn rate as controls."""
    px, py, th = x
    v, w = u
    return np.array([px + dt * v * np.cos(th),
                     py + dt * v * np.sin(th),
                     th + dt * w])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="certainty.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    Fx, Fu, Cx, Cu = system()
    Ks, Ps = lqr_backward(Fx, Fu, Cx, Cu, HORIZON)
    predicted = cost_to_go(Ps[0], X0)

    print(f"\nLQR on a double integrator, horizon {HORIZON}, "
          f"{args.seeds} seeds per noise level\n")
    print(f"  {'sigma':>6} {'gain identical to sigma=0':>26} "
          f"{'mean cost':>11} {'cost - deterministic':>21}")

    gains, costs = {}, {}
    for sigma in SIGMAS:
        # Recomputed from scratch for each sigma, to make the point that the
        # backward pass has no way of knowing about it.
        Ks_s, _ = lqr_backward(Fx, Fu, Cx, Cu, HORIZON)
        identical = all(np.array_equal(a, b) for a, b in zip(Ks, Ks_s))
        cs = [rollout(Fx, Fu, Cx, Cu, Ks_s, X0, sigma=sigma, seed=s)[1]
              for s in range(args.seeds)]
        gains[sigma] = np.array([K[0, 0] for K in Ks_s])
        costs[sigma] = np.array(cs)
        print(f"  {sigma:6.2f} {str(identical):>26} {np.mean(cs):11.3f} "
              f"{np.mean(cs) - predicted:21.3f}")

    print(f"\n  predicted deterministic cost  x0^T P_0 x0 = {predicted:.6f}")

    # The additive offset the theory predicts: sigma^2 * sum_t trace(P_{t+1}).
    print(f"\n  the noise adds a term that does not depend on x0:")
    print(f"  {'sigma':>6} {'measured extra':>15} {'std error':>10} "
          f"{'sigma^2 sum tr(P)':>18} {'gap in SE':>10}")
    trace_sum = sum(np.trace(P) for P in Ps[1:])
    for sigma in SIGMAS[1:]:
        extra = np.mean(costs[sigma]) - predicted
        se = np.std(costs[sigma]) / np.sqrt(len(costs[sigma]))
        theory = sigma ** 2 * trace_sum
        print(f"  {sigma:6.2f} {extra:15.3f} {se:10.3f} {theory:18.3f} "
              f"{(extra - theory) / se:+10.1f}")

    # iLQR on the unicycle: the same machinery where the system is not linear.
    x0 = np.array([2.0, 1.5, 0.0])
    Cx3, Cu3 = np.eye(3), 0.1 * np.eye(2)
    xs, us = ilqr(unicycle, Cx3, Cu3, x0, np.zeros((40, 2)), iters=15, alpha=0.7)
    cost = sum(float(x @ Cx3 @ x + u @ Cu3 @ u) for x, u in zip(xs[:-1], us))
    cost += float(xs[-1] @ Cx3 @ xs[-1])
    print(f"\n  iLQR on the unicycle: |x_0| = {np.linalg.norm(x0):.3f} "
          f"-> |x_T| = {np.linalg.norm(xs[-1]):.4f}, total cost {cost:.3f}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for sigma, style in zip(SIGMAS, ("-", "--", ":", "-.")):
        ax.plot(gains[sigma], style, lw=2.2, alpha=0.75,
                label=f"$\\sigma$ = {sigma}")
    ax.set_xlabel("t"); ax.set_ylabel("$K_t$ (position gain)")
    ax.set_title("four noise levels, four curves — they are the same curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    bx.errorbar(SIGMAS, [costs[s].mean() for s in SIGMAS],
                yerr=[costs[s].std() for s in SIGMAS], marker="o",
                capsize=4, label="measured cost")
    bx.plot(SIGMAS, [predicted + s ** 2 * trace_sum for s in SIGMAS], "k--",
            label=r"$x_0^\top P_0 x_0 + \sigma^2 \sum_t \mathrm{tr}(P_t)$")
    bx.set_xlabel("$\\sigma$"); bx.set_ylabel("cost")
    bx.set_title("the policy ignores the noise; the bill does not")
    bx.legend(fontsize=9); bx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
