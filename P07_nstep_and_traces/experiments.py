#!/usr/bin/env python3
"""P07 Part B — the theorem made visible: the forward and backward views are one
algorithm, and n has an optimum.

Left panel: the largest disagreement between the forward view (which needs the
whole future) and the backward view (which needs one number per state) over the
same episodes, as a function of lambda. It sits at machine precision for every
lambda, which is what "equivalent" means when it is stated properly.

Right panel: the root-mean-square error of n-step TD against the exact V^pi,
as a function of n, for four step sizes. The minimum is at an intermediate n and
it moves when alpha moves — neither end of the interval is the right answer.

    python experiments.py            # ~25 s, writes traces.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.traces import (            # noqa: E402
    backward_update, forward_update, gridworld, n_step_td,
    policy_evaluation_stochastic, rms_error, sample_episode, uniform_policy,
    visitable_states,
)

GAMMA = 0.9
SLIP = 0.1
GRID_SEED = 3
LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)
EQUIV_EPISODES = 20
NS = (1, 2, 3, 5, 8, 13, 21, 34, 60)
ALPHAS = (0.01, 0.02, 0.05, 0.1)
SEEDS = 30
EPISODES_PER_RUN = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="traces.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    P, R = gridworld(4, 4, slip=SLIP, seed=GRID_SEED)
    S = P.shape[0]
    terminal = S - 1
    pi = uniform_policy(S)
    truth = policy_evaluation_stochastic(P, R, pi, GAMMA)
    mask = visitable_states(P)
    mask[terminal] = False

    print(f"\nGridworld 4x4, slip = {SLIP}, gamma = {GAMMA}, uniform policy")
    print(f"  {int(mask.sum())} states the agent can actually be in, "
          f"out of {S}\n")

    # ------------------------------------------------- the equivalence
    print("Forward view vs backward view, same episodes, V frozen\n")
    print(f"  {'lambda':>7} {'max |fwd - bwd|':>16} {'in units of eps':>16}")
    print("  " + "-" * 42)
    rng = np.random.default_rng(0)
    episodes = [sample_episode(P, R, pi, rng, terminal)
                for _ in range(EQUIV_EPISODES)]
    V_frozen = np.random.default_rng(1).normal(size=S) * 0.1
    gaps = []
    for lam in LAMBDAS:
        gap = max(np.max(np.abs(forward_update(s, r, V_frozen, GAMMA, lam, 0.1)
                                - backward_update(s, r, V_frozen, GAMMA, lam, 0.1)))
                  for s, r in episodes)
        gaps.append(gap)
        print(f"  {lam:7.2f} {gap:16.3e} {gap / np.finfo(float).eps:16.1f}")
    print(f"\n  machine epsilon is {np.finfo(float).eps:.3e}. The largest "
          f"disagreement over\n  {EQUIV_EPISODES} episodes and "
          f"{len(LAMBDAS)} values of lambda is {max(gaps):.3e}.\n")

    # ------------------------------------------------- the U curve
    print(f"RMS error of n-step TD after {EPISODES_PER_RUN} episodes, "
          f"{args.seeds} seeds\n")
    header = "  " + f"{'alpha':>7} " + " ".join(f"{n:>7}" for n in NS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    curves = {}
    for alpha in ALPHAS:
        row = []
        for n in NS:
            errs = []
            for seed in range(args.seeds):
                r = np.random.default_rng(seed)
                eps = [sample_episode(P, R, pi, r, terminal)
                       for _ in range(EPISODES_PER_RUN)]
                errs.append(rms_error(n_step_td(eps, S, n, GAMMA, alpha),
                                      truth, mask))
            row.append(float(np.median(errs)))
        curves[alpha] = row
        best = NS[int(np.argmin(row))]
        print("  " + f"{alpha:7.2f} " + " ".join(f"{x:7.4f}" for x in row)
              + f"   best n = {best}")

    interior = [a for a in ALPHAS
                if 0 < int(np.argmin(curves[a])) < len(NS) - 1]
    print(f"\n  the best n moves left as alpha grows: "
          + ", ".join(f"{a} -> {NS[int(np.argmin(curves[a]))]}" for a in ALPHAS)
          + ".")
    print(f"  {len(interior)} of the {len(ALPHAS)} step sizes have the minimum "
          f"strictly inside the range;\n  the other{'s' if len(ALPHAS) - len(interior) != 1 else ''} "
          f"put it at an end, which is the honest way to say that a U shape\n"
          f"  is a statement about a region of (n, alpha) and not about n "
          f"alone.\n")

    # the step size at which offline accumulation stops being stable
    print("Where offline n-step TD stops converging\n")
    print(f"  {'alpha':>7} {'n = 1':>12} {'n = 8':>12} {'n = 60':>12}")
    print("  " + "-" * 46)
    for alpha in (0.1, 0.15, 0.2, 0.3):
        row = []
        for n in (1, 8, 60):
            r = np.random.default_rng(0)
            eps = [sample_episode(P, R, pi, r, terminal)
                   for _ in range(EPISODES_PER_RUN)]
            row.append(rms_error(n_step_td(eps, S, n, GAMMA, alpha), truth, mask))
        print(f"  {alpha:7.2f} " + " ".join(f"{x:12.4g}" for x in row))
    print("\n  updates are accumulated over the episode before being applied, so "
          "a state\n  visited k times moves by k alpha times its error. The "
          "README says what that\n  costs and what it buys.\n")

    # ------------------------------------------------------------- figure
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax.semilogy(LAMBDAS, np.maximum(gaps, 1e-20), "o-", lw=2,
                label="max |forward - backward|")
    ax.axhline(np.finfo(float).eps, color="k", ls="--", lw=1.4,
               label=r"machine $\varepsilon = 2.2\times10^{-16}$")
    ax.set_ylim(1e-19, 1e-6)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("largest disagreement")
    ax.set_title("two algorithms, one answer")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    for alpha in ALPHAS:
        bx.plot(NS, curves[alpha], "o-", lw=2, label=fr"$\alpha$ = {alpha}")
        i = int(np.argmin(curves[alpha]))
        bx.plot([NS[i]], [curves[alpha][i]], "k*", ms=13)
    bx.set_xscale("log")
    bx.set_xlabel("$n$"); bx.set_ylabel(r"RMS error against $V^\pi$")
    bx.set_title("neither TD(0) nor Monte Carlo: the star is the answer")
    bx.legend(fontsize=9); bx.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
