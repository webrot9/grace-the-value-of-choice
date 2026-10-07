#!/usr/bin/env python3
"""P08 Part B — the theorem made visible: Baird's counterexample.

Seven states, zero rewards, and a weight vector that can represent the true
value function exactly. Semi-gradient TD(0) still runs away from it — as long as
all three ingredients of the deadly triad are present. Remove any one and the
same code converges.

Nothing here is sampled: `expected_update` is the exact expectation of the TD
update, so the four curves are reproducible to the last digit with no seed.

    python experiments.py            # ~2 s, writes triad.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.triad import (             # noqa: E402
    baird_features, baird_mdp, behaviour_policy, key_matrix, policy_transition,
    run, stationary_distribution, target_policy, value_error,
)

GAMMA = 0.9        # Sutton & Barto use 0.99; the README says why this one is 0.9
ALPHA = 0.05
STEPS = 10_000
W0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default="triad.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    Phi = baird_features()
    P = baird_mdp()
    P_mu = policy_transition(P, behaviour_policy())
    P_pi = policy_transition(P, target_policy())
    d = stationary_distribution(P_mu)
    I7 = np.eye(7)

    print(f"\nBaird's star: 7 states, 8 weights, every reward 0, "
          f"gamma = {args.gamma}")
    print(f"  V^pi(s) = 0 for every s, and w = 0 represents it exactly.")
    print(f"  starting weights {W0.tolist()}")
    print(f"  starting predictions {(Phi @ W0).tolist()}\n")

    # The four runs differ in exactly one thing each.
    configs = [
        ("all three",       Phi, P_pi, True,  W0),
        ("on-policy",       Phi, P_mu, True,  W0),
        ("tabular",         I7,  P_pi, True,  Phi @ W0),
        ("no bootstrap",    Phi, P_pi, False, W0),
    ]

    print(f"  {'configuration':>14} {'missing leg':>22} "
          f"{'VE at 0':>9} {'VE at end':>12} {'min Re(eig A)':>14}")
    print("  " + "-" * 76)
    histories = {}
    missing = {"all three": "-- (the counterexample)",
               "on-policy": "off-policy",
               "tabular": "function approximation",
               "no bootstrap": "bootstrapping"}
    for name, features, P_target, bootstrap, w0 in configs:
        h = run(w0, features, d, P_target, args.gamma, ALPHA, args.steps,
                bootstrap)
        histories[name] = (features, h)
        ev = np.linalg.eigvals(
            key_matrix(features, d, P_target, args.gamma)).real
        ev_txt = f"{ev.min():+.5f}" if bootstrap else "n/a"
        print(f"  {name:>14} {missing[name]:>22} "
              f"{value_error(features, h[0], d):9.4f} "
              f"{value_error(features, h[-1], d):12.4g} {ev_txt:>14}")

    print(f"\n  every run starts from the same predictions "
          f"{(Phi @ W0).tolist()}, uses alpha = {ALPHA}\n  and "
          f"{args.steps} steps. One argument changes per row.\n")

    # Where the divergence comes from, and where it stops.
    print("The eigenvalue is the whole story\n")
    print("  A is singular for every gamma: the features have a redundant")
    print("  direction, so one eigenvalue is 0 and stays 0. What matters is "
          "whether\n  any eigenvalue is strictly negative.\n")
    print(f"  {'gamma':>7} | {'off-policy: min Re':>19} {'how many < 0':>13} "
          f"| {'on-policy: min Re':>19} {'how many < 0':>13}")
    print("  " + "-" * 78)
    for g in (0.5, 0.8, 0.88, 0.9, 0.95, 0.99):
        off = np.linalg.eigvals(key_matrix(Phi, d, P_pi, g)).real
        on = np.linalg.eigvals(key_matrix(Phi, d, P_mu, g)).real
        print(f"  {g:7.2f} | {off.min():19.6f} {int((off < -1e-9).sum()):13d} "
              f"| {on.min():19.6f} {int((on < -1e-9).sum()):13d}")

    lo, hi = 0.5, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.linalg.eigvals(key_matrix(Phi, d, P_pi, mid)).real.min() < -1e-12:
            hi = mid
        else:
            lo = mid
    print(f"\n  off-policy, the negative eigenvalue appears at "
          f"gamma = {lo:.6f}. Below it this\n  configuration does not diverge, "
          f"with all three ingredients still present.\n")

    # ------------------------------------------------------------- figure
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    styles = {"all three": ("tab:red", "-"), "on-policy": ("tab:blue", "--"),
              "tabular": ("tab:green", "-."), "no bootstrap": ("tab:purple", ":")}
    for name, (features, h) in histories.items():
        ve = np.array([value_error(features, w, d) for w in h])
        colour, ls = styles[name]
        ax.semilogy(np.maximum(ve, 1e-16), ls, color=colour, lw=2,
                    label=f"{name} ({missing[name].split(' (')[0]})")
    ax.set_xlabel("expected updates")
    ax.set_ylabel(r"$\sqrt{\sum_s d(s)\,\hat{V}(s)^2}$   (true value is 0)")
    ax.set_title("one ingredient removed per curve")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    h = histories["all three"][1]
    for i in range(8):
        bx.plot(h[:, i], lw=1.6, label=f"$w_{i+1}$" if i in (5, 6, 7) else None)
    bx.set_yscale("symlog", linthresh=1.0)
    bx.set_xlabel("expected updates"); bx.set_ylabel("weight")
    bx.set_title("the individual weights, all three ingredients present")
    bx.legend(fontsize=9); bx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
