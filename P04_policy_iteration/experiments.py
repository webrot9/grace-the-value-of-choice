#!/usr/bin/env python3
"""P04 Part B — the theorem made visible: improvement is monotone in every state,
and that is why policy iteration stops while value iteration does not.

Left panel: one line per state, the value of that state along the sequence of
policies. Every line is non-decreasing. Not "on average": every line.

Right panel: the two algorithms on the same problem and the same axis. Policy
iteration reaches V* exactly and halts after a handful of policies; value
iteration approaches it geometrically and never arrives.

    python experiments.py            # ~2 s, writes monotonicity.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.pi import (                # noqa: E402
    gridworld, is_monotone, policy_evaluation_exact, policy_improvement,
    policy_iteration, q_from_v,
)

GAMMA = 0.95
GRID = (5, 5)
SLIP = 0.15


def value_iteration_errors(P, R, gamma, v_star, sweeps=60):
    V = np.zeros(R.shape[0])
    out = [np.max(np.abs(V - v_star))]
    for _ in range(sweeps):
        V = q_from_v(P, R, V, gamma).max(axis=1)
        out.append(np.max(np.abs(V - v_star)))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--slip", type=float, default=SLIP)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="monotonicity.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()

    t0 = time.time()
    P, R = gridworld(*GRID, slip=args.slip, seed=1)
    pi, values = policy_iteration(P, R, args.gamma)
    v_star = values[-1]

    print(f"\ngridworld {GRID[0]}x{GRID[1]}, slip={args.slip}, "
          f"gamma={args.gamma}")
    print(f"  policies visited: {len(values)}")
    print(f"  monotone in every state: {is_monotone(values)}")
    worst = np.diff(values, axis=0).min()
    print(f"  smallest change of any state at any step: {worst:+.2e}"
          f"   (must be >= 0)")

    # How often does a step improve *every* state strictly? Almost never: most
    # steps leave most states untouched, which is the honest picture of what
    # 'monotone improvement' buys.
    steps = np.diff(values, axis=0)
    touched = (steps > 1e-9).sum(axis=1)
    print(f"  states strictly improved per step: {[int(x) for x in touched]} "
          f"of {len(v_star)}")

    # The comparison the calendar asks for: PI terminates, VI does not.
    errs = value_iteration_errors(P, R, args.gamma, v_star)
    reached = np.argmax(errs < 1e-9) if (errs < 1e-9).any() else -1
    print(f"\n  value iteration error after 60 sweeps: {errs[-1]:.2e}")
    print(f"  sweeps for value iteration to reach 1e-9: "
          f"{reached if reached > 0 else 'not reached'}")

    # Over many MDPs: is monotonicity ever violated, and how many policies?
    counts, violations = [], 0
    for s in range(args.seeds):
        Ps, Rs = gridworld(*GRID, slip=args.slip, seed=s)
        _, vals = policy_iteration(Ps, Rs, args.gamma)
        counts.append(len(vals))
        violations += (not is_monotone(vals))
    print(f"\n  over {args.seeds} random gridworlds: "
          f"{min(counts)}-{max(counts)} policies (median {int(np.median(counts))}), "
          f"monotonicity violated {violations} times")
    print(f"  the bound |A|^|S| for this grid is 4^{len(v_star)} "
          f"= {4.0 ** len(v_star):.1e}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for s in range(values.shape[1]):
        ax.plot(values[:, s], marker="o", ms=3, lw=1, alpha=0.8)
    ax.set_xlabel("policy index $k$"); ax.set_ylabel(r"$V^{\pi_k}(s)$")
    ax.set_title("one line per state — none of them ever goes down")
    ax.grid(alpha=0.3)

    bx.semilogy(np.maximum(errs, 1e-17), label="value iteration")
    pi_err = np.max(np.abs(values - v_star), axis=1)
    bx.semilogy(np.maximum(pi_err, 1e-17), marker="o", ms=4,
                label="policy iteration")
    bx.set_xlabel("iteration"); bx.set_ylabel(r"$\|V - V^{*}\|_\infty$ (log)")
    bx.set_title("PI hits zero and stops; VI keeps halving for ever")
    bx.legend(fontsize=9); bx.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
