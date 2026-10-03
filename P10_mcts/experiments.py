#!/usr/bin/env python3
"""P10 Part B — the theorem made visible: what a prior does to a search.

Tic-tac-toe, because minimax gives the exact answer: every measurement below is
against ground truth, not against another approximation.

Equal budget throughout. UCT and PUCT get the same number of simulations, the
same positions and the same seeds, and the question is not who plays better —
it is where each of them spends the simulations. PUCT runs twice, with the
centre prior and with a uniform prior that knows nothing, so that what the rule
does and what the prior does can be told apart.

    python experiments.py            # ~12 s, writes mcts.png
    python experiments.py --seeds 3  # quicker
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.search import (            # noqa: E402
    O, X, apply_move, centre_prior, initial_state, legal_moves, minimax,
    optimal_moves, other, recommended_move, search, tree_size, uniform_prior,
    visit_fractions, winner,
)

BUDGETS = (25, 50, 100, 200, 400, 800)
SEEDS = 6
N_POSITIONS = 12
PLIES = 3
C = 1.4
RULES = (("UCT", False, uniform_prior),
         ("PUCT + uniform prior", True, uniform_prior),
         ("PUCT + centre prior", True, centre_prior))


def test_positions(n: int, plies: int, seed: int = 0):
    """Positions reached by `plies` random moves in which some moves are
    optimal and some are not — from the empty board every move draws, so the
    empty board cannot distinguish a good search from a bad one."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        state, player = initial_state(), X
        for _ in range(plies):
            moves = legal_moves(state)
            state = apply_move(state, moves[int(rng.integers(len(moves)))],
                               player)
            player = other(player)
        if winner(state) is None and state not in seen and \
                len(optimal_moves(state, player)) < len(legal_moves(state)):
            seen.add(state)
            out.append((state, player))
    return out


def entropy(p) -> float:
    v = np.array(list(p))
    return float(-(v * np.log(v + 1e-15)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="mcts.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    positions = test_positions(N_POSITIONS, PLIES)
    print(f"\nTic-tac-toe, exact answers from minimax\n")
    print(f"  {len(positions)} positions after {PLIES} random plies, each with "
          f"some optimal and some\n  losing moves. "
          f"{args.seeds} seeds per position per budget.\n")
    print(f"  the empty board is a draw and all nine first moves are optimal, "
          f"which is\n  why the measurement is not taken there.\n")

    stats = {}
    for name, use_prior, prior in RULES:
        rows = []
        for budget in BUDGETS:
            sizes, fracs, ents, hits = [], [], [], []
            for state, player in positions:
                good = optimal_moves(state, player)
                for seed in range(args.seeds):
                    root = search(state, player, iters=budget, c=C,
                                  use_prior=use_prior, prior_fn=prior,
                                  seed=seed)
                    vf = visit_fractions(root)
                    sizes.append(tree_size(root))
                    fracs.append(sum(v for m, v in vf.items() if m in good))
                    ents.append(entropy(vf.values()))
                    hits.append(1.0 if recommended_move(root) in good else 0.0)
            rows.append((np.mean(sizes), np.mean(fracs), np.mean(ents),
                         np.mean(hits)))
        stats[name] = rows
        print(f"  {name}")
        print(f"  {'budget':>8} {'nodes visited':>13} {'visits on optimal':>19} "
              f"{'entropy':>9} {'picks optimal':>15}")
        print("  " + "-" * 69)
        for budget, (s, f, e, h) in zip(BUDGETS, rows):
            print(f"  {budget:8d} {s:13.1f} {f:19.3f} {e:9.3f} "
                  f"{100 * h:14.1f}%")
        print()

    a, u, b = (np.array(stats[name]) for name, _, _ in RULES)
    for budget in (100, 400):
        i = BUDGETS.index(budget)
        print(f"  at {budget} simulations, share of root visits on optimal moves: "
              f"UCT {a[i, 1]:.1%},\n  PUCT {u[i, 1]:.1%} with the uniform prior "
              f"and {b[i, 1]:.1%} with the centre prior.")
    print()

    # Does the exploration constant matter? In the tutors' notebook it does not,
    # because `uct` takes c and then writes 2*sqrt(...). Here it does.
    print("  the exploration constant, which a hard-coded 2 would hide\n")
    print(f"  {'c':>7} {'visits on optimal':>19} {'entropy':>9}")
    print("  " + "-" * 39)
    for c in (0.0, 0.5, 1.4, 3.0, 10.0):
        fracs, ents = [], []
        for state, player in positions:
            good = optimal_moves(state, player)
            for seed in range(args.seeds):
                root = search(state, player, iters=200, c=c, seed=seed)
                vf = visit_fractions(root)
                fracs.append(sum(v for m, v in vf.items() if m in good))
                ents.append(entropy(vf.values()))
        print(f"  {c:7.1f} {np.mean(fracs):19.3f} {np.mean(ents):9.3f}")
    print()

    # ------------------------------------------------------------- figure
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for (name, _, _), style in zip(RULES, ("o-", "s--", "^-")):
        r = np.array(stats[name])
        ax.plot(BUDGETS, r[:, 0], style, lw=2, label=name)
        bx.plot(BUDGETS, r[:, 1], style, lw=2, label=name)
        cx.plot(BUDGETS, r[:, 2], style, lw=2, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("simulations"); ax.set_ylabel("nodes visited")
    ax.set_title("same budget, smaller tree")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    bx.axhline(1.0, color="k", ls="--", lw=1.4, label="perfect play (minimax)")
    bx.set_xscale("log")
    bx.set_xlabel("simulations")
    bx.set_ylabel("fraction of root visits on optimal moves")
    bx.set_title("and spent where it matters")
    bx.legend(fontsize=9); bx.grid(alpha=0.3)

    # After three plies every position has six legal moves.
    cx.axhline(np.log(6), color="k", ls=":", lw=1.4,
               label=r"$\ln 6$: spread evenly over six moves")
    cx.set_xscale("log")
    cx.set_xlabel("simulations")
    cx.set_ylabel("entropy of the root visit distribution")
    cx.set_title("concentration, as one number")
    cx.legend(fontsize=9); cx.grid(alpha=0.3)

    for axis in (ax, bx, cx):
        axis.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        axis.tick_params(axis="x", which="minor", labelbottom=False)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
