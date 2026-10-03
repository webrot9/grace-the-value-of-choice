#!/usr/bin/env python3
"""P06 Part B — bias and variance, on the same episodes.

Two estimators of one number, V^pi(s_0), fed exactly the same data: the same
episodes, in the same order, sampled once and handed to both. Any difference in
what comes out is a difference between the estimators and nothing else.

The ground truth is (I - gamma P^pi)^{-1} r^pi, computed from the model, which no
learner is allowed to look at. With the truth in hand, "unbiased" and "lower
variance" stop being adjectives.

    python experiments.py                # ~50 s, writes bias_variance.png
    python experiments.py --reps 10      # a quick look
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.model_free import (        # noqa: E402
    bias_variance, discounted_returns, gridworld, mc_control, mc_evaluate,
    policy_evaluation_stochastic, q_from_v, q_learning, sample_episode,
    td0_evaluate, uniform_policy, value_iteration,
)

GAMMA = 0.9
SLIP = 0.1
GRID_SEED = 3                    # a layout whose obstacles trap nothing: see README
N_EPISODES = (5, 10, 20, 50, 100, 200)
REPS = 40                        # independent repetitions of the whole experiment
ALPHA_ONLINE, ALPHA_BATCH, BATCH_PASSES = 0.05, 0.02, 15
CHECKPOINTS = (500, 1500, 4000)
CONTROL_SEEDS = 5
EPS = 0.4                        # large on purpose: it moves Q*_eps away from Q*


def eps_soft_optimal(P, R, gamma, eps, tol=1e-13, max_iters=50_000):
    """The value of the best eps-soft policy, i.e. what on-policy control with a
    fixed eps is actually allowed to converge to. Given, because it is a model
    computation and the point here is to have the right target to compare
    against, not to implement it."""
    V = np.zeros(P.shape[0])
    for _ in range(max_iters):
        Q = R + gamma * (P @ V)
        V_new = (1 - eps) * Q.max(axis=1) + eps * Q.mean(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return R + gamma * (P @ V)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--control-seeds", type=int, default=CONTROL_SEEDS)
    ap.add_argument("--out", default="bias_variance.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    P, R = gridworld(4, 4, slip=SLIP, seed=GRID_SEED)
    S = P.shape[0]
    terminal = S - 1
    pi = uniform_policy(S)
    truth = policy_evaluation_stochastic(P, R, pi, GAMMA)[0]

    print(f"\nPolicy evaluation, uniform policy, gamma = {GAMMA}, "
          f"{args.reps} repetitions")
    print(f"  ground truth  V^pi(s_0) = {truth:.6f}   "
          f"(exact, from the model nobody learns from)\n")

    # The theory line: first-visit MC at the start state is the plain mean of n
    # i.i.d. returns, so its standard deviation is sigma / sqrt(n) with sigma the
    # standard deviation of a single return. Measure sigma once, from its own
    # episodes, and it becomes a dashed line rather than a slope-only reference.
    rng = np.random.default_rng(99)
    single = np.array([discounted_returns(
        sample_episode(P, R, pi, rng, terminal)[2], GAMMA)[0]
        for _ in range(5000)])
    sigma = float(single.std())
    print(f"  one episode is worth {single.mean():.4f} on average with "
          f"standard deviation {sigma:.4f}\n")

    est = {k: {n: [] for n in N_EPISODES} for k in ("mc", "td", "batch")}
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        episodes = [sample_episode(P, R, pi, rng, terminal)
                    for _ in range(max(N_EPISODES))]
        for n in N_EPISODES:
            data = episodes[:n]
            est["mc"][n].append(mc_evaluate(data, S, GAMMA)[0])
            est["td"][n].append(td0_evaluate(data, S, GAMMA,
                                             alpha=ALPHA_ONLINE, n_passes=1)[0])
            est["batch"][n].append(td0_evaluate(data, S, GAMMA,
                                                alpha=ALPHA_BATCH,
                                                n_passes=BATCH_PASSES)[0])

    stats = {k: {} for k in est}
    print(f"  {'episodes':>8} | {'MC bias':>9} {'MC std':>8} "
          f"{'sigma/sqrt(n)':>13} | {'TD bias':>9} {'TD std':>8} | "
          f"{'batch bias':>10} {'batch std':>9}")
    print("  " + "-" * 90)
    for n in N_EPISODES:
        row = []
        for k in ("mc", "td", "batch"):
            b, v, m = bias_variance(np.array(est[k][n]), truth)
            stats[k][n] = (b, v, m)
            row += [b, np.sqrt(v)]
        print(f"  {n:8d} | {row[0]:9.4f} {row[1]:8.4f} "
              f"{sigma / np.sqrt(n):13.4f} | "
              f"{row[2]:9.4f} {row[3]:8.4f} | {row[4]:10.4f} {row[5]:9.4f}")

    n0 = N_EPISODES[0]
    b0, v0, m0 = stats["td"][n0]
    print(f"\n  at {n0} episodes TD(0) with one pass has bias {b0:+.4f} and "
          f"standard deviation {np.sqrt(v0):.4f}.")
    print(f"  the truth is {truth:.4f}. Read those three numbers together "
          f"before you call it low variance.\n")

    dev = max(abs(np.sqrt(stats["mc"][n][1]) / (sigma / np.sqrt(n)) - 1)
              for n in N_EPISODES)
    print(f"  measured MC spread agrees with sigma/sqrt(n) to within "
          f"{100 * dev:.0f}% at every n. With {args.reps} repetitions a "
          f"standard deviation\n  is itself known to about "
          f"{100 / np.sqrt(2 * args.reps):.0f}%, so do not read more into that "
          f"number than it holds.\n")

    # ---------------------------------------------------------------- control
    V_star = value_iteration(P, R, GAMMA)
    Q_star = q_from_v(P, R, V_star, GAMMA)
    Q_eps = eps_soft_optimal(P, R, GAMMA, EPS)

    print(f"Control, epsilon = {EPS} fixed (never decayed), "
          f"{args.control_seeds} seeds\n")
    print(f"  {'episodes':>8} | {'MC vs Q*':>20} | {'MC vs Q*_eps':>20} | "
          f"{'Q-learning vs Q*':>20}")
    print("  " + "-" * 78)
    curves = {"mc_star": [], "mc_eps": [], "ql_star": []}
    for n in CHECKPOINTS:
        cols = {k: [] for k in curves}
        for seed in range(args.control_seeds):
            Qm = mc_control(P, R, GAMMA, terminal, n_episodes=n,
                            alpha=0.05, eps=EPS, seed=seed)
            Ql = q_learning(P, R, GAMMA, terminal, n_episodes=n,
                            alpha=0.05, eps=EPS, seed=seed)
            # At the start state only. Over the whole table the sup norm is
            # dominated by the four obstacle cells, which the agent can never
            # occupy and therefore never updates: comparing Q there measures
            # the layout, not the algorithm.
            cols["mc_star"].append(np.abs(Qm[0] - Q_star[0]).max())
            cols["mc_eps"].append(np.abs(Qm[0] - Q_eps[0]).max())
            cols["ql_star"].append(np.abs(Ql[0] - Q_star[0]).max())
            untouched = int((Ql == 0).all(axis=1).sum())
        for k in curves:
            curves[k].append(np.array(cols[k]))
        med = {k: np.median(cols[k]) for k in cols}
        iqr = {k: (np.percentile(cols[k], 25), np.percentile(cols[k], 75))
               for k in cols}
        print(f"  {n:8d} | {med['mc_star']:8.3f} "
              f"[{iqr['mc_star'][0]:.3f},{iqr['mc_star'][1]:.3f}] | "
              f"{med['mc_eps']:8.3f} [{iqr['mc_eps'][0]:.3f},{iqr['mc_eps'][1]:.3f}] | "
              f"{med['ql_star']:8.3f} "
              f"[{iqr['ql_star'][0]:.3f},{iqr['ql_star'][1]:.3f}]")
    print(f"\n  median and [25th, 75th] percentile of "
          f"max_a |Q(s_0,a) - target(s_0,a)| over {args.control_seeds} seeds.")
    print(f"  the two targets are {np.abs(Q_star[0] - Q_eps[0]).max():.4f} apart "
          f"at s_0: that gap is a property of epsilon = {EPS}, not of the data.")
    print(f"  {untouched} of {S} states were never updated by Q-learning "
          f"(the obstacles and the goal).\n")

    # ----------------------------------------------------------------- figure
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ns = np.array(N_EPISODES, dtype=float)

    for k, lab, style in (("mc", "MC (first visit)", "o-"),
                          ("td", "TD(0), one pass", "s-"),
                          ("batch", f"batch TD, {BATCH_PASSES} passes", "^-")):
        ax.plot(ns, [abs(stats[k][n][0]) for n in N_EPISODES], style, lw=2,
                label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("episodes"); ax.set_ylabel(r"$|\mathrm{bias}|$")
    ax.set_title("bias: MC has none to speak of")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    for k, lab, style in (("mc", "MC (first visit)", "o-"),
                          ("td", "TD(0), one pass", "s-"),
                          ("batch", f"batch TD, {BATCH_PASSES} passes", "^-")):
        bx.plot(ns, [np.sqrt(stats[k][n][1]) for n in N_EPISODES], style, lw=2,
                label=lab)
    bx.plot(ns, sigma / np.sqrt(ns), "k--", lw=1.4,
            label=r"$\sigma/\sqrt{n}$, measured $\sigma$")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel("episodes"); bx.set_ylabel("standard deviation")
    bx.set_title("variance: the dashed line is the theory")
    bx.legend(fontsize=8); bx.grid(alpha=0.3, which="both")

    xs = np.arange(len(CHECKPOINTS))
    for k, lab, colour in (("mc_star", r"MC control vs $Q^{*}$", "tab:blue"),
                           ("mc_eps", r"MC control vs $Q^{*}_{\epsilon}$", "tab:green"),
                           ("ql_star", r"Q-learning vs $Q^{*}$", "tab:red")):
        med = [np.median(c) for c in curves[k]]
        lo = [np.percentile(c, 25) for c in curves[k]]
        hi = [np.percentile(c, 75) for c in curves[k]]
        cx.plot(CHECKPOINTS, med, "o-", color=colour, lw=2, label=lab)
        cx.fill_between(CHECKPOINTS, lo, hi, color=colour, alpha=0.18)
    cx.set_xscale("log")
    cx.set_xlabel("episodes"); cx.set_ylabel(r"$\|Q - \mathrm{target}\|_\infty$")
    cx.set_title("the two algorithms are not chasing the same $Q$")
    cx.legend(fontsize=8); cx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
