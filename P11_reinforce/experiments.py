#!/usr/bin/env python3
"""P11 Part B — the theorem made visible: an unbiased gradient, and a baseline.

Panel 1: the sampled likelihood-ratio estimator against a reference that does
not sample at all — finite differences of the true objective, which is available
in closed form for a one-step problem. Unbiasedness stops being a claim.

Panel 2: the total variance of the estimator as a function of the baseline. It
is a parabola. The mean does not move along it; the variance moves by two orders
of magnitude.

Panel 3: the same baseline inside a training loop on a gridworld, four seeds,
median and interquartile band, with the gamma^t of the policy gradient (solid)
and without it (dashed), the way many implementations write it.

    python experiments.py            # ~30 s, writes reinforce.png
    python experiments.py --samples 20000 --seeds 2     # quicker
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.pg import (                # noqa: E402
    bandit_objective, exact_bandit_gradient, finite_difference_gradient,
    gridworld, optimal_constant_baseline, reinforce_samples, softmax,
    total_variance, train,
)

THETA = np.array([0.3, -0.2, 0.8, 0.0])
REWARDS = np.array([1.0, 0.5, -0.2, 2.0])
SAMPLES = 200_000
BASELINES = (-4.0, -2.0, 0.0, 0.5, 1.5, 3.0, 6.0, 10.0)
SEEDS = 4
ITERS = 200
BATCH = 10
ALPHA = 0.5
GAMMA = 0.95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="reinforce.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    exact = exact_bandit_gradient(THETA, REWARDS)
    fd = finite_difference_gradient(lambda t: bandit_objective(t, REWARDS),
                                    THETA, eps=1e-5)
    samples = reinforce_samples(THETA, REWARDS, args.samples, 0.0, rng)
    est = samples.mean(axis=0)
    se = samples.std(axis=0) / np.sqrt(args.samples)

    print(f"\nOne-step problem, {len(REWARDS)} actions, "
          f"pi = softmax({THETA.tolist()})")
    print(f"  rewards {REWARDS.tolist()},  J(theta) = "
          f"{bandit_objective(THETA, REWARDS):.6f}\n")
    print(f"  {'i':>3} {'exact grad':>13} {'finite diff':>13} "
          f"{'REINFORCE':>13} {'+- SE':>9} {'in SE':>7}")
    print("  " + "-" * 62)
    for i in range(len(THETA)):
        print(f"  {i:3d} {exact[i]:13.6f} {fd[i]:13.6f} {est[i]:13.6f} "
              f"{se[i]:9.6f} {(est[i] - exact[i]) / se[i]:+7.1f}")
    rel = np.abs(est - exact).max() / np.abs(exact).max()
    print(f"\n  exact vs finite differences: {np.abs(exact - fd).max():.2e} "
          f"(no sampling on either side)")
    print(f"  REINFORCE vs exact: {100 * rel:.2f}% relative error on "
          f"{args.samples} samples.\n")

    # ---------------------------------------------------- the baseline parabola
    b_star = optimal_constant_baseline(THETA, REWARDS)
    mean_r = float(softmax(THETA) @ REWARDS)
    print(f"Total variance as a function of the baseline\n")
    print(f"  {'b':>8} {'total variance':>16} {'bias, in SE':>13}")
    print("  " + "-" * 40)
    grid = sorted(set(BASELINES) | {round(b_star, 4), round(mean_r, 4)})
    variances = {}
    for b in grid:
        s = reinforce_samples(THETA, REWARDS, args.samples, b, rng)
        v = total_variance(s)
        variances[b] = v
        bias_se = np.abs((s.mean(axis=0) - exact) /
                         (s.std(axis=0) / np.sqrt(args.samples))).max()
        tag = ""
        if abs(b - b_star) < 1e-4:
            tag = "  <- b*, the variance-minimising baseline"
        elif abs(b - mean_r) < 1e-4:
            tag = "  <- the mean reward, which is not b*"
        print(f"  {b:8.4f} {v:16.4f} {bias_se:13.1f}{tag}")
    v0, vbest, vworst = variances[0.0], variances[round(b_star, 4)], variances[10.0]
    print(f"\n  b* = {b_star:.4f} and the mean reward is {mean_r:.4f}: not the "
          f"same number.")
    print(f"  the best baseline divides the variance by "
          f"{v0 / vbest:.1f}; b = 10 multiplies it by {vworst / v0:.0f}.")
    print(f"  the bias column stays at the level of sampling noise throughout: "
          f"every\n  constant baseline is unbiased, and only one of them is a "
          f"good idea.\n")

    # --------------------------------------------------------------- training
    P, R = gridworld(4, 4, slip=0.1, seed=3)
    terminal = P.shape[0] - 1
    print(f"REINFORCE on a 4x4 gridworld, {args.seeds} seeds, {ITERS} "
          f"iterations of {BATCH} episodes\n")
    runs = {}
    print(f"  {'':>30} {'return':>18} {'gradient spread':>19} {'0.9 at':>8}")
    for gamma_t in (True, False):
        for use_baseline in (False, True):
            name = (("gamma^t" if gamma_t else "no gamma^t") + ", "
                    + ("state baseline" if use_baseline else "no baseline"))
            rs, sp = [], []
            for seed in range(args.seeds):
                _, returns, spread = train(P, R, terminal, gamma=GAMMA,
                                           n_iters=ITERS, batch=BATCH,
                                           alpha=ALPHA,
                                           use_baseline=use_baseline,
                                           seed=seed, gamma_t=gamma_t)
                rs.append(returns)
                sp.append(spread)
            runs[name] = (np.array(rs), np.array(sp))
            r, s = runs[name]
            # first iteration at which the mean of the last ten, over seeds,
            # reaches 0.9
            smooth = np.convolve(r.mean(axis=0), np.ones(10) / 10, mode="valid")
            at = int(np.argmax(smooth >= 0.9)) + 10 if (smooth >= 0.9).any() else None
            print(f"  {name:>30}: {r[:, :10].mean():.4f} -> {r[:, -10:].mean():.4f}"
                  f"   {s[:, :10].mean():.4f} -> {s[:, -10:].mean():.4f}"
                  f"   {at if at is not None else '-':>6}")
    print()
    for tag in ("gamma^t", "no gamma^t"):
        a = runs[f"{tag}, no baseline"][1][:, -10:].mean()
        b = runs[f"{tag}, state baseline"][1][:, -10:].mean()
        print(f"  {tag}: the state baseline divides the final spread of the "
              f"batch gradients by {a / b:.1f}.")
    print()

    # ----------------------------------------------------------------- figure
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    idx = np.arange(len(THETA))
    ax.bar(idx - 0.25, exact, 0.25, label="exact gradient")
    ax.bar(idx, fd, 0.25, label="finite differences")
    ax.bar(idx + 0.25, est, 0.25, yerr=se * 3, capsize=4,
           label=f"REINFORCE, {args.samples} samples")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(idx); ax.set_xlabel(r"component of $\theta$")
    ax.set_ylabel(r"$\partial J / \partial \theta_i$")
    ax.set_title("three ways to the same number")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    bs = np.array(sorted(variances))
    bx.semilogy(bs, [variances[b] for b in bs], "o-", lw=2)
    bx.axvline(b_star, color="tab:green", ls="--", lw=1.6,
               label=f"$b^*$ = {b_star:.2f}")
    bx.axvline(mean_r, color="tab:orange", ls=":", lw=1.6,
               label=f"mean reward = {mean_r:.2f}")
    bx.axvline(0.0, color="k", ls="-.", lw=1.2, label="no baseline")
    bx.set_xlabel("baseline $b$"); bx.set_ylabel("total variance")
    bx.set_title("unbiased everywhere, sane in one place")
    bx.legend(fontsize=8); bx.grid(alpha=0.3, which="both")

    for name, (rs, sp) in runs.items():
        x = np.arange(sp.shape[1])
        colour = "tab:orange" if "state baseline" in name else "tab:blue"
        style = "--" if name.startswith("no gamma") else "-"
        cx.plot(x, np.median(sp, axis=0), style, color=colour, lw=2,
                label=name.replace("gamma^t", r"$\gamma^t$"))
        cx.fill_between(x, np.percentile(sp, 25, axis=0),
                        np.percentile(sp, 75, axis=0), color=colour, alpha=0.12)
    cx.set_yscale("log")
    cx.set_xlabel("iteration")
    cx.set_ylabel("std of the batch gradients")
    cx.set_title("the same effect, inside a training loop")
    cx.legend(fontsize=9); cx.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
