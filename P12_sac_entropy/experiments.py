#!/usr/bin/env python3
"""P12 Part B — the theorem made visible: the temperature is the exploration.

Left panel: the entropy of the soft-optimal policy against alpha, from 0 to
log|A|, with both limits as dashed lines. At alpha = 0 the policy is greedy and
the entropy is exactly zero: the collapse the session is named after.

Middle panel: the soft value against the hard one. log-sum-exp sits above the
max by at most alpha*log|A|, and the two coincide as alpha goes to zero.

Right panel: SAC's squashed Gaussian. The corrected density integrates to 1; the
one without the tanh Jacobian integrates to 0.66, and is therefore not a
density at all.

    python experiments.py            # ~5 s, writes entropy.png
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

from rl_lab.entropy import (           # noqa: E402
    boltzmann_policy, entropy, gaussian_log_prob, gridworld, policy_entropy,
    sample_squashed, soft_policy_iteration, soft_value, squashed_log_prob,
)

GAMMA = 0.9
ALPHAS = (0.0, 1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
STDS = (0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0)
MC = 200_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=MC)
    ap.add_argument("--out", default="entropy.png",
                    help="where to save the figure; for Part C, give a new name so the first one is kept")
    args = ap.parse_args()
    t0 = time.time()

    P, R = gridworld(4, 4, slip=0.1, seed=3)
    S, A = R.shape
    reachable = [s for s in range(S) if s != S - 1]

    print(f"\nSoft value iteration on a 4x4 gridworld, gamma = {GAMMA}, "
          f"{A} actions")
    print(f"  the entropy of a uniform policy over {A} actions is "
          f"log {A} = {np.log(A):.4f}\n")
    print(f"  {'alpha':>8} {'mean entropy':>13} {'max entropy?':>13} "
          f"{'V_soft(s0)':>12} {'V*(s0)':>10} {'gap':>10} {'bound':>10}")
    print("  " + "-" * 82)

    Q_hard, _ = soft_policy_iteration(P, R, GAMMA, 0.0)
    v_hard = float(Q_hard[0].max())
    rows = []
    for alpha in ALPHAS:
        Q, pi = soft_policy_iteration(P, R, GAMMA, alpha)
        h = float(np.mean([entropy(pi[s]) for s in reachable]))
        v = soft_value(Q[0], alpha)
        bound = alpha * np.log(A) / (1 - GAMMA)
        rows.append((alpha, h, v))
        print(f"  {alpha:8.4g} {h:13.6f} {h / np.log(A):12.1%} "
              f"{v:12.6f} {v_hard:10.6f} {v - v_hard:10.6f} {bound:10.6f}")

    print(f"\n  at alpha = 0 the entropy is exactly {rows[0][1]:.1e}: the "
          f"policy is deterministic and\n  the soft backup is the ordinary "
          f"one.")
    print(f"  the gap never exceeds alpha*log|A|/(1-gamma) = "
          f"{np.log(A) / (1 - GAMMA):.4f}*alpha. The per-step bonus is\n"
          f"  alpha*log|A|; over an infinite discounted horizon it accumulates "
          f"a factor\n  1/(1-gamma) = {1 / (1 - GAMMA):.0f}, and quoting the "
          f"per-step bound here would understate it\n  tenfold.\n")

    # ------------------------------------------------- the squashed Gaussian
    mu, log_std = np.array([0.3]), np.array([np.log(0.8)])
    a_grid = np.linspace(-0.99999, 0.99999, 200_001)
    u_grid = np.arctanh(a_grid)[:, None]
    dens = np.exp(squashed_log_prob(u_grid, mu, log_std))
    dens_bad = np.exp(gaussian_log_prob(u_grid, mu, log_std))
    mass = float(np.trapezoid(dens, a_grid))
    mass_bad = float(np.trapezoid(dens_bad, a_grid))

    print(f"SAC's policy: a = tanh(u), u ~ N({mu[0]}, {np.exp(log_std)[0]}^2)\n")
    print(f"  with the tanh Jacobian, the density integrates to {mass:.6f}")
    print(f"  without it,                                       {mass_bad:.6f}")
    print(f"  the second is not a density. An 'entropy bonus' computed from it "
          f"is not\n  an entropy, and the tutors' notebook for this practical "
          f"has no temperature\n  coefficient either: alpha is hard-coded to 1 "
          f"as `Q - log_prob`.\n")

    rng = np.random.default_rng(0)
    print(f"  entropy of the squashed policy against the spread it is "
          f"sampled with\n")
    print(f"  {'sigma':>8} {'H(pi), nats':>13} {'H of the Gaussian':>19}")
    print("  " + "-" * 44)
    ents = []
    for sd in STDS:
        ls = np.array([np.log(sd)])
        h = policy_entropy(mu, ls, args.samples, rng)
        gauss = 0.5 * np.log(2 * np.pi * np.e * sd ** 2)
        ents.append(h)
        print(f"  {sd:8.2f} {h:13.4f} {gauss:19.4f}")
    best = STDS[int(np.argmax(ents))]
    print(f"\n  the Gaussian's entropy grows without bound in sigma. The "
          f"squashed policy's\n  does not: it peaks at sigma = {best} with "
          f"{max(ents):.4f} nats, against a ceiling of\n  log 2 = "
          f"{np.log(2):.4f} for any distribution on (-1, 1), and falls away on "
          f"both sides.\n  Turning the noise up past that point makes the "
          f"policy *less* exploratory.\n")

    # ----------------------------------------------------------------- figure
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    xs = [max(a, 1e-5) for a, _, _ in rows]
    ax.semilogx(xs, [h for _, h, _ in rows], "o-", lw=2)
    ax.axhline(np.log(A), color="k", ls="--", lw=1.4,
               label=fr"$\log {A}$: uniform")
    ax.axhline(0.0, color="tab:red", ls=":", lw=1.6,
               label="0: deterministic")
    ax.set_xlabel(r"$\alpha$ (the leftmost point is $\alpha = 0$)")
    ax.set_ylabel(r"mean $\mathcal{H}(\pi(\cdot\mid s))$, nats")
    ax.set_title("the temperature is the exploration")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    bx.semilogx(xs, [v - v_hard for _, _, v in rows], "o-", lw=2,
                label=r"$V_{\mathrm{soft}} - V^{*}$")
    bx.semilogx(xs, [a * np.log(A) / (1 - GAMMA) for a in xs], "k--", lw=1.4,
                label=r"$\alpha \log |\mathcal{A}| / (1-\gamma)$")
    bx.set_yscale("log")
    bx.set_xlabel(r"$\alpha$"); bx.set_ylabel("gap at $s_0$")
    bx.set_title("log-sum-exp above the max, by a bounded amount")
    bx.legend(fontsize=9); bx.grid(alpha=0.3, which="both")

    _, samples = sample_squashed(mu, log_std, 100_000, rng)
    cx.hist(samples[:, 0], bins=120, density=True, alpha=0.45,
            label="sampled actions")
    cx.plot(a_grid, dens, lw=2,
            label=f"with the Jacobian (mass {mass:.3f})")
    cx.plot(a_grid, dens_bad, lw=2, ls="--",
            label=f"without it (mass {mass_bad:.3f})")
    cx.set_xlabel("$a = \\tanh(u)$"); cx.set_ylabel("density")
    cx.set_title("one term, and it is not optional")
    cx.legend(fontsize=8); cx.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"  figure -> {args.out}")
    print(f"\n  {time.time() - t0:.1f} s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
