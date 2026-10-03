r"""P12 — Maximum-entropy RL: the temperature is the exploration.

Reference implementation. `entropy.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Soft Actor-Critic optimises

$$J(\pi) = \mathbb{E}\Big[\sum_t r_t + \alpha\, \mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big],$$

and everything interesting about it is what $\alpha$ does. The soft Bellman
backup replaces the max with a **log-sum-exp**,

$$V(s) = \alpha \log \sum_a \exp\!\big(Q(s,a)/\alpha\big),$$

whose greedy policy is the Boltzmann distribution
$\pi(a\mid s) \propto \exp(Q(s,a)/\alpha)$. Both of those collapse onto the
familiar objects as $\alpha \to 0$ — the max and the greedy policy — and this lab
measures the collapse rather than asserting it.

The second half is the one line that is wrong in most implementations, the
tutors' notebook included: SAC's policy is a Gaussian squashed through a `tanh`,
and the change of variables costs a Jacobian term. Without it the "log
probability" is not the log of a density, and the entropy bonus it feeds into is
not an entropy.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.gridworld import gridworld


# --------------------------------------------------------------------------- #
# Q1: soft values and the Boltzmann policy
# --------------------------------------------------------------------------- #
def soft_value(q: np.ndarray, alpha: float) -> float:
    r"""$V = \alpha \log \sum_a \exp(Q_a/\alpha)$, computed stably.

    Factor the maximum out before exponentiating:
    $\alpha\log\sum_a e^{Q_a/\alpha} = \max_a Q_a + \alpha\log\sum_a
    e^{(Q_a - \max_a Q_a)/\alpha}$. Written naively, $\alpha = 0.001$ overflows
    and returns `inf`, which is the value the collapse is supposed to approach
    from below.

    `alpha = 0` must return $\max_a Q_a$ exactly, not `nan`.
    """
    # TODO: ...
    raise NotImplementedError


def boltzmann_policy(q: np.ndarray, alpha: float) -> np.ndarray:
    r"""$\pi(a) \propto \exp(Q_a/\alpha)$, the policy that maximises
    $\mathbb{E}_\pi[Q] + \alpha \mathcal{H}(\pi)$.

    At `alpha = 0` return the greedy policy, with the mass split evenly among
    ties. As $\alpha \to \infty$ it becomes uniform. Those two limits are the
    two ends of the figure in Part B.
    """
    # TODO: ...
    raise NotImplementedError


def entropy(p: np.ndarray) -> float:
    r"""$\mathcal{H}(p) = -\sum_a p_a \log p_a$, in nats, with
    $0 \log 0 = 0$.

    Its maximum is $\log |\mathcal{A}|$ for the uniform distribution and its
    minimum is $0$ for a deterministic one — the two dashed lines in the
    figure."""
    # TODO: ...
    raise NotImplementedError


def soft_policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                          alpha: float, tol: float = 1e-12,
                          max_iters: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    r"""The soft Bellman optimality backup, iterated to its fixed point:

    $$Q(s,a) \leftarrow r(s,a) + \gamma \sum_{s'} p(s'\mid s,a)\,
      \alpha \log \sum_{a'} \exp\!\big(Q(s',a')/\alpha\big)$$

    Return `(Q, pi)` with `pi` the Boltzmann policy of the fixed point.

    The backup is a $\gamma$-contraction for every $\alpha \ge 0$ — log-sum-exp
    is non-expansive in the sup norm, exactly like the max it generalises — so
    this converges for the same reason value iteration did in P03, and a test
    checks the rate.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the squashed Gaussian, and the term everybody forgets
# --------------------------------------------------------------------------- #
def gaussian_log_prob(u: np.ndarray, mu: np.ndarray,
                      log_std: np.ndarray) -> np.ndarray:
    r"""$\log \mathcal{N}(u \mid \mu, \sigma^2)$ summed over the action
    dimensions:

    $$-\tfrac12\Big(\tfrac{u-\mu}{\sigma}\Big)^2 - \log\sigma
      - \tfrac12\log 2\pi$$

    `u` has shape `(n, d)` and the result has shape `(n,)`.
    """
    # TODO: ...
    raise NotImplementedError


def squashed_log_prob(u: np.ndarray, mu: np.ndarray,
                      log_std: np.ndarray) -> np.ndarray:
    r"""The log density of $a = \tanh(u)$ when $u \sim \mathcal{N}(\mu,\sigma^2)$:

    $$\log \pi(a) = \log \mathcal{N}(u\mid\mu,\sigma^2)
                    - \sum_i \log\big(1 - \tanh^2(u_i)\big)$$

    The second term is the log-Jacobian of the squashing, and it is the line
    that is missing or wrong in most implementations — including the tutors'
    notebook for this practical, which also has no temperature coefficient at
    all. Without it the number is not the log of a density: Part B integrates
    both and shows one of them is not $1$.

    Note the sign. The Jacobian is $\mathrm{d}a/\mathrm{d}u = 1-\tanh^2 u \le 1$,
    so squashing *concentrates* probability mass and the density goes **up**;
    subtracting a log of something below one is an addition.

    Compute that log without forming $1 - \tanh^2 u$, since for $|u|$ above
    about 19 $\tanh u$ rounds to exactly $1$ and the log is $-\infty$. The
    identity
    $\log(1-\tanh^2 u) = 2\big(\log 2 - u - \operatorname{softplus}(-2u)\big)$,
    with $\operatorname{softplus}(x) = \log(1+e^x)$ = `np.logaddexp(0, x)`,
    is exact and finite everywhere. The common patch of a small $\epsilon$
    inside the log floors the correction at $\log\epsilon$ and gets the entropy
    of a wide policy wrong by several nats; a test checks the tails.
    """
    # TODO: ...
    raise NotImplementedError


def sample_squashed(mu: np.ndarray, log_std: np.ndarray, n: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    r"""Sample `n` actions with the reparameterisation trick:
    $u = \mu + \sigma\,\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$,
    then $a = \tanh(u)$.

    Return `(u, a)`, both of shape `(n, d)`. The pre-squash `u` is returned
    because the log density needs it: computing it from `a` by `arctanh` loses
    precision exactly where it matters, near the boundary.
    """
    # TODO: ...
    raise NotImplementedError


def policy_entropy(mu: np.ndarray, log_std: np.ndarray, n: int,
                   rng: np.random.Generator) -> float:
    r"""A Monte Carlo estimate of $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a)]$
    for the squashed Gaussian, using `n` samples.

    Unlike the discrete case there is no closed form, so this is an estimate
    with a standard error — and the standard error is worth reporting whenever
    the number is."""
    # TODO: ...
    raise NotImplementedError
