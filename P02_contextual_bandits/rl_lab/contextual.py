# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""P02 — contextual bandits and posterior sampling.

Reference implementation. `contextual.py` is generated from this file by
tools/strip_solutions.py, which deletes everything between `# TODO: ` and
tools/strip_solutions.py, which deletes everything between `raise NotImplementedError

The setting is the one from the practical: a course recommender. A visitor
arrives with a *context* (their background), we show one of K courses, and we
observe a click or no click. The click-through rate depends on both, so the
problem is no longer one bandit but one bandit per context.

Everything here is a pure function of the statistics, so every test is
deterministic.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Q1: the Beta posterior
# --------------------------------------------------------------------------- #
def posterior_mean(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    r"""Mean of a Beta posterior, elementwise.

    Starting from a $\mathrm{Beta}(1,1)$ prior and observing $k$ clicks out of
    $n$ views, the posterior is $\mathrm{Beta}(1+k,\,1+n-k)$ and its mean is

    $$\mathbb{E}[\mu] = \frac{\alpha}{\alpha+\beta} = \frac{k+1}{n+2},$$

    which is the empirical rate smoothed towards $1/2$ — Laplace's rule of
    succession. That smoothing is why a single click does not make an arm look
    perfect.
    """
    # TODO: return ...
    raise NotImplementedError


def posterior_variance(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    r"""Variance of a Beta posterior, elementwise:

    $$\mathrm{Var}[\theta] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.$$

    This is the quantity the whole lab is about. It shrinks like $1/n$ as
    evidence accumulates, and Part B shows that **it is the exploration**: an
    agent that keeps the mean and throws the variance away stops exploring.
    """
    # TODO: return ...
    raise NotImplementedError


def posterior_update(alpha: float, beta: float, reward: int) -> tuple[float, float]:
    r"""One Bernoulli observation into a Beta posterior: conjugacy in one line.

    $$\mathrm{Beta}(\alpha,\beta) \xrightarrow{\ r=1\ } \mathrm{Beta}(\alpha+1,\beta),
      \qquad \xrightarrow{\ r=0\ } \mathrm{Beta}(\alpha,\beta+1).$$
    """
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: acting on a posterior
# --------------------------------------------------------------------------- #
def select_thompson(alphas: np.ndarray, betas: np.ndarray, state: int,
                    rng: np.random.Generator) -> int:
    r"""Posterior sampling: draw one value per action **from its posterior** and
    take the argmax of the draws.

    $$\tilde\theta_a \sim \mathrm{Beta}(\alpha_{s,a}, \beta_{s,a}), \qquad
      a_t = \arg\max_a \tilde\theta_a$$

    The randomness is not noise added on top: it *is* the uncertainty. An action
    is chosen with exactly the probability that it is the best one given the
    data, which is why exploration fades on its own as the posteriors sharpen.
    Ties: lowest index.
    """
    # TODO: ...
    raise NotImplementedError


def select_posterior_greedy(alphas: np.ndarray, betas: np.ndarray,
                            state: int) -> int:
    r"""The same agent with the variance removed: act on the posterior *mean*.

    $$a_t = \arg\max_a \frac{\alpha_{s,a}}{\alpha_{s,a}+\beta_{s,a}}$$

    This is Part C. It is one line away from `select_thompson` and it is a
    different algorithm: it has no mechanism that can revisit an action whose
    mean happens to look bad early. Predict what its regret does before you run
    it. Ties: lowest index.
    """
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: regret, per context
# --------------------------------------------------------------------------- #
def contextual_regret(ctr: np.ndarray, states: np.ndarray,
                      actions: np.ndarray) -> float:
    r"""Pseudo-regret of a contextual run.

    The best action now depends on the context, so the gap is taken **inside**
    each context:

    $$\mathrm{Reg}(T) = \sum_{t=1}^{T}\Big[\max_a \mathrm{CTR}[s_t, a] - \mathrm{CTR}[s_t, a_t]\Big]$$

    A policy that is optimal on average over contexts but wrong inside one of
    them still pays linear regret, and that is the whole reason the context
    belongs in the state.
    """
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# The interaction loop (given: you do not need to modify it)
# --------------------------------------------------------------------------- #
def run_contextual(ctr: np.ndarray, horizon: int, algorithm: str,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run one contextual bandit episode.

    Returns `(states, actions)`, both of length `horizon`, which is everything
    `contextual_regret` needs.
    """
    rng = np.random.default_rng(seed)
    n_states, n_actions = ctr.shape
    alphas = np.ones((n_states, n_actions))
    betas = np.ones((n_states, n_actions))
    states = np.empty(horizon, dtype=int)
    actions = np.empty(horizon, dtype=int)

    for t in range(horizon):
        s = int(rng.integers(n_states))
        if algorithm == "thompson":
            a = select_thompson(alphas, betas, s, rng)
        elif algorithm == "posterior-greedy":
            a = select_posterior_greedy(alphas, betas, s)
        elif algorithm == "uniform":
            a = int(rng.integers(n_actions))
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        r = int(rng.random() < ctr[s, a])
        alphas[s, a], betas[s, a] = posterior_update(alphas[s, a], betas[s, a], r)
        states[t], actions[t] = s, a
    return states, actions
