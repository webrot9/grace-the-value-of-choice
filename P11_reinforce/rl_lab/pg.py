r"""P11 — REINFORCE: an unbiased gradient, and what a baseline does to it.

Reference implementation. `pg.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Two claims are made about the likelihood-ratio estimator, and they are usually
made in the same breath, which is a mistake because only one of them is a
theorem about the estimator:

  1. it is **unbiased**:
     $\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a)\, G\big]$
  2. subtracting a baseline $b$ that does not depend on the action leaves it
     unbiased and **reduces its variance**

The first is exact and this lab checks it against a finite-difference gradient
of the true objective — no sampling on the reference side at all, because the
objective is available in closed form.

The second is exact for the mean and *conditional* for the variance: the wrong
baseline increases it. Part B measures where the boundary is.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.gridworld import gridworld


# --------------------------------------------------------------------------- #
# Q1: the policy, and its gradient
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray) -> np.ndarray:
    r"""$\pi(a) = \dfrac{e^{\theta_a}}{\sum_{a'} e^{\theta_{a'}}}$.

    Subtract the maximum before exponentiating. Not for elegance: with logits
    around 800 the naive version returns `nan`, silently, in the middle of a
    training run that was going fine.
    """
    # TODO: ...
    raise NotImplementedError


def log_prob_grad(logits: np.ndarray, action: int) -> np.ndarray:
    r"""$\nabla_\theta \log \pi_\theta(a) = \mathbf{e}_a - \pi_\theta$.

    One line, and worth staring at: the gradient of the log-probability of the
    action you took pushes that action's logit up and every logit down in
    proportion to its current probability. It sums to zero, always, which is one
    of the tests.
    """
    # TODO: ...
    raise NotImplementedError


def bandit_objective(logits: np.ndarray, rewards: np.ndarray) -> float:
    r"""$J(\pi_\theta) = \sum_a \pi_\theta(a)\, r(a)$ for a one-step problem.

    Given in closed form on purpose: the point of the lab is to compare a
    *sampled* gradient against a reference that involves no sampling whatsoever.
    """
    return float(softmax(logits) @ np.asarray(rewards, dtype=float))


def exact_bandit_gradient(logits: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    r"""$\nabla_\theta J = \pi \odot \big(r - \pi^\top r\big)$, analytically.

    Derive it before you write it: it follows from
    $\partial \pi_a / \partial \theta_b = \pi_a(\delta_{ab} - \pi_b)$, and a test
    compares it against finite differences to $10^{-7}$.
    """
    # TODO: ...
    raise NotImplementedError


def finite_difference_gradient(f, theta: np.ndarray, eps: float = 1e-5
                               ) -> np.ndarray:
    r"""Central differences: $\dfrac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon}$.

    Central and not forward, for the same reason as in P05: the error is
    $O(\epsilon^2)$ rather than $O(\epsilon)$, and here it has to be small enough
    to be a *reference* rather than a second estimate.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the estimator, and the baseline
# --------------------------------------------------------------------------- #
def reinforce_samples(logits: np.ndarray, rewards: np.ndarray,
                      n_samples: int, baseline: float,
                      rng: np.random.Generator) -> np.ndarray:
    r"""`n_samples` independent one-sample estimates of $\nabla_\theta J$:

    $$\hat{g} = \nabla_\theta \log \pi_\theta(a)\,\big(r(a) - b\big),
      \qquad a \sim \pi_\theta$$

    Return an array of shape `(n_samples, len(logits))`. Do **not** average
    them: the variance is the object under study, and averaging here would throw
    it away.
    """
    # TODO: ...
    raise NotImplementedError


def optimal_constant_baseline(logits: np.ndarray, rewards: np.ndarray) -> float:
    r"""The constant baseline that minimises the total variance of the
    estimator:

    $$b^{*} = \frac{\mathbb{E}\big[\lVert\nabla\log\pi\rVert^2\, r\big]}
                   {\mathbb{E}\big[\lVert\nabla\log\pi\rVert^2\big]}$$

    A *weighted* mean of the rewards, weighted by how strongly each action moves
    the parameters — not the plain mean, and not the value function either,
    although the value function is the baseline everybody uses because it
    generalises to states.

    Computable exactly here, since the expectation is a sum over the actions.
    """
    # TODO: ...
    raise NotImplementedError


def total_variance(samples: np.ndarray) -> float:
    r"""$\sum_i \mathrm{Var}[\hat{g}_i]$ — the trace of the covariance, which is
    the scalar the baseline theorem is about. Population variance (`ddof=0`)."""
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: the same estimator, over episodes
# --------------------------------------------------------------------------- #
def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    r"""$G_t = r_t + \gamma G_{t+1}$, as in P06. Given back to you because
    everything below is built on it."""
    G = np.zeros(len(rewards))
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        G[t] = run
    return G


def episode_gradient(states: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, theta: np.ndarray, gamma: float,
                     baseline: np.ndarray | None = None,
                     gamma_t: bool = True) -> np.ndarray:
    r"""The REINFORCE gradient of one episode, for a tabular softmax policy
    `theta[state, action]`:

    $$\hat{g} = \sum_t \gamma^t\,\nabla_\theta \log \pi_\theta(a_t \mid s_t)
                \big(G_t - b(s_t)\big)$$

    `baseline` is a per-state array or `None`. Return an array shaped like
    `theta`.

    Note which $G$ appears: the return **from $t$ onwards**, not the return of
    the whole episode. Using the whole episode's return is also unbiased, and
    Part C measures what it costs.

    The $\gamma^t$ is there because the objective is
    $J(\theta) = \mathbb{E}[G_0]$, the return discounted to the start of the
    episode: a decision taken at step $t$ reaches $J$ only through rewards that
    $J$ discounts by $\gamma^t$ or more. With it, $\hat{g}$ is an unbiased
    estimate of $\nabla_\theta J$, and a test checks this exactly on a problem
    small enough to list every episode.

    `gamma_t=False` leaves the factor out, as many implementations do. Every
    step then counts as if the episode started there, so a decision at step 40
    weighs as much as one at step 0. The result is no longer an unbiased
    estimate of $\nabla_\theta J$: it follows the discounted return from every
    state the policy visits, and the discount only shortens how far ahead each
    return looks. Part B trains with both.
    """
    # TODO: ...
    raise NotImplementedError


def train(P: np.ndarray, R: np.ndarray, terminal: int, gamma: float = 0.95,
          n_iters: int = 300, batch: int = 10, alpha: float = 0.5,
          use_baseline: bool = False, seed: int = 0, max_steps: int = 60,
          gamma_t: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""REINFORCE on a tabular MDP, with an optional learned state baseline.

    Each iteration: roll out `batch` episodes, average their gradients, take one
    step. The baseline, when used, is the running mean of the returns seen from
    each state — the cheapest possible value estimate, and enough to make the
    point.

    `gamma_t` is passed to `episode_gradient`.

    Returns `(theta, returns_per_iter, grad_norms)`, where `grad_norms[i]` is
    the standard deviation of the batch's gradients: the quantity the baseline
    is supposed to shrink.
    """
    # TODO: ...
    raise NotImplementedError
