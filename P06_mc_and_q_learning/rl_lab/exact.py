"""Exact answers, computed from the model.

This file is **given** from P04 onwards. You wrote these functions yourself in
P03, and from then on you get them for free — because in every later lab they
stop being the exercise and become the *ground truth* that the exercise is
scored against.

That is the pattern worth noticing. A learner that never looks at `P` and `R`
estimates a value from samples; these functions compute the same value from the
model in one linear solve. Having both is what turns "the agent learned
something" into "the agent is 0.0343 below the right answer", and it is the
reason almost every lab in this course builds its MDP as two arrays instead of
wrapping a simulator.

Nothing here is allowed inside a learning algorithm. If a function you are
writing calls one of these, it is cheating — and, worse, it is measuring
nothing.
"""
from __future__ import annotations

import numpy as np


def policy_evaluation_exact(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                            gamma: float) -> np.ndarray:
    r"""$V^\pi$ for a **deterministic** policy given as one action per state.

    $$V^\pi = (\mathbf{I} - \gamma P^\pi)^{-1}\, \mathbf{r}^\pi$$

    `pi` is an array of integers, `pi[s]` being the action taken in `s`, so
    `P[idx, pi]` picks one row per state and the whole thing is a linear system
    of `|S|` equations. No iteration, no tolerance: this is the answer.
    """
    n = len(pi)
    idx = np.arange(n)
    return np.linalg.solve(np.eye(n) - gamma * P[idx, pi], R[idx, pi])


def policy_evaluation_stochastic(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                                 gamma: float) -> np.ndarray:
    r"""The same thing for a **stochastic** policy `pi[s, a]`.

    Average the dynamics and the reward over the policy first,

    $$P^\pi_{ss'} = \sum_a \pi(a\mid s)\, p(s'\mid s,a), \qquad
      r^\pi_s = \sum_a \pi(a\mid s)\, r(s,a),$$

    and then solve the same linear system. The `einsum` calls are those two sums
    written once each: `"sa,sat->st"` reads "for every s, sum over a of pi[s,a]
    times P[s,a,t]".
    """
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


def q_from_v(P: np.ndarray, R: np.ndarray, V: np.ndarray,
             gamma: float) -> np.ndarray:
    r"""$Q(s,a) = r(s,a) + \gamma \sum_{s'} p(s'\mid s,a)\, V(s')$, shape (S, A).

    One line, because `P @ V` already contracts the last axis of `P` with `V`:
    the result is the expected next-state value for every `(s, a)`.
    """
    return R + gamma * (P @ V)


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                    tol: float = 1e-12, max_iters: int = 10_000) -> np.ndarray:
    r"""$V^{*}$, by applying the Bellman optimality operator until it stops
    moving.

    $$V_{k+1}(s) = \max_a \Big( r(s,a) + \gamma \sum_{s'} p(s'\mid s,a) V_k(s') \Big)$$

    The operator is a $\gamma$-contraction in the sup norm — that is the theorem
    P03 makes visible — so this loop converges from any starting point, and the
    `tol` of $10^{-12}$ is reached long before `max_iters` on anything this
    small.
    """
    V = np.zeros(P.shape[0])
    for _ in range(max_iters):
        V_new = (R + gamma * (P @ V)).max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V
