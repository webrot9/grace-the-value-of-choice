# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""P04 — policy iteration: monotone improvement, and why it stops.

Reference implementation. `pi.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Same MDP representation as P03: `P[s, a, s']`, `R[s, a]`, `gamma`. The pieces you
wrote last week — the exact policy evaluation and the greedy step — are given
here, so that what you implement is only what is new: the improvement step
stated in terms of the advantage, the loop, and the two quantities that make its
behaviour visible.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.exact import policy_evaluation_exact, q_from_v
from rl_lab.gridworld import gridworld


# --------------------------------------------------------------------------- #
# Q1: improvement, stated with the advantage
# --------------------------------------------------------------------------- #
def advantage(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
              gamma: float) -> np.ndarray:
    r"""The advantage of every action against the current policy:

    $$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

    Shape (S, A). By construction $A^\pi(s, \pi(s)) = 0$ for every $s$, which is
    the first thing the tests check: an action you are already taking cannot be
    an improvement on itself.
    """
    # TODO: ...
    raise NotImplementedError


def policy_improvement(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                       gamma: float) -> np.ndarray:
    r"""One improvement step:

    $$\pi'(s) = \arg\max_a A^\pi(s,a) = \arg\max_a Q^\pi(s,a)$$

    The two argmaxes are the same because $V^\pi(s)$ does not depend on $a$.
    Ties: **keep the current action** if it is among the maximisers. In exact
    arithmetic the first maximiser would stop as well, but in floating point two
    tied actions can swap order between two policies of the same value, and the
    loop can alternate between them for ever; the rule is what makes "the policy
    stopped changing" a reliable stopping rule.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the loop
# --------------------------------------------------------------------------- #
def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                     pi0: np.ndarray | None = None, max_iters: int = 200
                     ) -> tuple[np.ndarray, np.ndarray]:
    r"""Alternate exact evaluation and improvement until the policy stops moving.

    Return `(pi, values)` where `values` has shape `(k+1, S)` and holds
    $V^{\pi_0}, V^{\pi_1}, \dots$ — one row per policy visited, including the
    starting one. The trace is what makes the monotonicity visible.

    Stop as soon as the improvement step returns the same policy: at that point
    $\pi$ is greedy with respect to its own value, which is the Bellman
    optimality equation, so $V^\pi = V^{*}$.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: the two things that make the behaviour visible
# --------------------------------------------------------------------------- #
def is_monotone(values: np.ndarray, tol: float = 1e-9) -> bool:
    r"""True if every step improved the value **in every state**:

    $$V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s) \qquad \forall s, \forall k$$

    The pointwise quantifier is the whole content of the theorem. A policy that
    is better on average but worse somewhere is not an improvement, and policy
    iteration never produces one.
    """
    # TODO: return ...
    raise NotImplementedError


def n_distinct_policies(P: np.ndarray, R: np.ndarray, gamma: float,
                        pi0: np.ndarray | None = None) -> int:
    r"""How many *different* policies the loop visits before stopping.

    Since every step is a strict improvement in at least one state until
    convergence, a policy can never be visited twice: the count is bounded by
    $|\mathcal{A}|^{|\mathcal{S}|}$, and in practice it is a handful. This is
    what "terminates in finitely many steps" means, and it is the property value
    iteration does not have.
    """
    # TODO: return ...
    raise NotImplementedError
