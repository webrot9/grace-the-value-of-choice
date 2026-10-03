# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""P03 — dynamic programming: the Bellman operator as a contraction.

Reference implementation. `dp.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

An MDP here is three arrays, and nothing else:

    P[s, a, s']   transition probabilities, each row P[s, a] sums to 1
    R[s, a]       expected reward, in [0, R_max]
    gamma         discount factor in [0, 1)

Everything is exact linear algebra on those arrays, which is what makes the
contraction visible to six decimal places instead of approximately.

Notation follows book/notation.md: the Bellman operators are written B and B_pi
(the book writes them $\\mathcal{B}$ and $\\mathcal{B}^\\pi$), and the optimal
value function is V*.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.gridworld import gridworld


# --------------------------------------------------------------------------- #
# Q1: the two Bellman operators
# --------------------------------------------------------------------------- #
def bellman_expectation(P: np.ndarray, R: np.ndarray, V: np.ndarray,
                        pi: np.ndarray, gamma: float) -> np.ndarray:
    r"""One application of $\mathcal{B}^\pi$ for a deterministic policy `pi`:

    $$(\mathcal{B}^\pi V)(s) = r(s, \pi(s)) + \gamma \sum_{s'} p(s' \mid s, \pi(s))\, V(s')$$

    `pi` is an array of length S holding one action index per state.
    """
    # TODO: return ...
    raise NotImplementedError


def bellman_optimality(P: np.ndarray, R: np.ndarray, V: np.ndarray,
                       gamma: float) -> np.ndarray:
    r"""One application of $\mathcal{B}$:

    $$(\mathcal{B}V)(s) = \max_a \Big[ r(s,a) + \gamma \sum_{s'} p(s'\mid s,a)\, V(s') \Big]$$

    Return the vector of length S. Do not loop over actions in Python if you can
    avoid it: `P @ V` already gives you the (S, A) matrix of expected next
    values.
    """
    # TODO: return ...
    raise NotImplementedError


def greedy_policy(P: np.ndarray, R: np.ndarray, V: np.ndarray,
                  gamma: float) -> np.ndarray:
    r"""The policy that is greedy with respect to `V`:

    $$\pi(s) = \arg\max_a \Big[ r(s,a) + \gamma \sum_{s'} p(s'\mid s,a)\, V(s') \Big]$$

    Ties: lowest action index. By the second Bellman optimality theorem, the
    greedy policy with respect to $V^{*}$ **is** optimal — which is why value
    iteration can ignore policies until the very last step.
    """
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: solving exactly, and solving by iterating
# --------------------------------------------------------------------------- #
def policy_evaluation_exact(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                            gamma: float) -> np.ndarray:
    r"""Solve $V^\pi$ in closed form.

    $V^\pi = \mathcal{B}^\pi V^\pi$ is a linear system, so

    $$V^\pi = (\mathbf{I} - \gamma P^\pi)^{-1}\, \mathbf{r}^\pi,$$

    where $P^\pi[s,s'] = p(s'\mid s,\pi(s))$ and $\mathbf{r}^\pi[s] = r(s,\pi(s))$.
    The inverse exists for every $\gamma < 1$ because $\gamma P^\pi$ has spectral
    radius below one.

    This costs $O(|\mathcal{S}|^3)$, which is exactly why value iteration exists.
    Here it is the ground truth the iterative methods are checked against.
    """
    # TODO: return ...
    raise NotImplementedError


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                    iters: int = 200, V0: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    r"""Run `iters` sweeps of $\mathcal{B}$ starting from `V0` (zeros by default).

    Return `(V, history)` where `history` has shape `(iters + 1, S)` and holds
    every iterate including the starting one — the whole point of this lab is to
    look at how the error shrinks, and for that you need the trace, not just the
    answer.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: the error, and the rate at which it falls
# --------------------------------------------------------------------------- #
def sup_norm(a: np.ndarray, b: np.ndarray) -> float:
    r"""$\lVert a - b \rVert_\infty$: the norm every contraction argument uses."""
    # TODO: return ...
    raise NotImplementedError


def error_ratios(history: np.ndarray, v_star: np.ndarray) -> np.ndarray:
    r"""The ratio between consecutive errors,

    $$\rho_k = \frac{\lVert V_{k+1} - V^{*}\rVert_\infty}{\lVert V_k - V^{*}\rVert_\infty}.$$

    The contraction theorem says $\rho_k \le \gamma$ for every $k$. On the
    deterministic grid of this lab the bound is attained, $\rho_k = \gamma$
    exactly, because the whole error sits on the single state farthest from the
    goal; with `--slip 0.1` the ratio stays well below $\gamma$.

    Ratios are dropped as soon as **either** error is below $10^{-12}$. On this
    grid the error becomes exactly zero after a few sweeps, and keeping those
    ratios would give $0/x = 0$ and then $0/0$.
    """
    # TODO: ...
    raise NotImplementedError
