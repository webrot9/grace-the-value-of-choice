r"""P09, first half — maximization bias, where it is impossible to miss.

Reference implementation. `maxbias.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

The MDP is Sutton & Barto's example 6.7 and it has almost nothing in it:

    A --left--> B --(any of 10 actions)--> terminal,  reward ~ N(-0.1, 1)
    A --right-> terminal,                             reward 0

So $q^{*}(A,\text{right}) = 0$ and $q^{*}(A,\text{left}) = -0.1$: right is
optimal, and an $\epsilon$-greedy agent with $\epsilon = 0.1$ should end up
taking left about 5% of the time. Q-learning takes it far more often, for a
while, and the reason is one line of the update.

$\max_a \hat{Q}(B,a)$ is an estimate of $\max_a q^{*}(B,a)$, and it is a biased
one: the maximum of ten noisy estimates of $-0.1$ is above $-0.1$ even when
every individual estimate is unbiased. The agent is not fooled by the
environment; it is fooled by its own arithmetic.
"""
from __future__ import annotations

import numpy as np

A, B, TERMINAL = 0, 1, 2
LEFT, RIGHT = 0, 1


# --------------------------------------------------------------------------- #
# Q1: the bias, on its own
# --------------------------------------------------------------------------- #
def max_of_estimates(rng: np.random.Generator, n_actions: int, mean: float,
                     sd: float, n_samples: int, n_reps: int) -> np.ndarray:
    r"""Draw `n_actions` sample means, each averaging `n_samples` draws from
    $\mathcal{N}(\text{mean}, \text{sd}^2)$, and return the maximum. Repeat
    `n_reps` times and return the array of maxima.

    Every one of those sample means is an unbiased estimate of `mean`. Their
    maximum is not an unbiased estimate of `mean`, and it is not an unbiased
    estimate of the maximum either — the true maximum here *is* `mean`, because
    all the actions have the same value.

    This function contains no reinforcement learning at all. That is the point:
    the bias is in the estimator, and it would be there in any field that took a
    maximum over noisy numbers.
    """
    # TODO: ...
    raise NotImplementedError


def double_estimate(rng: np.random.Generator, n_actions: int, mean: float,
                    sd: float, n_samples: int, n_reps: int) -> np.ndarray:
    r"""The same experiment, with the samples split in two halves: use the first
    half to choose which action looks best, and the **second** half to say what
    that action is worth.

    $$\hat{v} = \hat{\mu}^{(2)}\big[\arg\max_a \hat{\mu}^{(1)}_a\big]$$

    Same data, same amount of it, one extra line of bookkeeping. The selection
    noise and the evaluation noise are now independent, which is the entire idea
    behind Double Q-learning.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the same bias inside a control algorithm
# --------------------------------------------------------------------------- #
def q_learning_target(reward: float, q_next: np.ndarray, gamma: float,
                      done: bool) -> float:
    r"""The Q-learning target $r + \gamma \max_{a'} \hat{Q}(s',a')$, and $r$
    alone when $s'$ is terminal.

    One array, used twice: to pick the action and to say what it is worth.
    """
    # TODO: ...
    raise NotImplementedError


def double_q_target(reward: float, q_select: np.ndarray, q_evaluate: np.ndarray,
                    gamma: float, done: bool) -> float:
    r"""The Double Q-learning target:

    $$r + \gamma\, \hat{Q}^{(2)}\!\left(s', \arg\max_{a'} \hat{Q}^{(1)}(s',a')\right)$$

    `q_select` chooses the action; `q_evaluate` prices it. Two arrays, each used
    once. If you pass the same array twice you get the previous function back,
    which a test checks — the difference between the two algorithms really is
    that small.
    """
    # TODO: ...
    raise NotImplementedError


def run_bias_mdp(n_episodes: int = 300, n_b_actions: int = 10,
                 alpha: float = 0.1, eps: float = 0.1, gamma: float = 1.0,
                 double: bool = False, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    r"""Run Q-learning (or Double Q-learning) on the two-state MDP above.

    Returns `(left_taken, max_q_b)`, both of length `n_episodes`:
    `left_taken[e]` is 1 if the agent took left from A in episode `e`, and
    `max_q_b[e]` is $\max_a \hat{Q}(B,a)$ after that episode — the quantity
    whose true value is $-0.1$ and which the figure plots against it.

    For the double version, keep two tables and update one of them per step,
    chosen with probability one half; the behaviour policy is $\epsilon$-greedy
    on their **sum**, which is the standard formulation and is what makes the
    comparison against the single-table version fair.
    """
    # TODO: ...
    raise NotImplementedError


def overestimation(max_q: np.ndarray, truth: float) -> np.ndarray:
    r"""$\max_a \hat{Q} - \max_a q^{*}$, elementwise. Signed, because the sign
    is the finding: an unbiased algorithm gives a series that straddles zero and
    a biased one gives a series that sits on one side of it."""
    # TODO: return ...
    raise NotImplementedError
