# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""
P01 - Bandits and regret. REFERENCE SOLUTION.

This is the single source of truth. The student file is *generated* from this one
by tools/strip_solutions.py, which deletes everything between `# TODO: ` and
by tools/strip_solutions.py, which deletes everything between `raise NotImplementedError
student file by hand: edit this one and regenerate.

Convention adopted from the course TA repository (assignments/strip_solutions.py).
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Q1. Confidence radius (Hoeffding)
# --------------------------------------------------------------------------- #
def hoeffding_radius(n: int, delta: float) -> float:
    r"""Half-width of a $1-\delta$ confidence interval for the mean of a
    $[0,1]$-bounded variable after $n$ i.i.d. samples:

    $$\mathrm{rad}(n,\delta) = \sqrt{\frac{\log(2/\delta)}{2n}}$$

    Return `float('inf')` for `n == 0`: an arm never pulled has no information,
    and infinity is what makes the optimistic rule try it first.
    """
    if n == 0:
        return float("inf")
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2. Action selection rules (deterministic given the statistics)
# --------------------------------------------------------------------------- #
def select_greedy(means: np.ndarray, counts: np.ndarray) -> int:
    """Pull the arm with the highest empirical mean. Ties: lowest index.
    Arms never pulled count as mean 0 here (they are handled by the caller).

    This is the commit step of greedy and of explore-then-commit, and the greedy
    step of epsilon-greedy: which means it receives is the caller's business."""
    # TODO: return ...
    raise NotImplementedError


def select_etc(means: np.ndarray, counts: np.ndarray, t: int,
               explore_rounds: int) -> int:
    r"""Explore-then-commit: round-robin for the first $mK$ steps, then greedy.

    With $K$ arms and $m$ = `explore_rounds` pulls each, the rule is

    $$a_t = \begin{cases}
        (t-1) \bmod K & t \le mK \quad\text{(explore)}\\
        \arg\max_a \hat\mu(a) & t > mK \quad\text{(commit)}
    \end{cases}$$

    Two things worth noticing, because the tests check both. During exploration
    the choice does **not** look at `means`: that is what makes the exploration
    cost exactly $mK$ and independent of the data. And after the switch the
    choice never changes again, which is why a wrong commit is paid for the whole
    remaining horizon. That second property needs the right `means`: after the
    switch `run_bandit` passes the means as they were **at the end of the
    exploration**, not the running ones. With the running means the committed
    arm's estimate keeps moving, and "commit" quietly becomes greedy.

    Greedy, as in the lecture, is this same rule with $m = 1$: one pull per arm,
    then commit.

    `t` is 1-based, as everywhere in this lab.
    """
    # TODO: ...
    raise NotImplementedError


def select_ucb(means: np.ndarray, counts: np.ndarray, t: int, delta: float) -> int:
    r"""Upper confidence bound rule:

    $$a_t = \arg\max_a \left[\hat\mu(a) + \mathrm{rad}(N_t(a), \delta(t))\right]$$

    Any arm with `counts == 0` has an infinite radius, so it is pulled first.
    Ties: lowest index.

    Use `delta_t = delta / t**3`. This is the *anytime* version of the rule in
    the lecture: it does not need to know the horizon $T$. The intervals must
    hold at every step, for every arm and every number of pulls $n \le t$, and
    the union bound over all of them sums to
    $K \sum_t t\,\delta_t = K\delta\,\pi^2/6$: finite, so they hold at all
    times at once, with probability at least $1 - K\pi^2\delta/6$, the same
    guarantee up to a constant factor on $\delta$. Without the shrinking
    $\delta_t$ the sum diverges and the expected regret becomes linear; Q3 of the
    report asks you to break it.

    The lecture's version knows $T$ and gives each of the $KT$ intervals
    $\delta/(KT)$ instead: it is `select_ucb_known_horizon` below, given, and
    the figure plots the two side by side.
    """
    # TODO: ...
    raise NotImplementedError


def select_ucb_known_horizon(means: np.ndarray, counts: np.ndarray,
                             horizon: int, delta: float) -> int:
    r"""UCB as in the lecture (slide 52), with the horizon $T$ known in advance.
    GIVEN: you do not implement it; compare it with your `select_ucb`.

    $$a_t = \arg\max_a \left[\hat\mu(a) + \sqrt{\frac{\ln(KT/\delta)}{N_t(a)}}\right]$$

    Over the whole run UCB uses at most $KT$ intervals ($K$ arms, up to $T$
    pulls each), so the union bound gives each one $\delta/(KT)$. The bonus of an
    arm depends only on its own pulls, not on $t$: it needs $T$ before the run
    starts, which the anytime `select_ucb` does not. Same rate, $\sqrt{KT\ln T}$,
    different constants. Unpulled arms first, ties to the lowest index.
    """
    log_kt = math.log(len(means) * horizon / delta)
    bonus = np.array([math.inf if n == 0 else math.sqrt(log_kt / n) for n in counts])
    return int(np.argmax(means + bonus))


def select_epsilon_greedy(means: np.ndarray, counts: np.ndarray,
                          epsilon: float, rng: np.random.Generator) -> int:
    """With probability `epsilon` pull a uniformly random arm, otherwise greedy.
    Draw the uniform *first* with `rng.random()`, then the arm with
    `rng.integers(len(means))`, so that results are reproducible across
    implementations."""
    # TODO: ...
    raise NotImplementedError


def select_thompson_bernoulli(successes: np.ndarray, failures: np.ndarray,
                              rng: np.random.Generator) -> int:
    r"""Thompson sampling with a Beta(1,1) prior: draw
    $\tilde\mu(a) \sim \mathrm{Beta}(1+S_a,\, 1+F_a)$ and pull the argmax."""
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3. Regret
# --------------------------------------------------------------------------- #
def pseudo_regret(mu: np.ndarray, pulls: np.ndarray) -> float:
    r"""Pseudo-regret after the pulls recorded in `pulls` (an array of arm
    indices, in order):

    $$\mathrm{Reg}(T) = T\,\mu(a^{*}) - \sum_{t=1}^{T} \mu(a_t)
          = \sum_a N_T(a)\, \Delta_a, \qquad \Delta_a = \mu(a^{*}) - \mu(a)$$

    Note this uses the *true* means, not the observed rewards: it is the
    quantity the theory bounds, and it is not observable by the agent.
    """
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# The interaction loop (given: you do not need to modify it)
# --------------------------------------------------------------------------- #
def run_bandit(mu: np.ndarray, horizon: int, algorithm: str, seed: int,
               epsilon: float = 0.1, delta: float = 0.1,
               explore_rounds: int | None = None) -> np.ndarray:
    """Run one Bernoulli bandit episode. Returns the array of pulled arms.

    Greedy and explore-then-commit decide once, at the end of their exploration,
    from the means measured up to then, and never look at the data again: the
    loop freezes those means and passes them to `select_etc` from the switch on.
    For greedy the exploration is one pull per arm, whatever `explore_rounds` says.
    """
    rng = np.random.default_rng(seed)
    K = len(mu)
    counts = np.zeros(K)
    totals = np.zeros(K)
    succ = np.zeros(K)
    fail = np.zeros(K)
    pulls = np.empty(horizon, dtype=int)

    if explore_rounds is None:  # theory-optimal m for explore-then-commit
        explore_rounds = max(1, int(round((horizon / K) ** (2 / 3))))
    if algorithm == "greedy":   # lecture: try each arm once, then commit
        explore_rounds = 1
    committed = None            # the means at the end of exploration

    for t in range(1, horizon + 1):
        means = np.divide(totals, counts, out=np.zeros(K), where=counts > 0)
        if algorithm in ("greedy", "etc"):
            if t == explore_rounds * K + 1:
                committed = means.copy()
            a = select_etc(means if committed is None else committed, counts, t,
                           explore_rounds)
        elif algorithm == "ucb":
            a = select_ucb(means, counts, t, delta)
        elif algorithm == "ucb_T":
            a = select_ucb_known_horizon(means, counts, horizon, delta)
        elif algorithm == "eps":
            a = int(np.argmin(counts)) if counts.min() == 0 else \
                select_epsilon_greedy(means, counts, epsilon, rng)
        elif algorithm == "thompson":
            a = select_thompson_bernoulli(succ, fail, rng)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")

        r = float(rng.random() < mu[a])
        counts[a] += 1
        totals[a] += r
        succ[a] += r
        fail[a] += 1 - r
        pulls[t - 1] = a
    return pulls
