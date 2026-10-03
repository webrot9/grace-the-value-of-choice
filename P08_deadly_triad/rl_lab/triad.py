r"""P08 — Baird's counterexample: the deadly triad, and which leg to remove.

Reference implementation. `triad.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Seven states, eight weights, every reward zero. The true value function is
$V^\pi(s) = 0$ everywhere and $\mathbf{w} = \mathbf{0}$ represents it exactly, so
there is nothing hard to learn here. Semi-gradient TD(0) diverges anyway: the
weights grow without bound while chasing a target they could hit by standing
still.

Three ingredients have to be present at once:

  bootstrapping             the target contains the current estimate
  function approximation    states share weights, so an update leaks
  off-policy                the states you update are drawn from one policy and
                            the successors you bootstrap from come from another

Remove any one and it converges. The whole lab is four runs of the same
function with one argument changed, which is why nothing here is stochastic:
`expected_update` is the exact expectation of the TD update, so every number in
Part B is reproducible to the last digit with no seed at all.

Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., section 11.2
and figures 11.1-11.2.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Given: Baird's star
# --------------------------------------------------------------------------- #
N_STATES = 7
N_FEATURES = 8
DASHED, SOLID = 0, 1


def baird_features() -> np.ndarray:
    r"""The feature matrix $\Phi$, shape (7, 8).

    States 0-5 are the six upper states: $\phi_i = 2 e_i + e_7$.
    State 6 is the lower state: $\phi_6 = e_6 + 2 e_7$.

    Every state shares the last weight, and that sharing is the "function
    approximation" leg of the triad. Note that $\mathbf{w} = \mathbf{0}$ gives
    $\hat{V} = 0$ exactly: the representation is not the problem.
    """
    Phi = np.zeros((N_STATES, N_FEATURES))
    for i in range(6):
        Phi[i, i] = 2.0
        Phi[i, 7] = 1.0
    Phi[6, 6] = 1.0
    Phi[6, 7] = 2.0
    return Phi


def baird_mdp() -> np.ndarray:
    """`P[s, a, s']` for Baird's star: two actions from every state.

    `DASHED` sends the agent to one of the six upper states, uniformly.
    `SOLID` sends it to the lower state, state 6, with certainty.
    All rewards are zero, so there is no reward array to carry around.
    """
    P = np.zeros((N_STATES, 2, N_STATES))
    P[:, DASHED, :6] = 1.0 / 6.0
    P[:, SOLID, 6] = 1.0
    return P


def behaviour_policy() -> np.ndarray:
    """The policy that generates the data: dashed 6/7 of the time, solid 1/7.
    Its stationary distribution is uniform over the seven states, which is worth
    checking rather than believing — one of the tests does."""
    pi = np.zeros((N_STATES, 2))
    pi[:, DASHED] = 6.0 / 7.0
    pi[:, SOLID] = 1.0 / 7.0
    return pi


def target_policy() -> np.ndarray:
    """The policy being evaluated: always solid. Under it the agent sits in
    state 6 for ever, collecting nothing, so $V^\\pi \\equiv 0$."""
    pi = np.zeros((N_STATES, 2))
    pi[:, SOLID] = 1.0
    return pi


# --------------------------------------------------------------------------- #
# Q1: policies as matrices
# --------------------------------------------------------------------------- #
def policy_transition(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    r"""$P^\pi_{ss'} = \sum_a \pi(a\mid s)\, p(s'\mid s,a)$, shape (S, S).

    Every row must sum to one; a test checks it, because an unnormalised row is
    the kind of bug that turns a convergence proof into a divergence plot and
    looks like a discovery.
    """
    # TODO: return ...
    raise NotImplementedError


def stationary_distribution(P_pi: np.ndarray, tol: float = 1e-14,
                            max_iters: int = 100_000) -> np.ndarray:
    r"""The distribution $d$ with $d^\top P^\pi = d^\top$, reached by iterating
    from the uniform distribution.

    This is the distribution the updates are drawn from. In the off-policy
    setting it comes from the behaviour policy while the bootstrap target comes
    from the target policy, and that mismatch is the third leg of the triad —
    stated as two objects that disagree, rather than as the word "off-policy".
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the update, with each leg of the triad switchable
# --------------------------------------------------------------------------- #
def expected_update(w: np.ndarray, Phi: np.ndarray, d: np.ndarray,
                    P_target: np.ndarray, gamma: float, alpha: float,
                    bootstrap: bool = True) -> np.ndarray:
    r"""One **expected** semi-gradient TD(0) step: the exact expectation of the
    stochastic update, so there is nothing to average over.

    $$\delta_s = \gamma\, \big(P^{\pi}\Phi\, \mathbf{w}\big)_s - \phi_s^\top \mathbf{w},
      \qquad
      \Delta\mathbf{w} = \alpha \sum_s d(s)\, \delta_s\, \phi_s$$

    with all rewards zero. Return the **new** `w`, not the increment.

    `bootstrap=False` replaces $\gamma (P^\pi \Phi \mathbf{w})_s$ with the true
    return, which is $0$ here: that is the Monte Carlo target, and it turns the
    update into an ordinary gradient step on $\tfrac12 \sum_s d(s)(\hat{V}_s)^2$.

    "Semi-gradient" is the word for what is missing: the target depends on
    $\mathbf{w}$ too, and its derivative is not taken. That omission is the
    bootstrapping leg, and the whole counterexample lives in it.
    """
    # TODO: ...
    raise NotImplementedError


def run(w0: np.ndarray, Phi: np.ndarray, d: np.ndarray, P_target: np.ndarray,
        gamma: float, alpha: float, n_steps: int,
        bootstrap: bool = True) -> np.ndarray:
    r"""Iterate `expected_update` and return the whole weight history, shape
    `(n_steps + 1, len(w0))`.

    Return the history and not just the final weights: a single final number
    cannot tell divergence from slow convergence, and this lab is entirely about
    telling those two apart.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: reading the outcome
# --------------------------------------------------------------------------- #
def value_error(Phi: np.ndarray, w: np.ndarray, d: np.ndarray) -> float:
    r"""The $d$-weighted root-mean-square value error against the true
    $V^\pi \equiv 0$:

    $$\sqrt{\sum_s d(s)\,\big(\phi_s^\top \mathbf{w}\big)^2}$$

    The truth is zero everywhere, so this is just the weighted norm of the
    predictions — which is what makes the counterexample so blunt: any number
    above zero is pure error, and there is no approximation floor to hide
    behind.
    """
    # TODO: return ...
    raise NotImplementedError


def key_matrix(Phi: np.ndarray, d: np.ndarray, P_target: np.ndarray,
               gamma: float) -> np.ndarray:
    r"""The matrix that decides everything:
    $\mathbf{A} = \Phi^\top D (\mathbf{I} - \gamma P^\pi)\Phi$, shape (8, 8).

    The expected update is $\mathbf{w} \leftarrow (\mathbf{I} - \alpha \mathbf{A})\mathbf{w}$,
    so it converges for small $\alpha$ exactly when every eigenvalue of
    $\mathbf{A}$ has a positive real part. On-policy, with features of full
    column rank, $\mathbf{A}$ is provably positive definite. Baird's eight
    features on seven states do not have full rank, so on-policy the smallest
    eigenvalue is 0, and the run settles at a fixed point that depends on where
    it started. Off-policy nothing protects $\mathbf{A}$, and Part B prints the
    negative eigenvalue that does the damage.

    Given as one line so that the figure and the eigenvalue are the same claim
    twice: if the plot and the spectrum ever disagree, one of them is a bug.
    """
    return Phi.T @ (d[:, None] * (np.eye(len(d)) - gamma * P_target)) @ Phi
