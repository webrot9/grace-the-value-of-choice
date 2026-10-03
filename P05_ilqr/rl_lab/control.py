# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""P05 — LQR, iLQR, and certainty equivalence.

Reference implementation. `control.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Notation follows book/notation.md, which differs from every control textbook and
says why: the dynamics matrices are Fx, Fu and the cost matrices are Cx, Cu,
because Q and R are already the action-value function and the reward.

    x_{t+1} = Fx x_t + Fu u_t + w_t,      w_t ~ N(0, sigma^2 I)
    c(x, u) = x^T Cx x + u^T Cu u

The whole lab is built to make one fact impossible to miss: the optimal gain
does not depend on sigma. Not "barely depends": does not appear in the
recursion at all.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Q1: the Riccati recursion
# --------------------------------------------------------------------------- #
def riccati_step(Fx: np.ndarray, Fu: np.ndarray, Cx: np.ndarray, Cu: np.ndarray,
                 P_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""One step of the backward recursion, from $\mathbf{P}_{t+1}$ to
    $(\mathbf{K}_t, \mathbf{P}_t)$.

    $$\mathbf{K}_t = -\big(\mathbf{C}_u + \mathbf{F}_u^\top \mathbf{P}_{t+1}\mathbf{F}_u\big)^{-1}
                      \mathbf{F}_u^\top \mathbf{P}_{t+1}\mathbf{F}_x$$
    $$\mathbf{P}_t = \mathbf{C}_x + \mathbf{K}_t^\top \mathbf{C}_u \mathbf{K}_t
                   + (\mathbf{F}_x + \mathbf{F}_u\mathbf{K}_t)^\top \mathbf{P}_{t+1}
                     (\mathbf{F}_x + \mathbf{F}_u\mathbf{K}_t)$$

    Return `(K, P)`. Notice what is *not* in these two lines: the noise. It
    cannot be, because it has mean zero and the value function is quadratic —
    that is certainty equivalence, and Part B measures it.
    """
    # TODO: ...
    raise NotImplementedError


def lqr_backward(Fx: np.ndarray, Fu: np.ndarray, Cx: np.ndarray, Cu: np.ndarray,
                 horizon: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""The full backward pass over a finite horizon.

    Initialise $\mathbf{P}_T = \mathbf{C}_x$ and run the recursion down to
    $t = 0$. Return `(Ks, Ps)` with `len(Ks) == horizon` and
    `len(Ps) == horizon + 1`, both in **forward** time order, so `Ks[0]` is the
    gain applied at the first step.
    """
    # TODO: ...
    raise NotImplementedError


def cost_to_go(P: np.ndarray, x: np.ndarray) -> float:
    r"""$V_t^{*}(x) = x^\top \mathbf{P}_t\, x$ — the value function is quadratic,
    which is the induction hypothesis the whole derivation rests on."""
    # TODO: return ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: linearising a non-linear system
# --------------------------------------------------------------------------- #
def linearize(f, x: np.ndarray, u: np.ndarray, eps: float = 1e-6
              ) -> tuple[np.ndarray, np.ndarray]:
    r"""Jacobians of `f` at `(x, u)` by **central** differences:

    $$\mathbf{F}_x = \frac{\partial f}{\partial x}\Big|_{(x,u)}, \qquad
      \mathbf{F}_u = \frac{\partial f}{\partial u}\Big|_{(x,u)}$$

    Central and not forward differences: the error is $O(\epsilon^2)$ instead of
    $O(\epsilon)$, and on a linear system central differences recover the exact
    matrices to machine precision, which is what the test checks.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: running the controller
# --------------------------------------------------------------------------- #
def rollout(Fx: np.ndarray, Fu: np.ndarray, Cx: np.ndarray, Cu: np.ndarray,
            Ks: list[np.ndarray], x0: np.ndarray, sigma: float = 0.0,
            seed: int = 0) -> tuple[np.ndarray, float]:
    r"""Apply $u_t = \mathbf{K}_t x_t$ for the whole horizon and add noise
    $w_t \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ to the dynamics.

    Return `(states, total_cost)` with `states` of shape `(horizon + 1, n)`.
    The terminal cost $x_T^\top \mathbf{C}_x x_T$ is included.
    """
    # TODO: ...
    raise NotImplementedError


def lqr_backward_tv(Fxs: list[np.ndarray], Fus: list[np.ndarray],
                    Cx: np.ndarray, Cu: np.ndarray
                    ) -> list[np.ndarray]:
    r"""Backward pass for a **time-varying** linear system: one pair
    $(\mathbf{F}_x^{(t)}, \mathbf{F}_u^{(t)})$ per step, as produced by
    linearising a non-linear system along a trajectory.

    Same recursion as `lqr_backward`, only the matrices change with $t$. Return
    the gains in forward time order.
    """
    # TODO: ...
    raise NotImplementedError


def ilqr_backward(Fxs: list[np.ndarray], Fus: list[np.ndarray],
                  xs: np.ndarray, us: np.ndarray, Cx: np.ndarray,
                  Cu: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""The backward pass of iLQR, around a nominal trajectory
    $(\bar x_t, \bar u_t)$ = `(xs, us)`, with one linearisation
    $(\mathbf{F}_x^{(t)}, \mathbf{F}_u^{(t)})$ per step.

    The unknowns are the deviations $\delta x = x - \bar x$ and
    $\delta u = u - \bar u$. Around a trajectory that is not at the origin the
    cost has a gradient as well as a curvature, so the value function is
    quadratic *plus linear* in $\delta x$, and the policy is affine,
    $\delta u_t = \mathbf{k}_t + \mathbf{K}_t\,\delta x_t$. Start from the
    terminal cost, $V_x = 2\mathbf{C}_x \bar x_T$ and $V_{xx} = 2\mathbf{C}_x$,
    and for $t = T-1, \dots, 0$, with $V$ the one of step $t+1$:

    $$Q_x = 2\mathbf{C}_x\bar x_t + \mathbf{F}_x^\top V_x, \quad
      Q_u = 2\mathbf{C}_u\bar u_t + \mathbf{F}_u^\top V_x,$$
    $$Q_{xx} = 2\mathbf{C}_x + \mathbf{F}_x^\top V_{xx}\mathbf{F}_x, \quad
      Q_{uu} = 2\mathbf{C}_u + \mathbf{F}_u^\top V_{xx}\mathbf{F}_u, \quad
      Q_{ux} = \mathbf{F}_u^\top V_{xx}\mathbf{F}_x,$$
    $$\mathbf{k}_t = -Q_{uu}^{-1}Q_u, \qquad \mathbf{K}_t = -Q_{uu}^{-1}Q_{ux},$$
    $$V_x = Q_x + \mathbf{K}_t^\top Q_{uu}\mathbf{k}_t + \mathbf{K}_t^\top Q_u
            + Q_{ux}^\top\mathbf{k}_t, \quad
      V_{xx} = Q_{xx} + \mathbf{K}_t^\top Q_{uu}\mathbf{K}_t
               + \mathbf{K}_t^\top Q_{ux} + Q_{ux}^\top\mathbf{K}_t.$$

    The factors of 2 come from writing the cost as $x^\top\mathbf{C}_x x$,
    without the one half that many texts put in front. $\mathbf{K}_t$ is the
    gain of `lqr_backward_tv`: with $V_{xx} = 2\mathbf{P}$ the two recursions
    are the same one. What is new is $\mathbf{k}_t$, the step in the controls
    that the gradient asks for. Without it the forward pass would reproduce the
    nominal trajectory exactly, since $\delta x_t = 0$ all along it. Return
    `(ks, Ks)` in forward time order.
    """
    # TODO: ...
    raise NotImplementedError


def ilqr(f, Cx: np.ndarray, Cu: np.ndarray, x0: np.ndarray,
         us: np.ndarray, iters: int = 20, alpha: float = 1.0
         ) -> tuple[np.ndarray, np.ndarray]:
    r"""iLQR for a regulation problem: drive the state to the origin.

    Keep a nominal trajectory: the controls `us` and the states they produce
    from `x0`. Each iteration linearises `f` **at every step of that
    trajectory**, runs `ilqr_backward` around it, and rolls the true system
    forward with the affine policy

    $$u_t = \bar u_t + \alpha\,\mathbf{k}_t + \mathbf{K}_t\,(x_t - \bar x_t),
      \qquad x_{t+1} = f(x_t, u_t),$$

    which becomes the next nominal trajectory. The step size $\alpha$ scales
    only $\mathbf{k}_t$, the move the model asks for; the feedback term keeps
    the new trajectory close to what the model was built on. Part C is about
    $\alpha$: the linear model is only accurate near the trajectory it was
    taken on, and a full step can land somewhere the model knows nothing about.

    On a system that is already linear, one iteration with $\alpha = 1$ lands
    exactly on the LQR answer, and a test checks it. On a non-linear one, the
    point iLQR converges to is one where the total cost has zero gradient with
    respect to the controls, and a test checks that too.
    """
    # TODO: ...
    raise NotImplementedError
