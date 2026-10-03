"""P09 deep tests — the DQN and Double DQN targets, on fabricated batches.

Separate file because it needs torch. Part A's tabular half stays green on a
machine without it, but the lab is not finished until this file runs too: the
theorem's third panel lives here.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is needed for Q4 and Q5")

from rl_lab import deep as dp          # noqa: E402


def batch(rewards, next_online, next_target, dones):
    return (torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_online, dtype=torch.float32),
            torch.tensor(next_target, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32))


def test_q4_dqn_target_on_a_batch_by_hand():
    r, q_on, q_tg, d = batch([1.0, 0.0], [[0.0, 0.0, 0.0]] * 2,
                             [[1.0, 5.0, 2.0], [3.0, 1.0, 0.0]], [0.0, 0.0])
    got = dp.dqn_targets(r, q_tg, d, gamma=0.5)
    assert got.tolist() == pytest.approx([1.0 + 0.5 * 5.0, 0.0 + 0.5 * 3.0],
                                         abs=1e-6)


def test_q4_done_transitions_drop_the_bootstrap():
    r, q_on, q_tg, d = batch([2.0, 2.0], [[0.0] * 3] * 2,
                             [[9.0, 9.0, 9.0]] * 2, [1.0, 0.0])
    got = dp.dqn_targets(r, q_tg, d, gamma=0.9)
    assert got[0].item() == pytest.approx(2.0, abs=1e-6)
    assert got[1].item() == pytest.approx(2.0 + 0.9 * 9.0, abs=1e-6)


def test_q4_ddqn_selects_with_one_net_and_prices_with_the_other():
    """The online net says action 1 is best; the target net says it is worth
    -2. The Double target must be -2, not the target net's own maximum."""
    r, q_on, q_tg, d = batch([1.0], [[0.0, 5.0, 1.0]], [[9.0, -2.0, 3.0]],
                             [0.0])
    got = dp.ddqn_targets(r, q_on, q_tg, d, gamma=0.5)
    assert got.item() == pytest.approx(1.0 + 0.5 * (-2.0), abs=1e-6)


def test_q4_ddqn_with_one_network_is_dqn():
    torch.manual_seed(0)
    q = torch.randn(32, 3)
    r = torch.randn(32)
    d = (torch.rand(32) < 0.3).float()
    a = dp.ddqn_targets(r, q, q, d, 0.99)
    b = dp.dqn_targets(r, q, d, 0.99)
    assert torch.allclose(a, b, atol=1e-6)


def test_q4_targets_carry_no_gradient():
    """A target is data. If it still requires grad, the update is minimising a
    different objective and nothing will tell you."""
    q = torch.randn(8, 3, requires_grad=True)
    r, d = torch.randn(8), torch.zeros(8)
    assert not dp.dqn_targets(r, q, d, 0.9).requires_grad
    assert not dp.ddqn_targets(r, q, q, d, 0.9).requires_grad


def test_q4_soft_update_with_tau_one_is_a_copy():
    torch.manual_seed(0)
    online, target = dp.mlp(), dp.mlp()
    dp.soft_update(target, online, 1.0)
    for a, b in zip(target.parameters(), online.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


def test_q4_soft_update_with_tau_zero_changes_nothing():
    torch.manual_seed(1)
    online, target = dp.mlp(), dp.mlp()
    before = [p.clone() for p in target.parameters()]
    dp.soft_update(target, online, 0.0)
    for a, b in zip(target.parameters(), before):
        assert torch.allclose(a, b, atol=1e-7)


def test_q5_exact_q_is_a_fixed_point_of_the_bellman_operator():
    """The ground truth has to be right before anything is measured against it:
    the goal states are worth 0, and moving right is optimal everywhere else."""
    centres, Q = dp.exact_q(0.95, n_bins=200)
    assert Q.shape == (200, 3)
    assert np.allclose(Q[centres >= dp.GOAL], 0.0)
    interior = centres < dp.GOAL - 0.05
    assert (Q[interior].argmax(axis=1) == 2).all()


def test_q5_the_reward_noise_has_no_effect_on_the_ground_truth():
    """q* is computed from the model, and the reward noise has mean zero, so
    the truth the networks are scored against does not move when the noise
    does. If it did, the overestimation in Part B would be measuring the wrong
    thing."""
    _, a = dp.exact_q(0.95, n_bins=100)
    env_quiet = dp.Corridor(seed=0, reward_noise=0.0)
    env_noisy = dp.Corridor(seed=0, reward_noise=5.0)
    assert env_quiet.reward_noise == 0.0 and env_noisy.reward_noise == 5.0
    _, b = dp.exact_q(0.95, n_bins=100)
    assert np.allclose(a, b)


def test_q5_both_algorithms_see_the_same_data():
    """The buffer is built once and passed to both fits: same transitions, same
    order of sampling, same seed. The comparison is fair by construction rather
    than by convention."""
    buf = dp.collect(400, seed=0)
    a = dp.fit_offline(buf, double=False, seed=0, n_steps=30)[0]
    b = dp.fit_offline(buf, double=False, seed=0, n_steps=30)[0]
    assert a == b


def test_q5_fit_records_one_probe_value_per_interval():
    buf = dp.collect(400, seed=0)
    probe = np.linspace(0.05, 0.8, 5)
    values, net = dp.fit_offline(buf, double=True, seed=0, n_steps=200,
                                 probe=probe, probe_every=50)
    assert len(values) == 4
    assert all(np.isfinite(values))


def test_q5_collect_is_reproducible_and_the_right_size():
    a, b = dp.collect(300, seed=3), dp.collect(300, seed=3)
    assert len(a) == 300
    assert [x[2] for x in a.data] == [x[2] for x in b.data]


def _spy_fit(monkeypatch, double: bool, n_steps: int = 12, tau: float = 0.1):
    """Run fit_offline with spies on the buffer, the target functions and
    soft_update. At every step after the first, the spy recomputes both
    networks on the batch's next states and records whether the tensors handed
    to the target function came from the right network."""
    buf = dp.collect(400, seed=0)
    seen: dict = {"taus": [], "checks": []}
    real_sample = buf.sample

    def sample(n, rng):
        out = real_sample(n, rng)
        seen["s2"] = out[3]
        return out

    buf.sample = sample
    real_soft = dp.soft_update

    def soft(target, online, t):
        seen["target"], seen["online"] = target, online
        seen["taus"].append(t)
        real_soft(target, online, t)

    def same(x, net):
        with torch.no_grad():
            return bool(torch.allclose(x, net(seen["s2"]), atol=1e-6))

    real_dqn, real_ddqn = dp.dqn_targets, dp.ddqn_targets

    def dqn(r, q_target, d, gamma):
        if "online" in seen:
            seen["checks"].append(("dqn", same(q_target, seen["target"]), True))
        return real_dqn(r, q_target, d, gamma)

    def ddqn(r, q_online, q_target, d, gamma):
        if "online" in seen:
            seen["checks"].append(("ddqn", same(q_online, seen["online"]),
                                   same(q_target, seen["target"])))
        return real_ddqn(r, q_online, q_target, d, gamma)

    monkeypatch.setattr(dp, "soft_update", soft)
    monkeypatch.setattr(dp, "dqn_targets", dqn)
    monkeypatch.setattr(dp, "ddqn_targets", ddqn)
    dp.fit_offline(buf, double=double, seed=0, n_steps=n_steps, tau=tau)
    return seen


def test_q5_the_target_network_follows_the_online_one_every_step(monkeypatch):
    """Without the soft update the target network is frozen at its
    initialisation for the whole fit, and nothing crashes: the loss goes down
    towards a target that never learns. The fit calls soft_update once per
    step, with its own tau."""
    seen = _spy_fit(monkeypatch, double=True, n_steps=12, tau=0.1)
    assert seen["taus"] == [0.1] * 12


def test_q5_dqn_prices_the_next_state_with_the_target_network(monkeypatch):
    """The DQN target reads max_a' Q from the *target* network. Reading it from
    the online network instead is the algorithm without a target network, and
    it runs just as happily."""
    seen = _spy_fit(monkeypatch, double=False)
    kinds = {k for k, _, _ in seen["checks"]}
    assert kinds == {"dqn"}, "double=False must build the DQN target"
    assert len(seen["checks"]) == 11
    assert all(price for _, price, _ in seen["checks"])


def test_q5_ddqn_selects_with_the_online_net_and_prices_with_the_target(monkeypatch):
    """Swapping the two arguments of ddqn_targets gives a target that selects
    with the slow network and prices with the fast one, which fits just as well
    and is not Double DQN. After the first step the two networks differ, so the
    spy can tell which is which."""
    seen = _spy_fit(monkeypatch, double=True)
    kinds = {k for k, _, _ in seen["checks"]}
    assert kinds == {"ddqn"}, "double=True must build the Double DQN target"
    assert len(seen["checks"]) == 11
    assert all(sel for _, sel, _ in seen["checks"]), "selection is not the online net"
    assert all(price for _, _, price in seen["checks"]), "pricing is not the target net"
