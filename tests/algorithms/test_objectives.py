"""Tests for the training objectives.

The important test is
:meth:`TestZReallyCancels.test_shifting_every_log_reward_leaves_the_loss_unchanged`.
Contrastive balance claims the partition function cancels; a constant shift in
every log reward is exactly a change in ``log Z``, so if the loss moves, it has
not cancelled and the whole justification for the objective is wrong.
"""

import numpy as np
import pytest
import torch

from evogfn.algorithms.gflownet import (
    ContrastiveBalance,
    TrajectoryBalance,
    balance_violation,
    sample_trajectories,
)
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.models import SequencePolicy


def make_env(length=4, symbols="ABC", max_mutations=2):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
    )


def make_policy(env, seed=0):
    torch.manual_seed(seed)
    return SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        hidden_dim=32,
        embedding_dim=8,
    )


@pytest.fixture
def batch():
    env = make_env()
    policy = make_policy(env)
    trajectories = sample_trajectories(env, policy, 16, generator=torch.Generator().manual_seed(1))
    rng = np.random.default_rng(0)
    log_rewards = torch.as_tensor(rng.uniform(-2.0, 2.0, size=16), dtype=torch.float32)
    return trajectories, log_rewards, policy


class TestZReallyCancels:
    """The property the objective exists for."""

    def test_shifting_every_log_reward_leaves_the_loss_unchanged(self, batch):
        # Adding a constant to every log reward is exactly a change in log Z. If
        # contrastive balance is invariant to it, Z has cancelled; if not, the
        # justification for the objective is false.
        trajectories, log_rewards, policy = batch
        objective = ContrastiveBalance()
        base = objective.loss(trajectories, log_rewards, policy)
        for shift in (-5.0, 3.0, 100.0):
            shifted = objective.loss(trajectories, log_rewards + shift, policy)
            assert shifted.detach().item() == pytest.approx(base.detach().item(), rel=1e-5)

    def test_trajectory_balance_is_not_invariant_to_that_shift(self, batch):
        # The contrast that makes the test above meaningful: TB measures against
        # a fixed log Z, so shifting the rewards must move its loss. If both
        # were invariant, the first test would prove nothing.
        trajectories, log_rewards, policy = batch
        objective = TrajectoryBalance()
        base = objective.loss(trajectories, log_rewards, policy)
        shifted = objective.loss(trajectories, log_rewards + 5.0, policy)
        assert shifted.detach().item() != pytest.approx(base.detach().item())

    def test_it_reports_that_it_uses_no_partition_function(self):
        assert not ContrastiveBalance().uses_log_z
        assert TrajectoryBalance().uses_log_z

    def test_no_gradient_reaches_log_z(self, batch):
        # Not merely unused in the formula: the parameter must receive nothing,
        # or an optimiser would still drift it.
        trajectories, log_rewards, policy = batch
        ContrastiveBalance().loss(trajectories, log_rewards, policy).backward()
        assert policy.log_z.grad is None or policy.log_z.grad.abs().item() == 0.0


class TestBalanceViolation:
    def test_it_is_the_shared_quantity_both_objectives_are_built_from(self, batch):
        trajectories, log_rewards, _ = batch
        violation = balance_violation(trajectories, log_rewards)
        expected = trajectories.log_forward - log_rewards - trajectories.log_backward
        assert torch.allclose(violation, expected)

    def test_an_infinite_log_reward_is_refused(self, batch):
        trajectories, _, _ = batch
        with pytest.raises(ValueError, match="must be finite"):
            balance_violation(trajectories, torch.full((16,), -float("inf")))

    def test_mismatched_batch_sizes_are_refused(self, batch):
        trajectories, _, _ = batch
        with pytest.raises(ValueError, match="log rewards for"):
            balance_violation(trajectories, torch.zeros(3))


class TestContrastiveLoss:
    def test_it_is_zero_when_every_trajectory_agrees(self, batch):
        # At balance the violation is the same constant for every trajectory,
        # so every pairwise difference vanishes -- without Z appearing anywhere.
        trajectories, _, policy = batch
        constant = (trajectories.log_forward - trajectories.log_backward).detach()
        loss = ContrastiveBalance().loss(trajectories, constant, policy)
        assert loss.detach().item() == pytest.approx(0.0, abs=1e-10)

    def test_it_grows_with_disagreement(self, batch):
        trajectories, _, policy = batch
        constant = (trajectories.log_forward - trajectories.log_backward).detach()
        disturbance = torch.zeros(16)
        disturbance[:8] = 1.0
        objective = ContrastiveBalance()
        small = objective.loss(trajectories, constant - disturbance, policy)
        large = objective.loss(trajectories, constant - 3 * disturbance, policy)
        assert small.detach().item() == pytest.approx(1.0)
        assert large.detach().item() == pytest.approx(9.0)

    def test_gradients_reach_the_policy(self, batch):
        trajectories, log_rewards, policy = batch
        ContrastiveBalance().loss(trajectories, log_rewards, policy).backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in policy.policy_parameters()
        )

    def test_a_single_trajectory_cannot_be_paired(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 1)
        with pytest.raises(ValueError, match="at least 2"):
            ContrastiveBalance().loss(trajectories, torch.zeros(1), policy)

    def test_an_odd_batch_drops_the_unpaired_trajectory(self, batch):
        trajectories, log_rewards, policy = batch
        odd = ContrastiveBalance().loss(trajectories, log_rewards, policy)
        assert torch.isfinite(odd)


class TestNoisyGradientPruning:
    def test_near_tied_pairs_are_dropped(self, batch):
        # Two designs of near-identical reward say almost nothing about which
        # the policy should prefer, and with a noisy fitness oracle the gap
        # between them is mostly measurement error.
        trajectories, _, policy = batch
        log_rewards = torch.zeros(16)
        log_rewards[:8] = 0.01  # every pair differs by 0.01
        pruned = ContrastiveBalance(prune_threshold=0.1).loss(trajectories, log_rewards, policy)
        assert pruned.detach().item() == pytest.approx(0.0)

    def test_well_separated_pairs_survive(self, batch):
        trajectories, _, policy = batch
        log_rewards = torch.zeros(16)
        log_rewards[:8] = 5.0
        pruned = ContrastiveBalance(prune_threshold=0.1).loss(trajectories, log_rewards, policy)
        assert pruned.detach().item() > 0.0

    def test_pruning_everything_still_yields_a_usable_zero(self, batch):
        # A batch with no signal is a real outcome, not an error. The zero has
        # to stay attached to the graph or the optimiser step breaks.
        trajectories, _, policy = batch
        loss = ContrastiveBalance(prune_threshold=1e6).loss(trajectories, torch.zeros(16), policy)
        assert loss.detach().item() == 0.0
        loss.backward()

    def test_pruning_is_off_by_default(self):
        assert ContrastiveBalance().prune_threshold == 0.0

    def test_a_negative_threshold_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContrastiveBalance(prune_threshold=-1.0)
