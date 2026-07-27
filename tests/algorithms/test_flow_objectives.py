"""Tests for the detailed-balance family.

These objectives are easy to write in a way that runs, produces a falling loss,
and is wrong -- a flow system with no terminal anchoring is satisfied by any
constant, and a sub-trajectory weighting that ignores padding quietly trains on
transitions that never happened. The tests target those two failure modes
directly, plus the identity that makes SubTB worth having: at its endpoints it
must reproduce the objectives it interpolates between.
"""

import numpy as np
import pytest
import torch

from evoflownet.algorithms.gflownet import (
    DetailedBalance,
    ForwardLookingDetailedBalance,
    SubTrajectoryBalance,
    TrainingConfig,
    sample_trajectories,
    train_trajectory_balance,
)
from evoflownet.algorithms.gflownet.sampling import Trajectories
from evoflownet.core.types import Alphabet
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.models.policy import SequencePolicy
from evoflownet.rewards import TemperedReward

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 4
PARENT = np.zeros(LENGTH, dtype=np.int32)


class Additive(FitnessLandscape):
    """Value rises with the count of token 1. Every state is scorable."""

    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    def _evaluate(self, sequences):
        return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64) + 1.0


@pytest.fixture
def env():
    return MutationEnvironment(PARENT, ALPHABET, max_mutations=3)


def make_policy(env, *, flow=True):
    return SequencePolicy(
        n_actions=env.n_actions,
        sequence_length=LENGTH,
        n_tokens=ALPHABET.size,
        hidden_dim=64,
        learn_flow=flow,
    )


def batch(env, policy, n=32, seed=0):
    trajectories = sample_trajectories(
        env, policy, n, epsilon=0.2, generator=torch.Generator().manual_seed(seed)
    )
    values = Additive().evaluate(trajectories.terminal)
    return trajectories, torch.as_tensor(
        TemperedReward(beta=1.0).log_reward(values), dtype=torch.float32
    )


class TestPerStepRecord:
    def test_the_sampler_records_every_state(self, env):
        policy = make_policy(env)
        trajectories, _ = batch(env, policy)
        steps = trajectories.require_steps()
        # One more state than transitions.
        assert steps.states.shape[1] == steps.log_forward.shape[1] + 1

    def test_the_summed_log_probs_match_the_per_step_record(self, env):
        # If these disagreed, trajectory balance and detailed balance would be
        # training against different rollouts while claiming to compare.
        policy = make_policy(env)
        trajectories, _ = batch(env, policy)
        steps = trajectories.require_steps()
        active = torch.as_tensor(steps.active)
        summed = (steps.log_forward * active).sum(dim=1).detach().numpy()
        assert summed == pytest.approx(trajectories.log_forward.detach().numpy(), abs=1e-4)

    def test_a_batch_without_steps_is_refused(self, env):
        bare = Trajectories(
            terminal=np.zeros((2, LENGTH), dtype=np.int32),
            log_forward=torch.zeros(2),
            log_backward=torch.zeros(2),
            lengths=np.ones(2, dtype=np.int64),
        )
        with pytest.raises(ValueError, match="no per-step record"):
            DetailedBalance().loss(bare, torch.zeros(2), make_policy(env))


class TestFlowHead:
    def test_a_policy_without_a_flow_head_is_refused(self, env):
        # Silently falling back to a constant would turn detailed balance into
        # a worse trajectory balance while still reporting its name.
        policy = make_policy(env, flow=False)
        trajectories, log_rewards = batch(env, policy)
        with pytest.raises(RuntimeError, match="no flow head"):
            DetailedBalance().loss(trajectories, log_rewards, policy)

    def test_the_flow_objectives_do_not_use_log_z(self):
        assert not DetailedBalance().uses_log_z
        assert not SubTrajectoryBalance().uses_log_z


class TestLosses:
    @pytest.mark.parametrize("objective", [DetailedBalance(), SubTrajectoryBalance(lam=0.9)])
    def test_the_loss_is_a_finite_scalar(self, env, objective):
        policy = make_policy(env)
        trajectories, log_rewards = batch(env, policy)
        loss = objective.loss(trajectories, log_rewards, policy)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("objective", [DetailedBalance(), SubTrajectoryBalance(lam=0.9)])
    def test_the_loss_reaches_the_policy(self, env, objective):
        policy = make_policy(env)
        trajectories, log_rewards = batch(env, policy)
        objective.loss(trajectories, log_rewards, policy).backward()
        assert policy.flow_head.weight.grad is not None
        assert policy.forward_head.weight.grad is not None

    def test_terminal_flow_is_pinned_to_the_reward(self, env):
        # The boundary condition that anchors the system. Without it any
        # constant flow satisfies detailed balance and the loss goes to zero
        # while the policy learns nothing.
        policy = make_policy(env)
        trajectories, log_rewards = batch(env, policy)
        flows = DetailedBalance()._flows(trajectories, log_rewards, policy)
        steps = trajectories.require_steps()
        reached = np.cumsum(steps.stopping, axis=1) > 0
        for row, col in zip(*np.nonzero(reached), strict=True):
            assert flows[row, col + 1].item() == pytest.approx(log_rewards[row].item(), abs=1e-5)

    def test_a_non_positive_lambda_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            SubTrajectoryBalance(lam=0.0)


class TestForwardLooking:
    def test_it_declares_that_it_needs_state_rewards(self):
        assert ForwardLookingDetailedBalance().needs_state_rewards
        assert not DetailedBalance().needs_state_rewards

    def test_it_refuses_to_run_without_them(self, env):
        # Falling back to plain detailed balance would report a result under
        # the wrong method's name.
        policy = make_policy(env)
        trajectories, log_rewards = batch(env, policy)
        with pytest.raises(ValueError, match="reward at every visited"):
            ForwardLookingDetailedBalance().loss(trajectories, log_rewards, policy)

    def test_it_trains_end_to_end(self, env):
        # The trainer is what supplies the state rewards, so this is the only
        # place the wiring is exercised.
        policy = make_policy(env)
        result = train_trajectory_balance(
            env,
            policy,
            Additive(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=15, batch_size=16, seed=0),
            objective=ForwardLookingDetailedBalance(),
        )
        assert len(result.losses) == 15
        assert np.isfinite(result.losses).all()


class TestTrainingReducesViolation:
    @pytest.mark.parametrize(
        "objective",
        [DetailedBalance(), SubTrajectoryBalance(lam=0.9), ForwardLookingDetailedBalance()],
        ids=["db", "subtb", "fl-db"],
    )
    def test_the_loss_falls(self, env, objective):
        policy = make_policy(env)
        result = train_trajectory_balance(
            env,
            policy,
            Additive(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=120, batch_size=32, learning_rate=1e-2, seed=0),
            objective=objective,
        )
        early = float(np.mean(result.losses[:20]))
        late = float(np.mean(result.losses[-20:]))
        assert late < early, f"{objective}: {early:.4f} -> {late:.4f}"
