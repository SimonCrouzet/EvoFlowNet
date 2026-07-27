"""Tests for scoring externally supplied sequences.

This is what makes off-policy training possible -- a genetic algorithm, a replay
buffer or an existing assay produces sequences, not trajectories, and the loss
needs a path.
"""

import math

import numpy as np
import pytest
import torch

from evoflownet.algorithms.gflownet import (
    replay_trajectories,
    sample_trajectories,
    trajectory_balance_loss,
)
from evoflownet.core import Alphabet
from evoflownet.env.base import State
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.models import SequencePolicy


def make_env(length=5, symbols="ABC", max_mutations=3, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
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


class TestRoundTrip:
    def test_replaying_a_sampled_terminal_returns_that_terminal(self):
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 8, generator=torch.Generator().manual_seed(1))
        replayed = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(2)
        )
        assert np.array_equal(replayed.terminal, sampled.terminal)

    def test_backward_log_probability_is_exactly_minus_log_factorial(self):
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 8, generator=torch.Generator().manual_seed(1))
        replayed = replay_trajectories(env, policy, sampled.terminal)
        counts = (sampled.terminal != env.parent[None, :]).sum(axis=1)
        expected = np.array([-math.log(math.factorial(int(k))) for k in counts])
        assert replayed.log_backward.detach().numpy() == pytest.approx(expected, abs=1e-5)

    def test_backward_probability_does_not_depend_on_the_ordering_drawn(self):
        # log P_B(tau|x) = -log k! for every ordering, which is what makes the
        # off-policy estimator well behaved. log P_F does vary by ordering.
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 6, generator=torch.Generator().manual_seed(3))
        first = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(10)
        )
        second = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(11)
        )
        assert torch.allclose(first.log_backward, second.log_backward, atol=1e-6)

    def test_trajectory_length_is_the_mutation_count_plus_the_stop(self):
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 8, generator=torch.Generator().manual_seed(1))
        replayed = replay_trajectories(env, policy, sampled.terminal)
        counts = (sampled.terminal != env.parent[None, :]).sum(axis=1)
        assert replayed.lengths.tolist() == (counts + 1).tolist()


class TestStoppedStatesHaveOneParent:
    """Regression tests for a bug that trajectory balance could not see.

    The backward mask used to mark a stopped state's mutations *and* its stop
    action, giving a terminal ``k + 1`` parents instead of one. Sampling never
    exercised it, because the sampler discards the stop action's ``P_B`` term --
    but it admits paths that undo a mutation while stopped, which is not an edge
    of this graph, and it makes uniform ``P_B`` wrong at every terminal.
    """

    def test_a_stopped_state_has_exactly_one_parent(self):
        env = make_env()
        state = env.initial(1)
        state = env.step(state, np.array([0 * 3 + 1]))
        state = env.step(state, np.array([1 * 3 + 2]))
        stopped = env.step(state, np.array([env.stop_action]))
        mask = env.backward_mask(stopped)
        assert mask.sum() == 1
        assert mask[0, env.stop_action]

    def test_the_same_state_unstopped_has_its_mutations_as_parents(self):
        env = make_env()
        state = env.initial(1)
        state = env.step(state, np.array([0 * 3 + 1]))
        state = env.step(state, np.array([1 * 3 + 2]))
        mask = env.backward_mask(state)
        assert mask.sum() == 2
        assert not mask[0, env.stop_action]

    def test_a_mixed_batch_is_handled_per_row(self):
        env = make_env()
        state = env.step(env.initial(2), np.array([1, 1]))
        state = env.step(state, np.array([env.stop_action, 4]))
        mask = env.backward_mask(state)
        assert mask[0].sum() == 1
        assert mask[0, env.stop_action]
        assert mask[1].sum() == 2
        assert not mask[1, env.stop_action]


class TestUnderFeasibilityMasking:
    """Where drawing a path backward matters rather than permuting uniformly.

    With a transition constraint, some orderings of a mutation set pass through
    states the environment forbids. A uniform permutation would propose paths
    that do not exist; walking backward through the backward mask can only
    traverse real edges.
    """

    def transitions(self, vocab, forbidden):
        matrix = np.ones((vocab, vocab), dtype=np.float64)
        for a, b in forbidden:
            matrix[a, b] = 0.0
        return matrix

    def test_replayed_paths_are_legal_under_a_transition_constraint(self):
        matrix = self.transitions(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        env = make_env(length=6, symbols="ABCD", max_mutations=3, transitions=matrix)
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 16, generator=torch.Generator().manual_seed(4))
        # Would raise from env.step if any replayed action were masked.
        replayed = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(5)
        )
        assert np.array_equal(replayed.terminal, sampled.terminal)


class TestUsableForTraining:
    def test_gradients_flow_through_a_replayed_trajectory(self):
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 8, generator=torch.Generator().manual_seed(1))
        replayed = replay_trajectories(env, policy, sampled.terminal)
        loss = trajectory_balance_loss(replayed, torch.zeros(8), policy.log_z)
        loss.backward()
        assert policy.log_z.grad is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in policy.policy_parameters()
        )

    def test_the_parent_itself_can_be_replayed(self):
        # Zero mutations, one stop action. The degenerate path.
        env = make_env()
        policy = make_policy(env)
        replayed = replay_trajectories(env, policy, env.parent[None, :])
        assert replayed.lengths.tolist() == [1]
        assert replayed.log_backward.detach().item() == pytest.approx(0.0)


class TestValidation:
    def test_an_empty_batch_is_allowed(self):
        env = make_env()
        policy = make_policy(env)
        replayed = replay_trajectories(env, policy, np.zeros((0, 5), dtype=np.int32))
        assert len(replayed) == 0

    def test_the_wrong_sequence_width_is_refused(self):
        env = make_env()
        policy = make_policy(env)
        with pytest.raises(ValueError, match="expected shape"):
            replay_trajectories(env, policy, np.zeros((2, 9), dtype=np.int32))

    def test_a_sequence_beyond_the_mutation_budget_is_refused(self):
        # It is not in the space the policy is defined over, so training on it
        # would be meaningless rather than merely inaccurate.
        env = make_env(length=5, symbols="ABC", max_mutations=1)
        policy = make_policy(env)
        far = np.array([[1, 2, 1, 2, 1]], dtype=np.int32)
        with pytest.raises(ValueError, match="not constructible"):
            replay_trajectories(env, policy, far)


class TestAgreementWithSampling:
    def test_a_replayed_path_is_scored_as_a_sampled_one_would_be(self):
        # Replay draws its own ordering, so log P_F need not match the sampled
        # trajectory. What must hold is that it is a valid log-probability of a
        # real path.
        env = make_env()
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 16, generator=torch.Generator().manual_seed(1))
        replayed = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(6)
        )
        assert torch.isfinite(replayed.log_forward).all()
        assert (replayed.log_forward < 0).all()

    def test_replaying_a_single_ordering_environment_reproduces_log_pf_exactly(self):
        # With at most one mutation there is only one ordering, so replay must
        # agree with sampling exactly. Any disagreement here is a scoring bug
        # rather than an ordering difference.
        env = make_env(length=4, symbols="ABC", max_mutations=1)
        policy = make_policy(env)
        sampled = sample_trajectories(env, policy, 12, generator=torch.Generator().manual_seed(2))
        replayed = replay_trajectories(env, policy, sampled.terminal)
        counts = (sampled.terminal != env.parent[None, :]).sum(axis=1)
        single = counts <= 1
        assert single.all()
        assert torch.allclose(
            replayed.log_forward[torch.as_tensor(single)],
            sampled.log_forward[torch.as_tensor(single)],
            atol=1e-5,
        )


def terminal_state(sequences):
    return State(sequences=sequences, stopped=np.ones(sequences.shape[0], dtype=bool))
