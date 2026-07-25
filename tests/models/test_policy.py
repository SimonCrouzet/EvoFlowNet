"""Tests for the policy network.

These check the properties trajectory balance depends on -- that masked actions
carry no probability, that the distributions normalise, that a stopped
trajectory produces no nan -- rather than anything about the architecture.
"""

import numpy as np
import pytest
import torch

from evoflownet.core import Alphabet
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.models import SequencePolicy, to_tensor


@pytest.fixture
def env():
    return MutationEnvironment(
        np.zeros(4, dtype=np.int32), Alphabet.from_string("ABC"), max_mutations=3
    )


@pytest.fixture
def policy(env):
    torch.manual_seed(0)
    return SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        hidden_dim=32,
        embedding_dim=8,
    )


def masked_log_probs(policy, env, state):
    sequences = to_tensor(state.sequences)
    forward = torch.as_tensor(env.forward_mask(state))
    backward = torch.as_tensor(env.backward_mask(state))
    return policy.log_probs(sequences, forward, backward)


class TestShapes:
    def test_log_probs_cover_the_action_space(self, policy, env):
        forward, backward = masked_log_probs(policy, env, env.initial(5))
        assert forward.shape == (5, env.n_actions)
        assert backward.shape == (5, env.n_actions)

    def test_the_representation_is_shared_between_heads(self, policy, env):
        # Malkin et al. share everything but the final layer; this checks the
        # trunk is actually reached rather than the heads reading raw input.
        hidden = policy(to_tensor(env.initial(3).sequences))
        assert hidden.shape == (3, 32)


class TestMasking:
    def test_masked_actions_have_zero_probability(self, policy, env):
        state = env.initial(4)
        mask = torch.as_tensor(env.forward_mask(state))
        forward, _ = masked_log_probs(policy, env, state)
        assert torch.isneginf(forward[~mask]).all()

    def test_permitted_actions_normalise_to_one(self, policy, env):
        forward, _ = masked_log_probs(policy, env, env.initial(4))
        total = forward.exp().sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_masking_cannot_be_bypassed(self, policy, env):
        # log_probs is the only way to get probabilities out, and it requires
        # the masks. A caller who forgot would otherwise get a policy placing
        # mass on edges that do not exist, with no crash to reveal it.
        assert not hasattr(policy, "action_log_probs")
        with pytest.raises(TypeError):
            policy.log_probs(to_tensor(env.initial(2).sequences))

    def test_a_stopped_trajectory_yields_no_nan(self, policy, env):
        # Its mask is empty in both directions. Naive log-softmax over an
        # all-masked row gives nan, which would contaminate the whole batch
        # through the loss.
        stopped = env.step(env.initial(1), np.array([env.stop_action]))
        forward, backward = masked_log_probs(policy, env, stopped)
        assert not torch.isnan(forward).any()
        assert not torch.isnan(backward).any()
        assert torch.isneginf(forward).all()

    def test_a_mask_of_the_wrong_width_is_refused(self, policy, env):
        sequences = to_tensor(env.initial(2).sequences)
        wrong = torch.ones((2, 3), dtype=torch.bool)
        with pytest.raises(ValueError, match="must have shape"):
            policy.log_probs(sequences, wrong, wrong)

    def test_a_non_boolean_mask_is_refused(self, policy, env):
        sequences = to_tensor(env.initial(2).sequences)
        numeric = torch.ones((2, env.n_actions))
        with pytest.raises(ValueError, match="must be boolean"):
            policy.log_probs(sequences, numeric, numeric)


class TestBackwardPolicy:
    def test_uniform_backward_is_one_over_the_parent_count(self, policy, env):
        # On the subset lattice a state with k mutations has exactly k parents,
        # so P_B is 1/k in closed form with no model involved.
        state = env.initial(1)
        for expected_parents in (1, 2, 3):
            state = env.step(state, np.array([(expected_parents - 1) * 3 + 1]))
            _, backward = masked_log_probs(policy, env, state)
            permitted = backward[torch.isfinite(backward)]
            assert permitted.numel() == expected_parents
            assert torch.allclose(
                permitted.exp(),
                torch.full_like(permitted, 1.0 / expected_parents),
                atol=1e-6,
            )

    def test_the_uniform_backward_policy_has_no_parameters(self, policy):
        assert not policy.learns_backward
        assert policy.backward_head is None

    def test_a_learned_backward_policy_still_normalises(self, env):
        torch.manual_seed(1)
        learned = SequencePolicy(
            n_tokens=env.alphabet.size,
            sequence_length=env.sequence_length,
            n_actions=env.n_actions,
            hidden_dim=32,
            embedding_dim=8,
            learn_backward=True,
        )
        state = env.step(env.initial(2), np.array([1, 4]))
        _, backward = masked_log_probs(learned, env, state)
        total = backward.exp().sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_learning_the_backward_policy_adds_parameters(self, env):
        shared = {
            "n_tokens": env.alphabet.size,
            "sequence_length": env.sequence_length,
            "n_actions": env.n_actions,
            "hidden_dim": 32,
            "embedding_dim": 8,
        }
        uniform = SequencePolicy(**shared)
        learned = SequencePolicy(**shared, learn_backward=True)
        assert len(learned.policy_parameters()) > len(uniform.policy_parameters())


class TestLogZ:
    def test_it_starts_at_zero(self, policy):
        # Z = 1. It is expected to travel a long way from here, which is why it
        # gets its own learning rate.
        assert policy.log_z.item() == pytest.approx(0.0)

    def test_it_is_a_scalar_not_a_head(self, policy):
        # log Z is the total flow through the DAG and does not depend on state.
        assert policy.log_z.shape == ()

    def test_it_is_excluded_from_the_policy_parameter_group(self, policy):
        # So the optimiser can give it a higher learning rate, per Malkin et al.
        names = {id(p) for p in policy.policy_parameters()}
        assert id(policy.log_z) not in names
        assert len(names) + 1 == len(list(policy.parameters()))

    def test_it_is_trainable(self, policy):
        assert policy.log_z.requires_grad


class TestGradients:
    def test_the_loss_reaches_both_the_policy_and_log_z(self, policy, env):
        forward, _ = masked_log_probs(policy, env, env.initial(3))
        surrogate = forward[torch.isfinite(forward)].sum() + policy.log_z
        surrogate.backward()
        assert policy.log_z.grad is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in policy.policy_parameters()
        )

    def test_masked_entries_do_not_produce_nan_gradients(self, policy, env):
        # -inf entries in the output must not propagate nan backwards when the
        # loss touches only the finite ones.
        forward, _ = masked_log_probs(policy, env, env.initial(3))
        forward[torch.isfinite(forward)].sum().backward()
        for parameter in policy.policy_parameters():
            if parameter.grad is not None:
                assert not torch.isnan(parameter.grad).any()


class TestValidation:
    @pytest.mark.parametrize(
        "field", ["n_tokens", "sequence_length", "n_actions", "hidden_dim", "n_layers"]
    )
    def test_degenerate_sizes_are_refused(self, field):
        sizes = {
            "n_tokens": 4,
            "sequence_length": 4,
            "n_actions": 17,
            "hidden_dim": 8,
            "n_layers": 1,
        }
        with pytest.raises(ValueError, match="at least 1"):
            SequencePolicy(**{**sizes, field: 0})  # type: ignore[arg-type]


class TestDeterminism:
    def test_the_same_seed_gives_the_same_initial_policy(self, env):
        sizes = {
            "n_tokens": env.alphabet.size,
            "sequence_length": env.sequence_length,
            "n_actions": env.n_actions,
            "hidden_dim": 16,
            "embedding_dim": 4,
        }
        torch.manual_seed(7)
        first = SequencePolicy(**sizes)
        torch.manual_seed(7)
        second = SequencePolicy(**sizes)
        state = env.initial(3)
        a, _ = masked_log_probs(first, env, state)
        b, _ = masked_log_probs(second, env, state)
        assert torch.equal(a, b)
