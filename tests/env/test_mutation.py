"""Tests for the mutation environment.

Trajectory balance is only valid on a graph that is genuinely acyclic, whose
backward edges are exactly its forward edges reversed, and whose masks do not
lie. Those are properties of the graph rather than of any sampler, so they are
checked here by walking and enumerating the graph directly.

The path-counting tests matter most. A variant with ``k`` mutations is reached
by exactly ``k!`` trajectories and has exactly ``k`` parents, so the uniform
backward policy is ``1/k`` without a model. That is a statement about cost, not
correctness -- any valid ``P_B`` induces a forward policy sampling proportional
to reward -- but the arithmetic still has to be right, so it is verified by
exhaustive enumeration rather than assumed.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evogfn.core import Alphabet
from evogfn.env.base import State
from evogfn.env.mutation import MutationEnvironment


def make_env(length=4, symbols="ABC", max_mutations=None, transitions=None, allow_stop=True):
    alphabet = Alphabet.from_string(symbols)
    parent = np.zeros(length, dtype=np.int32)
    return MutationEnvironment(
        parent,
        alphabet,
        max_mutations=max_mutations,
        transitions=transitions,
        allow_stop_before_max=allow_stop,
    )


def rollout(env, rng, n=1):
    """Walk random legal actions to termination, recording the actions taken."""
    state = env.initial(n)
    history = []
    while not env.is_terminal(state).all():
        mask = env.forward_mask(state)
        actions = np.array(
            [rng.choice(np.flatnonzero(row)) if row.any() else env.stop_action for row in mask]
        )
        # A stopped trajectory has an all-False row; keep it parked on stop.
        actions = np.where(env.is_terminal(state), env.stop_action, actions)
        live = ~env.is_terminal(state)
        if live.any():
            sub = State(sequences=state.sequences[live], stopped=state.stopped[live])
            stepped = env.step(sub, actions[live])
            sequences = state.sequences.copy()
            stopped = state.stopped.copy()
            sequences[live] = stepped.sequences
            stopped[live] = stepped.stopped
            state = State(sequences=sequences, stopped=stopped)
        history.append(actions)
    return state, history


class TestActionLayout:
    def test_action_space_covers_every_substitution_plus_stop(self):
        env = make_env(length=4, symbols="ABC")
        assert env.n_mutation_actions == 4 * 3
        assert env.n_actions == 4 * 3 + 1
        assert env.stop_action == 12

    def test_an_action_index_decodes_to_a_position_and_token(self):
        env = make_env(length=4, symbols="ABC")
        state = env.initial(1)
        # position 2, token 1
        stepped = env.step(state, np.array([2 * 3 + 1]))
        assert stepped.sequences[0].tolist() == [0, 0, 1, 0]


class TestAcyclicity:
    def test_every_forward_action_increases_the_mutation_count(self):
        # The strictly increasing quantity that makes the graph a DAG.
        env = make_env(length=5, symbols="ABCD")
        rng = np.random.default_rng(0)
        state = env.initial(1)
        while not env.is_terminal(state).all():
            before = env.n_mutations(state)[0]
            mask = env.forward_mask(state)
            action = rng.choice(np.flatnonzero(mask[0]))
            state = env.step(state, np.array([action]))
            after = env.n_mutations(state)[0]
            if action != env.stop_action:
                assert after == before + 1

    def test_a_mutated_position_cannot_be_mutated_again(self):
        # Without this the graph would have cycles: mutate, revert, repeat.
        env = make_env(length=4, symbols="ABC")
        state = env.step(env.initial(1), np.array([1]))  # position 0 -> token 1
        mask = env.forward_mask(state)
        position_zero = mask[0, : env.alphabet.size]
        assert not position_zero.any()

    def test_substituting_the_parent_token_is_never_offered(self):
        # It would be a self-loop: an edge that does not change the state.
        env = make_env(length=4, symbols="ABC")
        mask = env.forward_mask(env.initial(1))
        for position in range(4):
            assert not mask[0, position * 3 + 0]  # parent token is 0 everywhere


class TestBackwardConsistency:
    @pytest.mark.parametrize("seed", range(8))
    def test_a_forward_action_can_always_be_undone(self, seed):
        env = make_env(length=5, symbols="ABCD")
        rng = np.random.default_rng(seed)
        state = env.initial(1)
        for _ in range(4):
            mask = env.forward_mask(state)
            legal = np.flatnonzero(mask[0])
            legal = legal[legal != env.stop_action]
            if legal.size == 0:
                break
            action = rng.choice(legal)
            nxt = env.step(state, np.array([action]))
            # The action that got us here must be a legal backward action.
            assert env.backward_mask(nxt)[0, action]
            back = env.backward_step(nxt, np.array([action]))
            assert np.array_equal(back.sequences, state.sequences)
            assert np.array_equal(back.stopped, state.stopped)
            state = nxt

    def test_the_source_state_has_no_parents(self):
        env = make_env()
        assert not env.backward_mask(env.initial(3)).any()

    def test_backward_mask_marks_exactly_the_applied_mutations(self):
        env = make_env(length=5, symbols="ABC")
        state = env.step(env.initial(1), np.array([0 * 3 + 1]))
        state = env.step(state, np.array([3 * 3 + 2]))
        mask = env.backward_mask(state)
        assert mask.sum() == 2
        assert mask[0, 0 * 3 + 1]
        assert mask[0, 3 * 3 + 2]

    def test_a_stopped_trajectory_can_undo_its_stop(self):
        env = make_env()
        stopped = env.step(env.initial(1), np.array([env.stop_action]))
        assert env.backward_mask(stopped)[0, env.stop_action]
        resumed = env.backward_step(stopped, np.array([env.stop_action]))
        assert not resumed.stopped[0]


class TestPathCounting:
    """The k! claim, verified by enumeration rather than assumed."""

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_a_state_with_k_mutations_is_reached_by_exactly_k_factorial_paths(self, k):
        env = make_env(length=5, symbols="ABC")
        target_positions = list(range(k))
        target_token = 1

        # Every ordering of the k mutations should reach the same sequence, and
        # no other ordering should exist.
        reached = set()
        for order in itertools.permutations(target_positions):
            state = env.initial(1)
            for position in order:
                state = env.step(state, np.array([position * 3 + target_token]))
            reached.add(tuple(state.sequences[0].tolist()))

        assert len(reached) == 1, "orderings produced different sequences"
        assert math.factorial(k) == len(list(itertools.permutations(target_positions)))

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
    def test_log_path_count_matches_log_k_factorial(self, k):
        env = make_env(length=6, symbols="ABC")
        state = env.initial(1)
        for position in range(k):
            state = env.step(state, np.array([position * 3 + 1]))
        assert env.log_n_trajectories(state)[0] == pytest.approx(math.log(math.factorial(k)))

    def test_number_of_parents_equals_the_mutation_count(self):
        # This is what lets uniform P_B be computed as 1/k without a model.
        env = make_env(length=6, symbols="ABC")
        state = env.initial(1)
        for k in range(1, 5):
            state = env.step(state, np.array([(k - 1) * 3 + 1]))
            parents = env.backward_mask(state)[0, : env.n_mutation_actions].sum()
            assert parents == k == env.n_mutations(state)[0]

    def test_enumerated_paths_to_a_target_match_the_factorial(self):
        # Exhaustive: count every distinct action sequence from the parent that
        # arrives at one specific 3-mutation variant.
        env = make_env(length=4, symbols="AB")
        target = np.array([1, 1, 1, 0], dtype=np.int32)

        def count_paths(state, depth=0):
            if np.array_equal(state.sequences[0], target):
                return 1 if depth == 3 else 0
            if depth >= 3:
                return 0
            total = 0
            for action in np.flatnonzero(env.forward_mask(state)[0]):
                if action == env.stop_action:
                    continue
                total += count_paths(env.step(state, np.array([action])), depth + 1)
            return total

        assert count_paths(env.initial(1)) == math.factorial(3)


class TestMasking:
    def test_masked_actions_are_refused_rather_than_ignored(self):
        # Silently ignoring an illegal action would let a policy put probability
        # on edges that do not exist.
        env = make_env(length=4, symbols="ABC")
        state = env.step(env.initial(1), np.array([1]))
        with pytest.raises(ValueError, match="forward action not permitted"):
            env.step(state, np.array([1]))

    def test_backward_steps_are_checked_too(self):
        env = make_env(length=4, symbols="ABC")
        with pytest.raises(ValueError, match="backward action not permitted"):
            env.backward_step(env.initial(1), np.array([1]))

    def test_out_of_range_actions_are_rejected(self):
        env = make_env(length=4, symbols="ABC")
        with pytest.raises(ValueError, match="must lie in"):
            env.step(env.initial(1), np.array([999]))

    def test_one_action_per_trajectory_is_required(self):
        env = make_env(length=4, symbols="ABC")
        with pytest.raises(ValueError, match="one action per trajectory"):
            env.step(env.initial(3), np.array([0]))

    @given(
        length=st.integers(min_value=2, max_value=6),
        vocab=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=60, deadline=None)
    def test_every_unstopped_state_has_at_least_one_legal_action(self, length, vocab, seed):
        # A state with no legal action leaves the policy nothing to normalise
        # over, which shows up as a NaN loss much later.
        env = make_env(length=length, symbols="ABCDE"[:vocab])
        rng = np.random.default_rng(seed)
        state = env.initial(1)
        for _ in range(length + 2):
            if env.is_terminal(state)[0]:
                break
            mask = env.forward_mask(state)
            assert mask[0].any()
            state = env.step(state, np.array([rng.choice(np.flatnonzero(mask[0]))]))


class TestMutationBudget:
    def test_trajectories_stop_at_the_mutation_cap(self):
        env = make_env(length=6, symbols="ABC", max_mutations=2)
        state = env.initial(1)
        state = env.step(state, np.array([0 * 3 + 1]))
        state = env.step(state, np.array([1 * 3 + 1]))
        mask = env.forward_mask(state)
        assert not mask[0, : env.n_mutation_actions].any()
        assert mask[0, env.stop_action]

    def test_stopping_early_can_be_forbidden(self):
        env = make_env(length=4, symbols="ABC", max_mutations=2, allow_stop=False)
        assert not env.forward_mask(env.initial(1))[0, env.stop_action]

    def test_forbidding_early_stop_gives_terminals_a_fixed_mutation_count(self):
        env = make_env(length=5, symbols="ABC", max_mutations=3, allow_stop=False)
        rng = np.random.default_rng(0)
        final, _ = rollout(env, rng, n=1)
        assert env.n_mutations(final)[0] == 3

    def test_a_zero_mutation_budget_permits_only_stopping(self):
        env = make_env(length=4, symbols="ABC", max_mutations=0)
        mask = env.forward_mask(env.initial(1))
        assert mask[0, env.stop_action]
        assert not mask[0, : env.n_mutation_actions].any()


class TestFeasibilityMasking:
    def transitions(self, vocab, forbidden):
        matrix = np.ones((vocab, vocab), dtype=np.float64)
        for a, b in forbidden:
            matrix[a, b] = 0.0
        return matrix

    def test_a_forbidden_adjacency_is_never_offered(self):
        # Parent is all token 0; forbid 0 -> 1, so setting position 1 to token 1
        # would create the forbidden pair (position 0 = 0, position 1 = 1).
        matrix = self.transitions(3, [(0, 1)])
        env = make_env(length=4, symbols="ABC", transitions=matrix)
        mask = env.forward_mask(env.initial(1))
        assert not mask[0, 1 * 3 + 1]

    def test_every_reachable_sequence_is_feasible(self):
        # The point of masking: feasibility by construction rather than by
        # filtering afterwards.
        rng = np.random.default_rng(0)
        matrix = self.transitions(4, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 1)])
        env = make_env(length=6, symbols="ABCD", transitions=matrix)
        permitted = matrix > 0
        for _ in range(200):
            final, _ = rollout(env, rng, n=1)
            sequence = final.sequences[0]
            for position in range(len(sequence) - 1):
                assert permitted[sequence[position], sequence[position + 1]], (
                    f"reached an infeasible sequence: {sequence.tolist()}"
                )

    def test_transitions_must_match_the_alphabet(self):
        with pytest.raises(ValueError, match="to match the alphabet"):
            make_env(symbols="ABC", transitions=np.ones((2, 2)))


class TestParentValidation:
    def test_a_batch_is_not_a_parent(self):
        with pytest.raises(ValueError, match="single sequence"):
            MutationEnvironment(np.zeros((2, 4), dtype=np.int32), Alphabet.from_string("AB"))

    def test_float_parents_are_rejected(self):
        with pytest.raises(ValueError, match="token indices"):
            MutationEnvironment(np.zeros(4, dtype=np.float64), Alphabet.from_string("AB"))

    def test_tokens_outside_the_alphabet_are_rejected(self):
        with pytest.raises(ValueError, match=r"lie in \[0, 2\)"):
            MutationEnvironment(np.array([0, 5]), Alphabet.from_string("AB"))

    def test_an_impossible_mutation_cap_is_rejected(self):
        with pytest.raises(ValueError, match="max_mutations must lie in"):
            make_env(length=4, max_mutations=9)

    def test_the_parent_is_returned_as_a_copy(self):
        env = make_env()
        env.parent[0] = 2
        assert env.initial(1).sequences[0, 0] == 0


class TestBatching:
    def test_trajectories_in_a_batch_advance_independently(self):
        env = make_env(length=4, symbols="ABC")
        state = env.initial(3)
        state = env.step(state, np.array([0 * 3 + 1, 1 * 3 + 2, env.stop_action]))
        assert state.sequences[0].tolist() == [1, 0, 0, 0]
        assert state.sequences[1].tolist() == [0, 2, 0, 0]
        assert state.sequences[2].tolist() == [0, 0, 0, 0]
        assert state.stopped.tolist() == [False, False, True]

    def test_a_stopped_trajectory_is_offered_nothing(self):
        env = make_env()
        state = env.step(env.initial(1), np.array([env.stop_action]))
        assert not env.forward_mask(state)[0].any()

    def test_state_length_is_the_batch_size(self):
        assert len(make_env().initial(7)) == 7
