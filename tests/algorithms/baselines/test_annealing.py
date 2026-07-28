"""Tests for simulated annealing.

The two properties that make it a different baseline from hill climbing are that
it accepts a worse design when it is hot and refuses one when it is cold. Both
are checked directly rather than through downstream behaviour, because a bug in
the acceptance rule shows up as "annealing is just a noisy hill climber", which
is exactly the thing that would look fine in an aggregate benchmark result.
"""

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines import SimulatedAnnealing
from evoflownet.core import Alphabet
from evoflownet.env.mutation import MutationEnvironment


def make_env(length=6, symbols="ABCD", max_mutations=3, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def constrained_transitions(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def toy_landscape(sequences):
    """Reward sequences for containing token 1, so improvement is detectable."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(SimulatedAnnealing(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert SimulatedAnnealing(env).propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        # A candidate outside the mutation budget is one no GFlowNet could have
        # produced, so scoring it would make the comparison meaningless.
        env = make_env(max_mutations=2)
        sampler = SimulatedAnnealing(env, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_proposals_are_counted(self):
        sampler = SimulatedAnnealing(make_env())
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        left = SimulatedAnnealing(env, seed=3)
        right = SimulatedAnnealing(env, seed=3)
        assert np.array_equal(left.propose(16), right.propose(16))

    def test_a_different_seed_gives_different_proposals(self):
        env = make_env()
        left = SimulatedAnnealing(env, seed=0).propose(32)
        right = SimulatedAnnealing(env, seed=1).propose(32)
        assert not np.array_equal(left, right)

    def test_a_whole_campaign_is_reproducible(self):
        env = make_env(length=8, max_mutations=5)
        runs = []
        for _ in range(2):
            sampler = SimulatedAnnealing(env, seed=7)
            batches = []
            for _ in range(4):
                proposals = sampler.propose(24)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])


class TestTheMetropolisRule:
    """The one thing annealing has that hill climbing does not."""

    def test_a_hot_chain_moves_downhill(self):
        env = make_env()
        sampler = SimulatedAnnealing(
            env, initial_temperature=1e6, cooling_rate=1.0, min_temperature=1e-6, seed=0
        )
        good = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32)
        worse = np.array([[2, 0, 0, 0, 0, 0]], dtype=np.int32)
        sampler.observe(good, np.array([[10.0]]))
        sampler.observe(worse, np.array([[9.0]]))
        # exp(-1 / 1e6) is indistinguishable from 1, so the move is certain.
        assert sampler.current_value == pytest.approx(9.0)

    def test_a_cold_chain_refuses_to_move_downhill(self):
        env = make_env()
        sampler = SimulatedAnnealing(
            env, initial_temperature=1e-9, cooling_rate=1.0, min_temperature=1e-9, seed=0
        )
        good = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32)
        worse = np.array([[2, 0, 0, 0, 0, 0]], dtype=np.int32)
        sampler.observe(good, np.array([[10.0]]))
        sampler.observe(worse, np.array([[0.0]]))
        assert sampler.current_value == pytest.approx(10.0)

    def test_an_uphill_move_is_always_taken(self):
        sampler = SimulatedAnnealing(make_env(), initial_temperature=1e-9, min_temperature=1e-9)
        sampler.observe(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[1.0]]))
        sampler.observe(np.array([[2, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[5.0]]))
        assert sampler.current_value == pytest.approx(5.0)

    def test_a_failed_assay_neither_moves_the_chain_nor_is_recorded(self):
        # -inf is what an infeasible design scores. Treating it as a measurement
        # would let one bad well drag the chain off a good design.
        sampler = SimulatedAnnealing(make_env(), seed=0)
        sampler.observe(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[4.0]]))
        sampler.observe(np.array([[3, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[-np.inf]]))
        assert sampler.current_value == pytest.approx(4.0)

    def test_the_best_seen_is_tracked_separately_from_the_current_point(self):
        # The chain may be sitting below its best, which is the whole point.
        sampler = SimulatedAnnealing(
            make_env(), initial_temperature=1e6, cooling_rate=1.0, min_temperature=1e-6, seed=0
        )
        sampler.observe(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[10.0]]))
        sampler.observe(np.array([[2, 0, 0, 0, 0, 0]], dtype=np.int32), np.array([[3.0]]))
        assert sampler.best_value == pytest.approx(10.0)
        assert sampler.current_value == pytest.approx(3.0)


class TestTheCoolingSchedule:
    def test_the_temperature_falls_geometrically_once_per_round(self):
        sampler = SimulatedAnnealing(
            make_env(), initial_temperature=1.0, cooling_rate=0.5, min_temperature=1e-6
        )
        seen = [sampler.temperature]
        for _ in range(3):
            sampler.observe(np.zeros((1, 6), dtype=np.int32), np.array([[1.0]]))
            seen.append(sampler.temperature)
        assert seen == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_the_floor_holds(self):
        # Without it the acceptance ratio eventually divides by zero.
        sampler = SimulatedAnnealing(
            make_env(), initial_temperature=1.0, cooling_rate=0.1, min_temperature=0.05
        )
        for _ in range(10):
            sampler.observe(np.zeros((1, 6), dtype=np.int32), np.array([[1.0]]))
        assert sampler.temperature == pytest.approx(0.05)

    def test_a_rate_of_one_is_the_fixed_temperature_metropolis_sampler(self):
        # The regime in which annealing targets exp(f / T), which is the same
        # target a GFlowNet with that reward claims to sample.
        sampler = SimulatedAnnealing(make_env(), initial_temperature=2.0, cooling_rate=1.0)
        for _ in range(5):
            sampler.observe(np.zeros((1, 6), dtype=np.int32), np.array([[1.0]]))
        assert sampler.temperature == pytest.approx(2.0)


class TestTheNeighbourhood:
    """The trajectory constraint is not a constraint on states.

    An annealer that could not revise an earlier substitution could not undo a
    move it accepted, which for a method whose defining feature is accepting
    moves it may need to undo is not a limitation but a contradiction.
    """

    def test_it_can_revise_an_existing_substitution(self):
        env = make_env(length=4, max_mutations=4)
        sampler = SimulatedAnnealing(env, seed=0)
        sampler.observe(np.array([[1, 0, 0, 0]], dtype=np.int32), np.array([[10.0]]))
        proposals = sampler.propose(400)
        revised = (proposals[:, 0] != 0) & (proposals[:, 0] != 1)
        assert revised.any(), "annealing never revised the existing mutation"

    def test_the_neighbourhood_does_not_collapse_as_it_moves(self):
        env = make_env(length=6, max_mutations=3)
        sampler = SimulatedAnnealing(env, seed=2)
        sizes = []
        for _ in range(5):
            proposals = sampler.propose(500)
            sizes.append(len({row.tobytes() for row in np.ascontiguousarray(proposals)}))
            sampler.observe(proposals, np.arange(500, dtype=float)[:, None])
        assert min(sizes) > 1, f"neighbourhood collapsed: {sizes}"

    def test_it_climbs_when_it_is_cold(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = SimulatedAnnealing(env, initial_temperature=0.5, cooling_rate=0.5, seed=0)
        first = last = 0.0
        for index in range(12):
            proposals = sampler.propose(32)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first


class TestFeasibility:
    def test_a_feasible_only_chain_only_emits_constructible_designs(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        sampler = SimulatedAnnealing(env, feasible_only=True, seed=0)
        for _ in range(4):
            proposals = sampler.propose(64)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_the_label_says_whether_feasibility_is_enforced(self):
        env = make_env()
        assert SimulatedAnnealing(env).name == "SimulatedAnnealing"
        assert SimulatedAnnealing(env, feasible_only=True).name == "SimulatedAnnealing (feasible)"


class TestTheScheduleIsValidated:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("initial_temperature", 0.0, "initial_temperature must be positive"),
            ("initial_temperature", -1.0, "initial_temperature must be positive"),
            ("min_temperature", 0.0, "min_temperature must be positive"),
            ("cooling_rate", 0.0, "cooling_rate must lie in"),
            ("cooling_rate", 1.5, "cooling_rate must lie in"),
        ],
    )
    def test_an_impossible_schedule_is_refused(self, field, value, message):
        with pytest.raises(ValueError, match=message):
            SimulatedAnnealing(make_env(), **{field: value})

    def test_a_floor_above_the_start_is_refused(self):
        with pytest.raises(ValueError, match="exceeds initial_temperature"):
            SimulatedAnnealing(make_env(), initial_temperature=0.1, min_temperature=1.0)
