"""Tests for the classical baselines.

These are not strawmen to be cleared. Directed evolution *is* a genetic
algorithm, so the GA here is the incumbent, and the tests check it is a real one:
that it improves, that it respects the same graph the GFlowNet does, and that its
hyperparameters are the ones its own authors published.
"""

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines import (
    GeneticAlgorithm,
    HillClimbing,
    RandomMutagenesis,
)
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


ALL = [
    lambda env: RandomMutagenesis(env, seed=0),
    lambda env: HillClimbing(env, seed=0),
    lambda env: GeneticAlgorithm(env, population_size=32, seed=0),
]


class TestTheSharedInterface:
    @pytest.mark.parametrize("make", ALL)
    def test_every_baseline_is_a_sampler(self, make):
        # One interface is what makes the benchmark a comparison rather than a
        # measurement of two different harnesses.
        assert isinstance(make(make_env()), Sampler)

    @pytest.mark.parametrize("make", ALL)
    def test_proposals_have_the_right_shape(self, make):
        env = make_env()
        proposals = make(env).propose(16)
        assert proposals.shape == (16, env.sequence_length)

    @pytest.mark.parametrize("make", ALL)
    def test_proposals_stay_inside_the_environment_graph(self, make):
        # A candidate outside the mutation budget is one no GFlowNet could have
        # produced, so scoring it would make the comparison meaningless.
        env = make_env(max_mutations=2)
        sampler = make(env)
        for _ in range(5):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    @pytest.mark.parametrize("make", ALL)
    def test_proposals_are_counted(self, make):
        sampler = make(make_env())
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made >= 17

    @pytest.mark.parametrize("make", ALL)
    def test_observing_is_optional_to_the_caller_but_never_crashes(self, make):
        env = make_env()
        sampler = make(env)
        proposals = sampler.propose(8)
        sampler.observe(proposals, toy_landscape(proposals))


class TestTheyActuallySearch:
    @pytest.mark.parametrize("make", [ALL[1], ALL[2]])
    def test_an_adaptive_sampler_improves_over_rounds(self, make):
        # Random mutagenesis is excluded deliberately: it does not adapt, which
        # is the point of having it.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = make(env)
        first = last = 0.0
        for round_index in range(12):
            proposals = sampler.propose(32)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if round_index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first

    def test_random_mutagenesis_does_not_adapt(
        self,
    ):
        # The null. If it improved, `observe` would be doing something it should
        # not, and the floor would no longer be a floor.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = RandomMutagenesis(env, seed=0)
        before = sampler.propose(64)
        sampler.observe(before, toy_landscape(before))
        after = sampler.propose(64)
        assert toy_landscape(after).mean() == pytest.approx(toy_landscape(before).mean(), abs=0.6)

    def test_hill_climbing_collapses_onto_one_design(self):
        # The behaviour the project exists to contrast against: a pure optimiser
        # returns one peak, however many designs it is asked for.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = HillClimbing(env, patience=100, seed=0)
        for _ in range(20):
            proposals = sampler.propose(32)
            sampler.observe(proposals, toy_landscape(proposals))
        final = sampler.propose(32)
        distinct = len({tuple(row) for row in final})
        assert distinct < 32


class TestGeneticAlgorithm:
    def test_the_published_hyperparameters_are_the_defaults(self):
        # Stanton et al. report p_m = p_r = 1/L. Comparing against anything else
        # would be comparing against a configuration convenient to us.
        env = make_env(length=10)
        ga = GeneticAlgorithm(env)
        assert ga._mutation_prob == pytest.approx(1 / 10)
        assert ga._recombine_prob == pytest.approx(1 / 10)

    def test_selection_never_loses_ground(self):
        # Parents and offspring are ranked together, so the population's best
        # cannot get worse between generations.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        ga = GeneticAlgorithm(env, population_size=32, seed=0)
        best_so_far = -np.inf
        for _ in range(10):
            proposals = ga.propose(32)
            ga.observe(proposals, toy_landscape(proposals))
            population_best = float(toy_landscape(ga.population).max())
            assert population_best >= best_so_far - 1e-9
            best_so_far = population_best

    def test_offspring_beyond_the_mutation_budget_are_reverted(self):
        env = make_env(length=10, symbols="ABCD", max_mutations=2)
        ga = GeneticAlgorithm(env, population_size=16, mutation_prob=0.9, seed=0)
        proposals = ga.propose(64)
        counts = (proposals != env.parent[None, :]).sum(axis=1)
        assert counts.max() <= 2

    @pytest.mark.parametrize(
        ("field", "value"),
        [("mutation_prob", 1.5), ("recombine_prob", -0.1), ("survival_quantile", 2.0)],
    )
    def test_probabilities_outside_the_unit_interval_are_refused(self, field, value):
        with pytest.raises(ValueError, match="must lie in"):
            GeneticAlgorithm(make_env(), **{field: value})

    def test_an_empty_population_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            GeneticAlgorithm(make_env(), population_size=0)


class TestRejectionSampling:
    """The control that decides whether feasibility masking is an advantage."""

    def test_a_rejection_sampler_only_emits_feasible_designs(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        for sampler in (
            RandomMutagenesis(env, feasible_only=True, seed=0),
            GeneticAlgorithm(env, population_size=32, feasible_only=True, seed=0),
        ):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all(), sampler.name

    def test_rejection_costs_proposals_rather_than_oracle_calls(self):
        # The heart of the comparison: rejection wastes no evaluations, so at
        # equal oracle budget it may match a masked policy. What it burns is
        # proposals, and that is what has to be reported.
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        plain = RandomMutagenesis(env, seed=0)
        rejecting = RandomMutagenesis(env, feasible_only=True, seed=0)
        plain.propose(32)
        rejecting.propose(32)
        assert rejecting.proposals_made > plain.proposals_made

    def test_an_impossible_constraint_raises_rather_than_returning_junk(self):
        # Silently emitting infeasible designs would corrupt the very experiment
        # this class exists for. Failure here is also itself a result: it means
        # rejection sampling has become impractical at that density.
        forbidden = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
        env = make_env(
            length=8,
            symbols="ABCD",
            max_mutations=3,
            transitions=constrained_transitions(4, forbidden),
        )
        with pytest.raises(RuntimeError, match="feasible"):
            RandomMutagenesis(env, feasible_only=True, max_attempts=3, seed=0).propose(32)


class TestReproducibility:
    @pytest.mark.parametrize("make", ALL)
    def test_the_same_seed_gives_the_same_proposals(self, make):
        env = make_env()
        assert np.array_equal(make(env).propose(16), make(env).propose(16))
