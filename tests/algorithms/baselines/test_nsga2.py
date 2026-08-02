"""Tests for NSGA-II.

Two of these tests exist because of specific ways NSGA-II implementations go
wrong quietly. The non-dominated sort must produce the paper's fronts, so it is
checked against hand-computed cases and against an independent reference. The
crowding distance must set boundary solutions to infinity *per objective*, so
that the extremes of a front survive truncation -- omit that and a run loses
spread gradually, with no error and a perfectly plausible-looking front.
"""

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines import (
    NSGA2,
    PUBLISHED_CROSSOVER_PROB,
    PUBLISHED_POPULATION_SIZE,
    TOURNAMENT_SIZE,
    crowding_distance,
    fast_non_dominated_sort,
)
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes import MultiEhrlichLandscape
from evogfn.metrics.pareto import hypervolume, igd_plus, non_dominated


def make_env(length=8, symbols="ABCD", max_mutations=6, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def two_objectives(sequences):
    """Count token 1 and token 2. They compete for the same finite positions."""
    designs = np.asarray(sequences)
    return np.stack([(designs == 1).sum(axis=1), (designs == 2).sum(axis=1)], axis=1).astype(
        np.float64
    )


def scored(sampler):
    """The population's objective values, past the pre-scoring `None`."""
    values = sampler.values
    assert values is not None, "the sampler has not observed anything yet"
    return values


def reference_fronts(values):
    """Rank by repeatedly peeling the non-dominated set.

    Deliberately the O(M N^3) method NSGA-II exists to avoid. It is a correct but
    slow definition of the same fronts, which makes it exactly the right thing to
    check the fast algorithm against: an independent implementation that shares
    no code with it.
    """
    values = np.asarray(values, dtype=np.float64)
    ranks = np.full(values.shape[0], -1, dtype=int)
    remaining = np.arange(values.shape[0])
    rank = 0
    while remaining.size:
        keep = non_dominated(values[remaining])
        ranks[remaining[keep]] = rank
        remaining = remaining[~keep]
        rank += 1
    return ranks


class TestFastNonDominatedSort:
    def test_a_hand_computed_chain(self):
        # (2,2) beats both others; (1,1) beats (0,0). Three fronts of one.
        values = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
        assert fast_non_dominated_sort(values).tolist() == [1, 0, 2]

    def test_a_hand_computed_trade_off(self):
        # Neither of the first two beats the other; both beat the third.
        values = np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
        assert fast_non_dominated_sort(values).tolist() == [0, 0, 1]

    def test_mutually_non_dominated_designs_share_one_front(self):
        values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        assert fast_non_dominated_sort(values).tolist() == [0, 0, 0, 0]

    def test_duplicates_are_not_dominated_by_their_copies(self):
        # a >= b everywhere but never strictly better, so neither dominates.
        values = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        assert fast_non_dominated_sort(values).tolist() == [0, 0, 1]

    def test_a_single_design_is_the_whole_first_front(self):
        assert fast_non_dominated_sort(np.array([[1.0, 2.0]])).tolist() == [0]

    def test_an_empty_population_sorts_to_nothing(self):
        assert fast_non_dominated_sort(np.zeros((0, 2))).shape == (0,)

    @pytest.mark.parametrize("n_objectives", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_agrees_with_the_slow_definition(self, n_objectives, seed):
        # The check that matters: the fast algorithm must produce the same fronts
        # as repeatedly peeling the non-dominated set, which is the definition.
        rng = np.random.default_rng(seed)
        # Coarse values so there are plenty of ties and duplicates to disagree on.
        values = rng.integers(0, 4, size=(60, n_objectives)).astype(np.float64)
        assert fast_non_dominated_sort(values).tolist() == reference_fronts(values).tolist()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_every_design_beyond_the_first_front_has_a_dominator_one_rank_up(self, seed):
        # The defining property of the fronts. If ranks were assigned by anything
        # other than "your last dominator retired", this is what would break.
        rng = np.random.default_rng(seed)
        values = rng.integers(0, 5, size=(50, 3)).astype(np.float64)
        ranks = fast_non_dominated_sort(values)
        for index in np.flatnonzero(ranks > 0):
            above = values[ranks == ranks[index] - 1]
            dominated = (above >= values[index]).all(axis=1) & (above > values[index]).any(axis=1)
            assert dominated.any(), f"design {index} entered its front with no dominator above"

    def test_every_design_gets_a_front(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(40, 2))
        ranks = fast_non_dominated_sort(values)
        assert (ranks >= 0).all()

    def test_infeasible_designs_rank_below_everything_finite(self):
        # -inf on every objective is what an infeasible Ehrlich design scores.
        values = np.array([[1.0, 0.0], [-np.inf, -np.inf], [0.0, 1.0]])
        ranks = fast_non_dominated_sort(values)
        assert ranks.tolist() == [0, 1, 0]

    def test_nan_is_refused_rather_than_ranked(self):
        with pytest.raises(ValueError, match="nan"):
            fast_non_dominated_sort(np.array([[1.0, np.nan]]))

    def test_a_flat_array_is_refused_as_ambiguous(self):
        with pytest.raises(ValueError, match="n_objectives"):
            fast_non_dominated_sort(np.array([1.0, 2.0, 3.0]))


class TestCrowdingDistance:
    def test_a_hand_computed_front(self):
        # Four evenly spaced points on a line. Objective 0 sorts them 0,1,2,3 and
        # objective 1 sorts them in reverse, so each interior point collects
        # (2 - 0) / 3 twice: 2/3 + 2/3 = 4/3. The two ends are boundaries on both.
        values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        distance = crowding_distance(values)
        assert np.isinf(distance[[0, 3]]).all()
        assert distance[1] == pytest.approx(4 / 3)
        assert distance[2] == pytest.approx(4 / 3)

    def test_a_lonely_design_scores_above_a_crowded_one(self):
        # Objective 0 places the designs at 0, 1, 2, 10 and objective 1 spaces
        # them evenly, so only the first breaks the tie. The design at 2 has a
        # gap of 9 to its right; the one at 1 is squeezed between 0 and 2.
        values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [10.0, 0.0]])
        distance = crowding_distance(values)
        assert distance[1] == pytest.approx((2 - 0) / 10 + (3 - 1) / 3)
        assert distance[2] == pytest.approx((10 - 1) / 10 + (2 - 0) / 3)
        assert distance[2] > distance[1]

    def test_the_boundaries_are_infinite_on_every_objective_separately(self):
        # The failure this test exists for: applying the boundary rule on the
        # first objective only. The best design on objective 1 sits in the
        # *interior* of objective 0, so it is finite unless objective 1 is also
        # given its boundaries -- and it is what carries the front's spread.
        values = np.array(
            [
                [0.0, 5.0],  # best on objective 1, interior-ish on objective 0
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 0.0],  # best on objective 0
            ]
        )
        distance = crowding_distance(values)
        assert np.isinf(distance[0]), "the extreme of objective 1 was not made a boundary"
        assert np.isinf(distance[4]), "the extreme of objective 0 was not made a boundary"
        assert np.isfinite(distance[1:4]).all()

    def test_an_interior_design_extreme_on_one_objective_is_still_a_boundary(self):
        # Three objectives; design 1 is middling on the first two and best on the
        # third. It must survive truncation, so it must be infinite.
        values = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 9.0],
                [2.0, 2.0, 1.0],
            ]
        )
        assert np.isinf(crowding_distance(values)).all()  # n = 3 with two at each end

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_a_front_of_two_or_fewer_is_all_boundary(self, n):
        distance = crowding_distance(np.zeros((n, 2)))
        assert distance.shape == (n,)
        assert np.isinf(distance).all()

    def test_a_constant_objective_contributes_nothing_and_produces_no_nan(self):
        # Dividing by a zero range is the obvious way to get nan here, and a nan
        # distance silently randomises the tournament rather than raising.
        values = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        distance = crowding_distance(values)
        assert not np.isnan(distance).any()
        assert np.isfinite(distance[[1, 2]]).all()

    def test_an_infinite_objective_value_produces_no_nan(self):
        # -inf minus -inf is nan. An infeasible design in the front is how that
        # arrives in practice.
        values = np.array([[-np.inf, 1.0], [0.0, 0.5], [1.0, 0.0], [2.0, -1.0]])
        distance = crowding_distance(values)
        assert not np.isnan(distance).any()

    def test_nan_is_refused(self):
        with pytest.raises(ValueError, match="nan"):
            crowding_distance(np.array([[1.0, np.nan], [0.0, 0.0], [2.0, 2.0]]))


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(NSGA2(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert NSGA2(env, population_size=16).propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        # A candidate outside the mutation budget is one no GFlowNet could have
        # produced, so scoring it would make the comparison meaningless.
        env = make_env(length=10, max_mutations=3)
        sampler = NSGA2(env, population_size=32, mutation_prob=0.8, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, two_objectives(proposals))

    def test_proposals_are_counted(self):
        sampler = NSGA2(make_env(), population_size=16)
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        first, second = NSGA2(env, seed=3), NSGA2(env, seed=3)
        assert np.array_equal(first.propose(16), second.propose(16))

    def test_the_same_seed_gives_the_same_run(self):
        # Reproducibility has to survive selection, not just the first draw.
        env = make_env()

        def run(seed):
            sampler = NSGA2(env, population_size=16, seed=seed)
            for _ in range(5):
                batch = sampler.propose(16)
                sampler.observe(batch, two_objectives(batch))
            return sampler.population

        assert np.array_equal(run(4), run(4))
        assert not np.array_equal(run(4), run(5))


class TestPublishedHyperparameters:
    def test_the_population_size_is_debs(self):
        assert PUBLISHED_POPULATION_SIZE == 100
        assert NSGA2(make_env())._population_size == 100

    def test_the_crossover_probability_is_debs(self):
        assert PUBLISHED_CROSSOVER_PROB == 0.9
        assert NSGA2(make_env())._crossover_prob == pytest.approx(0.9)

    def test_the_tournament_is_binary(self):
        # Part of the method, not a knob.
        assert TOURNAMENT_SIZE == 2

    def test_the_mutation_rate_is_one_over_length(self):
        # Deb et al.'s p_m = 1/l for binary-coded GAs, and the same value the
        # GA baseline in this package carries from Stanton et al.
        assert NSGA2(make_env(length=10))._mutation_prob == pytest.approx(1 / 10)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("mutation_prob", 1.5), ("crossover_prob", -0.1), ("recombine_prob", 2.0)],
    )
    def test_probabilities_outside_the_unit_interval_are_refused(self, field, value):
        with pytest.raises(ValueError, match="must lie in"):
            NSGA2(make_env(), **{field: value})

    def test_an_empty_population_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            NSGA2(make_env(), population_size=0)


class TestItActuallySearches:
    def test_it_improves_a_two_objective_front_over_generations(self):
        # The point of the baseline. Hypervolume against a fixed reference is
        # the indicator that is monotone in dominance, so a set that improves
        # cannot score lower.
        env = make_env(length=12, symbols="ABCD", max_mutations=10)
        sampler = NSGA2(env, population_size=32, seed=0)
        volumes = []
        for _ in range(25):
            batch = sampler.propose(32)
            sampler.observe(batch, two_objectives(batch))
            volumes.append(hypervolume(scored(sampler), reference=[-1.0, -1.0]))
        assert volumes[-1] > volumes[0], f"the front never improved: {volumes}"

    def test_the_front_never_loses_ground(self):
        # Elitism: parents and offspring are ranked together, so hypervolume is
        # non-decreasing between generations.
        env = make_env(length=12, symbols="ABCD", max_mutations=10)
        sampler = NSGA2(env, population_size=32, seed=1)
        previous = -np.inf
        for _ in range(15):
            batch = sampler.propose(32)
            sampler.observe(batch, two_objectives(batch))
            current = hypervolume(scored(sampler), reference=[-1.0, -1.0])
            assert current >= previous - 1e-9
            previous = current

    def test_it_keeps_a_spread_of_trade_offs_rather_than_one_design(self):
        # What separates it from every scalarising baseline here: the population
        # it ends with covers both objectives rather than one corner.
        env = make_env(length=12, symbols="ABCD", max_mutations=10)
        sampler = NSGA2(env, population_size=32, seed=2)
        for _ in range(25):
            batch = sampler.propose(32)
            sampler.observe(batch, two_objectives(batch))
        front = scored(sampler)[sampler.ranks == 0]
        assert len(np.unique(front, axis=0)) > 1, "the front collapsed onto one trade-off"

    def test_it_closes_on_the_known_front_of_a_real_landscape(self):
        # The two deliverables together, and the only place a *known* target is
        # available: IGD+ is measured against the enumerated exact front rather
        # than against whatever the run itself happened to find.
        landscape = MultiEhrlichLandscape.with_conflict(
            sequence_length=8,
            vocab_size=3,
            n_motifs=2,
            motif_length=4,
            max_spacing=1,
            transition_density=1.0,
            conflict=1.0,
            seed=0,
        )
        env = MutationEnvironment(
            landscape.feasible_sequence(seed=1),
            landscape.alphabet,
            max_mutations=8,
            transitions=landscape.transition_matrix,
        )
        target = landscape.exact_pareto_front()
        assert target.shape[0] > 1, "the instance has to be in tension for this to mean anything"

        sampler = NSGA2(env, population_size=32, seed=0)
        distances = []
        for _ in range(30):
            batch = sampler.propose(32)
            sampler.observe(batch, landscape.evaluate(batch))
            reached = scored(sampler)
            distances.append(igd_plus(reached[np.isfinite(reached).all(axis=1)], target))
        assert distances[-1] < distances[0], f"never approached the known front: {distances}"

    def test_the_front_accessor_returns_the_non_dominated_designs(self):
        env = make_env(length=10, max_mutations=8)
        sampler = NSGA2(env, population_size=16, seed=0)
        for _ in range(8):
            batch = sampler.propose(16)
            sampler.observe(batch, two_objectives(batch))
        assert non_dominated(two_objectives(sampler.front())).all()


class TestSurvival:
    def test_the_population_size_is_held_exactly(self):
        env = make_env()
        sampler = NSGA2(env, population_size=10, seed=0)
        for _ in range(4):
            batch = sampler.propose(30)
            sampler.observe(batch, two_objectives(batch))
            assert sampler.population.shape[0] == 10
            assert scored(sampler).shape == (10, 2)

    def test_truncating_an_overfull_front_keeps_its_extremes(self):
        # The consequence of the boundary rule, at the place it bites: a front
        # larger than the population has to be cut, and the cut must not take the
        # tips off. Nothing raises if it does -- the front just quietly narrows.
        env = make_env(length=6, symbols="AB", max_mutations=6)
        sampler = NSGA2(env, population_size=4, seed=0)
        designs = np.zeros((9, 6), dtype=np.int32)
        # Nine mutually non-dominated trade-offs along a line.
        values = np.stack([np.arange(9.0), 8.0 - np.arange(9.0)], axis=1)
        sampler.observe(designs, values)

        kept = scored(sampler)
        assert kept.shape[0] == 4
        assert np.isclose(kept, [0.0, 8.0]).all(axis=1).any(), (
            "the objective-1 extreme was discarded"
        )
        assert np.isclose(kept, [8.0, 0.0]).all(axis=1).any(), (
            "the objective-0 extreme was discarded"
        )

    def test_unscored_founders_are_displaced_by_anything_real(self):
        env = make_env()
        sampler = NSGA2(env, population_size=4, seed=0)
        designs = np.zeros((4, env.sequence_length), dtype=np.int32)
        designs[:, 0] = [0, 1, 2, 3]
        sampler.observe(designs, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
        assert np.isfinite(scored(sampler)).all()

    def test_ranks_and_crowding_describe_the_surviving_population(self):
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(16)
        sampler.observe(batch, two_objectives(batch))
        assert sampler.ranks.shape == (8,)
        assert sampler.crowding_distances.shape == (8,)
        assert (sampler.ranks == 0).any()

    def test_the_values_accessor_is_empty_before_anything_is_scored(self):
        assert NSGA2(make_env()).values is None

    def test_accessors_return_copies(self):
        # Handing out internal arrays would let a caller silently rewrite the
        # population between rounds.
        env = make_env()
        sampler = NSGA2(env, population_size=4, seed=0)
        batch = sampler.propose(4)
        sampler.observe(batch, two_objectives(batch))
        sampler.population[0, 0] = 3
        scored(sampler)[0, 0] = 999.0
        sampler.ranks[0] = 99
        assert scored(sampler).max() < 999.0
        assert sampler.ranks.max() < 99


class TestMultiObjectiveValuesAreRequired:
    def test_two_objectives_are_accepted_rather_than_refused(self):
        # The opposite of every other baseline here, and the whole reason this
        # one exists: NSGA-II ranks by dominance, so it needs the vector.
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        sampler.observe(batch, two_objectives(batch))
        assert scored(sampler).shape == (8, 2)

    def test_a_single_objective_degenerates_to_a_dominance_ranked_ga(self):
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        sampler.observe(batch, (batch == 1).sum(axis=1, keepdims=True).astype(float))
        assert scored(sampler).shape == (8, 1)

    def test_a_flat_array_is_refused_as_ambiguous(self):
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        with pytest.raises(ValueError, match="n_objectives"):
            sampler.observe(batch, np.zeros(8))

    def test_mismatched_rows_are_refused(self):
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        with pytest.raises(ValueError, match="line up"):
            sampler.observe(batch, np.zeros((4, 2)))

    def test_changing_the_objective_count_midway_is_refused(self):
        # The landscape cannot grow an objective between rounds, and dominance
        # between a two-vector and a three-vector has no meaning.
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        sampler.observe(batch, two_objectives(batch))
        with pytest.raises(ValueError, match="objectives"):
            sampler.observe(batch, np.zeros((8, 3)))

    def test_nan_values_are_refused(self):
        env = make_env()
        sampler = NSGA2(env, population_size=8, seed=0)
        batch = sampler.propose(8)
        values = two_objectives(batch)
        values[0, 0] = np.nan
        with pytest.raises(ValueError, match="nan"):
            sampler.observe(batch, values)


class TestFeasibility:
    def constrained(self, vocab, forbidden):
        matrix = np.ones((vocab, vocab), dtype=np.float64)
        for a, b in forbidden:
            matrix[a, b] = 0.0
        return matrix

    def test_a_rejection_sampler_only_emits_feasible_designs(self):
        transitions = self.constrained(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, max_mutations=3, transitions=transitions)
        sampler = NSGA2(env, population_size=32, feasible_only=True, seed=0)
        assert env.is_reachable(sampler.propose(32)).all()

    def test_rejection_costs_proposals_rather_than_oracle_calls(self):
        transitions = self.constrained(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, max_mutations=3, transitions=transitions)
        plain = NSGA2(env, population_size=32, seed=0)
        rejecting = NSGA2(env, population_size=32, feasible_only=True, seed=0)
        plain.propose(32)
        rejecting.propose(32)
        assert rejecting.proposals_made > plain.proposals_made

    def test_an_impossible_constraint_raises_rather_than_returning_junk(self):
        forbidden = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
        env = make_env(length=8, max_mutations=3, transitions=self.constrained(4, forbidden))
        sampler = NSGA2(
            env, population_size=32, mutation_prob=0.9, feasible_only=True, max_attempts=3, seed=0
        )
        with pytest.raises(RuntimeError, match="feasible"):
            sampler.propose(32)

    def test_the_name_says_whether_rejection_is_on(self):
        env = make_env()
        assert NSGA2(env).name == "NSGA-II"
        assert NSGA2(env, feasible_only=True).name == "NSGA-II (rejection)"


class TestMutationBudget:
    def test_offspring_beyond_the_budget_are_reverted(self):
        env = make_env(length=10, max_mutations=2)
        sampler = NSGA2(env, population_size=16, mutation_prob=0.9, seed=0)
        proposals = sampler.propose(64)
        assert (proposals != env.parent[None, :]).sum(axis=1).max() <= 2
