"""Tests for the multi-objective Ehrlich landscape.

The point of a synthetic is that the right answer is knowable, so the Pareto
front here is checked against exhaustive search rather than against
plausible-looking output. The conflict dial is checked by measuring what it
claims to control: the front collapses to a single point when the objectives are
aligned, and spreads when they are not.
"""

import numpy as np
import pytest

from evogfn.landscapes import EhrlichLandscape, MultiEhrlichLandscape
from evogfn.metrics.pareto import non_dominated

#: A two-objective instance small enough to enumerate: 3^8 = 6,561 sequences.
#: Motifs of four tokens leave two objectives genuinely competing for an
#: eight-position sequence, which is what makes the conflict dial visible at all.
SMALL = {
    "sequence_length": 8,
    "vocab_size": 3,
    "n_motifs": 2,
    "motif_length": 4,
    "max_spacing": 1,
    "transition_density": 1.0,
}

#: Seeds every conflict measurement averages over. One instance is one draw of
#: the motifs, and a single draw says nothing about a distributional claim.
SEEDS = range(8)


def small(**overrides):
    return MultiEhrlichLandscape.with_conflict(**(SMALL | overrides))


def ehrlich(**overrides):
    """One constituent, built directly, for the composition checks."""
    return EhrlichLandscape(**(SMALL | overrides))


def front_sizes(conflict, **overrides):
    return [
        small(conflict=conflict, seed=seed, **overrides).exact_pareto_front().shape[0]
        for seed in SEEDS
    ]


class TestShape:
    def test_it_returns_one_column_per_objective(self):
        landscape = small(conflict=1.0, seed=0)
        values = landscape.evaluate(np.zeros((5, 8), dtype=np.int32))
        assert values.shape == (5, 2)

    @pytest.mark.parametrize("n_objectives", [2, 3, 4])
    def test_the_column_count_follows_n_objectives(self, n_objectives):
        landscape = small(n_objectives=n_objectives, conflict=1.0, seed=0)
        assert landscape.n_objectives == n_objectives
        assert landscape.evaluate(np.zeros((3, 8), dtype=np.int32)).shape == (3, n_objectives)
        assert len(landscape.objective_names) == n_objectives

    def test_one_objective_is_refused_as_the_wrong_class(self):
        with pytest.raises(ValueError, match="at least 2 objectives"):
            small(n_objectives=1)

    def test_composing_fewer_than_two_landscapes_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 objectives"):
            MultiEhrlichLandscape([ehrlich(seed=0)])

    def test_the_optimum_is_the_ideal_point(self):
        landscape = small(conflict=1.0, seed=0)
        assert landscape.optimum.tolist() == [1.0, 1.0]

    def test_every_objective_attains_one_at_its_own_planted_optimum(self):
        landscape = small(conflict=1.0, seed=0)
        for index, constituent in enumerate(landscape.landscapes):
            score = landscape.evaluate(constituent.optimal_sequence[None, :])[0, index]
            assert score == pytest.approx(1.0)


class TestOneFeasibleSet:
    def test_the_constituents_share_a_transition_matrix(self):
        landscape = small(conflict=1.0, transition_density=0.5, seed=2)
        shared = landscape.transition_matrix
        for constituent in landscape.landscapes:
            assert np.array_equal(constituent.transition_matrix, shared)

    @pytest.mark.parametrize("conflict", [0.0, 0.5, 1.0])
    def test_sharing_holds_across_the_whole_dial(self, conflict):
        # The divergence machinery redirects the random stream *after* the
        # transition matrix is drawn. If it ever redirected it before, this is
        # the test that says so.
        landscape = small(conflict=conflict, transition_density=0.4, seed=5)
        shared = landscape.transition_matrix
        assert all(np.array_equal(c.transition_matrix, shared) for c in landscape.landscapes)

    def test_landscapes_with_different_transition_matrices_are_refused(self):
        # Enforced rather than assumed: with two feasible sets, `is_feasible` has
        # to pick an objective to believe and the hypervolume is computed over a
        # set whose membership nobody agrees on.
        first = ehrlich(transition_density=0.5, seed=0)
        second = ehrlich(transition_density=0.5, seed=1)
        assert not np.array_equal(first.transition_matrix, second.transition_matrix)
        with pytest.raises(ValueError, match="different transition matrix"):
            MultiEhrlichLandscape([first, second])

    def test_landscapes_of_different_lengths_are_refused(self):
        first = ehrlich(seed=0)
        second = ehrlich(sequence_length=12, seed=0)
        with pytest.raises(ValueError, match="length"):
            MultiEhrlichLandscape([first, second])

    def test_landscapes_over_different_alphabets_are_refused(self):
        first = ehrlich(seed=0)
        second = ehrlich(vocab_size=4, seed=0)
        with pytest.raises(ValueError, match="alphabet"):
            MultiEhrlichLandscape([first, second])

    def test_infeasible_sequences_score_minus_infinity_on_every_objective(self):
        # Only coherent because the feasible set is shared. A design infeasible
        # for one objective and scored by another would be nonsense.
        landscape = small(conflict=1.0, transition_density=0.3, seed=1)
        space = landscape.enumerate()
        infeasible = ~landscape.is_feasible(space)
        assert infeasible.any(), "test needs at least one infeasible sequence"
        assert np.isneginf(landscape.evaluate(space)[infeasible]).all()

    def test_feasibility_agrees_with_every_constituent(self):
        landscape = small(conflict=1.0, transition_density=0.4, seed=3)
        space = landscape.enumerate()
        shared = landscape.is_feasible(space)
        for constituent in landscape.landscapes:
            assert np.array_equal(constituent.is_feasible(space), shared)

    def test_the_starting_point_is_feasible_and_is_not_an_optimum(self):
        landscape = small(conflict=1.0, transition_density=0.4, seed=1)
        start = landscape.feasible_sequence(seed=11)
        assert landscape.is_feasible(start[None, :])[0]
        # The divergence machinery must switch itself off after construction, or
        # `feasible_sequence` would hand back a planted optimum and leak the
        # answer into every campaign's starting point.
        assert not (landscape.optimal_sequences == start[None, :]).all(axis=1).any()


class TestTheConflictDial:
    def test_aligned_objectives_share_one_planted_optimum(self):
        landscape = small(conflict=0.0, seed=0)
        optima = landscape.optimal_sequences
        assert len({row.tobytes() for row in optima}) == 1

    def test_conflicting_objectives_have_different_planted_optima(self):
        landscape = small(conflict=1.0, seed=0)
        optima = landscape.optimal_sequences
        assert len({row.tobytes() for row in optima}) == 2

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_front_collapses_to_a_point_when_aligned(self, seed):
        # Not "approximately collapses": at conflict 0 every objective is carved
        # from the same sequence, so that sequence scores 1.0 on all of them and
        # nothing else can be non-dominated.
        landscape = small(conflict=0.0, seed=seed)
        front = landscape.exact_pareto_front()
        assert front.shape[0] == 1
        assert front[0].tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_front_spreads_when_the_objectives_are_in_tension(self, seed):
        landscape = small(conflict=1.0, seed=seed)
        assert landscape.exact_pareto_front().shape[0] > 1

    def test_partial_overlap_sits_between_the_extremes(self):
        # The dial has to be a dial, not a switch. Averaged over seeds, because
        # one draw of the motifs is one sample of a distributional claim.
        aligned = np.mean(front_sizes(0.0))
        partial = np.mean(front_sizes(0.5))
        contested = np.mean(front_sizes(1.0))
        assert aligned == 1.0
        assert aligned < partial < contested, f"{aligned} / {partial} / {contested}"

    def test_the_ideal_point_is_attainable_only_when_aligned(self):
        # The concrete statement of what conflict means: at zero, one sequence
        # maxes every objective; at one, generally none does.
        assert all(size == 1 for size in front_sizes(0.0))
        attained = [
            (small(conflict=1.0, seed=seed).exact_pareto_front() == 1.0).all(axis=1).any()
            for seed in SEEDS
        ]
        assert not any(attained)

    def test_conflict_outside_the_unit_interval_is_refused(self):
        for value in (-0.1, 1.5):
            with pytest.raises(ValueError, match="conflict must lie in"):
                small(conflict=value)

    def test_the_instance_reports_the_dial_it_was_built_with(self):
        # A benchmark that cannot state its parameters cannot be reproduced from
        # a results table.
        assert small(conflict=0.25, seed=0).conflict == pytest.approx(0.25)
        assert MultiEhrlichLandscape([ehrlich(seed=0)] * 2).conflict is None


class TestTheExactFront:
    def test_the_front_is_exactly_the_non_dominated_feasible_set(self):
        # Recomputed from scratch over the whole enumerated space: nothing
        # feasible dominates a front point, and every front point is really in
        # the space.
        landscape = small(conflict=1.0, seed=0)
        space = landscape.enumerate()
        values = landscape.evaluate(space)
        feasible = values[np.isfinite(values).all(axis=1)]
        expected = np.unique(feasible[non_dominated(feasible)], axis=0)
        assert np.array_equal(landscape.exact_pareto_front(), expected)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_nothing_in_the_space_dominates_a_front_point(self, seed):
        landscape = small(conflict=1.0, seed=seed)
        front = landscape.exact_pareto_front()
        values = landscape.evaluate(landscape.enumerate())
        for point in front:
            beats = (values >= point).all(axis=1) & (values > point).any(axis=1)
            assert not beats.any(), f"{point} is dominated by {values[beats][0]}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_feasible_design_is_covered_by_the_front(self, seed):
        landscape = small(conflict=1.0, seed=seed)
        front = landscape.exact_pareto_front()
        values = landscape.evaluate(landscape.enumerate())
        feasible = values[np.isfinite(values).all(axis=1)]
        covered = (front[None, :, :] >= feasible[:, None, :]).all(axis=2).any(axis=1)
        assert covered.all()

    def test_the_pareto_set_attains_the_front(self):
        landscape = small(conflict=1.0, seed=0)
        sequences, values = landscape.exact_pareto_set()
        assert sequences.shape[1] == landscape.sequence_length
        assert np.array_equal(landscape.evaluate(sequences), values)
        assert np.array_equal(np.unique(values, axis=0), landscape.exact_pareto_front())

    def test_it_can_be_restricted_to_a_candidate_set(self):
        # What a campaign under a mutation budget can actually attain is a subset
        # of the space, and regret against the unreachable front is not regret.
        landscape = small(conflict=1.0, seed=0)
        space = landscape.enumerate()
        subset = space[:200]
        restricted = landscape.exact_pareto_front(subset)
        values = landscape.evaluate(subset)
        feasible = values[np.isfinite(values).all(axis=1)]
        assert np.array_equal(restricted, np.unique(feasible[non_dominated(feasible)], axis=0))

    def test_a_search_set_with_nothing_feasible_has_an_empty_front(self):
        # Two infeasible designs do not dominate each other, so a naive
        # implementation would report both as "the front".
        landscape = small(conflict=1.0, transition_density=0.3, seed=1)
        space = landscape.enumerate()
        infeasible = space[~landscape.is_feasible(space)][:20]
        assert landscape.exact_pareto_front(infeasible).shape[0] == 0

    def test_asking_a_space_too_large_to_enumerate_is_refused(self):
        landscape = MultiEhrlichLandscape.with_conflict(sequence_length=32, vocab_size=20)
        with pytest.raises(ValueError, match="enumeration limit"):
            landscape.exact_pareto_front()

    def test_repeated_calls_agree(self):
        # The result is cached; a cache that hands out its own array would let a
        # caller edit the front in place.
        landscape = small(conflict=1.0, seed=0)
        first = landscape.exact_pareto_front()
        first[0, 0] = 99.0
        assert landscape.exact_pareto_front().max() <= 1.0


class TestReproducibility:
    def test_the_same_seed_gives_the_same_landscape(self):
        first, second = small(conflict=0.5, seed=3), small(conflict=0.5, seed=3)
        assert np.array_equal(first.optimal_sequences, second.optimal_sequences)
        assert np.array_equal(first.transition_matrix, second.transition_matrix)
        assert np.array_equal(first.exact_pareto_front(), second.exact_pareto_front())

    def test_different_seeds_give_different_landscapes(self):
        first, second = small(conflict=1.0, seed=0), small(conflict=1.0, seed=1)
        assert not np.array_equal(first.optimal_sequences, second.optimal_sequences)

    def test_accessors_return_copies(self):
        landscape = small(conflict=1.0, seed=0)
        landscape.transition_matrix[0, 0] = 99.0
        landscape.optimal_sequences[0, 0] = 2
        assert landscape.transition_matrix.max() <= 1.0
        for index, constituent in enumerate(landscape.landscapes):
            score = landscape.evaluate(constituent.optimal_sequence[None, :])[0, index]
            assert score == pytest.approx(1.0)


class TestInputValidation:
    def test_a_single_sequence_without_a_batch_dimension_is_rejected(self):
        landscape = small(conflict=1.0, seed=0)
        with pytest.raises(ValueError, match="ndim 2"):
            landscape.evaluate(np.zeros(8, dtype=np.int32))

    def test_wrong_sequence_length_is_rejected(self):
        with pytest.raises(ValueError, match="length 8"):
            small(conflict=1.0, seed=0).evaluate(np.zeros((1, 5), dtype=np.int32))

    def test_out_of_alphabet_tokens_are_rejected(self):
        with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
            small(conflict=1.0, seed=0).evaluate(np.full((1, 8), 7, dtype=np.int32))
