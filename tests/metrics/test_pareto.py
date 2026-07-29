"""Tests for the multi-objective indicators."""

import numpy as np
import pytest

from evogfn.metrics import (
    gd_plus,
    hypervolume,
    igd_plus,
    non_dominated,
    pareto_front,
    r2_indicator,
)
from evogfn.metrics.pareto import MAX_INCLUSION_EXCLUSION_POINTS


class TestNonDominated:
    def test_a_single_point_is_its_own_front(self):
        assert non_dominated(np.array([[1.0, 2.0]])).tolist() == [True]

    def test_a_dominated_point_is_excluded(self):
        # (1, 1) is beaten on both objectives by (2, 2).
        values = np.array([[2.0, 2.0], [1.0, 1.0]])
        assert non_dominated(values).tolist() == [True, False]

    def test_a_point_better_on_one_objective_survives(self):
        values = np.array([[2.0, 0.0], [0.0, 2.0]])
        assert non_dominated(values).tolist() == [True, True]

    def test_weak_dominance_does_not_exclude(self):
        # Equal on one objective and equal on the other: neither strictly beats
        # the other, so duplicates are both kept.
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert non_dominated(values).tolist() == [True, True]

    def test_a_tie_on_one_objective_is_broken_by_the_other(self):
        values = np.array([[1.0, 2.0], [1.0, 1.0]])
        assert non_dominated(values).tolist() == [True, False]

    def test_the_front_is_never_empty_for_a_non_empty_set(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            values = rng.normal(size=(20, 3))
            assert non_dominated(values).any()

    def test_no_front_point_is_dominated_by_any_other_point(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(40, 3))
        front = pareto_front(values)
        for point in front:
            beaten = (values >= point).all(axis=1) & (values > point).any(axis=1)
            assert not beaten.any()

    def test_the_front_keeps_the_input_order(self):
        values = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        assert pareto_front(values) == pytest.approx(values)

    def test_infeasible_designs_are_dominated(self):
        values = np.array([[1.0, 1.0], [-np.inf, -np.inf]])
        assert non_dominated(values).tolist() == [True, False]

    def test_a_one_dimensional_input_is_refused_as_ambiguous(self):
        with pytest.raises(ValueError, match="ambiguous"):
            non_dominated(np.array([1.0, 2.0]))

    def test_missing_measurements_are_refused(self):
        with pytest.raises(ValueError, match="undefined"):
            non_dominated(np.array([[1.0, np.nan]]))


class TestHypervolume:
    def test_a_single_point_encloses_its_box(self):
        assert hypervolume(np.array([[3.0, 2.0]]), reference=[0.0, 0.0]) == pytest.approx(6.0)

    def test_a_known_two_objective_front(self):
        # Rectangles: 3x1 + 2x1 + 1x1 = 6, checkable by hand.
        values = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        assert hypervolume(values, reference=[0.0, 0.0]) == pytest.approx(6.0)

    def test_a_known_three_objective_front(self):
        # Two boxes of volume 2 overlapping in a unit cube: 2 + 2 - 1 = 3.
        values = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]])
        assert hypervolume(values, reference=[0.0, 0.0, 0.0]) == pytest.approx(3.0)

    def test_one_objective_is_the_gap_to_the_best_design(self):
        assert hypervolume(np.array([[1.0], [4.0]]), reference=[0.5]) == pytest.approx(3.5)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_adding_a_dominating_point_never_lowers_it(self, seed):
        # The property that makes hypervolume worth reporting: it is monotone
        # with respect to Pareto dominance. Any implementation that can decrease
        # here is wrong regardless of how plausible its numbers look.
        rng = np.random.default_rng(seed)
        values = rng.uniform(0.0, 1.0, size=(8, 2))
        reference = [-0.1, -0.1]
        before = hypervolume(values, reference=reference)
        dominating = values.max(axis=0) + 0.5
        after = hypervolume(np.vstack([values, dominating]), reference=reference)
        assert after > before

    @pytest.mark.parametrize("seed", [0, 1])
    def test_adding_a_dominated_point_leaves_it_unchanged(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.uniform(1.0, 2.0, size=(6, 3))
        reference = [0.0, 0.0, 0.0]
        before = hypervolume(values, reference=reference)
        dominated = values.min(axis=0) - 0.5
        after = hypervolume(np.vstack([values, dominated]), reference=reference)
        assert after == pytest.approx(before)

    def test_the_two_dimensional_sweep_agrees_with_inclusion_exclusion(self):
        # The two objective case uses a different algorithm from the general one.
        # Padding a 2D front with a constant third objective makes the two
        # comparable: the extra unit thickness leaves the volume unchanged.
        rng = np.random.default_rng(3)
        values = rng.uniform(0.0, 1.0, size=(7, 2))
        swept = hypervolume(values, reference=[0.0, 0.0])
        padded = np.hstack([values, np.ones((values.shape[0], 1))])
        sliced = hypervolume(padded, reference=[0.0, 0.0, 0.0])
        assert sliced == pytest.approx(swept)

    def test_designs_that_do_not_beat_the_reference_contribute_nothing(self):
        values = np.array([[3.0, 2.0], [-1.0, 5.0]])
        with_bad = hypervolume(values, reference=[0.0, 0.0])
        without = hypervolume(values[:1], reference=[0.0, 0.0])
        assert with_bad == pytest.approx(without)

    def test_it_is_zero_when_nothing_beats_the_reference(self):
        assert hypervolume(np.array([[1.0, 1.0]]), reference=[5.0, 5.0]) == 0.0

    def test_infeasible_designs_contribute_nothing(self):
        values = np.array([[3.0, 2.0], [-np.inf, -np.inf]])
        assert hypervolume(values, reference=[0.0, 0.0]) == pytest.approx(6.0)

    def test_a_large_front_in_three_objectives_is_refused_not_approximated(self):
        # Numerical honesty: past the exact method's limit this must say so
        # rather than return a plausible wrong number.
        k = MAX_INCLUSION_EXCLUSION_POINTS + 1
        i = np.arange(k, dtype=np.float64)
        values = np.stack([i + 1.0, float(k) - i, np.ones(k)], axis=1)
        with pytest.raises(NotImplementedError, match="inclusion-exclusion"):
            hypervolume(values, reference=[-1.0, -1.0, -1.0])

    def test_a_front_at_the_limit_is_still_computed(self):
        k = MAX_INCLUSION_EXCLUSION_POINTS
        i = np.arange(k, dtype=np.float64)
        values = np.stack([i + 1.0, float(k) - i, np.ones(k)], axis=1)
        assert hypervolume(values, reference=[0.0, 0.0, 0.0]) > 0.0

    def test_many_points_in_two_objectives_are_not_refused(self):
        # The limit belongs to the general method, not to the sweep.
        rng = np.random.default_rng(4)
        values = rng.uniform(size=(500, 2))
        assert hypervolume(values, reference=[0.0, 0.0]) > 0.0

    def test_a_reference_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            hypervolume(np.array([[1.0, 2.0]]), reference=[0.0, 0.0, 0.0])

    def test_a_non_finite_reference_is_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            hypervolume(np.array([[1.0, 2.0]]), reference=[-np.inf, 0.0])


class TestR2Indicator:
    def test_a_set_containing_the_ideal_scores_zero(self):
        values = np.array([[1.0, 1.0], [0.5, 0.5]])
        weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        assert r2_indicator(values, weights, ideal=[1.0, 1.0]) == pytest.approx(0.0)

    def test_lower_is_better(self):
        weights = np.array([[0.5, 0.5]])
        close = np.array([[0.9, 0.9]])
        far = np.array([[0.1, 0.1]])
        ideal = [1.0, 1.0]
        assert r2_indicator(close, weights, ideal=ideal) < r2_indicator(far, weights, ideal=ideal)

    def test_a_known_value(self):
        # max_i λ_i |z*_i - a_i| = max(0.5 · 0.4, 0.5 · 0.2) = 0.2, one weight
        # vector so the average is that.
        values = np.array([[0.6, 0.8]])
        assert r2_indicator(values, [[0.5, 0.5]], ideal=[1.0, 1.0]) == pytest.approx(0.2)

    def test_adding_a_point_can_only_help(self):
        # The min over designs is taken per weight vector, so a new design can
        # improve some directions and worsen none.
        rng = np.random.default_rng(5)
        values = rng.uniform(size=(6, 2))
        weights = np.array([[w, 1.0 - w] for w in np.linspace(0.0, 1.0, 11)])
        extra = np.vstack([values, rng.uniform(size=(1, 2))])
        ideal = [1.0, 1.0]
        assert (
            r2_indicator(extra, weights, ideal=ideal)
            <= r2_indicator(values, weights, ideal=ideal) + 1e-12
        )

    def test_a_single_weight_vector_is_accepted(self):
        assert r2_indicator(np.array([[0.6, 0.8]]), [0.5, 0.5], ideal=[1.0, 1.0]) == pytest.approx(
            0.2
        )

    def test_the_ideal_defaults_to_the_best_seen_per_objective(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert r2_indicator(values, [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(0.0)

    def test_an_infeasible_design_never_wins_a_direction(self):
        # 0 · inf is nan, and a nan utility would silently make the infeasible
        # design the best in that direction.
        values = np.array([[0.6, 0.8], [-np.inf, -np.inf]])
        assert r2_indicator(values, [[1.0, 0.0]], ideal=[1.0, 1.0]) == pytest.approx(0.4)

    def test_an_empty_set_is_refused(self):
        with pytest.raises(ValueError, match="empty set"):
            r2_indicator(np.zeros((0, 2)), [[0.5, 0.5]])

    def test_weights_off_the_simplex_are_refused(self):
        with pytest.raises(ValueError, match="sum to 1"):
            r2_indicator(np.array([[1.0, 1.0]]), [[0.5, 0.2]])

    def test_weights_of_the_wrong_width_are_refused(self):
        with pytest.raises(ValueError, match=r"shape \(m, 2\)"):
            r2_indicator(np.array([[1.0, 1.0]]), [[0.5, 0.25, 0.25]])


class TestIgdPlus:
    def test_a_set_covering_the_reference_front_scores_zero(self):
        front = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert igd_plus(front, front) == pytest.approx(0.0)

    def test_overshooting_the_reference_front_is_not_penalised(self):
        # The whole point of the "+" modification (Ishibuchi et al., 2015): plain
        # IGD charges Euclidean distance and would rank this *worse* than the
        # front it dominates.
        front = np.array([[1.0, 1.0]])
        better = np.array([[3.0, 3.0]])
        assert igd_plus(better, front) == pytest.approx(0.0)

    def test_falling_short_is_charged(self):
        front = np.array([[1.0, 1.0]])
        worse = np.array([[0.0, 1.0]])
        assert igd_plus(worse, front) == pytest.approx(1.0)

    def test_it_charges_only_the_objectives_that_fall_short(self):
        # Shortfall of 1 on the first objective, an overshoot of 5 on the second.
        front = np.array([[1.0, 1.0]])
        values = np.array([[0.0, 6.0]])
        assert igd_plus(values, front) == pytest.approx(1.0)

    def test_a_dominating_set_never_scores_worse(self):
        rng = np.random.default_rng(6)
        front = rng.uniform(size=(5, 2))
        values = rng.uniform(size=(7, 2))
        dominating = values + 0.3
        assert igd_plus(dominating, front) <= igd_plus(values, front) + 1e-12

    def test_it_notices_a_front_that_is_only_partly_covered(self):
        # One excellent design covering one end of the front: convergence is
        # perfect, coverage is not, and this is the metric that says so.
        front = np.array([[1.0, 0.0], [0.0, 1.0]])
        one_corner = np.array([[1.0, 0.0]])
        assert igd_plus(one_corner, front) > 0.0

    def test_an_empty_reference_front_is_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            igd_plus(np.array([[1.0, 1.0]]), np.zeros((0, 2)))

    def test_a_non_finite_reference_front_is_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            igd_plus(np.array([[1.0, 1.0]]), np.array([[np.inf, 1.0]]))

    def test_mismatched_objective_counts_are_refused(self):
        with pytest.raises(ValueError, match="same objectives"):
            igd_plus(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0, 1.0]]))


class TestGdPlus:
    def test_a_design_on_the_front_scores_zero(self):
        front = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert gd_plus(np.array([[1.0, 0.0]]), front) == pytest.approx(0.0)

    def test_it_measures_convergence_and_not_coverage(self):
        # The complement of the IGD+ test above: one corner converges perfectly
        # and covers nothing, and GD+ cannot see the difference. Reporting either
        # alone hides half the picture.
        front = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert gd_plus(np.array([[1.0, 0.0]]), front) == pytest.approx(0.0)
        assert igd_plus(np.array([[1.0, 0.0]]), front) > 0.0

    def test_a_design_below_the_front_is_charged(self):
        front = np.array([[1.0, 1.0]])
        assert gd_plus(np.array([[1.0, 0.0]]), front) == pytest.approx(1.0)
