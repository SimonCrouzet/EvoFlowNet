"""Tests for performance metrics.

Expected values are computed by hand from the definitions in the source papers,
not read off the implementation.
"""

import numpy as np
import pytest

from evoflownet.metrics import (
    cumulative_regret,
    feasible_fraction,
    simple_regret,
    top_k_performance,
)


class TestSimpleRegret:
    def test_it_is_the_gap_from_the_best_found_to_the_optimum(self):
        assert simple_regret(np.array([0.2, 0.7, 0.5]), optimum=1.0) == pytest.approx(0.3)

    def test_finding_the_optimum_gives_zero(self):
        assert simple_regret(np.array([0.1, 1.0]), optimum=1.0) == pytest.approx(0.0)

    def test_a_single_objective_column_is_accepted(self):
        assert simple_regret(np.array([[0.2], [0.7]]), optimum=1.0) == pytest.approx(0.3)

    def test_infeasible_designs_are_ignored_not_averaged(self):
        # -inf marks a design that cannot be built; it is not a bad score.
        values = np.array([-np.inf, 0.6, -np.inf])
        assert simple_regret(values, optimum=1.0) == pytest.approx(0.4)

    def test_producing_nothing_feasible_is_infinite_regret(self):
        # Reported as inf rather than as a number that would average misleadingly.
        assert simple_regret(np.array([-np.inf, -np.inf]), optimum=1.0) == float("inf")

    def test_multi_objective_input_is_refused(self):
        with pytest.raises(ValueError, match="must be scalarised"):
            simple_regret(np.zeros((3, 2)), optimum=1.0)


class TestCumulativeRegret:
    def test_it_sums_the_regret_of_the_running_best(self):
        # Running best: 0.2, 0.7, 0.7 -> regrets 0.8, 0.3, 0.3 -> 1.4
        values = np.array([0.2, 0.7, 0.5])
        assert cumulative_regret(values, optimum=1.0) == pytest.approx(1.4)

    def test_finding_the_optimum_sooner_costs_less(self):
        early = cumulative_regret(np.array([1.0, 0.1, 0.1, 0.1]), optimum=1.0)
        late = cumulative_regret(np.array([0.1, 0.1, 0.1, 1.0]), optimum=1.0)
        assert early < late
        assert early == pytest.approx(0.0)

    def test_order_matters_unlike_simple_regret(self):
        forwards = np.array([0.1, 0.9])
        backwards = np.array([0.9, 0.1])
        assert simple_regret(forwards, 1.0) == simple_regret(backwards, 1.0)
        assert cumulative_regret(forwards, 1.0) != cumulative_regret(backwards, 1.0)

    def test_it_is_undefined_with_nothing_finite(self):
        with pytest.raises(ValueError, match="undefined"):
            cumulative_regret(np.array([-np.inf]), optimum=1.0)


class TestTopKPerformance:
    def test_it_averages_the_best_k(self):
        # top 2 of [0.1, 0.9, 0.5, 0.7] are 0.9 and 0.7 -> 0.8
        assert top_k_performance(np.array([0.1, 0.9, 0.5, 0.7]), k=2) == pytest.approx(0.8)

    def test_it_is_a_mean_not_a_maximum(self):
        # One excellent design among poor ones is a worse round than several
        # good ones, and the metric has to say so.
        spiky = top_k_performance(np.array([1.0, 0.0, 0.0, 0.0]), k=4)
        even = top_k_performance(np.array([0.4, 0.4, 0.4, 0.4]), k=4)
        assert spiky < even

    def test_k_larger_than_the_batch_uses_the_whole_batch(self):
        assert top_k_performance(np.array([0.2, 0.4]), k=10) == pytest.approx(0.3)

    def test_infeasible_designs_are_excluded(self):
        assert top_k_performance(np.array([-np.inf, 0.5, 0.3]), k=2) == pytest.approx(0.4)

    def test_a_nonpositive_k_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            top_k_performance(np.array([0.5]), k=0)


class TestFeasibleFraction:
    def test_it_is_the_share_that_are_constructible(self):
        assert feasible_fraction(np.array([True, True, False, False])) == pytest.approx(0.5)

    def test_all_feasible_is_one(self):
        assert feasible_fraction(np.ones(5, dtype=bool)) == pytest.approx(1.0)

    def test_an_empty_batch_is_zero_not_an_error(self):
        assert feasible_fraction(np.zeros(0, dtype=bool)) == 0.0
