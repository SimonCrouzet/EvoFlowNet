"""Tests for multi-objective scalarisation and the reward built on it."""

import numpy as np
import pytest

from evogfn.rewards import (
    ScalarizedReward,
    Tchebycheff,
    TemperedReward,
    WeightedLogSum,
    WeightedSum,
)
from evogfn.rewards.base import Reward

ALL_SCALARIZATIONS = [WeightedSum(), Tchebycheff(), WeightedLogSum()]


class TestOneHotPreferenceReducesToOneObjective:
    """The defining property: ω = e_j must mean "optimise objective j alone"."""

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_a_one_hot_preference_returns_that_objective(self, scalarization, index):
        values = np.array([[0.2, 0.7, 0.5], [1.0, 0.1, 0.9], [0.4, 0.4, 0.4]])
        preference = np.eye(3)[index]
        assert scalarization.scalarize(values, preference) == pytest.approx(values[:, index])

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_single_objective_problem_is_the_identity(self, scalarization):
        # A one-objective landscape scalarised with ω = [1] must be exactly the
        # single-objective path, or multi-objective code could not be the default.
        values = np.array([[0.5], [2.0], [7.0]])
        assert scalarization.scalarize(values, [1.0]) == pytest.approx(values[:, 0])

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_zero_weighted_infeasible_objective_does_not_poison_the_result(self, scalarization):
        # An infeasible objective scores -inf and 0 * -inf is nan, so the naive
        # implementation returns "unmeasured" for a design the user asked to
        # ignore that objective for.
        values = np.array([[1.0, -np.inf], [4.0, -np.inf]])
        scalarized = scalarization.scalarize(values, [1.0, 0.0])
        assert np.isfinite(scalarized).all()
        assert scalarized == pytest.approx([1.0, 4.0])


class TestWeightedSum:
    def test_it_is_the_linear_combination(self):
        values = np.array([[1.0, 3.0], [2.0, 0.0]])
        expected = 0.25 * values[:, 0] + 0.75 * values[:, 1]
        assert WeightedSum().scalarize(values, [0.25, 0.75]) == pytest.approx(expected)

    def test_it_cannot_select_a_point_in_a_concave_dent(self):
        # Miettinen (1999), Thm 3.1.4: a weighted sum only reaches the convex hull
        # of the front. C is non-dominated but never optimal at any preference,
        # which is the reason Tchebycheff exists.
        values = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]])
        for w in np.linspace(0.0, 1.0, 21):
            scalarized = WeightedSum().scalarize(values, [w, 1.0 - w])
            assert int(np.argmax(scalarized)) != 2

    def test_it_is_monotone_in_every_objective(self):
        better = np.array([[1.0, 2.0]])
        worse = np.array([[1.0, 1.0]])
        preference = [0.5, 0.5]
        assert WeightedSum().scalarize(better, preference) > WeightedSum().scalarize(
            worse, preference
        )


class TestTchebycheff:
    def test_it_selects_a_point_a_weighted_sum_cannot(self):
        # The same concave dent as above: at an even preference the balanced
        # design wins, which is what makes the whole front reachable.
        values = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]])
        scalarized = Tchebycheff().scalarize(values, [0.5, 0.5])
        assert int(np.argmax(scalarized)) == 2

    def test_it_is_the_smallest_weighted_improvement_over_the_reference(self):
        values = np.array([[3.0, 5.0]])
        # min(0.5 * |3 - 1|, 0.5 * |5 - 1|) = min(1.0, 2.0)
        scalarized = Tchebycheff(reference=[1.0, 1.0]).scalarize(values, [0.5, 0.5])
        assert scalarized == pytest.approx([1.0])

    def test_the_reference_shifts_the_scalarised_value(self):
        values = np.array([[3.0, 5.0]])
        at_zero = Tchebycheff().scalarize(values, [0.5, 0.5])
        at_one = Tchebycheff(reference=[1.0, 1.0]).scalarize(values, [0.5, 0.5])
        assert at_one < at_zero

    def test_a_reference_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="reference has 3 entries"):
            Tchebycheff(reference=[0.0, 0.0, 0.0]).scalarize(np.zeros((2, 2)), [0.5, 0.5])

    def test_a_non_finite_reference_is_refused(self):
        with pytest.raises(ValueError, match="reference must be finite"):
            Tchebycheff(reference=[0.0, -np.inf])

    def test_a_two_dimensional_reference_is_refused(self):
        with pytest.raises(ValueError, match="ndim 1"):
            Tchebycheff(reference=[[0.0, 0.0]])


class TestWeightedLogSum:
    def test_it_is_the_weighted_geometric_mean(self):
        values = np.array([[4.0, 9.0]])
        assert WeightedLogSum().scalarize(values, [0.5, 0.5]) == pytest.approx([6.0])

    def test_a_dead_objective_cannot_be_compensated(self):
        # The substantive difference from a weighted sum: an unfoldable variant
        # with excellent predicted binding is not a partial success.
        values = np.array([[0.0, 100.0], [1.0, 1.0]])
        scalarized = WeightedLogSum(floor=1e-8).scalarize(values, [0.5, 0.5])
        assert scalarized[0] < scalarized[1]

    def test_the_floor_keeps_a_dead_objective_finite(self):
        scalarized = WeightedLogSum(floor=1e-4).scalarize(np.array([[0.0, 1.0]]), [0.5, 0.5])
        assert np.isfinite(scalarized).all()
        assert scalarized == pytest.approx([np.sqrt(1e-4)])

    def test_a_zero_floor_is_refused(self):
        with pytest.raises(ValueError, match="floor must be positive"):
            WeightedLogSum(floor=0.0)


class TestPreferenceValidation:
    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_negative_preference_is_refused(self, scalarization):
        with pytest.raises(ValueError, match="non-negative"):
            scalarization.scalarize(np.ones((2, 2)), [1.5, -0.5])

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_preference_that_does_not_sum_to_one_is_refused(self, scalarization):
        with pytest.raises(ValueError, match="must sum to 1"):
            scalarization.scalarize(np.ones((2, 2)), [0.5, 0.2])

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_preference_of_the_wrong_width_is_refused(self, scalarization):
        with pytest.raises(ValueError, match="covers 3 objectives"):
            scalarization.scalarize(np.ones((2, 2)), [1 / 3, 1 / 3, 1 / 3])

    def test_rounding_in_a_written_down_preference_is_tolerated(self):
        # A preference typed into a config file sums to 0.999999...; rejecting it
        # would be pedantry, not safety.
        scalarized = WeightedSum().scalarize(np.ones((1, 3)), [0.333333, 0.333333, 0.333334])
        assert scalarized == pytest.approx([1.0])

    def test_one_preference_per_design_is_accepted(self):
        # MOGFN-PC draws a preference per trajectory, so a batch of preferences
        # has to be as valid as a single shared one.
        values = np.array([[1.0, 0.0], [1.0, 0.0]])
        preferences = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert WeightedSum().scalarize(values, preferences) == pytest.approx([1.0, 0.0])

    def test_a_batch_of_preferences_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="for 3 designs"):
            WeightedSum().scalarize(np.ones((3, 2)), np.array([[1.0, 0.0], [0.0, 1.0]]))


class TestObjectiveValidation:
    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_a_one_dimensional_input_is_refused_as_ambiguous(self, scalarization):
        with pytest.raises(ValueError, match="ambiguous"):
            scalarization.scalarize(np.array([1.0, 2.0]), [1.0])

    @pytest.mark.parametrize("scalarization", ALL_SCALARIZATIONS)
    def test_missing_measurements_are_refused(self, scalarization):
        with pytest.raises(ValueError, match="missing measurement"):
            scalarization.scalarize(np.array([[1.0, np.nan]]), [0.5, 0.5])

    def test_a_matrix_with_no_objectives_is_refused(self):
        with pytest.raises(ValueError, match="at least one objective"):
            WeightedSum().scalarize(np.zeros((3, 0)), [])


class TestScalarizedReward:
    def test_it_is_a_reward_like_any_other(self):
        # The point of the composition: nothing in the training loop needs to
        # know there was ever more than one objective.
        reward = ScalarizedReward(WeightedSum(), [0.5, 0.5])
        assert isinstance(reward, Reward)

    def test_it_is_the_scalar_reward_of_the_scalarised_values(self):
        values = np.array([[1.0, 3.0], [2.0, 2.0]])
        scalarization = WeightedSum()
        preference = [0.5, 0.5]
        tempered = TemperedReward(beta=3.0)
        reward = ScalarizedReward(scalarization, preference, reward=tempered)
        expected = tempered.log_reward(scalarization.scalarize(values, preference))
        assert reward.log_reward(values) == pytest.approx(expected)

    def test_a_one_hot_preference_matches_the_single_objective_reward(self):
        values = np.array([[0.5, 99.0], [4.0, 0.0]])
        reward = ScalarizedReward(WeightedSum(), [1.0, 0.0], reward=TemperedReward(beta=2.0))
        assert reward.log_reward(values) == pytest.approx(
            TemperedReward(beta=2.0).log_reward(values[:, 0])
        )

    def test_log_rewards_stay_finite_on_dead_and_infeasible_designs(self):
        values = np.array([[0.0, 0.0], [-np.inf, 1.0], [-1.0, -1.0]])
        for scalarization in ALL_SCALARIZATIONS:
            reward = ScalarizedReward(scalarization, [0.5, 0.5])
            assert np.isfinite(reward.log_reward(values)).all()

    def test_the_preference_can_be_swapped_without_rebuilding_the_reward(self):
        # MOGFN-PC resamples the preference every batch.
        reward = ScalarizedReward(WeightedSum(), [1.0, 0.0])
        swapped = reward.with_preference([0.0, 1.0])
        values = np.array([[1.0, 4.0]])
        assert swapped.log_reward(values) == pytest.approx(np.log([4.0]))
        assert reward.log_reward(values) == pytest.approx(np.log([1.0]))

    def test_swapping_keeps_the_scalarisation_and_the_scalar_reward(self):
        reward = ScalarizedReward(
            Tchebycheff(), [1.0, 0.0], reward=TemperedReward(beta=2.0, floor=1e-3)
        )
        swapped = reward.with_preference([0.5, 0.5])
        assert swapped.scalarization is reward.scalarization
        assert swapped.scalar_reward is reward.scalar_reward

    def test_the_preference_cannot_be_mutated_through_the_property(self):
        reward = ScalarizedReward(WeightedSum(), [1.0, 0.0])
        reward.preference[0] = 0.0
        assert reward.preference == pytest.approx([1.0, 0.0])

    def test_an_invalid_preference_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            ScalarizedReward(WeightedSum(), [0.5, 0.9])

    def test_a_two_dimensional_preference_is_refused_at_construction(self):
        # The reward carries one preference; a batch of them belongs to the
        # caller resampling per batch, not to a single reward object.
        with pytest.raises(ValueError, match="ndim 1"):
            ScalarizedReward(WeightedSum(), [[1.0, 0.0]])
