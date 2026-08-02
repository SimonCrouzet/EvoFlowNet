"""Tests for the tempered reward transform."""

import numpy as np
import pytest

from evogfn.metrics import target_distribution
from evogfn.rewards import TemperedReward


class TestLogReward:
    def test_at_beta_one_it_is_the_log_of_the_value(self):
        reward = TemperedReward(beta=1.0)
        values = np.array([1.0, np.e, 10.0])
        assert reward.log_reward(values) == pytest.approx(np.log(values))

    def test_beta_scales_the_log_reward(self):
        values = np.array([2.0, 5.0])
        assert TemperedReward(beta=3.0).log_reward(values) == pytest.approx(
            3.0 * TemperedReward(beta=1.0).log_reward(values)
        )

    def test_beta_zero_makes_every_design_equally_rewarded(self):
        reward = TemperedReward(beta=0.0)
        assert reward.log_reward(np.array([0.1, 1.0, 99.0])) == pytest.approx(np.zeros(3))

    def test_a_single_objective_column_is_accepted(self):
        reward = TemperedReward()
        assert reward.log_reward(np.array([[2.0], [4.0]])) == pytest.approx(
            reward.log_reward(np.array([2.0, 4.0]))
        )

    def test_reward_is_the_exponential_of_log_reward(self):
        reward = TemperedReward(beta=2.0)
        values = np.array([1.5, 3.0])
        assert reward.reward(values) == pytest.approx(values**2)


class TestFiniteness:
    def test_dead_designs_get_the_floor_not_negative_infinity(self):
        # ~20% of measured GB1 variants are exactly zero, so this is the common
        # case; -inf would reach the loss as nan at the first subtraction.
        reward = TemperedReward(beta=1.0, floor=1e-8)
        assert reward.log_reward(np.array([0.0])) == pytest.approx(np.log(1e-8))

    def test_infeasible_designs_get_the_floor(self):
        reward = TemperedReward(beta=2.0, floor=1e-6)
        assert reward.log_reward(np.array([-np.inf])) == pytest.approx(2.0 * np.log(1e-6))

    def test_negative_values_get_the_floor(self):
        reward = TemperedReward(floor=1e-4)
        assert reward.log_reward(np.array([-5.0])) == pytest.approx(np.log(1e-4))

    @pytest.mark.parametrize("beta", [0.0, 1.0, 3.0, 16.0])
    def test_log_rewards_are_always_finite(self, beta):
        # The whole point of the floor. Any non-finite entry here becomes nan in
        # the trajectory balance loss.
        reward = TemperedReward(beta=beta)
        awkward = np.array([-np.inf, -1.0, 0.0, 1e-30, 1.0, 1e6])
        assert np.isfinite(reward.log_reward(awkward)).all()

    def test_missing_measurements_are_refused_rather_than_floored(self):
        # nan is not a small value, it is an absent one. Flooring it would
        # silently present an unmeasured design as a dead one.
        with pytest.raises(ValueError, match="missing measurement"):
            TemperedReward().log_reward(np.array([1.0, np.nan]))


class TestOrdering:
    def test_the_transform_is_monotonic_above_the_floor(self):
        # Tempering may sharpen the target but must not reorder designs.
        reward = TemperedReward(beta=3.0)
        values = np.array([0.1, 0.5, 1.0, 2.0, 8.0])
        assert np.all(np.diff(reward.log_reward(values)) > 0)

    def test_the_best_design_stays_best_at_every_beta(self):
        values = np.array([0.2, 9.0, 1.0])
        for beta in (0.5, 1.0, 3.0, 10.0):
            assert int(np.argmax(TemperedReward(beta=beta).log_reward(values))) == 1


class TestAgreementWithTheTargetDistribution:
    """The reward and the metric must describe the same target."""

    @pytest.mark.parametrize("beta", [1.0, 3.0])
    def test_softmax_of_log_reward_matches_the_metric_target(self, beta):
        # metrics.target_distribution computes p*(x) for evaluation; the reward
        # defines what training aims at. If these disagreed, the model would be
        # measured against a distribution it was never asked to learn.
        values = np.array([0.5, 1.0, 4.0, 2.0])
        log_rewards = TemperedReward(beta=beta).log_reward(values)
        shifted = np.exp(log_rewards - log_rewards.max())
        from_reward = shifted / shifted.sum()
        assert from_reward == pytest.approx(target_distribution(values, beta=beta))

    def test_a_larger_beta_concentrates_the_target(self):
        values = np.array([1.0, 2.0])
        best_mass = []
        for beta in (1.0, 4.0):
            log_rewards = TemperedReward(beta=beta).log_reward(values)
            weights = np.exp(log_rewards - log_rewards.max())
            best_mass.append((weights / weights.sum())[1])
        assert best_mass[1] > best_mass[0]

    @pytest.mark.parametrize("floor", [1e-8, 1e-3])
    def test_they_agree_on_dead_designs_when_the_floor_is_passed_through(self, floor):
        # The two differ in what they give a dead design: the reward floors it,
        # the metric zeroes it by default. At a tiny floor the gap is invisible,
        # but at a larger one it is not -- so the contract is that passing the
        # floor as min_reward makes them identical. Roughly a fifth of measured
        # GB1 variants are exactly zero, so this is not a corner case.
        values = np.array([0.0, -np.inf, 1.0, 4.0])
        log_rewards = TemperedReward(beta=1.0, floor=floor).log_reward(values)
        weights = np.exp(log_rewards - log_rewards.max())
        from_reward = weights / weights.sum()
        assert from_reward == pytest.approx(target_distribution(values, beta=1.0, min_reward=floor))

    def test_a_large_floor_visibly_diverges_from_the_metric_default(self):
        # Guards the test above from being vacuous: if the two always agreed
        # regardless, passing the floor through would prove nothing.
        values = np.array([0.0, 1.0])
        log_rewards = TemperedReward(beta=1.0, floor=0.5).log_reward(values)
        weights = np.exp(log_rewards - log_rewards.max())
        from_reward = weights / weights.sum()
        assert from_reward != pytest.approx(target_distribution(values, beta=1.0))


class TestValidation:
    def test_a_negative_beta_is_refused(self):
        with pytest.raises(ValueError, match="beta must be non-negative"):
            TemperedReward(beta=-1.0)

    def test_a_zero_floor_is_refused(self):
        with pytest.raises(ValueError, match="floor must be positive"):
            TemperedReward(floor=0.0)

    def test_multi_objective_input_is_refused(self):
        with pytest.raises(ValueError, match="need a scalarisation"):
            TemperedReward().log_reward(np.zeros((3, 2)))

    def test_the_configuration_is_readable(self):
        reward = TemperedReward(beta=2.5, floor=1e-5)
        assert reward.beta == 2.5
        assert reward.floor == 1e-5
