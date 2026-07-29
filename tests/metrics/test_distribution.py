"""Tests for exact distribution comparison.

This is the metric a hill-climber cannot pass, so the tests include a
demonstration that it separates a proportional sampler from a greedy one.
"""

import numpy as np
import pytest

from evogfn.metrics import (
    empirical_distribution,
    expected_l1_from_sampling_noise,
    l1_distance,
    target_distribution,
)


class TestTargetDistribution:
    def test_it_normalises_to_one(self):
        target = target_distribution(np.array([1.0, 2.0, 3.0]))
        assert target.sum() == pytest.approx(1.0)

    def test_at_beta_one_it_is_proportional_to_reward(self):
        target = target_distribution(np.array([1.0, 2.0, 3.0]), beta=1.0)
        assert target == pytest.approx(np.array([1 / 6, 2 / 6, 3 / 6]))

    def test_beta_concentrates_on_the_best(self):
        rewards = np.array([1.0, 2.0])
        flat = target_distribution(rewards, beta=1.0)
        peaked = target_distribution(rewards, beta=8.0)
        assert peaked[1] > flat[1]
        assert peaked[1] > 0.99

    def test_beta_zero_is_uniform(self):
        target = target_distribution(np.array([1.0, 5.0, 100.0]), beta=0.0)
        assert target == pytest.approx(np.full(3, 1 / 3))

    def test_a_single_objective_column_is_accepted(self):
        assert target_distribution(np.array([[1.0], [3.0]])) == pytest.approx([0.25, 0.75])

    def test_infeasible_and_zero_designs_get_no_mass(self):
        target = target_distribution(np.array([-np.inf, 0.0, 4.0]))
        assert target == pytest.approx([0.0, 0.0, 1.0])

    def test_a_large_beta_does_not_overflow(self):
        # Computed in log space: naive exponentiation of a wide dynamic range
        # would overflow to inf/inf and produce NaN.
        target = target_distribution(np.array([1.0, 1e3]), beta=100.0)
        assert np.isfinite(target).all()
        assert target.sum() == pytest.approx(1.0)

    def test_an_all_zero_landscape_is_refused(self):
        with pytest.raises(ValueError, match="undefined"):
            target_distribution(np.zeros(4))


class TestEmpiricalDistribution:
    def test_frequencies_match_the_enumeration_order(self):
        space = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
        samples = np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int32)
        assert empirical_distribution(samples, space) == pytest.approx([0, 2 / 3, 0, 1 / 3])

    def test_it_normalises_to_one(self):
        space = np.array([[0], [1]], dtype=np.int32)
        samples = np.array([[0], [1], [1]], dtype=np.int32)
        assert empirical_distribution(samples, space).sum() == pytest.approx(1.0)

    def test_no_samples_gives_all_zeros(self):
        space = np.array([[0], [1]], dtype=np.int32)
        empty = np.zeros((0, 1), dtype=np.int32)
        assert empirical_distribution(empty, space).tolist() == [0.0, 0.0]

    def test_a_sample_outside_the_space_is_refused(self):
        # Silently dropping it would make the comparison quietly wrong.
        space = np.array([[0], [1]], dtype=np.int32)
        with pytest.raises(ValueError, match="do not describe the same landscape"):
            empirical_distribution(np.array([[7]], dtype=np.int32), space)


class TestL1Distance:
    def test_identical_distributions_are_zero_apart(self):
        p = np.array([0.25, 0.75])
        assert l1_distance(p, p) == pytest.approx(0.0)

    def test_disjoint_distributions_are_two_apart(self):
        assert l1_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(2.0)

    def test_it_is_the_sum_of_absolute_differences(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.25, 0.75])
        assert l1_distance(a, b) == pytest.approx(0.5)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes"):
            l1_distance(np.array([0.5, 0.5]), np.array([1.0]))


class TestSamplingNoiseFloor:
    def test_the_floor_shrinks_as_samples_grow(self):
        target = target_distribution(np.arange(1.0, 17.0))
        few = expected_l1_from_sampling_noise(target, n_samples=50)
        many = expected_l1_from_sampling_noise(target, n_samples=5000)
        assert few > many > 0.0

    def test_it_is_reproducible(self):
        target = target_distribution(np.arange(1.0, 9.0))
        a = expected_l1_from_sampling_noise(target, n_samples=100, seed=3)
        b = expected_l1_from_sampling_noise(target, n_samples=100, seed=3)
        assert a == b

    def test_a_nonpositive_sample_count_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            expected_l1_from_sampling_noise(np.array([0.5, 0.5]), n_samples=0)


class TestItSeparatesSamplingFromOptimising:
    """The reason this module exists."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.space = np.arange(64, dtype=np.int32)[:, None]
        self.values = rng.uniform(0.1, 1.0, size=64)
        self.target = target_distribution(self.values, beta=1.0)
        self.rng = rng

    def test_a_proportional_sampler_scores_near_the_noise_floor(self):
        n = 20_000
        drawn = self.rng.choice(64, size=n, p=self.target)
        empirical = empirical_distribution(self.space[drawn], self.space)
        distance = l1_distance(empirical, self.target)
        floor = expected_l1_from_sampling_noise(self.target, n_samples=n)
        assert distance < 3 * floor

    def test_a_greedy_optimiser_scores_far_worse_despite_finding_the_best(self):
        # It finds the optimum every time and has zero regret, yet is nowhere
        # near the target distribution. No performance metric would notice.
        n = 20_000
        best = int(np.argmax(self.values))
        empirical = empirical_distribution(np.full((n, 1), best, dtype=np.int32), self.space)
        greedy_distance = l1_distance(empirical, self.target)

        drawn = self.rng.choice(64, size=n, p=self.target)
        sampler_distance = l1_distance(
            empirical_distribution(self.space[drawn], self.space), self.target
        )
        assert greedy_distance > 10 * sampler_distance
        assert greedy_distance > 1.5
