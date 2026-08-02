"""Tests for preference conditioning: thermometer encoding and Dirichlet sampling."""

import numpy as np
import pytest

from evogfn.models.conditioning import (
    encoding_dim,
    preference_encoding,
    sample_preferences,
    thermometer_encode,
)


class TestThermometerEncode:
    def test_the_minimum_encodes_to_all_zeros(self):
        assert thermometer_encode(0.0, n_bins=8) == pytest.approx(np.zeros(8))

    def test_the_maximum_encodes_to_all_ones(self):
        assert thermometer_encode(1.0, n_bins=8) == pytest.approx(np.ones(8))

    def test_no_bin_is_wasted(self):
        # Spacing the edges over the closed interval, as the reference
        # implementation does, leaves the last coordinate at zero for every value
        # in range: one bin of the conditioning signal carrying nothing.
        assert thermometer_encode(1.0, n_bins=4) == pytest.approx(np.ones(4))
        assert thermometer_encode(0.5, n_bins=4) == pytest.approx([1.0, 1.0, 0.0, 0.0])
        assert thermometer_encode(0.625, n_bins=4) == pytest.approx([1.0, 1.0, 0.5, 0.0])

    def test_it_is_monotone_in_the_value(self):
        # The property that makes it an encoding of an ordered quantity rather
        # than a categorical one.
        low = thermometer_encode(0.3, n_bins=16)
        high = thermometer_encode(0.7, n_bins=16)
        assert np.all(high >= low)
        assert high.sum() > low.sum()

    def test_each_encoding_fills_from_the_left(self):
        # Non-increasing along the bins: a filled bin is never to the right of an
        # empty one, which is what "thermometer" means.
        encoded = thermometer_encode(0.42, n_bins=12)
        assert np.all(np.diff(encoded) <= 1e-12)

    def test_every_entry_lies_in_the_unit_interval(self):
        for value in (0.0, 0.1, 0.5, 0.9, 1.0):
            encoded = thermometer_encode(value, n_bins=10)
            assert np.all((encoded >= 0.0) & (encoded <= 1.0))

    def test_nearby_values_encode_to_nearby_vectors(self):
        # Unlike a one-hot binning, which jumps at a bin edge. This continuity is
        # what lets a policy interpolate to a preference it never saw.
        edge = 1.0 / 16.0  # a bin boundary for n_bins=16 over [0, 1]
        below = thermometer_encode(edge - 1e-4, n_bins=16)
        above = thermometer_encode(edge + 1e-4, n_bins=16)
        assert np.abs(above - below).max() < 0.01

    def test_values_outside_the_range_saturate(self):
        assert thermometer_encode(5.0, n_bins=4, vmin=0.0, vmax=1.0) == pytest.approx(np.ones(4))
        assert thermometer_encode(-5.0, n_bins=4, vmin=0.0, vmax=1.0) == pytest.approx(np.zeros(4))

    def test_it_adds_a_trailing_axis_to_any_shape(self):
        assert thermometer_encode(np.zeros((3, 2)), n_bins=5).shape == (3, 2, 5)

    def test_the_range_can_be_widened(self):
        # The same encoder handles beta, which is not confined to [0, 1].
        assert thermometer_encode(16.0, n_bins=8, vmin=0.0, vmax=32.0) == pytest.approx(
            thermometer_encode(0.5, n_bins=8)
        )

    def test_a_single_bin_is_refused(self):
        with pytest.raises(ValueError, match="conditions the policy on nothing"):
            thermometer_encode(0.5, n_bins=1)

    def test_an_empty_range_is_refused(self):
        with pytest.raises(ValueError, match="vmax must exceed vmin"):
            thermometer_encode(0.5, vmin=1.0, vmax=1.0)

    def test_non_finite_values_are_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            thermometer_encode(np.array([np.inf]))


class TestPreferenceEncoding:
    def test_it_flattens_to_the_declared_width(self):
        encoded = preference_encoding([0.25, 0.75], n_bins=8)
        assert encoded.shape == (encoding_dim(2, n_bins=8),)

    def test_a_batch_keeps_its_leading_axis(self):
        preferences = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        assert preference_encoding(preferences, n_bins=4).shape == (3, encoding_dim(2, n_bins=4))

    def test_the_layout_is_objective_major(self):
        # ω = (1, 0): the first objective's block is full, the second's is empty.
        encoded = preference_encoding([1.0, 0.0], n_bins=4)
        assert encoded[:4] == pytest.approx(np.ones(4))
        assert encoded[4:] == pytest.approx(np.zeros(4))

    def test_different_preferences_encode_differently(self):
        # The failure this module exists to prevent is a policy that cannot tell
        # two trade-offs apart.
        a = preference_encoding([0.9, 0.1], n_bins=16)
        b = preference_encoding([0.1, 0.9], n_bins=16)
        assert np.abs(a - b).max() > 0.5

    def test_a_preference_off_the_simplex_is_refused(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            preference_encoding([0.5, 0.2])

    def test_a_negative_preference_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            preference_encoding([1.5, -0.5])

    def test_a_three_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="ndim 3"):
            preference_encoding(np.ones((1, 1, 2)) / 2.0)

    def test_encoding_dim_matches_what_the_encoder_produces(self):
        for n_objectives, n_bins in ((1, 2), (2, 16), (4, 7)):
            preference = np.full(n_objectives, 1.0 / n_objectives)
            assert preference_encoding(preference, n_bins=n_bins).shape == (
                encoding_dim(n_objectives, n_bins=n_bins),
            )

    def test_encoding_dim_refuses_the_sizes_the_encoder_refuses(self):
        with pytest.raises(ValueError, match="n_bins must be at least"):
            encoding_dim(2, n_bins=1)
        with pytest.raises(ValueError, match="n_objectives must be at least"):
            encoding_dim(0)


class TestSamplePreferences:
    def test_every_draw_lies_on_the_simplex(self):
        preferences = sample_preferences(3, 64, seed=0)
        assert preferences.shape == (64, 3)
        assert (preferences >= 0.0).all()
        assert preferences.sum(axis=1) == pytest.approx(np.ones(64))

    def test_the_draws_are_reproducible_from_a_seed(self):
        assert sample_preferences(3, 8, seed=7) == pytest.approx(sample_preferences(3, 8, seed=7))

    def test_different_seeds_give_different_draws(self):
        assert sample_preferences(3, 8, seed=1) != pytest.approx(sample_preferences(3, 8, seed=2))

    def test_a_small_alpha_concentrates_near_the_corners(self):
        # alpha < 1 asks mostly for single-objective specialists; alpha > 1 asks
        # mostly for balanced designs. The largest weight in a draw separates them.
        corners = sample_preferences(3, 200, alpha=0.1, seed=0).max(axis=1).mean()
        centre = sample_preferences(3, 200, alpha=10.0, seed=0).max(axis=1).mean()
        assert corners > centre

    def test_a_per_objective_alpha_tilts_the_draws(self):
        preferences = sample_preferences(2, 200, alpha=[10.0, 1.0], seed=0)
        assert preferences[:, 0].mean() > preferences[:, 1].mean()

    def test_a_single_objective_always_gets_the_whole_weight(self):
        assert sample_preferences(1, 4, seed=0) == pytest.approx(np.ones((4, 1)))

    def test_the_draws_encode_without_further_validation(self):
        # Sampling and encoding are used together every batch, so a draw that the
        # encoder rejects would be a broken interface between them.
        for preference in sample_preferences(4, 16, seed=3):
            assert preference_encoding(preference, n_bins=8).shape == (32,)

    def test_a_non_positive_alpha_is_refused(self):
        with pytest.raises(ValueError, match="finite and positive"):
            sample_preferences(2, alpha=0.0)

    def test_an_alpha_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="or have 2 entries"):
            sample_preferences(2, alpha=[1.0, 1.0, 1.0])

    @pytest.mark.parametrize(("n_objectives", "n_samples"), [(0, 1), (2, 0)])
    def test_non_positive_sizes_are_refused(self, n_objectives, n_samples):
        with pytest.raises(ValueError, match="must be at least 1"):
            sample_preferences(n_objectives, n_samples)
