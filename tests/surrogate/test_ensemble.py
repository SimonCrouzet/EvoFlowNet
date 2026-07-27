"""Tests for the deep-ensemble surrogate.

The property that matters is not accuracy but *calibrated ignorance*: the
ensemble must disagree where it has no data. With a few hundred observations in
a space of 10^14 the model is uninformed almost everywhere, and an acquisition
function relies on the spread to know that.
"""

import numpy as np
import pytest

from evoflownet.surrogate import DeepEnsemble


def additive_landscape(sequences):
    """A learnable target: value rises with the count of token 1."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


@pytest.fixture(scope="module")
def trained():
    """An ensemble fitted to a small sample of a 4-position, 5-token space."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, 5, size=(200, 4))
    surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=120, seed=0)
    surrogate.fit(train, additive_landscape(train))
    return surrogate


class TestItLearns:
    def test_predictions_track_the_truth(self, trained):
        rng = np.random.default_rng(1)
        held_out = rng.integers(0, 5, size=(200, 4))
        mean, _ = trained.predict(held_out)
        truth = additive_landscape(held_out)[:, 0]
        assert np.corrcoef(mean, truth)[0, 1] > 0.8

    def test_predictions_are_on_the_original_scale(self, trained):
        # Targets are standardised internally; a caller comparing a prediction
        # against a measured value must not have to know that.
        rng = np.random.default_rng(2)
        held_out = rng.integers(0, 5, size=(64, 4))
        mean, _ = trained.predict(held_out)
        truth = additive_landscape(held_out)[:, 0]
        assert abs(mean.mean() - truth.mean()) < 1.0

    def test_it_fits_from_a_few_hundred_observations(self):
        # The realistic regime. A surrogate that needs 10^5 points is useless
        # for a campaign that can afford 10^2.
        rng = np.random.default_rng(3)
        train = rng.integers(0, 5, size=(96, 4))
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=120, seed=0)
        surrogate.fit(train, additive_landscape(train))
        mean, _ = surrogate.predict(train)
        assert np.corrcoef(mean, additive_landscape(train)[:, 0])[0, 1] > 0.7


class TestUncertainty:
    def test_the_ensemble_disagrees_somewhere(self, trained):
        # A spread of exactly zero everywhere would make every acquisition rule
        # that uses uncertainty equivalent to greedy, silently.
        rng = np.random.default_rng(4)
        _, spread = trained.predict(rng.integers(0, 5, size=(256, 4)))
        assert spread.max() > 0.0

    def test_uncertainty_is_higher_away_from_the_training_data(self):
        # The property acquisition depends on. Trained only on sequences of
        # token 0 and 1, the model should be visibly less sure about token 4.
        rng = np.random.default_rng(5)
        train = rng.integers(0, 2, size=(200, 4))
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=150, seed=0)
        surrogate.fit(train, additive_landscape(train))

        _, near = surrogate.predict(rng.integers(0, 2, size=(200, 4)))
        _, far = surrogate.predict(np.full((200, 4), 4))
        assert far.mean() > near.mean()

    def test_a_single_member_reports_no_spread(self):
        # Honest rather than convenient: one network cannot disagree with
        # itself, and a fabricated spread would be worse than none.
        rng = np.random.default_rng(6)
        train = rng.integers(0, 5, size=(64, 4))
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, n_members=1, epochs=40, seed=0)
        surrogate.fit(train, additive_landscape(train))
        _, spread = surrogate.predict(train)
        assert np.all(spread == 0.0)

    def test_the_default_ensemble_size_follows_the_literature(self):
        assert DeepEnsemble(n_tokens=5, sequence_length=4).n_members == 5


class TestFitting:
    def test_predicting_before_fitting_is_refused(self):
        # An unfitted network returns its initialisation, which looks exactly
        # like a prediction and is not one.
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4)
        assert not surrogate.is_fitted
        with pytest.raises(RuntimeError, match="not been fitted"):
            surrogate.predict(np.zeros((2, 4), dtype=np.int32))

    def test_infeasible_observations_are_excluded(self):
        # -inf marks a design that cannot be built. Keeping it would drag every
        # prediction toward negative infinity.
        rng = np.random.default_rng(7)
        train = rng.integers(0, 5, size=(128, 4))
        values = additive_landscape(train)
        values[:20] = -np.inf
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=80, seed=0)
        surrogate.fit(train, values)
        mean, _ = surrogate.predict(train)
        assert np.isfinite(mean).all()

    def test_fitting_with_nothing_finite_is_refused(self):
        train = np.zeros((8, 4), dtype=np.int32)
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4)
        with pytest.raises(ValueError, match="no observation is finite"):
            surrogate.fit(train, np.full((8, 1), -np.inf))

    def test_fitting_with_no_data_is_refused(self):
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4)
        with pytest.raises(ValueError, match="no observations"):
            surrogate.fit(np.zeros((0, 4), dtype=np.int32), np.zeros((0, 1)))

    def test_mismatched_shapes_are_refused(self):
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4)
        with pytest.raises(ValueError, match="sequences and"):
            surrogate.fit(np.zeros((8, 4), dtype=np.int32), np.zeros((3, 1)))

    def test_the_wrong_sequence_width_is_refused(self):
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4)
        with pytest.raises(ValueError, match="expected shape"):
            surrogate.fit(np.zeros((8, 9), dtype=np.int32), np.zeros((8, 1)))


class TestReproducibility:
    def test_the_same_seed_gives_the_same_predictions(self):
        rng = np.random.default_rng(8)
        train = rng.integers(0, 5, size=(96, 4))
        values = additive_landscape(train)
        predictions = []
        for _ in range(2):
            surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=40, seed=11)
            surrogate.fit(train, values)
            predictions.append(surrogate.predict(train)[0])
        assert np.allclose(predictions[0], predictions[1])

    def test_members_are_not_identical(self):
        # If every member initialised the same way, the spread would be zero and
        # the ensemble would be an expensive single network.
        rng = np.random.default_rng(9)
        train = rng.integers(0, 5, size=(96, 4))
        surrogate = DeepEnsemble(n_tokens=5, sequence_length=4, epochs=40, seed=0)
        surrogate.fit(train, additive_landscape(train))
        _, spread = surrogate.predict(rng.integers(0, 5, size=(64, 4)))
        assert spread.mean() > 0.0
