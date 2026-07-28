"""Tests for the count-based selection noise wrapper.

The central test is :class:`TestFlightedSignature`. FLIGHTED (Sundar et al.,
bioRxiv 2024) reports that the ~1,000 highest-enrichment GB1 variants show
essentially zero correlation between measured and true fitness, while the
landscape as a whole correlates well. Reproducing that is the entire reason
:class:`SelectionNoisy` exists -- a noise model that fails it is
:class:`Noisy` with extra steps -- so it is asserted directly, and asserted
against :class:`Noisy` as a control.

The comparison is always made *within bands of true fitness*, never within bands
of measured fitness. Selecting on the measurement would depress the correlation
for any noise model at all, through regression to the mean, and would therefore
prove nothing about this one.
"""

import numpy as np
import pytest

from evoflownet.core.types import Alphabet
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.landscapes.wrappers import Noisy, SelectionNoisy

#: Sequences per band. Ten bands of 1,000 mirrors the size of the slice FLIGHTED
#: reports on.
BAND = 1_000


class RampLandscape(FitnessLandscape):
    """A landscape whose 10,000 sequences tile ``[0, 1]`` at even spacing.

    Deliberately uniform. Correlation inside a band depends on both the noise and
    the spread of true fitness within that band, so a landscape with a long tail
    would make the top band look uninformative through range restriction alone.
    Even spacing gives every band identical spread, which leaves the noise as the
    only thing that can differ between them.
    """

    @property
    def alphabet(self):
        return Alphabet.from_string("ABCDEFGHIJ")

    @property
    def sequence_length(self):
        return 4

    @property
    def optimum(self):
        return np.array([1.0])

    def _evaluate(self, sequences):
        weights = 10 ** np.arange(self.sequence_length - 1, -1, -1)
        return (np.asarray(sequences, dtype=np.intp) @ weights / 9_999.0)[:, None]


@pytest.fixture(scope="module")
def landscape():
    return RampLandscape()


@pytest.fixture(scope="module")
def every(landscape):
    return landscape.enumerate()


@pytest.fixture(scope="module")
def truth(landscape, every):
    return landscape.evaluate(every)[:, 0]


def bands(truth):
    """Indices of the ten equal-count bands of true fitness, worst band first."""
    order = np.argsort(truth)
    return [order[i * BAND : (i + 1) * BAND] for i in range(10)]


def band_correlations(truth, measured):
    """Pearson r between measurement and truth inside each band."""
    return np.array([np.corrcoef(truth[i], measured[i])[0, 1] for i in bands(truth)])


def band_errors(truth, measured):
    """Standard deviation of the measurement error inside each band."""
    return np.array([np.std(measured[i] - truth[i]) for i in bands(truth)])


class TestFlightedSignature:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_the_top_of_the_landscape_is_where_measurement_fails(
        self, landscape, every, truth, seed
    ):
        # The finding, stated as an assertion: the worst 1,000 variants are
        # measured informatively, the best 1,000 are barely measured at all.
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=100.0, seed=seed)
        correlations = band_correlations(truth, assay.evaluate(every)[:, 0])
        assert correlations[0] > 0.4, "the bottom of the landscape should be measurable"
        assert correlations[-1] < 0.2, "the top of the landscape should not be"
        assert correlations[-1] < 0.5 * correlations[0]

    def test_the_whole_landscape_still_correlates_well(self, landscape, every, truth):
        # The degradation has to be local to the top. A model that simply added
        # more noise everywhere would also fail the band test above, so this is
        # what separates "uninformative at the top" from "uninformative".
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=100.0, seed=0)
        assert np.corrcoef(truth, assay.evaluate(every)[:, 0])[0, 1] > 0.85

    def test_measurement_error_grows_with_fitness(self, landscape, every, truth):
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=100.0, seed=0)
        errors = band_errors(truth, assay.evaluate(every)[:, 0])
        assert (np.diff(errors) >= -1e-3).all(), f"error should not fall with fitness: {errors}"
        assert errors[-1] > 2.0 * errors[0]

    def test_gaussian_noise_cannot_produce_the_signature(self, landscape, every, truth):
        # The control. Homoscedastic noise at a *better* overall correlation
        # leaves the top band no worse measured than the bottom one, which is
        # exactly the property that makes it the wrong model of an assay.
        gaussian = Noisy(landscape, scale=0.1, seed=0)
        measured = gaussian.evaluate(every)[:, 0]
        assert np.corrcoef(truth, measured)[0, 1] > 0.85

        correlations = band_correlations(truth, measured)
        assert correlations[-1] > 0.8 * correlations[0]
        errors = band_errors(truth, measured)
        assert errors[-1] == pytest.approx(errors[0], rel=0.1)

    def test_saturation_is_the_mechanism(self, landscape):
        # The wrapper is only uninformative at the top because survival
        # saturates there. If this stopped being true the signature would vanish
        # silently, so it is asserted rather than assumed.
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, top_survival=0.995)
        assert assay.survival_probability(np.array([0.0]))[0] == pytest.approx(0.5)
        assert assay.survival_probability(np.array([1.0]))[0] == pytest.approx(0.995)


class TestForwardModel:
    def test_unlimited_reads_recover_the_truth(self, landscape, every, truth):
        # The estimator inverts the forward model exactly, so all of the error
        # comes from finite counts and none from the link.
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=1e7, seed=0)
        assert np.allclose(assay.evaluate(every)[:, 0], truth, atol=0.02)

    def test_fewer_reads_mean_a_worse_assay(self, landscape, every, truth):
        deep = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=1000.0, seed=0)
        shallow = SelectionNoisy.calibrated(landscape, top_fitness=1.0, reads=30.0, seed=0)
        assert (
            np.corrcoef(truth, deep.evaluate(every)[:, 0])[0, 1]
            > np.corrcoef(truth, shallow.evaluate(every)[:, 0])[0, 1]
        )

    def test_survival_saturates_without_overflowing(self, landscape):
        # Warnings are errors in this suite, so an overflow in the logistic
        # would fail here rather than return a silently wrong probability.
        assay = SelectionNoisy(landscape, slope=1.0)
        extreme = np.array([-1e4, -700.0, 0.0, 700.0, 1e4])
        probabilities = assay.survival_probability(extreme)
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
        assert (np.diff(probabilities) >= 0).all()

    def test_infeasible_sequences_stay_infeasible(self, landscape, every):
        # Mirrors Noisy: -inf is not a noisy measurement of anything.
        class Infeasible(FitnessLandscape):
            alphabet = landscape.alphabet
            sequence_length = landscape.sequence_length

            def _evaluate(self, sequences):
                return np.full((sequences.shape[0], 1), -np.inf)

        assay = SelectionNoisy(Infeasible(), seed=0)
        assert np.isneginf(assay.evaluate(every[:10])[:, 0]).all()


class TestReproducibility:
    def test_the_same_seed_gives_the_same_assay(self, landscape, every):
        a = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=7)
        b = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=7)
        assert np.array_equal(a.evaluate(every), b.evaluate(every))

    def test_different_seeds_give_different_assays(self, landscape, every):
        a = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=7)
        b = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=8)
        assert not np.array_equal(a.evaluate(every), b.evaluate(every))

    def test_measurements_vary_between_calls(self, landscape, every):
        # Re-measuring a variant re-runs the assay, as in a real repeat.
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=0)
        assert not np.array_equal(assay.evaluate(every[:100]), assay.evaluate(every[:100]))


class TestCalibration:
    def test_the_optimum_is_used_when_no_top_fitness_is_given(self, landscape):
        inferred = SelectionNoisy.calibrated(landscape)
        explicit = SelectionNoisy.calibrated(landscape, top_fitness=1.0)
        assert inferred.slope == pytest.approx(explicit.slope)

    def test_an_unknown_optimum_is_an_error_rather_than_a_guess(self, landscape):
        class Unknown(FitnessLandscape):
            alphabet = landscape.alphabet
            sequence_length = landscape.sequence_length

            def _evaluate(self, sequences):
                return np.zeros((sequences.shape[0], 1))

        with pytest.raises(ValueError, match="does not know its optimum"):
            SelectionNoisy.calibrated(Unknown())

    @pytest.mark.parametrize("top_survival", [0.5, 0.4, 1.0, 1.5])
    def test_a_top_survival_outside_the_open_upper_half_is_rejected(self, landscape, top_survival):
        with pytest.raises(ValueError, match="top_survival"):
            SelectionNoisy.calibrated(landscape, top_fitness=1.0, top_survival=top_survival)

    def test_a_top_fitness_below_the_midpoint_is_rejected(self, landscape):
        with pytest.raises(ValueError, match="must exceed neutral_fitness"):
            SelectionNoisy.calibrated(landscape, neutral_fitness=1.0, top_fitness=0.5)

    def test_a_stricter_selection_saturates_sooner(self, landscape):
        lenient = SelectionNoisy.calibrated(landscape, top_fitness=1.0, top_survival=0.9)
        strict = SelectionNoisy.calibrated(landscape, top_fitness=1.0, top_survival=0.999)
        assert strict.slope > lenient.slope


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"slope": 0.0}, "slope must be positive"),
            ({"slope": -1.0}, "slope must be positive"),
            ({"reads": 0.0}, "reads must be positive"),
            ({"dispersion": 0.0}, "dispersion must be positive"),
        ],
    )
    def test_parameters_that_would_break_the_model_are_rejected(self, landscape, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SelectionNoisy(landscape, **kwargs)

    def test_structure_is_forwarded_unchanged(self, landscape):
        assay = SelectionNoisy(landscape)
        assert assay.alphabet == landscape.alphabet
        assert assay.sequence_length == landscape.sequence_length
        assert assay.n_objectives == landscape.n_objectives
        assert assay.inner is landscape

    def test_the_truth_stays_reachable(self, landscape, every, truth):
        # Metrics are computed against ground truth while the search sees noise.
        assay = SelectionNoisy.calibrated(landscape, top_fitness=1.0, seed=0)
        assert np.array_equal(assay.inner.evaluate(every)[:, 0], truth)
