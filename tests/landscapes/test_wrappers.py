"""Tests for the landscape wrappers."""

import numpy as np
import pytest

from evoflownet.landscapes import EhrlichLandscape
from evoflownet.landscapes.wrappers import (
    Budgeted,
    BudgetExhaustedError,
    Cached,
    Noisy,
)


@pytest.fixture
def landscape():
    return EhrlichLandscape(
        sequence_length=8, vocab_size=4, n_motifs=1, motif_length=2, transition_density=1.0, seed=0
    )


@pytest.fixture
def sequences(landscape):
    rng = np.random.default_rng(0)
    return rng.integers(0, landscape.alphabet.size, size=(10, landscape.sequence_length))


class TestDelegation:
    @pytest.mark.parametrize(
        "wrap",
        [
            lambda inner: Noisy(inner, scale=0.1),
            lambda inner: Budgeted(inner, max_evaluations=100),
            Cached,
        ],
    )
    def test_structure_is_forwarded_unchanged(self, landscape, wrap):
        wrapped = wrap(landscape)
        assert wrapped.alphabet == landscape.alphabet
        assert wrapped.sequence_length == landscape.sequence_length
        assert wrapped.n_objectives == landscape.n_objectives
        assert wrapped.objective_names == landscape.objective_names
        assert np.array_equal(wrapped.optimum, landscape.optimum)

    def test_the_wrapped_landscape_stays_reachable(self, landscape):
        # Metrics need ground truth even when the search only sees noise.
        wrapped = Noisy(landscape, scale=0.5)
        assert wrapped.inner is landscape


class TestNoisy:
    def test_measurements_vary_between_calls(self, landscape, sequences):
        noisy = Noisy(landscape, scale=0.1, seed=0)
        assert not np.allclose(noisy.evaluate(sequences), noisy.evaluate(sequences))

    def test_noise_is_centred_on_the_true_value(self, landscape, sequences):
        noisy = Noisy(landscape, scale=0.1, seed=0)
        truth = landscape.evaluate(sequences)
        repeats = np.stack([noisy.evaluate(sequences) for _ in range(400)])
        assert np.allclose(repeats.mean(axis=0), truth, atol=0.02)

    def test_zero_scale_leaves_values_untouched(self, landscape, sequences):
        noisy = Noisy(landscape, scale=0.0)
        assert np.array_equal(noisy.evaluate(sequences), landscape.evaluate(sequences))

    def test_infeasible_sequences_stay_infeasible(self):
        # An infeasible sequence is not a noisy measurement of a feasible one.
        constrained = EhrlichLandscape(
            sequence_length=12, vocab_size=5, motif_length=2, transition_density=0.3, seed=2
        )
        rng = np.random.default_rng(0)
        candidates = rng.integers(0, 5, size=(300, 12))
        infeasible = ~constrained.is_feasible(candidates)
        assert infeasible.any(), "test needs at least one infeasible sequence"
        noisy = Noisy(constrained, scale=0.5, seed=1)
        assert np.isneginf(noisy.evaluate(candidates)[infeasible, 0]).all()

    def test_clipping_is_off_by_default(self, landscape):
        # Silently clipping overshoot would make regret look better than it is.
        noisy = Noisy(landscape, scale=0.5, seed=0)
        best = landscape.optimal_sequence[None, :]
        repeats = np.array([noisy.evaluate(best)[0, 0] for _ in range(200)])
        assert (repeats > landscape.optimum[0]).any()

    def test_clipping_caps_at_the_optimum_when_requested(self, landscape):
        noisy = Noisy(landscape, scale=0.5, seed=0, clip_to_optimum=True)
        best = landscape.optimal_sequence[None, :]
        repeats = np.array([noisy.evaluate(best)[0, 0] for _ in range(200)])
        assert (repeats <= landscape.optimum[0] + 1e-12).all()

    def test_negative_scale_is_rejected(self, landscape):
        with pytest.raises(ValueError, match="non-negative"):
            Noisy(landscape, scale=-0.1)

    def test_same_seed_reproduces_the_measurement_stream(self, landscape, sequences):
        a = Noisy(landscape, scale=0.2, seed=42)
        b = Noisy(landscape, scale=0.2, seed=42)
        assert np.array_equal(a.evaluate(sequences), b.evaluate(sequences))


class TestBudgeted:
    def test_evaluations_are_counted(self, landscape, sequences):
        budgeted = Budgeted(landscape, max_evaluations=100)
        budgeted.evaluate(sequences)
        assert budgeted.used == len(sequences)
        assert budgeted.remaining == 100 - len(sequences)

    def test_exceeding_the_budget_raises(self, landscape, sequences):
        # An error, not a silent clamp: a run that quietly stops measuring looks
        # like a completed experiment.
        budgeted = Budgeted(landscape, max_evaluations=5)
        with pytest.raises(BudgetExhaustedError, match=r"requested 10 .* 5 remaining"):
            budgeted.evaluate(sequences)

    def test_a_rejected_batch_costs_nothing(self, landscape, sequences):
        # Partly serving a batch would hand back fewer results than asked for.
        budgeted = Budgeted(landscape, max_evaluations=5)
        with pytest.raises(BudgetExhaustedError):
            budgeted.evaluate(sequences)
        assert budgeted.used == 0

    def test_the_budget_can_be_spent_exactly(self, landscape, sequences):
        budgeted = Budgeted(landscape, max_evaluations=len(sequences))
        budgeted.evaluate(sequences)
        assert budgeted.remaining == 0

    def test_reset_restores_the_budget(self, landscape, sequences):
        budgeted = Budgeted(landscape, max_evaluations=20)
        budgeted.evaluate(sequences)
        budgeted.reset()
        assert budgeted.used == 0

    def test_negative_budget_is_rejected(self, landscape):
        with pytest.raises(ValueError, match="non-negative"):
            Budgeted(landscape, max_evaluations=-1)


class TestCached:
    def test_repeated_sequences_are_scored_once(self, landscape, sequences):
        cached = Cached(landscape)
        cached.evaluate(sequences)
        cached.evaluate(sequences)
        assert cached.size == len(sequences)
        assert cached.hits == len(sequences)

    def test_cached_results_match_uncached_ones(self, landscape, sequences):
        cached = Cached(landscape)
        assert np.array_equal(cached.evaluate(sequences), landscape.evaluate(sequences))
        assert np.array_equal(cached.evaluate(sequences), landscape.evaluate(sequences))

    def test_a_partly_seen_batch_is_reassembled_in_order(self, landscape, sequences):
        cached = Cached(landscape)
        cached.evaluate(sequences[:4])
        mixed = cached.evaluate(sequences)
        assert np.array_equal(mixed, landscape.evaluate(sequences))

    def test_an_empty_batch_is_handled(self, landscape):
        cached = Cached(landscape)
        empty = np.zeros((0, landscape.sequence_length), dtype=np.int32)
        assert cached.evaluate(empty).shape == (0, 1)

    def test_caching_a_noisy_landscape_is_refused(self, landscape):
        # It would freeze the first measurement of each sequence forever,
        # removing exactly the repeated sampling that makes noise meaningful.
        with pytest.raises(ValueError, match="freeze the first measurement"):
            Cached(Noisy(landscape))


class TestComposition:
    def test_cache_inside_budget_makes_repeats_free(self, landscape, sequences):
        # Models looking up a result you already have.
        assay = Budgeted(Cached(landscape), max_evaluations=len(sequences))
        assay.evaluate(sequences)
        assert assay.remaining == 0
        # The budget is spent, so a repeat cannot be served however cheap it is.
        with pytest.raises(BudgetExhaustedError):
            assay.evaluate(sequences)

    def test_budget_inside_cache_makes_repeats_cost_nothing_twice(self, landscape, sequences):
        # Models a lab where re-running the assay costs money, but the campaign
        # remembers what it already measured.
        budget = Budgeted(landscape, max_evaluations=len(sequences))
        assay = Cached(budget)
        assay.evaluate(sequences)
        assay.evaluate(sequences)
        assert budget.used == len(sequences)

    def test_noise_and_budget_compose(self, landscape, sequences):
        assay = Budgeted(Noisy(landscape, scale=0.05, seed=0), max_evaluations=50)
        values = assay.evaluate(sequences)
        assert values.shape == (len(sequences), 1)
        assert assay.used == len(sequences)

    def test_ground_truth_survives_two_layers(self, landscape):
        assay = Budgeted(Noisy(landscape, scale=0.1), max_evaluations=10)
        noisy = assay.inner
        assert isinstance(noisy, Noisy)
        assert noisy.inner is landscape
