"""Tests for Ehrlich test functions.

The point of this landscape is that the right answer is knowable, so these
tests check against it rather than against plausible-looking output.
"""

import numpy as np
import pytest

from evogfn.landscapes import EhrlichLandscape


def small(**overrides):
    """An instance small enough to enumerate exhaustively."""
    params = {
        "sequence_length": 4,
        "vocab_size": 3,
        "n_motifs": 1,
        "motif_length": 2,
        "max_spacing": 1,
        "transition_density": 1.0,
        "seed": 7,
    }
    return EhrlichLandscape(**(params | overrides))


class TestPlantedOptimum:
    def test_optimal_sequence_scores_exactly_one(self):
        landscape = small()
        score = landscape.evaluate(landscape.optimal_sequence[None, :])
        assert score.shape == (1, 1)
        assert score[0, 0] == pytest.approx(1.0)

    def test_optimal_sequence_is_feasible(self):
        # An optimum that cannot be constructed is not an optimum.
        landscape = EhrlichLandscape(
            sequence_length=24, vocab_size=6, transition_density=0.4, seed=11
        )
        assert landscape.is_feasible(landscape.optimal_sequence[None, :])[0]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_nothing_in_the_whole_space_beats_the_claimed_optimum(self, seed):
        # The strongest available check: exhaustive search. If the planted
        # optimum were not global, every regret number would be wrong.
        landscape = small(seed=seed)
        every_score = landscape.evaluate(landscape.enumerate())[:, 0]
        assert every_score.max() == pytest.approx(landscape.optimum[0])

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_optimum_is_actually_attained_somewhere(self, seed):
        landscape = small(seed=seed)
        every_score = landscape.evaluate(landscape.enumerate())[:, 0]
        assert np.isclose(every_score, 1.0).any()


class TestQuantization:
    @pytest.mark.parametrize(("q", "expected_levels"), [(4, 4), (2, 3), (1, 2)])
    def test_scores_fall_on_q_evenly_spaced_levels(self, q, expected_levels):
        # q controls how sparse the signal is: q=k pays for each extra matched
        # token, q=1 pays nothing until a motif is matched in full.
        landscape = EhrlichLandscape(
            sequence_length=24,
            vocab_size=5,
            n_motifs=1,
            motif_length=4,
            quantization=q,
            transition_density=1.0,
            seed=5,
        )
        rng = np.random.default_rng(0)
        scores = landscape.evaluate(rng.integers(0, 5, size=(4000, 24)))[:, 0]
        levels = np.unique(np.round(scores, 6))
        assert len(levels) <= expected_levels
        # Every observed score is a multiple of 1/q.
        assert np.allclose(levels * q, np.round(levels * q))

    def test_quantization_must_divide_motif_length(self):
        with pytest.raises(ValueError, match="divide motif_length"):
            EhrlichLandscape(motif_length=4, quantization=3)

    def test_quantization_defaults_to_the_dense_signal(self):
        landscape = EhrlichLandscape(motif_length=4)
        assert landscape.evaluate(landscape.optimal_sequence[None, :])[0, 0] == pytest.approx(1.0)


class TestFeasibility:
    def test_a_fully_dense_transition_matrix_forbids_nothing(self):
        landscape = small(transition_density=1.0)
        rng = np.random.default_rng(0)
        sequences = rng.integers(0, landscape.alphabet.size, size=(200, 4))
        assert landscape.is_feasible(sequences).all()

    def test_uniform_random_sequences_are_almost_all_infeasible(self):
        # This is the property that makes the benchmark interesting: naive
        # search spends most of its budget on sequences that cannot be built.
        landscape = EhrlichLandscape(
            sequence_length=32, vocab_size=8, transition_density=0.5, seed=2
        )
        rng = np.random.default_rng(0)
        sequences = rng.integers(0, 8, size=(2000, 32))
        assert landscape.is_feasible(sequences).mean() < 0.01

    def test_markov_walk_produces_feasible_sequences(self):
        # The same chain that defines feasibility can generate it, which is what
        # an action-masking policy exploits.
        landscape = EhrlichLandscape(
            sequence_length=32, vocab_size=8, transition_density=0.3, seed=4
        )
        assert landscape.is_feasible(landscape.optimal_sequence[None, :]).all()

    def test_infeasible_sequences_score_negative_infinity(self):
        landscape = EhrlichLandscape(
            sequence_length=8, vocab_size=5, motif_length=2, transition_density=0.3, seed=6
        )
        rng = np.random.default_rng(0)
        sequences = rng.integers(0, 5, size=(500, 8))
        infeasible = ~landscape.is_feasible(sequences)
        assert infeasible.any(), "test needs at least one infeasible sequence"
        assert np.isneginf(landscape.evaluate(sequences)[infeasible, 0]).all()


class TestScoreRange:
    def test_feasible_scores_lie_in_the_unit_interval(self):
        landscape = small(transition_density=1.0)
        scores = landscape.evaluate(landscape.enumerate())[:, 0]
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_more_motifs_make_a_perfect_score_rarer(self):
        # The score is a product over motifs, so missing any one zeroes it.
        # That product is where the epistasis comes from.
        rng = np.random.default_rng(0)
        perfect_counts = []
        for n_motifs in (1, 2):
            landscape = EhrlichLandscape(
                sequence_length=24,
                vocab_size=4,
                n_motifs=n_motifs,
                motif_length=2,
                quantization=1,
                transition_density=1.0,
                seed=9,
            )
            scores = landscape.evaluate(rng.integers(0, 4, size=(3000, 24)))[:, 0]
            perfect_counts.append(int((scores == 1.0).sum()))
        assert perfect_counts[1] < perfect_counts[0]


class TestReproducibility:
    def test_the_same_seed_gives_the_same_landscape(self):
        a, b = small(seed=3), small(seed=3)
        assert np.array_equal(a.motifs, b.motifs)
        assert np.array_equal(a.spacings, b.spacings)
        assert np.array_equal(a.optimal_sequence, b.optimal_sequence)
        assert np.allclose(a.transition_matrix, b.transition_matrix)

    def test_different_seeds_give_different_landscapes(self):
        a, b = EhrlichLandscape(seed=0), EhrlichLandscape(seed=1)
        assert not np.array_equal(a.optimal_sequence, b.optimal_sequence)

    def test_accessors_return_copies(self):
        # Handing out internal arrays would let a caller silently redefine the
        # landscape partway through a run.
        landscape = small()
        landscape.motifs[0, 0] = 99
        landscape.optimal_sequence[0] = 99
        assert landscape.evaluate(landscape.optimal_sequence[None, :])[0, 0] == pytest.approx(1.0)


class TestParameterValidation:
    def test_motifs_must_fit_inside_their_block(self):
        with pytest.raises(ValueError, match="spans up to"):
            EhrlichLandscape(sequence_length=8, n_motifs=4, motif_length=4, max_spacing=3)

    def test_vocabulary_must_be_large_enough_to_constrain(self):
        with pytest.raises(ValueError, match="vocab_size must be at least"):
            EhrlichLandscape(vocab_size=2)

    @pytest.mark.parametrize(("field", "value"), [("n_motifs", 0), ("motif_length", 0)])
    def test_degenerate_sizes_are_rejected(self, field, value):
        with pytest.raises(ValueError, match="at least 1"):
            EhrlichLandscape(**{field: value})

    def test_spacing_must_be_positive(self):
        with pytest.raises(ValueError, match="max_spacing must be at least 1"):
            EhrlichLandscape(max_spacing=0)


class TestInputValidation:
    def test_a_single_sequence_without_a_batch_dimension_is_rejected(self):
        landscape = small()
        with pytest.raises(ValueError, match="ndim 2"):
            landscape.evaluate(landscape.optimal_sequence)

    def test_wrong_sequence_length_is_rejected(self):
        with pytest.raises(ValueError, match="length 4"):
            small().evaluate(np.zeros((1, 5), dtype=np.int32))

    def test_float_input_is_rejected(self):
        with pytest.raises(ValueError, match="integer token indices"):
            small().evaluate(np.zeros((1, 4), dtype=np.float64))

    def test_out_of_alphabet_tokens_are_rejected(self):
        with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
            small().evaluate(np.full((1, 4), 7, dtype=np.int32))


class TestEnumeration:
    def test_enumerate_covers_the_whole_space_exactly_once(self):
        landscape = small()
        sequences = landscape.enumerate()
        assert sequences.shape == (landscape.search_space_size, landscape.sequence_length)
        assert len({tuple(row) for row in sequences}) == landscape.search_space_size

    def test_enumerate_refuses_a_space_it_cannot_hold(self):
        landscape = EhrlichLandscape(sequence_length=64, vocab_size=20)
        with pytest.raises(ValueError, match="enumeration limit"):
            landscape.enumerate()


class TestParameterAccessors:
    def test_the_instance_reports_its_own_definition(self):
        # A benchmark that cannot state its parameters cannot be reproduced from
        # a results table.
        landscape = EhrlichLandscape(
            sequence_length=24, vocab_size=5, n_motifs=2, motif_length=4, quantization=2
        )
        assert landscape.n_motifs == 2
        assert landscape.motif_length == 4
        assert landscape.quantization == 2
        assert landscape.motifs.shape == (2, 4)
        assert landscape.spacings.shape == (2, 4)

    def test_length_one_sequences_have_no_adjacency_to_constrain(self):
        landscape = EhrlichLandscape(
            sequence_length=1, vocab_size=3, n_motifs=1, motif_length=1, max_spacing=1, seed=0
        )
        assert landscape.is_feasible(np.zeros((3, 1), dtype=np.int32)).all()
