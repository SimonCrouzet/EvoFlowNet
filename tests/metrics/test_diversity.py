"""Tests for diversity, novelty and mode counting."""

import numpy as np
import pytest

from evoflownet.metrics import distinct_modes, diversity, hamming_distances, novelty


def seqs(*rows):
    return np.array(rows, dtype=np.int32)


class TestHammingDistances:
    def test_distances_are_counted_per_position(self):
        d = hamming_distances(seqs([0, 0, 0]), seqs([0, 0, 0], [1, 0, 0], [1, 1, 1]))
        assert d.tolist() == [[0, 1, 3]]

    def test_a_sequence_is_zero_from_itself(self):
        assert hamming_distances(seqs([1, 2, 3]), seqs([1, 2, 3]))[0, 0] == 0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="sequence lengths differ"):
            hamming_distances(seqs([0, 0]), seqs([0, 0, 0]))


class TestDiversity:
    def test_identical_designs_have_zero_diversity(self):
        # The failure mode this project exists to detect: a batch that looks
        # like many designs but is one.
        assert diversity(seqs([0, 0, 0], [0, 0, 0], [0, 0, 0])) == pytest.approx(0.0)

    def test_it_is_the_mean_over_ordered_pairs(self):
        # Distances: (a,b)=1, (a,c)=3, (b,c)=2, each counted twice -> 12/6 = 2
        assert diversity(seqs([0, 0, 0], [1, 0, 0], [1, 1, 1])) == pytest.approx(2.0)

    def test_completely_different_designs_reach_the_sequence_length(self):
        assert diversity(seqs([0, 0], [1, 1])) == pytest.approx(2.0)

    def test_a_single_design_has_no_pairs(self):
        assert diversity(seqs([0, 1, 2])) == 0.0

    def test_an_empty_batch_has_no_pairs(self):
        assert diversity(np.zeros((0, 3), dtype=np.int32)) == 0.0


class TestNovelty:
    def test_it_is_the_mean_distance_to_the_nearest_known_sequence(self):
        known = seqs([0, 0, 0])
        proposed = seqs([1, 0, 0], [1, 1, 1])
        assert novelty(proposed, known) == pytest.approx(2.0)

    def test_reproposing_known_sequences_scores_zero(self):
        known = seqs([0, 0, 0], [1, 1, 1])
        assert novelty(known, known) == pytest.approx(0.0)

    def test_it_takes_the_nearest_not_the_average_reference(self):
        # A design sitting on top of one known sequence is not novel, however
        # far it is from the rest of the reference set.
        known = seqs([0, 0, 0], [1, 1, 1])
        assert novelty(seqs([1, 1, 1]), known) == pytest.approx(0.0)

    def test_diversity_and_novelty_measure_different_things(self):
        # A batch can be internally varied while exploring only what was already
        # given, which is why both are reported.
        known = seqs([0, 0, 0], [1, 1, 1])
        batch = seqs([0, 0, 0], [1, 1, 1])
        assert diversity(batch) > 0
        assert novelty(batch, known) == pytest.approx(0.0)

    def test_an_empty_reference_set_is_refused(self):
        with pytest.raises(ValueError, match="non-empty reference"):
            novelty(seqs([0, 0]), np.zeros((0, 2), dtype=np.int32))


class TestDistinctModes:
    def test_repeated_copies_of_one_design_count_once(self):
        # The headline distinction: an optimiser returning a thousand copies of
        # one good variant has found one mode, not a thousand.
        sequences = seqs([1, 1, 1], [1, 1, 1], [1, 1, 1])
        values = np.array([0.9, 0.9, 0.9])
        assert distinct_modes(sequences, values, threshold=0.5) == 1

    def test_separated_designs_count_separately(self):
        sequences = seqs([0, 0, 0], [1, 1, 1], [2, 2, 2])
        values = np.array([0.9, 0.8, 0.7])
        assert distinct_modes(sequences, values, threshold=0.5) == 3

    def test_designs_below_the_threshold_do_not_count(self):
        sequences = seqs([0, 0, 0], [1, 1, 1])
        values = np.array([0.9, 0.1])
        assert distinct_modes(sequences, values, threshold=0.5) == 1

    def test_a_minimum_distance_merges_nearby_designs(self):
        # Neighbours of one peak are the same discovery, not two.
        sequences = seqs([0, 0, 0], [0, 0, 1], [1, 1, 1])
        values = np.array([0.9, 0.9, 0.9])
        assert distinct_modes(sequences, values, threshold=0.5, min_distance=1) == 3
        assert distinct_modes(sequences, values, threshold=0.5, min_distance=2) == 2

    def test_the_best_design_represents_its_region(self):
        # Considered best-first, so a region is counted by its strongest member.
        sequences = seqs([0, 0, 0], [0, 0, 1])
        values = np.array([0.6, 0.95])
        assert distinct_modes(sequences, values, threshold=0.5, min_distance=2) == 1

    def test_nothing_above_the_threshold_is_zero_modes(self):
        assert distinct_modes(seqs([0, 0]), np.array([0.1]), threshold=0.5) == 0

    def test_infeasible_designs_never_count(self):
        sequences = seqs([0, 0], [1, 1])
        values = np.array([-np.inf, 0.9])
        assert distinct_modes(sequences, values, threshold=0.5) == 1

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError, match="they must match"):
            distinct_modes(seqs([0, 0], [1, 1]), np.array([0.5]), threshold=0.1)

    def test_a_nonpositive_minimum_distance_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            distinct_modes(seqs([0, 0]), np.array([0.9]), threshold=0.5, min_distance=0)
