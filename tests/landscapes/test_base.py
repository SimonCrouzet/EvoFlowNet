"""Tests for the FitnessLandscape base class itself.

Exercised through a minimal subclass rather than through Ehrlich, so that the
defaults the interface provides are tested as defaults -- an implementation that
overrides them cannot show whether they work.
"""

import numpy as np
import pytest

from evogfn.core import Alphabet
from evogfn.landscapes.base import FitnessLandscape


class CountingLandscape(FitnessLandscape):
    """Scores a sequence by how many tokens equal zero. Overrides nothing else."""

    def __init__(self, length=4, symbols="ABC"):
        self._alphabet = Alphabet.from_string(symbols)
        self._length = length

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sequence_length(self):
        return self._length

    def _evaluate(self, sequences):
        return (sequences == 0).sum(axis=1, keepdims=True).astype(np.float64)


class MisbehavingLandscape(CountingLandscape):
    """Returns the wrong shape, to prove the wrapper notices."""

    def _evaluate(self, sequences):
        return (sequences == 0).sum(axis=1).astype(np.float64)  # (n,) not (n, 1)


class TestDefaults:
    def test_ground_truth_is_absent_unless_provided(self):
        # None is the honest answer for a landscape that cannot know, and is
        # what metrics branch on to decide whether exact regret is available.
        assert CountingLandscape().optimum is None

    def test_everything_is_feasible_by_default(self):
        landscape = CountingLandscape()
        sequences = np.zeros((5, 4), dtype=np.int32)
        assert landscape.is_feasible(sequences).all()

    def test_single_objective_is_the_default(self):
        assert CountingLandscape().n_objectives == 1
        assert CountingLandscape().objective_names == ("fitness",)

    def test_search_space_size_is_alphabet_to_the_length(self):
        assert CountingLandscape(length=4, symbols="ABC").search_space_size == 81


class TestEvaluateWrapper:
    def test_a_wrong_output_shape_is_caught(self):
        # A landscape returning (n,) instead of (n, 1) would broadcast against
        # objective weights rather than fail, producing plausible nonsense.
        with pytest.raises(ValueError, match=r"_evaluate returned \(3,\), expected \(3, 1\)"):
            MisbehavingLandscape().evaluate(np.zeros((3, 4), dtype=np.int32))

    def test_validation_runs_before_the_subclass_sees_anything(self):
        with pytest.raises(ValueError, match="ndim 2"):
            CountingLandscape().evaluate(np.zeros(4, dtype=np.int32))

    def test_an_empty_batch_is_allowed(self):
        # Zero-length batches arise naturally when every candidate is filtered.
        result = CountingLandscape().evaluate(np.zeros((0, 4), dtype=np.int32))
        assert result.shape == (0, 1)

    def test_feasibility_validates_its_input_too(self):
        with pytest.raises(ValueError, match="length 4"):
            CountingLandscape().is_feasible(np.zeros((1, 9), dtype=np.int32))


class TestEnumeration:
    def test_enumeration_is_ordered_with_the_last_position_fastest(self):
        sequences = CountingLandscape(length=2, symbols="AB").enumerate()
        assert sequences.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]

    def test_enumeration_is_refused_when_it_would_not_fit(self):
        with pytest.raises(ValueError, match="enumeration limit"):
            CountingLandscape(length=40, symbols="ABCDEFGHIJKLMNOPQRST").enumerate()
