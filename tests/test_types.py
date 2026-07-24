"""Tests for the alphabet and sequence encoding."""

import numpy as np
import pytest

from evoflownet.core import Alphabet


class TestConstruction:
    def test_from_string_indexes_in_order(self):
        alphabet = Alphabet.from_string("ACGT")
        assert alphabet.size == 4
        assert alphabet.index_of("A") == 0
        assert alphabet.index_of("T") == 3

    def test_protein_alphabet_has_twenty_amino_acids(self):
        assert Alphabet.protein().size == 20

    def test_dna_alphabet_has_four_bases(self):
        assert Alphabet.dna().size == 4

    def test_empty_alphabet_is_rejected(self):
        with pytest.raises(ValueError, match="at least one symbol"):
            Alphabet(())

    def test_duplicate_symbols_are_rejected(self):
        # Silently deduplicating would shift every index after the duplicate,
        # reinterpreting any model or cached landscape built on this alphabet.
        with pytest.raises(ValueError, match="unique"):
            Alphabet.from_string("ACGA")

    def test_alphabet_is_immutable(self):
        alphabet = Alphabet.from_string("ACGT")
        with pytest.raises(AttributeError):
            alphabet.symbols = ("A",)  # type: ignore[misc]


class TestEncoding:
    def test_round_trip_preserves_the_sequence(self):
        alphabet = Alphabet.protein()
        sequence = "MKTAYIAKQR"
        assert alphabet.decode(alphabet.encode(sequence)) == sequence

    def test_encode_produces_integer_indices(self):
        tokens = Alphabet.from_string("ACGT").encode("GATTACA")
        assert np.issubdtype(tokens.dtype, np.integer)
        assert tokens.tolist() == [2, 0, 3, 3, 0, 1, 0]

    def test_encode_rejects_symbols_outside_the_alphabet(self):
        with pytest.raises(KeyError, match="not in this alphabet"):
            Alphabet.dna().encode("ACGU")  # U is RNA

    def test_encode_many_stacks_into_a_matrix(self):
        tokens = Alphabet.dna().encode_many(["ACGT", "TGCA"])
        assert tokens.shape == (2, 4)

    def test_encode_many_rejects_ragged_input(self):
        # A ragged batch would otherwise become an object array and fail much
        # later, somewhere with no useful context.
        with pytest.raises(ValueError, match="same length"):
            Alphabet.dna().encode_many(["ACGT", "ACG"])

    def test_encode_many_handles_no_sequences(self):
        assert Alphabet.dna().encode_many([]).shape == (0, 0)

    def test_decode_many_round_trips(self):
        alphabet = Alphabet.dna()
        sequences = ["ACGT", "TGCA"]
        assert alphabet.decode_many(alphabet.encode_many(sequences)) == sequences


class TestDecodingErrors:
    def test_decode_rejects_a_batch(self):
        with pytest.raises(ValueError, match="ndim 1"):
            Alphabet.dna().decode(np.zeros((2, 4), dtype=np.int32))

    def test_decode_many_rejects_a_single_sequence(self):
        with pytest.raises(ValueError, match="ndim 2"):
            Alphabet.dna().decode_many(np.zeros(4, dtype=np.int32))

    def test_decode_rejects_out_of_range_indices(self):
        # Reported rather than wrapping around, which is what raw indexing with
        # a negative value would silently do.
        with pytest.raises(IndexError, match="outside alphabet"):
            Alphabet.dna().decode(np.array([0, 9], dtype=np.int32))

    def test_decode_rejects_negative_indices(self):
        with pytest.raises(IndexError, match="outside alphabet"):
            Alphabet.dna().decode(np.array([-1], dtype=np.int32))
