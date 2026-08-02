"""Fundamental types for representing sequences and their fitness.

Sequences are integer arrays of token indices, never strings. Every hot path in
this library -- masking actions, scoring a batch, computing pairwise distances --
is array work, and converting to and from text at each step would dominate the
cost. [Alphabet][evogfn.core.types.Alphabet] is the single place that knows how
indices relate to letters, so text appears only at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable

# PEP 695 (`type X = ...`) would be tidier, but it requires Python 3.12 and this
# package supports 3.11.
#: Token indices into an :class:`Alphabet`, shaped ``(..., length)``.
#: Batches are ``(n, length)``; a single sequence is ``(length,)``.
Tokens: TypeAlias = npt.NDArray[np.integer]

#: Objective values, shaped ``(n, n_objectives)``.
#: Single-objective landscapes return ``(n, 1)`` rather than ``(n,)``, so that
#: downstream code never needs to branch on how many objectives there are.
Fitness: TypeAlias = npt.NDArray[np.floating]

#: The 20 standard proteinogenic amino acids, in the conventional order.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: The four DNA bases.
DNA_BASES = "ACGT"

# Expected dimensionality of a single sequence and of a batch of them.
_SEQUENCE_NDIM = 1
_BATCH_NDIM = 2


@dataclass(frozen=True, slots=True)
class Alphabet:
    """An ordered, immutable set of tokens that sequences are written in.

    Token *index* is what the rest of the library works with; the symbol is for
    display and for reading data files. Order is significant and fixed: indices
    are used directly as positions in policy logits and transition matrices, so
    reordering an alphabet would silently reinterpret every trained model and
    cached landscape.

    Args:
        symbols: The tokens, in index order. Must be non-empty and unique.

    Raises:
        ValueError: If ``symbols`` is empty or contains duplicates.
    """

    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that the symbols form a usable alphabet."""
        if not self.symbols:
            raise ValueError("alphabet must contain at least one symbol")
        duplicates = {s for s in self.symbols if self.symbols.count(s) > 1}
        if duplicates:
            raise ValueError(f"alphabet symbols must be unique, repeated: {sorted(duplicates)}")

    @classmethod
    def from_string(cls, symbols: str) -> Alphabet:
        """Build an alphabet from a string, one character per token.

        Args:
            symbols: Characters to use as tokens, in index order.

        Returns:
            An alphabet whose token ``i`` is ``symbols[i]``.

        Example:
            >>> Alphabet.from_string("ACGT").size
            4
        """
        return cls(tuple(symbols))

    @classmethod
    def protein(cls) -> Alphabet:
        """The 20 standard amino acids.

        Returns:
            An alphabet over `AMINO_ACIDS`.
        """
        return cls.from_string(AMINO_ACIDS)

    @classmethod
    def dna(cls) -> Alphabet:
        """The four DNA bases.

        Returns:
            An alphabet over `DNA_BASES`.
        """
        return cls.from_string(DNA_BASES)

    @property
    def size(self) -> int:
        """Number of distinct tokens."""
        return len(self.symbols)

    def index_of(self, symbol: str) -> int:
        """Look up the index of a single symbol.

        Args:
            symbol: The token to look up.

        Returns:
            Its index in this alphabet.

        Raises:
            KeyError: If the symbol is not in the alphabet.
        """
        try:
            return self.symbols.index(symbol)
        except ValueError:
            raise KeyError(f"{symbol!r} is not in this alphabet") from None

    def encode(self, sequence: str) -> Tokens:
        """Convert a string to token indices.

        Args:
            sequence: Text written in this alphabet.

        Returns:
            A ``(len(sequence),)`` array of indices.

        Raises:
            KeyError: If any character is not in the alphabet.
        """
        return np.array([self.index_of(c) for c in sequence], dtype=np.int32)

    def encode_many(self, sequences: Iterable[str]) -> Tokens:
        """Convert several equal-length strings to a token matrix.

        Args:
            sequences: Texts written in this alphabet, all the same length.

        Returns:
            An ``(n, length)`` array of indices.

        Raises:
            KeyError: If any character is not in the alphabet.
            ValueError: If the sequences are not all the same length.
        """
        encoded = [self.encode(s) for s in sequences]
        if not encoded:
            return np.zeros((0, 0), dtype=np.int32)
        lengths = {len(e) for e in encoded}
        if len(lengths) > 1:
            raise ValueError(f"sequences must all be the same length, got {sorted(lengths)}")
        return np.stack(encoded)

    def decode(self, tokens: Tokens) -> str:
        """Convert one sequence of token indices back to text.

        Args:
            tokens: A one-dimensional array of indices.

        Returns:
            The corresponding string.

        Raises:
            ValueError: If ``tokens`` is not one-dimensional.
            IndexError: If any index is outside the alphabet.
        """
        array = np.asarray(tokens)
        if array.ndim != _SEQUENCE_NDIM:
            raise ValueError(f"expected a single sequence with ndim 1, got ndim {array.ndim}")
        return "".join(self._symbol_at(int(i)) for i in array)

    def decode_many(self, tokens: Tokens) -> list[str]:
        """Convert a batch of token indices back to text.

        Args:
            tokens: An ``(n, length)`` array of indices.

        Returns:
            One string per row.

        Raises:
            ValueError: If ``tokens`` is not two-dimensional.
            IndexError: If any index is outside the alphabet.
        """
        array = np.asarray(tokens)
        if array.ndim != _BATCH_NDIM:
            raise ValueError(f"expected a batch with ndim 2, got ndim {array.ndim}")
        return [self.decode(row) for row in array]

    def _symbol_at(self, index: int) -> str:
        """Return the symbol at ``index``, with a clearer error than a raw lookup."""
        if not 0 <= index < self.size:
            raise IndexError(f"token index {index} is outside alphabet of size {self.size}")
        return self.symbols[index]
