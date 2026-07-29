"""The fitness landscape interface.

A landscape is the thing being optimised against: it maps sequences to objective
values. In a real campaign it stands in for an assay, and evaluating it is the
expensive step that the whole method exists to economise on.

Two properties are deliberately part of the interface even though most real
landscapes cannot provide them:

* [FitnessLandscape.optimum][evogfn.landscapes.base.FitnessLandscape.optimum] --
  the best attainable objective values, when known by construction or by
  exhaustive measurement. It makes *regret* exact rather than relative to the
  best sequence seen so far.
*
  [FitnessLandscape.enumerate][evogfn.landscapes.base.FitnessLandscape.enumerate]
  -- every sequence in the space, when the space is small enough. It makes the
  target distribution ``p*(x)`` computable in closed form, which is the only way
  to check that a sampler is sampling rather than hill-climbing.

Landscapes that cannot answer these return ``None`` and raise respectively; the
benchmarks in this package were chosen precisely because they can.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Fitness, Tokens

#: Refuse to enumerate a space larger than this. Enumeration materialises an
#: ``(N, L)`` integer array, so the guard exists to turn "this would have needed
#: 40TB of RAM" into an error that names the number.
MAX_ENUMERABLE_SIZE = 5_000_000


class FitnessLandscape(ABC):
    """Maps sequences to objective values.

    Subclasses implement `_evaluate`. The public
    [evaluate][evogfn.landscapes.base.FitnessLandscape.evaluate] validates its
    input first, so no subclass has to repeat those checks and none can forget
    them.
    """

    @property
    @abstractmethod
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        """Length of every sequence this landscape scores."""

    @property
    def n_objectives(self) -> int:
        """Number of objectives. Single-objective landscapes return 1."""
        return 1

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Names of the objectives, for labelling output.

        Returns:
            One name per objective.
        """
        if self.n_objectives == 1:
            return ("fitness",)
        return tuple(f"objective_{i}" for i in range(self.n_objectives))

    @property
    def search_space_size(self) -> int:
        """Total number of distinct sequences, feasible or not."""
        return int(self.alphabet.size**self.sequence_length)

    @property
    def optimum(self) -> Fitness | None:
        """Best attainable objective values, or ``None`` if unknown.

        Returns:
            A ``(n_objectives,)`` array, or ``None`` when the landscape cannot
            say -- which is the honest answer for most real assays.
        """
        return None

    @abstractmethod
    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score a validated batch of sequences.

        Args:
            sequences: An ``(n, sequence_length)`` array of token indices,
                already checked for shape and range.

        Returns:
            An ``(n, n_objectives)`` array of objective values.
        """

    def evaluate(self, sequences: Tokens) -> Fitness:
        """Score a batch of sequences.

        Args:
            sequences: An ``(n, sequence_length)`` array of token indices.

        Returns:
            An ``(n, n_objectives)`` array of objective values.

        Raises:
            ValueError: If the input is not a two-dimensional array of the
                expected width, or contains indices outside the alphabet.
        """
        checked = self._validate(sequences)
        values = self._evaluate(checked)
        expected = (checked.shape[0], self.n_objectives)
        if values.shape != expected:
            raise ValueError(
                f"{type(self).__name__}._evaluate returned {values.shape}, expected {expected}"
            )
        return values

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences the landscape considers constructible.

        Landscapes with no feasibility notion accept everything. Where a
        constraint does exist it is a property of the *landscape*, and the
        matching environment masks it during generation so that infeasible
        sequences are never proposed in the first place.

        Args:
            sequences: An ``(n, sequence_length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.

        Raises:
            ValueError: If the input fails validation.
        """
        checked = self._validate(sequences)
        return np.ones(checked.shape[0], dtype=np.bool_)

    def enumerate(self) -> Tokens:
        """Every sequence in the search space.

        Only usable on small spaces; this is what makes exact distributional
        comparison possible on the benchmark landscapes.

        Returns:
            An ``(search_space_size, sequence_length)`` array of token indices,
            in odometer order with the last position varying fastest.

        Raises:
            ValueError: If the space is larger than `MAX_ENUMERABLE_SIZE`.
        """
        size = self.search_space_size
        if size > MAX_ENUMERABLE_SIZE:
            raise ValueError(
                f"search space has {size:,} sequences, above the "
                f"{MAX_ENUMERABLE_SIZE:,} enumeration limit"
            )
        grids = np.meshgrid(
            *([np.arange(self.alphabet.size, dtype=np.int32)] * self.sequence_length),
            indexing="ij",
        )
        return np.stack([g.ravel() for g in grids], axis=-1)

    def _validate(self, sequences: Tokens) -> Tokens:
        """Check shape and token range, returning the array as integers."""
        array = np.asarray(sequences)
        if array.ndim != 2:  # noqa: PLR2004 - a batch is two-dimensional, by definition
            raise ValueError(f"expected a batch with ndim 2, got ndim {array.ndim}")
        if array.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected sequences of length {self.sequence_length}, got {array.shape[1]}"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"expected integer token indices, got dtype {array.dtype}")
        if array.size and (array.min() < 0 or array.max() >= self.alphabet.size):
            raise ValueError(
                f"token indices must lie in [0, {self.alphabet.size}), got "
                f"[{array.min()}, {array.max()}]"
            )
        return array
