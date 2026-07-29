"""Choosing which candidates to spend the budget on.

An acquisition function turns a surrogate's prediction and uncertainty into a
single score, and a batch selector turns those scores into the set actually
sent for measurement. Both matter, and the second is the one usually skipped.

Why the batch step is separate
------------------------------

Taking the top ``k`` by score is not the same as choosing a good batch of ``k``.
The top ``k`` are typically near-identical -- neighbouring points on one peak --
so a round buys one bet repeated ``k`` times. Whether that is bad is an
empirical question and a contested one, which is precisely why selection is a
replaceable component here rather than a hard-coded ``argsort``.

The evidence, honestly
----------------------

Greenman et al. (PLOS Comput Biol 2025) report that uncertainty-based methods
*"never outperform greedily sampling the sequences with the highest predicted
values"* -- but with sequential, single-point acquisition. GameOpt (ICLR 2025)
sweeps batch size on GB1 and reports a crossover: greedy wins at batch 1, the
diverse method wins increasingly as the batch grows. Both cannot be generalised
past their own regimes, so both selectors ship and the comparison is run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from evogfn.core.types import Tokens


class Acquisition(ABC):
    """Scores candidates from a surrogate's mean and uncertainty."""

    @abstractmethod
    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],
        *,
        best_observed: float,
    ) -> npt.NDArray[np.floating]:
        """Rank candidates. Higher is better.

        Args:
            mean: ``(n,)`` predicted values.
            std: ``(n,)`` predictive standard deviations.
            best_observed: Best value measured so far, for improvement-based
                rules.

        Returns:
            An ``(n,)`` array of scores.
        """

    def __repr__(self) -> str:
        """Name the acquisition function."""
        return f"{type(self).__name__}()"


class BatchSelector(ABC):
    """Chooses which scored candidates to actually measure."""

    @abstractmethod
    def select(
        self,
        candidates: Tokens,
        scores: npt.NDArray[np.floating],
        k: int,
    ) -> npt.NDArray[np.intp]:
        """Pick ``k`` candidates to send for measurement.

        Args:
            candidates: An ``(n, length)`` array of proposals.
            scores: An ``(n,)`` array of acquisition scores.
            k: How many to select.

        Returns:
            Indices of the chosen candidates.
        """

    def __repr__(self) -> str:
        """Name the selector."""
        return f"{type(self).__name__}()"
