"""The learned stand-in for an expensive oracle.

This is the component that makes a realistic budget workable. A wet-lab round
screens hundreds of variants, not millions, so a sampler trained directly
against the assay sees almost no data. The published arrangement (Jain et al.,
ICML 2022) instead fits a surrogate to the few real measurements and trains the
sampler against *that*, which is free. Only the surrogate's training set costs
budget.

That distinction is easy to lose and expensive to lose. Charging a GFlowNet's
gradient steps against the oracle budget makes it look absurdly sample-hungry
compared with a genetic algorithm, when the two are not being asked to pay for
the same thing.

Uncertainty is not optional here
--------------------------------

With a few hundred observations in a space of 10^14, a surrogate is uninformed
almost everywhere. A point prediction off that tiny manifold is not a weak
estimate, it is very nearly noise, and optimising it confidently is how a method
spends a whole round on an artefact. Every surrogate here therefore returns a
spread as well as a mean, and the acquisition layer is expected to use it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from evoflownet.core.types import Fitness, Tokens


class Surrogate(ABC):
    """Predicts objective values, with a measure of its own uncertainty."""

    @abstractmethod
    def fit(self, sequences: Tokens, values: Fitness) -> None:
        """Train on everything measured so far.

        Called once per round with the full accumulated dataset rather than the
        newest batch, so that a surrogate is free to refit from scratch. Warm
        starting across rounds is an optimisation, not a requirement.

        Args:
            sequences: An ``(n, length)`` array of measured sequences.
            values: An ``(n, n_objectives)`` array of their measured values.
        """

    @abstractmethod
    def predict(
        self, sequences: Tokens
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Predict values and epistatic uncertainty.

        Args:
            sequences: An ``(n, length)`` array to score.

        Returns:
            Mean predictions ``(n,)`` and standard deviations ``(n,)``. The
            spread must reflect *model* uncertainty -- disagreement about what
            the value is -- not assay noise, since it is what tells an
            acquisition function where the model is guessing.
        """

    @property
    def is_fitted(self) -> bool:
        """Whether the surrogate has seen any data."""
        return True

    def __repr__(self) -> str:
        """Name the surrogate."""
        return f"{type(self).__name__}()"
