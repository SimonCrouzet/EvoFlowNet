"""A landscape backed by a surrogate, so a sampler can train without an oracle.

This is the seam that keeps the budget honest. The GFlowNet trainer takes a
:class:`~evogfn.landscapes.base.FitnessLandscape` and calls it thousands of
times; pointing it at the real oracle would spend a 384-call campaign budget in
the first few gradient steps. Wrapping the surrogate in the same interface lets
the entire existing training path run unchanged against the proxy, and makes it
structurally impossible to confuse the two -- a ``ProxyLandscape`` has no oracle
to call.

This is what GFN-AL does, and what the design-build-test-learn loop means: the
model is fitted on what the assay measured, the sampler is optimised against the
model, and the assay is spent only on the batch that comes back.

The proxy is deliberately a live view, not a snapshot. It holds the surrogate
instance the campaign refits each round, so a sampler constructed once at the
start of a campaign trains against the current model every round without anyone
having to hand it a new one.

Uncertainty is available but off by default
-------------------------------------------

``optimism`` adds ``kappa * std`` to the prediction, making the proxy an
optimistic model of the landscape rather than a mean one. At zero -- the default
-- the sampler chases the surrogate's mean and exploration comes from the
GFlowNet's own reward-proportional sampling, which is the comparison worth
running first: it isolates what the *sampler* contributes from what an
exploration bonus contributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    from evogfn.core.types import Alphabet, Fitness, Tokens
    from evogfn.surrogate.base import Surrogate


class ProxyLandscape(FitnessLandscape):
    """Presents a fitted surrogate as a landscape, costing nothing to evaluate.

    Args:
        surrogate: The model to score with. Held by reference, so refitting it
            elsewhere updates this proxy.
        alphabet: The alphabet sequences are written in.
        sequence_length: Length of the sequences being scored.
        optimism: Weight on the surrogate's uncertainty. Zero scores the mean.

    Raises:
        ValueError: If ``optimism`` is negative.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        *,
        alphabet: Alphabet,
        sequence_length: int,
        optimism: float = 0.0,
    ) -> None:
        """Wrap the surrogate without copying it."""
        if optimism < 0:
            raise ValueError(f"optimism must be non-negative, got {optimism}")
        self._surrogate = surrogate
        self._alphabet = alphabet
        self._length = sequence_length
        self._optimism = optimism

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of every sequence this proxy scores."""
        return self._length

    @property
    def surrogate(self) -> Surrogate:
        """The model being used as the reward, for inspection."""
        return self._surrogate

    @property
    def is_ready(self) -> bool:
        """Whether the underlying surrogate has been fitted.

        A campaign's first round has nothing to fit on, so a sampler should
        check this rather than assume the proxy can be trained against.
        """
        return self._surrogate.is_fitted

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score with the surrogate's prediction, optionally made optimistic.

        Raises:
            RuntimeError: If the surrogate has not been fitted. Returning an
                unfitted network's output would look like a reward signal and
                would train the policy toward its initialisation.
        """
        mean, spread = self._surrogate.predict(sequences)
        combined = np.asarray(mean, dtype=np.float64)
        if self._optimism:
            combined = combined + self._optimism * np.asarray(spread, dtype=np.float64)
        return combined.reshape(-1, 1)

    def __repr__(self) -> str:
        """Name the proxy and its optimism."""
        return f"ProxyLandscape({self._surrogate!r}, optimism={self._optimism})"
