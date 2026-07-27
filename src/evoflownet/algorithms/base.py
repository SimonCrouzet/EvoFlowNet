"""The sampler interface, shared by GFlowNets and the classical baselines.

One interface, deliberately. A benchmark in which the method under test and the
methods it is compared against run through different harnesses is not a
comparison -- the budget accounting drifts, the stopping conditions differ, and
the result measures the harness as much as the method. Here every sampler is
driven by the same loop and charged for the same thing.

Two calls, matching what a design round actually does:

* :meth:`Sampler.propose` -- hand me candidates to evaluate.
* :meth:`Sampler.observe` -- here is what they scored.

A genetic algorithm keeps a population between the two; a GFlowNet updates a
policy; random mutagenesis ignores ``observe`` entirely. None of that is visible
to the harness, which is the point.

Proposals are not the same as oracle calls
------------------------------------------

:attr:`Sampler.proposals_made` counts candidates *generated*, which can far
exceed the number *evaluated* when a sampler rejects its own output -- a
rejection-sampling GA under a feasibility constraint being the case this
distinction exists for. Reporting only oracle calls would make such a sampler
look free; reporting only proposals would make an expensive oracle look cheap.
Both are recorded so the trade can be seen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evoflownet.core.types import Fitness, Tokens


class Sampler(ABC):
    """Proposes candidate sequences and, optionally, learns from their scores."""

    def __init__(self) -> None:
        """Initialise proposal accounting."""
        self._proposals_made = 0

    @property
    def proposals_made(self) -> int:
        """Candidates generated, including any the sampler itself discarded.

        Differs from oracle calls whenever a sampler filters its own output.
        """
        return self._proposals_made

    @property
    def name(self) -> str:
        """Short label for reporting."""
        return type(self).__name__

    @abstractmethod
    def propose(self, n: int) -> Tokens:
        """Generate ``n`` candidates to be evaluated.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array of token indices.
        """

    def observe(self, sequences: Tokens, values: Fitness) -> None:  # noqa: B027
        """Learn from evaluated candidates.

        Deliberately a no-op default rather than abstract: samplers that do
        not adapt should not be forced to write an empty override, and
        random mutagenesis is the honest case for that.

        Args:
            sequences: The candidates that were evaluated.
            values: An ``(n, n_objectives)`` array of their objective values.
        """

    def _count(self, n: int) -> None:
        """Record that ``n`` candidates were generated."""
        self._proposals_made += n

    def __repr__(self) -> str:
        """Name the sampler."""
        return f"{self.name}()"
