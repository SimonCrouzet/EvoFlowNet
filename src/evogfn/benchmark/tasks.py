"""What a benchmark task is: a landscape, a protocol, and a reason to run it.

A benchmark is not a landscape and a number. It is a set of *tests*, each chosen
because it can settle a question the others cannot, run under a protocol a wet
lab would recognise. This module holds only the shape of one such test; the
tests themselves live in [evogfn.benchmark.suite][].

That separation is the point, and it is enforced by there being nothing else
here. A second list of tasks alongside the one the results come from is not
redundant, it is wrong: this module previously carried its own suite in which
"the feasibility task" was 32 residues while ``suite.MAIN`` runs it at 64, so
the name meant two different experiments depending on which import a reader
followed, and nothing in either definition said which had produced the numbers.

The one field a task cannot omit is `Task.purpose`. A suite is only as
good as its ability to distinguish methods, so a row that cannot say what it
decides that the others do not should be deleted rather than kept for
completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape

if TYPE_CHECKING:
    from collections.abc import Callable

    from evogfn.benchmark.protocol import Protocol
    from evogfn.core.types import Tokens
    from evogfn.landscapes.base import FitnessLandscape


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark test: a landscape, a protocol, and a reason to run it.

    Attributes:
        name: Short identifier.
        purpose: What this task can decide that the others cannot. Present so a
            suite cannot silently accumulate rows that measure the same thing.
        build: Makes the landscape. A factory rather than an instance, so each
            seed can draw its own instance where that is meaningful.
        protocol: Rounds and batch size.
        max_mutations: How far a design may stray from the parent.
        parent_seed: Seeds the starting sequence.
    """

    name: str
    purpose: str
    build: Callable[[], FitnessLandscape]
    protocol: Protocol
    max_mutations: int
    parent_seed: int = 0

    def landscape(self) -> FitnessLandscape:
        """Build this task's landscape."""
        return self.build()

    def parent(self, landscape: FitnessLandscape) -> Tokens:
        """The wild type a campaign starts from.

        On a landscape with a feasibility constraint this must be a
        constructible sequence: an infeasible parent scores minus infinity and
        leaves a mutation-based sampler with nothing to climb from.

        Args:
            landscape: The landscape being searched.

        Returns:
            A starting sequence of the landscape's length.
        """
        if isinstance(landscape, EhrlichLandscape):
            return landscape.feasible_sequence(self.parent_seed)
        if isinstance(landscape, GB1Landscape):
            return landscape.wild_type
        raise TypeError(f"no wild type defined for {type(landscape).__name__}")

    @property
    def constrains_search(self) -> bool:
        """Whether the mutation budget actually restricts the reachable set."""
        return self.protocol.constrains_search(self.build().sequence_length)

    def __repr__(self) -> str:
        """Name the task and its budget."""
        return f"{self.name} ({self.protocol!r})"
