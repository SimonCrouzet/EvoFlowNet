"""The benchmark proper: a set of tasks, each able to decide something.

A benchmark is not a landscape and a number. It is a set of *tests*, each chosen
because it can settle a question the others cannot, run under a protocol a wet
lab would recognise. A suite where one task dominates measures that task.

Why GB1 is one row here and not the centre
-------------------------------------------

GB1 is the empirical anchor -- real measurements, a known optimum, 149,361 of
160,000 variants assayed -- and it is also the *easiest geometry in the suite*.
Four sites with a four-mutation budget means every sequence is reachable, so the
mutation constraint is vacuous; the space is 160,000, so a 384-call campaign
samples 0.24% of it; and there is no feasibility constraint at all. A method can
win on GB1 while being unable to do the three things directed evolution actually
demands: respect a constructibility constraint, search a space it cannot
enumerate, and work within a mutation budget far tighter than the sequence.

Ehrlich supplies those. It is closed-form, so the optimum is known by
construction rather than by exhaustive assay; it carries a discrete Markov
process whose transition matrix makes most of the space *infeasible*; and its
length, motif count and quantisation are free parameters, so difficulty can be
dialled rather than hoped for. Stanton et al. report quantisation dominating
difficulty, and their own genetic-algorithm baseline running at a feasible
fraction of 0.2-0.7 -- which under honest budget accounting is most of a
campaign spent on constructs that cannot be built.

What each task decides
----------------------

``gb1-anchor``
    Do the numbers hold on real measurements? The only task with an empirical
    landscape, and the weakest as a test of search.
``ehrlich-feasibility``
    Can a method stay inside the constructible set? Low transition density
    makes most sequences infeasible, so a rejection sampler burns its budget
    while a masked sampler cannot. This is the headline claim as an experiment.
``ehrlich-sparse``
    Does the method survive a sparse signal? Quantisation 1 means partial
    motifs score nothing, so there is no gradient to follow until a motif is
    complete.
``ehrlich-constrained``
    Does the method work when the mutation budget genuinely binds? Length 64
    with four mutations, so the reachable set is a vanishing shell rather than
    the whole space -- the case GB1 cannot express.
``ehrlich-epistatic``
    Four interacting motifs, so partial credit on one is worthless without the
    others. Interaction is what makes directed evolution hard.
``tiny-budget``
    Eight rounds of twelve, after EVOLVEpro. Ninety-six assays total. Methods
    that need a few hundred observations to fit a surrogate should fail here,
    and it is worth knowing which do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evoflownet.benchmark.protocol import PLATE, Protocol
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.landscapes.gb1 import GB1Landscape

if TYPE_CHECKING:
    from collections.abc import Callable

    from evoflownet.core.types import Tokens
    from evoflownet.landscapes.base import FitnessLandscape


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


def _ehrlich(**kwargs: object) -> Callable[[], FitnessLandscape]:
    """A factory for an Ehrlich instance with fixed parameters."""

    def build() -> FitnessLandscape:
        return EhrlichLandscape(**kwargs)  # type: ignore[arg-type]

    return build


#: The suite. Each row states what it decides; a row that decides nothing the
#: others do not should be removed rather than kept for completeness.
SUITE: tuple[Task, ...] = (
    Task(
        name="gb1-anchor",
        purpose="Do the numbers hold on real measurements? Empirical, but the "
        "easiest geometry here: 4 sites, no feasibility constraint, and a "
        "mutation budget that reaches everything.",
        build=GB1Landscape,
        protocol=Protocol(rounds=4, batch_size=PLATE, max_mutations=4, label="four plates"),
        max_mutations=4,
    ),
    Task(
        name="ehrlich-feasibility",
        purpose="Can a method stay inside the constructible set? Sparse "
        "transitions make most sequences infeasible, so rejection sampling "
        "burns budget where masking cannot.",
        build=_ehrlich(
            sequence_length=32,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.15,
            seed=0,
        ),
        protocol=Protocol(rounds=4, batch_size=PLATE, max_mutations=8, label="four plates"),
        max_mutations=8,
    ),
    Task(
        name="ehrlich-sparse",
        purpose="Does the method survive a sparse signal? Quantisation 1 gives "
        "partial motifs no credit, so there is no gradient until one completes. "
        "Stanton et al. report this parameter dominating difficulty.",
        build=_ehrlich(
            sequence_length=32,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            quantization=1,
            transition_density=0.5,
            seed=1,
        ),
        protocol=Protocol(rounds=4, batch_size=PLATE, max_mutations=8, label="four plates"),
        max_mutations=8,
    ),
    Task(
        name="ehrlich-constrained",
        purpose="Does the method work when the mutation budget genuinely binds? "
        "Length 64 with four mutations makes the reachable set a vanishing "
        "shell -- the case GB1 cannot express.",
        build=_ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        protocol=Protocol(rounds=4, batch_size=PLATE, max_mutations=4, label="four plates"),
        max_mutations=4,
    ),
    Task(
        name="ehrlich-epistatic",
        purpose="Four interacting motifs, so partial credit on one is worthless "
        "without the others. Interaction is what makes the problem hard.",
        build=_ehrlich(
            sequence_length=48,
            vocab_size=20,
            n_motifs=4,
            motif_length=4,
            transition_density=0.5,
            seed=3,
        ),
        protocol=Protocol(rounds=4, batch_size=PLATE, max_mutations=8, label="four plates"),
        max_mutations=8,
    ),
    Task(
        name="tiny-budget",
        purpose="Ninety-six assays total, after EVOLVEpro's eight rounds of "
        "twelve. Methods needing a few hundred observations to fit a surrogate "
        "should fail here, and it is worth knowing which do.",
        build=_ehrlich(
            sequence_length=32,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=4,
        ),
        protocol=Protocol(rounds=8, batch_size=12, max_mutations=8, label="EVOLVEpro"),
        max_mutations=8,
    ),
)

#: Tasks by name, for selecting a subset without retyping the definitions.
BY_NAME: dict[str, Task] = {task.name: task for task in SUITE}
