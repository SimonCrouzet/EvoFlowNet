"""What a campaign spent, and on what.

The oracle budget is the whole constraint. A directed-evolution round measures
somewhere between a few dozen and a few thousand variants, and the surveyed
literature puts a realistic total in the hundreds -- so a result reported at
20,000 evaluations is not a result about directed evolution. The ledger exists
so that the number every claim is indexed by cannot be quietly wrong.

Four counts are kept separate because they diverge, and the gaps are the
interesting part:

* **proposals** -- candidates the sampler generated. Free. A rejection-sampling
  baseline under a feasibility constraint can generate ten times what it keeps,
  and reporting only oracle calls would hide that entirely.
* **screened** -- proposals that survived deduplication and reached the
  selector. A sampler that has collapsed onto one mode re-proposes what it has
  already measured, and the gap from proposals to screened is where that shows.
* **evaluated** -- oracle calls charged. The constrained resource.
* **feasible** -- of those evaluated, how many the landscape could actually
  build.

Infeasible designs are charged
------------------------------

They cost the same to synthesise as feasible ones, so a method that proposes
them has spent the budget. Stanton et al. report their genetic-algorithm
baseline running at a feasible population fraction of 0.2-0.7, which under this
accounting is most of a budget spent on constructs that cannot be made. Charging
for them is what turns a masked sampler's feasibility-by-construction from a
stated property into a measured advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evoflownet.core.types import Fitness, Tokens


@dataclass(frozen=True)
class RoundRecord:
    """What one design-build-test-learn round did.

    Attributes:
        index: Zero-based round number. Round 0 is the initial design, which is
            charged like any other -- seed data is not free.
        proposed: Candidates the sampler generated.
        screened: Proposals that reached the selector after deduplication.
        evaluated: Oracle calls charged this round.
        feasible: How many of the evaluated candidates were constructible.
        best_in_round: Best objective value measured this round.
        best_so_far: Best measured since the campaign began.
        mean_in_round: Mean objective value over the feasible measurements.
        batch_diversity: Mean pairwise Hamming distance within the batch. What
            a lab actually receives: a batch of near-duplicates is one
            experiment repeated, whatever its mean predicted value.
        surrogate_correlation: Pearson correlation between what the surrogate
            predicted for this batch and what the oracle measured. ``nan``
            before a surrogate exists. This is the single most useful
            diagnostic in the ledger: it separates a method failing because its
            sampler proposes badly from one failing because its model cannot
            tell good designs from bad, and those call for opposite fixes.
    """

    index: int
    proposed: int
    screened: int
    evaluated: int
    feasible: int
    best_in_round: float
    best_so_far: float
    mean_in_round: float
    batch_diversity: float
    surrogate_correlation: float = float("nan")

    @property
    def feasible_fraction(self) -> float:
        """Share of the round's oracle calls spent on constructible designs."""
        return self.feasible / self.evaluated if self.evaluated else 0.0

    @property
    def rejection_ratio(self) -> float:
        """Proposals generated per oracle call charged.

        One means the sampler proposed exactly what was measured. Large values
        mean it is discarding most of its own output, which is free in compute
        terms and worth seeing.
        """
        return self.proposed / self.evaluated if self.evaluated else float("inf")


@dataclass(frozen=True)
class CampaignResult:
    """Everything a campaign measured, and the ledger of how it was spent.

    Attributes:
        sampler: Name of the sampler under test.
        rounds: One record per completed round, in order.
        sequences: Every sequence evaluated, in evaluation order.
        values: Their objective values, aligned with ``sequences``.
        optimum: The landscape's best attainable value, when it knows it.
    """

    sampler: str
    rounds: tuple[RoundRecord, ...]
    sequences: Tokens
    values: Fitness
    optimum: float | None = None

    @property
    def oracle_calls(self) -> int:
        """Total oracle calls charged -- the number every claim is indexed by."""
        return sum(record.evaluated for record in self.rounds)

    @property
    def proposals(self) -> int:
        """Total candidates generated, including those never evaluated."""
        return sum(record.proposed for record in self.rounds)

    @property
    def best_value(self) -> float:
        """Best objective value measured, or ``-inf`` if nothing was feasible."""
        finite = self.values[np.isfinite(self.values)]
        return float(finite.max()) if finite.size else float("-inf")

    @property
    def feasible_fraction(self) -> float:
        """Share of the whole budget spent on constructible designs."""
        calls = self.oracle_calls
        return sum(r.feasible for r in self.rounds) / calls if calls else 0.0

    @property
    def simple_regret(self) -> float | None:
        """Distance from the best measurement to the true optimum.

        Returns:
            ``optimum - best_value``, or ``None`` when the landscape does not
            know its optimum -- which is the honest answer for a real assay.
        """
        if self.optimum is None:
            return None
        return self.optimum - self.best_value

    def trace(self) -> list[float]:
        """Best-so-far after each round, for plotting a budget curve."""
        return [record.best_so_far for record in self.rounds]

    def summary(self) -> dict[str, float]:
        """Flat metrics for logging, keyed for a tracker."""
        metrics = {
            "oracle_calls": float(self.oracle_calls),
            "proposals": float(self.proposals),
            "best_value": self.best_value,
            "feasible_fraction": self.feasible_fraction,
            "rounds": float(len(self.rounds)),
        }
        if (regret := self.simple_regret) is not None:
            metrics["simple_regret"] = regret
        return metrics
