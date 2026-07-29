"""The design-build-test-learn round engine.

A campaign is what directed evolution actually is: a few rounds, each measuring
a batch of variants, each informed by everything measured before. This module
runs that loop identically for every sampler, so a difference in results is a
difference between methods rather than between harnesses.

What a round does
-----------------

#. The sampler proposes a **pool** of candidates -- far more than can be
   measured. Generating them is free.
#. The surrogate scores the pool; the acquisition rule turns predictions and
   uncertainty into one number per candidate.
#. The batch selector picks the ``batch_size`` that will actually be measured.
#. The oracle evaluates exactly those, and only those are charged.
#. The surrogate is refitted on everything measured so far, and the sampler is
   told what its proposals scored.

Why the sampler does not touch the oracle
-----------------------------------------

Training a GFlowNet takes thousands of reward evaluations. Charging those
against the oracle budget would exhaust a realistic 384-call campaign before the
first round finished, and no published method does it -- GFN-AL trains the
sampler against a learned proxy and spends the real budget only on the selected
batch. Getting this wrong does not produce an error; it produces a benchmark in
which the GFlowNet appears catastrophically sample-inefficient for a reason that
has nothing to do with GFlowNets.

The seam is deliberately implicit: a sampler that wants to train against the
surrogate is constructed with *the same surrogate instance* the campaign holds.
Refitting mutates it in place, so the sampler sees each round's model without
the campaign needing to know which samplers care.

Defaults
--------

Four rounds of 96, so 384 oracle calls. That is not a round number picked for
convenience -- it is the size of real ML-guided campaigns. ALDE (Arnold lab,
2025) screened 396 variants as six 96-well plates over three rounds; LaMBO-2's
wet-lab campaign measured 374 over three rounds. The iterative-benchmark
convention of 1,000-10,000 evaluations sits above even *classical* directed
evolution, and well above the regime where MLDE's advantage is claimed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.acquisition.rules import Greedy, TopK
from evogfn.loop.ledger import CampaignResult, RoundRecord
from evogfn.loop.provenance import write_manifest, write_round
from evogfn.metrics.diversity import diversity
from evogfn.tracking.base import NoOpTracker

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.acquisition.base import Acquisition, BatchSelector
    from evogfn.algorithms.base import Sampler
    from evogfn.core.types import Fitness, Tokens
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.surrogate.base import Surrogate
    from evogfn.tracking.base import Tracker

#: Variants per round. One 96-well plate, which is the unit a wet lab works in.
DEFAULT_BATCH_SIZE = 96

#: Rounds per campaign. Three to four is what published wet-lab campaigns run.
DEFAULT_ROUNDS = 4

#: Candidates generated per round before selection. Free, so generous.
DEFAULT_POOL_SIZE = 2048

#: Below two sequences, pairwise diversity is undefined rather than zero.
_MIN_FOR_DIVERSITY = 2


class Campaign:
    """Runs a sampler against a landscape under a fixed oracle budget.

    Args:
        landscape: The oracle. Every call to it is charged.
        sampler: The method under test.
        surrogate: Model fitted to the measurements and used to score the pool.
            ``None`` runs the sampler unassisted -- the ablation that says how
            much of the result is the surrogate rather than the sampler.
        acquisition: Turns predictions and uncertainty into one score. Defaults
            to [Greedy][evogfn.acquisition.rules.Greedy], the baseline the
            published nulls favour.
        selector: Picks the batch to measure. Defaults to
            [TopK][evogfn.acquisition.rules.TopK].
        rounds: How many design-build-test-learn cycles.
        batch_size: Variants measured per round.
        pool_size: Candidates generated per round before selection.
        initial_design: Sequences to measure in round 0. ``None`` takes them
            from the sampler, unassisted.
        skip_measured: Drop candidates already measured, and duplicates
            within the pool, before selection. A lab does not re-order a variant
            it has already assayed, and a sampler that has collapsed onto one
            mode would otherwise spend its whole budget re-measuring it. Setting
            this false disables deduplication entirely, which is the ablation
            that says how much of a method's apparent efficiency is the
            screening rather than the method.
        tracker: Where per-round metrics go.
        artifact_dir: Where to write each round's batch, as a chained artifact.
            ``None`` writes nothing, which is right for a benchmark sweep where
            the aggregate is the product. A campaign wants the opposite -- the
            designs that went to the lab in round three and what came back --
            and that is what this records.

    Raises:
        ValueError: If any size is not positive, or the pool is smaller than
            the batch it must be selected from.
    """

    def __init__(  # noqa: PLR0913 - a campaign is defined by its protocol
        self,
        *,
        landscape: FitnessLandscape,
        sampler: Sampler,
        surrogate: Surrogate | None = None,
        acquisition: Acquisition | None = None,
        selector: BatchSelector | None = None,
        rounds: int = DEFAULT_ROUNDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        initial_design: Tokens | None = None,
        skip_measured: bool = True,
        tracker: Tracker | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        """Configure the campaign without running it."""
        for name, value in [
            ("rounds", rounds),
            ("batch_size", batch_size),
            ("pool_size", pool_size),
        ]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if pool_size < batch_size:
            raise ValueError(
                f"pool_size {pool_size} is smaller than batch_size {batch_size}; "
                "there would be nothing to select from"
            )

        self._landscape = landscape
        self._sampler = sampler
        self._surrogate = surrogate
        self._acquisition = acquisition or Greedy()
        self._selector = selector or TopK()
        self._rounds = rounds
        self._batch_size = batch_size
        self._pool_size = pool_size
        self._initial_design = initial_design
        self._skip_measured = skip_measured
        self._tracker = tracker or NoOpTracker()
        self._artifact_dir = artifact_dir

    @property
    def budget(self) -> int:
        """Total oracle calls the campaign may spend."""
        return self._rounds * self._batch_size

    @property
    def sampler(self) -> Sampler:
        """The method under test, for reading its own accounting after a run."""
        return self._sampler

    def run(self) -> CampaignResult:
        """Execute every round and return the ledger.

        Returns:
            The measurements and the accounting behind them.
        """
        measured: list[Tokens] = []
        values: list[Fitness] = []
        seen: set[bytes] = set()
        records: list[RoundRecord] = []
        best_so_far = float("-inf")
        spent = 0

        for index in range(self._rounds):
            remaining = self.budget - spent
            if remaining <= 0:
                break

            proposed, screened, batch, predicted = self._design(
                index, measured, values, seen, remaining
            )
            if batch.shape[0] == 0:
                # The sampler cannot produce anything unmeasured. Stopping is
                # honest; padding the batch with repeats would spend the budget
                # to manufacture a full-looking ledger.
                break

            scores = self._landscape.evaluate(batch)
            spent += batch.shape[0]
            measured.append(batch)
            values.append(scores)
            seen.update(row.tobytes() for row in np.ascontiguousarray(batch))
            self._sampler.observe(batch, scores)

            record = self._record(
                index=index,
                proposed=proposed,
                screened=screened,
                batch=batch,
                scores=scores,
                previous_best=best_so_far,
                predicted=predicted,
            )
            if self._artifact_dir is not None:
                write_round(
                    self._artifact_dir,
                    record=record,
                    sequences=batch,
                    values=scores,
                    predicted=predicted,
                    tracker=self._tracker,
                )
            best_so_far = record.best_so_far
            records.append(record)
            self._tracker.log_metrics(
                {
                    "best_so_far": record.best_so_far,
                    "best_in_round": record.best_in_round,
                    "batch_diversity": record.batch_diversity,
                    "feasible_fraction": record.feasible_fraction,
                    "oracle_calls": float(spent),
                },
                step=index,
            )

        if self._artifact_dir is not None and records:
            write_manifest(self._artifact_dir, tuple(records))

        optimum = self._landscape.optimum
        return CampaignResult(
            sampler=self._sampler.name,
            rounds=tuple(records),
            sequences=(
                np.concatenate(measured)
                if measured
                else np.zeros((0, self._landscape.sequence_length), dtype=np.int32)
            ),
            values=(
                np.concatenate(values) if values else np.zeros((0, self._landscape.n_objectives))
            ),
            optimum=float(np.max(optimum)) if optimum is not None else None,
        )

    def _design(
        self,
        index: int,
        measured: list[Tokens],
        values: list[Fitness],
        seen: set[bytes],
        remaining: int,
    ) -> tuple[int, int, Tokens, npt.NDArray[np.floating] | None]:
        """Choose the batch, returning proposals generated, pool screened, and batch.

        Round 0 has nothing to fit a surrogate on, so it is the sampler's own
        proposals unassisted. Every later round refits on the accumulated
        measurements first. Deduplication applies from round 0 onward: a plate
        of eight copies of one variant is one experiment however it arose, and
        exempting the first round would let a collapsed sampler book a full
        opening plate for a single measurement's worth of information.
        """
        size = min(self._batch_size, remaining)
        fitted = False
        if index > 0 and self._surrogate is not None:
            # A method can fail to produce a single buildable design in a whole
            # round -- on a sparse feasible set an unmasked sampler routinely
            # does. There is then nothing to fit, and that is a result about the
            # method rather than an error: it proceeds unassisted and the ledger
            # records a feasible fraction of zero. Raising here would turn the
            # finding into a traceback and lose the rest of the campaign.
            history = np.concatenate(values)
            if np.isfinite(history).any():
                # In place, so any sampler holding this instance sees the update.
                self._surrogate.fit(np.concatenate(measured), history)
                fitted = True

        pool = self._pool(index)
        proposed = pool.shape[0]
        if self._skip_measured:
            pool = self._unmeasured(pool, seen)
        screened = pool.shape[0]
        # Round 0 has no model to score with, so the pool order stands.
        if screened == 0 or index == 0 or self._surrogate is None or not fitted:
            return proposed, screened, pool[:size], None

        mean, spread = self._surrogate.predict(pool)
        best_observed = self._best_observed(values)
        scored = self._acquisition.score(mean, spread, best_observed=best_observed)
        chosen = self._selector.select(pool, scored, size)
        return proposed, screened, pool[chosen], mean[chosen]

    def _pool(self, index: int) -> Tokens:
        """The candidates this round selects from, before deduplication."""
        if index == 0 and self._initial_design is not None:
            return np.asarray(self._initial_design)
        return self._sampler.propose(self._pool_size)

    def _unmeasured(self, pool: Tokens, seen: set[bytes]) -> Tokens:
        """Drop candidates already measured, and duplicates within the pool."""
        array = np.ascontiguousarray(pool)
        keep: list[int] = []
        batch_seen: set[bytes] = set()
        for position, row in enumerate(array):
            key = row.tobytes()
            if key in batch_seen or key in seen:
                continue
            batch_seen.add(key)
            keep.append(position)
        return array[keep]

    def _record(  # noqa: PLR0913 - a round record is its fields
        self,
        *,
        index: int,
        proposed: int,
        screened: int,
        batch: Tokens,
        scores: Fitness,
        previous_best: float,
        predicted: npt.NDArray[np.floating] | None = None,
    ) -> RoundRecord:
        """Summarise a completed round."""
        flat = np.asarray(scores, dtype=np.float64).reshape(scores.shape[0], -1).max(axis=1)
        finite = flat[np.isfinite(flat)]
        best_in_round = float(finite.max()) if finite.size else float("-inf")
        return RoundRecord(
            index=index,
            proposed=proposed,
            screened=screened,
            evaluated=batch.shape[0],
            feasible=int(np.isfinite(flat).sum()),
            best_in_round=best_in_round,
            best_so_far=max(previous_best, best_in_round),
            mean_in_round=float(finite.mean()) if finite.size else float("-inf"),
            batch_diversity=(diversity(batch) if batch.shape[0] >= _MIN_FOR_DIVERSITY else 0.0),
            surrogate_correlation=_correlation(predicted, flat),
        )

    @staticmethod
    def _best_observed(values: list[Fitness]) -> float:
        """Best finite measurement so far, for improvement-based acquisition.

        Args:
            values: One ``(n, 1)`` array of objective values per completed round.

        Returns:
            The largest finite value measured, or ``0.0`` if nothing finite has
            been measured yet -- the incumbent an improvement rule falls back to
            before there is one.

        Raises:
            ValueError: If the measurements carry more than one objective, where
                "the best value so far" is not defined without a scalarisation.
        """
        flat = _single_objective(np.concatenate(values))
        finite = flat[np.isfinite(flat)]
        return float(finite.max()) if finite.size else 0.0

    def __repr__(self) -> str:
        """Name the sampler and the budget it is held to."""
        return (
            f"Campaign(sampler={self._sampler.name}, rounds={self._rounds}, "
            f"batch_size={self._batch_size}, budget={self.budget})"
        )


def _correlation(
    predicted: npt.NDArray[np.floating] | None, measured: npt.NDArray[np.floating]
) -> float:
    """Pearson correlation between prediction and measurement, or ``nan``.

    Returns ``nan`` rather than zero when it cannot be computed -- no surrogate,
    fewer than two finite measurements, or a constant on either side. Zero would
    read as "the model is useless", which is a different claim from "there was
    nothing to correlate".
    """
    if predicted is None:
        return float("nan")
    usable = np.isfinite(predicted) & np.isfinite(measured)
    if usable.sum() < _MIN_FOR_DIVERSITY:
        return float("nan")
    left, right = predicted[usable], measured[usable]
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _single_objective(values: Fitness) -> npt.NDArray[np.float64]:
    """Accept ``(n,)`` or single-objective ``(n, 1)`` and return ``(n,)``.

    Args:
        values: Objective values for the measurements made so far.

    Returns:
        An ``(n,)`` array, one value per measured design.

    Raises:
        ValueError: If the input has more than one objective. Flattening it
            would hand the acquisition rule an incumbent taken across objectives
            of different scales, which no expected-improvement calculation is
            defined against.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:  # noqa: PLR2004 - a single objective
        return array[:, 0]
    raise ValueError(
        f"expected shape (n,) or (n, 1), got {array.shape}; multi-objective "
        f"measurements must be scalarised before an improvement-based acquisition "
        f"rule has an incumbent to improve on"
    )
