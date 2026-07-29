"""Per-round artifacts, so a campaign is a chain rather than a summary.

A benchmark needs aggregate numbers. A campaign needs the opposite: the actual
designs that went out to the lab in round three, the values that came back, and
what the model believed at the time. Those are what someone asks for six months
later when a variant turns out to matter, and they are exactly what a mean
regret discards.

Round *N* → round *N+1* is a lineage, not a sequence of independent batches: the
batch proposed in round three exists because of what rounds zero to two
measured. Writing each round as its own artifact, named by index and carrying
the model's prediction alongside the assay result, is what makes that navigable
after the fact -- and what makes a disagreement between prediction and
measurement attributable to a round rather than to the campaign as a whole.

Why files rather than tracker calls alone
------------------------------------------

:meth:`~evogfn.tracking.base.Tracker.log_artifact` takes a path, so
something has to write the file. Doing it here rather than inside the tracker
keeps the format one thing instead of one per backend, and means a run with
``tracker=noop`` still leaves a readable trail on disk -- which is the case that
actually matters when a campaign is being debugged rather than demonstrated.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from evogfn.core.types import Fitness, Tokens
    from evogfn.loop.ledger import RoundRecord
    from evogfn.tracking.base import Tracker

#: Columns every round file carries, in order.
FIELDS = ("index", "sequence", "predicted", "measured", "feasible")


def write_round(  # noqa: PLR0913 - a round is defined by what it produced
    directory: Path,
    *,
    record: RoundRecord,
    sequences: Tokens,
    values: Fitness,
    predicted: np.ndarray | None = None,
    tracker: Tracker | None = None,
) -> Path:
    """Write one round's batch to disk and register it with the tracker.

    Args:
        directory: Where to write. Created if absent.
        record: The round's ledger entry, for the index and its summary.
        sequences: The designs measured this round.
        values: What the oracle returned for them.
        predicted: What the surrogate expected, when there was a surrogate.
            Kept beside the measurement because the two disagreeing is the most
            useful signal a round produces, and reconstructing it later means
            re-fitting the model as it stood at the time.
        tracker: Where to register the artifact. ``None`` writes the file
            without registering, which is the right behaviour for a local run.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"round-{record.index:03d}.csv"

    designs = np.asarray(sequences)
    measured = np.asarray(values, dtype=np.float64).reshape(designs.shape[0], -1).max(axis=1)
    expected = np.full(designs.shape[0], np.nan) if predicted is None else np.asarray(predicted)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for position, design in enumerate(designs):
            writer.writerow(
                [
                    record.index,
                    " ".join(str(int(token)) for token in design),
                    f"{expected[position]:.6g}",
                    f"{measured[position]:.6g}",
                    int(np.isfinite(measured[position])),
                ]
            )

    if tracker is not None:
        # Naming by index is what makes the chain navigable: an artifact
        # browser sorts them into the order the campaign actually ran.
        tracker.log_artifact(path, name=f"round-{record.index:03d}")
    return path


def write_manifest(directory: Path, records: tuple[RoundRecord, ...]) -> Path:
    """Write the round-by-round summary that ties the batch files together.

    One row per round, so the lineage can be read without opening every batch:
    what was proposed, what survived screening, what was measured, and how well
    the model predicted it.

    Args:
        directory: Where to write. Created if absent.
        records: The campaign's rounds, in order.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rounds.csv"
    columns = (
        "index",
        "proposed",
        "screened",
        "evaluated",
        "feasible",
        "feasible_fraction",
        "best_in_round",
        "best_so_far",
        "batch_diversity",
        "surrogate_correlation",
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for record in records:
            writer.writerow(
                [
                    record.index,
                    record.proposed,
                    record.screened,
                    record.evaluated,
                    record.feasible,
                    f"{record.feasible_fraction:.6g}",
                    f"{record.best_in_round:.6g}",
                    f"{record.best_so_far:.6g}",
                    f"{record.batch_diversity:.6g}",
                    f"{record.surrogate_correlation:.6g}",
                ]
            )
    return path
