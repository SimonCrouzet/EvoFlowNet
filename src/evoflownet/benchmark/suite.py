"""The suite: main tests that carry claims, diagnostics that inform them.

Two tiers, because they answer different kinds of question and should not be
read the same way.

**Main tests** are the headline. Each is a landscape and a protocol chosen so
that a result on it means something a lab would recognise, and each is sized so
the claim survives the strongest control available. These go in the paper's main
table.

**Diagnostics** vary one axis on a fixed, cheap landscape. They decide things --
which objective to carry into the main table, whether the ranking survives a
change of budget, whether rounds matter at fixed total. They are how choices get
made, not what gets claimed.

Sequence lengths follow published practice rather than convenience
------------------------------------------------------------------

Stanton et al.'s own base configuration is ``L = 256, c = 4, k = 8, q = 4``, and
they sweep around it. HDBO uses ``L = 5, 15, 64``, and reports two published
Bayesian-optimisation methods running out of memory at 64. holo-bench ships
``dim = 7`` defaults for quick enumerable use. So:

* the flagship large-space task uses **Stanton's base configuration**, which
  makes our numbers directly comparable to the benchmark's authors;
* the mid-size tasks use **L = 64**, the setting where the published field
  degrades;
* diagnostics use **L = 32**, cheap enough to sweep an axis at 50 seeds.

Mutation budget is 4 everywhere, so a conclusion drawn on a diagnostic transfers
to the main table rather than being confounded by a different search radius.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evoflownet.benchmark.protocol import PLATE, Protocol, round_sweep
from evoflownet.benchmark.tasks import Task
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.landscapes.gb1 import GB1Landscape
from evoflownet.metrics.diversity import diversity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evoflownet.benchmark.methods import Methodology
    from evoflownet.benchmark.store import ResultStore, RunRecord
    from evoflownet.landscapes.base import FitnessLandscape
    from evoflownet.loop.ledger import CampaignResult

#: Mutation budget, held constant so diagnostics transfer to the main table.
MUTATIONS = 4


def _ehrlich(**kwargs: object) -> Callable[[], FitnessLandscape]:
    """A factory for an Ehrlich instance with fixed parameters."""

    def build() -> FitnessLandscape:
        return EhrlichLandscape(**kwargs)  # type: ignore[arg-type]

    return build


def _task(
    name: str,
    purpose: str,
    build: Callable[[], FitnessLandscape],
    protocol: Protocol,
) -> Task:
    """A task at the shared mutation budget."""
    return Task(
        name=name,
        purpose=purpose,
        build=build,
        protocol=protocol,
        max_mutations=MUTATIONS,
    )


#: Stanton et al.'s base configuration, so our numbers sit beside theirs.
STANTON_BASE = _ehrlich(
    sequence_length=256,
    vocab_size=20,
    n_motifs=4,
    motif_length=8,
    quantization=4,
    transition_density=0.5,
    seed=0,
)

MAIN: tuple[Task, ...] = (
    _task(
        "gb1-anchor",
        "Do the numbers hold on real measurements? The empirical anchor, and "
        "the easiest geometry here: four sites, no feasibility constraint, and "
        "a mutation budget that reaches every sequence.",
        GB1Landscape,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=MUTATIONS, label="four plates"),
    ),
    _task(
        "large-space",
        "Can the method search a space it cannot enumerate? Stanton et al.'s "
        "base configuration, L=256 with four motifs of length eight. The "
        "reachable set is ~10^13 designs and the budget is 384.",
        STANTON_BASE,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=MUTATIONS, label="four plates"),
    ),
    _task(
        "feasibility",
        "Can the method stay inside the constructible set? A sparse transition "
        "matrix makes most sequences unbuildable, so rejection sampling spends "
        "the budget on designs that cannot be made while masking cannot.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.15,
            seed=1,
        ),
        Protocol(rounds=4, batch_size=PLATE, max_mutations=MUTATIONS, label="four plates"),
    ),
    _task(
        "protocol-alde",
        "Does the ranking survive the shape a real campaign takes? Three rounds "
        "of 132, after ALDE's six 96-well plates over three rounds.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(rounds=3, batch_size=132, max_mutations=MUTATIONS, label="ALDE"),
    ),
    _task(
        "protocol-evolvepro",
        "The opposite shape at a comparable budget: eight rounds of twelve, "
        "after EVOLVEpro. Many small rounds against few large ones, on the same "
        "landscape as protocol-alde so only the shape differs.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(rounds=8, batch_size=48, max_mutations=MUTATIONS, label="EVOLVEpro-like"),
    ),
)

#: The cheap landscape every diagnostic varies an axis on.
DIAGNOSTIC_LANDSCAPE = _ehrlich(
    sequence_length=32,
    vocab_size=20,
    n_motifs=2,
    motif_length=4,
    transition_density=0.5,
    seed=7,
)


def budget_gradient() -> tuple[Task, ...]:
    """Tasks spanning the wet-lab regime to the machine-learning convention.

    C5 as a curve rather than an assertion: if the ranking of methods flips
    somewhere between 96 assays and 10,000, that location is the finding.

    Returns:
        One task per budget, on the shared diagnostic landscape.
    """
    return tuple(
        _task(
            f"budget-{rounds * batch}",
            f"Budget gradient at {rounds * batch} calls.",
            DIAGNOSTIC_LANDSCAPE,
            Protocol(rounds=rounds, batch_size=batch, max_mutations=MUTATIONS),
        )
        for rounds, batch in ((8, 12), (4, PLATE), (10, 100), (10, 1000))
    )


def rounds_curve(budget: int = 384) -> tuple[Task, ...]:
    """Tasks splitting one budget across different numbers of rounds.

    Args:
        budget: Total oracle calls to hold fixed.

    Returns:
        One task per split, on the shared diagnostic landscape.
    """
    return tuple(
        _task(
            f"rounds-{protocol.rounds}x{protocol.batch_size}",
            f"Response curve: {protocol.rounds} rounds of {protocol.batch_size}.",
            DIAGNOSTIC_LANDSCAPE,
            protocol,
        )
        for protocol in round_sweep(budget)
    )


def objective_task() -> Task:
    """The single task the GFlowNet objectives are compared on."""
    return _task(
        "objectives",
        "Which training objective, at equal budget? GFlowNet-only, since a "
        "classical baseline has no objective to vary.",
        DIAGNOSTIC_LANDSCAPE,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=MUTATIONS),
    )


@dataclass(frozen=True, slots=True)
class Tier:
    """A group of tasks run at one seed count, with a reason to exist.

    Attributes:
        name: Short identifier.
        tasks: What to run.
        seeds: Seeds per arm.
        headline: Whether results here carry claims or only inform choices.
    """

    name: str
    tasks: tuple[Task, ...]
    seeds: tuple[int, ...]
    headline: bool

    def __repr__(self) -> str:
        """Name the tier, its size and its standing."""
        kind = "main" if self.headline else "diagnostic"
        return f"{self.name} ({kind}, {len(self.tasks)} tasks x {len(self.seeds)} seeds)"


def run_task(
    task: Task,
    methods: Mapping[str, Methodology],
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever is missing for one task, storing each result as it lands.

    Args:
        task: What to run.
        methods: Methodologies by name.
        store: Where results go, and what says which are already held.
        seeds: Seeds wanted.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.
    """
    landscape = task.landscape()
    optimum = landscape.optimum
    best_possible = float(np.max(optimum)) if optimum is not None else None
    ran = 0

    for name, method in methods.items():
        outstanding = store.missing(task.name, name, seeds)
        if not outstanding:
            report(f"  {task.name}/{name}: {len(seeds)} seeds cached")
            continue
        started = time.perf_counter()
        for seed in outstanding:
            campaign = method(task, seed)
            result = campaign.run()
            method_sampler = campaign.sampler
            feasible = (
                float(landscape.is_feasible(result.sequences).mean())
                if len(result.sequences)
                else 0.0
            )
            store.append(
                store.stamp(
                    task=task.name,
                    method=name,
                    seed=seed,
                    protocol=repr(task.protocol),
                    best=result.best_value,
                    regret=(
                        best_possible - result.best_value if best_possible is not None else None
                    ),
                    diversity=(diversity(result.sequences) if len(result.sequences) > 1 else 0.0),
                    feasible_fraction=feasible,
                    oracle_calls=result.oracle_calls,
                    proposals=result.proposals,
                    proxy_calls=int(getattr(method_sampler, "proxy_calls", 0)),
                    top_sequences=_top_designs(result),
                    trace=result.trace(),
                    rounds=[
                        {
                            "index": float(record.index),
                            "proposed": float(record.proposed),
                            "screened": float(record.screened),
                            "evaluated": float(record.evaluated),
                            "feasible": float(record.feasible),
                            "best_in_round": record.best_in_round,
                            "best_so_far": record.best_so_far,
                            "mean_in_round": record.mean_in_round,
                            "batch_diversity": record.batch_diversity,
                            "surrogate_correlation": record.surrogate_correlation,
                        }
                        for record in result.rounds
                    ],
                )
            )
            ran += 1
        elapsed = time.perf_counter() - started
        report(
            f"  {task.name}/{name}: ran {len(outstanding)} "
            f"({elapsed / max(len(outstanding), 1):.1f}s each), "
            f"{len(seeds) - len(outstanding)} cached"
        )
    return ran


def run_tier(
    tier: Tier,
    methods: Mapping[str, Methodology],
    store: ResultStore,
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run every task in a tier.

    Args:
        tier: What to run.
        methods: Methodologies by name.
        store: Where results go.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.
    """
    report(f"{tier!r}")
    return sum(run_task(task, methods, store, tier.seeds, report=report) for task in tier.tasks)


def records_to_metric(
    records: Mapping[int, RunRecord], seeds: Sequence[int], metric: str
) -> np.ndarray:
    """Pull one metric out of stored records, in seed order.

    Args:
        records: Records by seed.
        seeds: The order to return them in.
        metric: Field name.

    Returns:
        An array with one entry per seed present.
    """
    values = []
    for seed in seeds:
        record = records.get(seed)
        if record is None:
            continue
        value = getattr(record, metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def _top_designs(result: CampaignResult, k: int = 10) -> list[list[int]]:
    """The best designs a campaign found, for inspection.

    Storing every measured sequence would be hundreds of megabytes across a
    suite; storing the best ten is what anyone actually looks at when a number
    surprises them.

    Args:
        result: A completed campaign.
        k: How many to keep.

    Returns:
        Token lists, best first.
    """
    if not len(result.sequences):
        return []
    values = np.asarray(result.values, dtype=np.float64).reshape(len(result.sequences), -1)
    order = np.argsort(-values.max(axis=1))[:k]
    return [[int(t) for t in result.sequences[i]] for i in order]
