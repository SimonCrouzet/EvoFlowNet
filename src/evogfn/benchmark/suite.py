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

The mutation budget is per task, and that is not a convenience
--------------------------------------------------------------

It used to be 4 everywhere, on the argument that a shared search radius lets a
conclusion drawn on a diagnostic transfer to the main table. The argument was
right and the number was wrong, because a budget is only comparable across tasks
if it means the same thing on each -- and on an Ehrlich instance it did not.

Reaching reward 1.0 there means placing every residue of every motif, and the
parent is drawn independently of the planted optimum, so the two differ in
roughly ``L * (1 - 1/v)`` positions: 248 on the flagship task, 61 to 62 on the
mid-size ones. At a budget of 4 the planted optimum was outside the search space
of every task in ``MAIN`` except the GB1 anchor. Regret was then measured
against a target no method could reach, so most of the column was a constant
that said nothing about any method, and the arms were separated by whatever
fraction was left.

Each task therefore carries its own budget, set to the distance its own planted
optimum sits at. They are stated as named constants rather than derived, so the
number a run used is readable without instantiating a landscape, and
``tests/benchmark/test_attainable.py`` asserts against every task in ``MAIN``
that the planted optimum is inside the budget -- which is the check whose
absence is what let this stand.

[evogfn.benchmark.attainable][] is the other half: reachability is *not*
implied by budget, so what a task can attain still has to be audited rather than
assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import is_deterministic
from evogfn.benchmark.protocol import PLATE, Protocol, round_sweep
from evogfn.benchmark.tasks import Task
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape
from evogfn.metrics.diversity import diversity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.store import ResultStore, RunRecord
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.loop.ledger import CampaignResult

#: GB1's four measured sites. Equal to the sequence length, so every variant in
#: the published table is reachable and the anchor exercises no search radius.
GB1_MUTATIONS = 4

#: Distance from the flagship task's parent to its planted optimum, at L=256.
#: Below 256, so the radius still constrains the search; at 248 it no longer
#: excludes the answer.
LARGE_SPACE_MUTATIONS = 248

#: Same, for the feasibility task at L=64.
FEASIBILITY_MUTATIONS = 62

#: Same, for the two protocol tasks -- one landscape, so one distance.
PROTOCOL_MUTATIONS = 61

#: Same, for the shared diagnostic landscape at L=32. It reaches the sequence
#: length, which is what a parent drawn independently of the planted optimum
#: costs at this size: the two agree in fewer than one position on average.
DIAGNOSTIC_MUTATIONS = 32

#: Where a campaign's code actually starts, for staleness. Everything a run
#: touches is reachable from these two: the methodology table pulls in every
#: sampler, surrogate, acquisition rule and landscape, and the loop pulls in the
#: ledger and its metrics. Declaring them rather than hashing the whole package
#: tree is what stops an unrelated addition -- a new Pareto indicator, say --
#: invalidating a genetic-algorithm result it cannot possibly have influenced.
CAMPAIGN_ENTRY_POINTS = (
    "evogfn.benchmark.methods",
    "evogfn.loop.campaign",
)

#: What a stored campaign's result can depend on. Every methodology is built in
#: ``benchmark.methods``, and every campaign runs through ``loop.campaign``, so
#: their transitive imports bound what could have changed the number. Stated
#: here rather than derived, because a wrong entry point silently shrinks the
#: dependency set -- and a record that under-declares what it depends on is
#: exactly the stale-result failure the fingerprint exists to prevent.
RESULT_DEPENDENCIES = (
    "evogfn.benchmark.methods",
    "evogfn.loop.campaign",
)


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
    """A task whose search radius is the one its protocol was costed at.

    Taking the budget from the protocol rather than accepting it separately is
    what stops the two disagreeing. They are read by different code -- the
    environment uses the task's, `Protocol.constrains_search` reports the
    protocol's -- and a task that searched at one radius while reporting another
    would be undetectable from the stored record, which carries only the
    protocol's repr.

    Args:
        name: Short identifier.
        purpose: What this task decides that the others cannot.
        build: Makes the landscape.
        protocol: Rounds, batch size and mutation budget.

    Returns:
        The task.

    Raises:
        ValueError: If the protocol states no mutation budget. A task without
            one searches the whole space, which is a decision worth writing
            down rather than defaulting into.
    """
    if protocol.max_mutations is None:
        raise ValueError(f"{name}: protocol must state a mutation budget")
    return Task(
        name=name,
        purpose=purpose,
        build=build,
        protocol=protocol,
        max_mutations=protocol.max_mutations,
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
        Protocol(rounds=4, batch_size=PLATE, max_mutations=GB1_MUTATIONS, label="four plates"),
    ),
    _task(
        "large-space",
        "Can the method search a space it cannot enumerate? Stanton et al.'s "
        "base configuration, L=256 with four motifs of length eight. The budget "
        "is 384 designs against a reachable set with no useful upper digit -- "
        "248 substitutions of 256 positions over an alphabet of 20.",
        STANTON_BASE,
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=LARGE_SPACE_MUTATIONS, label="four plates"
        ),
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
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=FEASIBILITY_MUTATIONS, label="four plates"
        ),
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
        Protocol(rounds=3, batch_size=132, max_mutations=PROTOCOL_MUTATIONS, label="ALDE"),
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
        Protocol(rounds=8, batch_size=48, max_mutations=PROTOCOL_MUTATIONS, label="EVOLVEpro-like"),
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
            Protocol(rounds=rounds, batch_size=batch, max_mutations=DIAGNOSTIC_MUTATIONS),
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
            # `round_sweep` varies the shape and says nothing about the search
            # radius, so the diagnostic's own is filled in here rather than
            # letting the sweep decide an axis it is not varying.
            replace(protocol, max_mutations=DIAGNOSTIC_MUTATIONS),
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
        Protocol(rounds=4, batch_size=PLATE, max_mutations=DIAGNOSTIC_MUTATIONS),
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


def _scores(task: Task, result: CampaignResult, best_possible: float | None) -> dict[str, object]:
    """The two numbers a stored record is indexed by, whatever it measured.

    `RunRecord` has one larger-is-better field and one smaller-is-better field,
    and every table built off the store reads them positionally. So the only
    safe thing to put in them is a pair that means the same thing down the whole
    column, and what that pair *is* differs by objective count:

    * **One objective.** Best value measured, and the distance from it to the
      landscape's optimum. Unchanged, and still the honest reading only once
      [evogfn.benchmark.attainable][] says the optimum was reachable.
    * **More than one.** Hypervolume above the campaign's reference point, and
      IGD+ against its reference front. These are the multi-objective
      counterparts with the same orientation -- hypervolume rises as the set
      improves, IGD+ falls to zero when the front is covered -- so a column
      built from them is at least internally coherent.

    [CampaignResult.best_value][evogfn.loop.ledger.CampaignResult.best_value]
    now raises on a multi-objective result rather than returning the maximum
    over designs *and* objectives, which is why this branch exists at all.

    Args:
        task: The task being run, named in any error.
        result: The completed campaign.
        best_possible: The landscape's optimum, or ``None`` when unknown.

    Returns:
        ``best`` and ``regret``, ready to pass to
        [ResultStore.stamp][evogfn.benchmark.store.ResultStore.stamp].

    Raises:
        ValueError: If a multi-objective campaign supplied no reference point.
            Storing such a run under a ``best`` that means something different
            from every other row in the column is precisely the silent
            mixing the store exists to prevent, so it is refused rather than
            filled with ``nan``.
    """
    if not result.is_multi_objective:
        return {
            "best": result.best_value,
            "regret": (best_possible - result.best_value if best_possible is not None else None),
        }
    volume = result.hypervolume
    if volume is None:
        raise ValueError(
            f"{task.name} measured {result.n_objectives} objectives but supplied no reference "
            f"point, so its result has no indicator to be stored under; give the campaign a "
            f"reference_point, or score it single-objective"
        )
    return {"best": volume, "regret": result.igd_plus}


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
                    # Declaring entry points is what makes the fingerprint pay: a
                    # record then goes stale only when something it can
                    # actually reach changed, instead of when any package did.
                    # Without this the mechanism is correct and useless --
                    # adding an unrelated file invalidated ~3,900 campaigns.
                    depends_on=RESULT_DEPENDENCIES,
                    task=task.name,
                    method=name,
                    seed=seed,
                    protocol=repr(task.protocol),
                    **_scores(task, result, best_possible),
                    diversity=(diversity(result.sequences) if len(result.sequences) > 1 else 0.0),
                    feasible_fraction=feasible,
                    oracle_calls=result.oracle_calls,
                    proposals=result.proposals,
                    proxy_calls=int(getattr(method_sampler, "proxy_calls", 0)),
                    deterministic=is_deterministic(),
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
                            "hypervolume": record.hypervolume,
                            # How far this round's anchor sat from the wild
                            # type. Flat at zero says the campaign searched one
                            # Hamming ball for its whole life, which is the
                            # difference between a budget of `max_mutations` and
                            # a budget of `max_mutations` per round.
                            "anchor_distance": float(record.anchor_distance),
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
