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

The search radius is per round, and the anchor moves
----------------------------------------------------

The radius used to be 4 everywhere and the anchor never moved, which made every
Ehrlich task in the suite unwinnable by construction. Reaching reward 1.0 on an
Ehrlich instance means placing every residue of every motif, and the parent is
drawn independently of the planted optimum, so the two differ in roughly
``L * (1 - 1/v)`` positions: 248 on the flagship task, 61 to 62 on the mid-size
ones. Against one fixed Hamming ball of radius 4 the answer was not merely hard
to find, it was absent. Regret was then reported against a target no method
could reach, most of the column was a constant that said nothing about any
method, and the arms were separated by whatever fraction was left.

The fix is not a wider radius. Real directed evolution keeps the radius small --
four or five substitutions is what a round of site-saturation mutagenesis buys --
and moves the *anchor*: round two starts from the best variant round one
produced. Distance from the wild type then accumulates while the per-round
budget does not. So every Ehrlich task here re-anchors, and its radius is the
smallest one the audit measured to put its own optimum inside the campaign's
reach:

| task | landscape | protocol | per round | re-anchors | attainable optimum |
| --- | --- | --- | --- | --- | --- |
| `gb1-anchor` | GB1, L=4 | 4x96 | 4 | no | the landscape's own optimum |
| `large-space` | Ehrlich, L=256 | 4x96 | 62 | yes | [0.2812, 1.0] |
| `feasibility` | Ehrlich, L=64, density 0.15 | 4x96 | 4 | no | 0.3750 exact |
| `protocol-alde` | Ehrlich, L=64 | 3x132 | 21 | yes | 1.0 exact |
| `protocol-evolvepro` | the same L=64 instance | 8x48 | 4 | yes | 1.0 exact |
| diagnostics (7) | Ehrlich, L=32 | various | 4 | yes | 1.0 exact |

Two rows are not like the others, and both for stated reasons. ``gb1-anchor``
has four sites and a budget of four, so its ball is the whole space and there is
nothing for an anchor to move towards. ``feasibility`` keeps a fixed anchor
because what binds there is the transition matrix, not the radius: the
constructible set within four substitutions holds 26,580 designs and tops out at
0.375, and a re-anchored chain searching outward from it was measured to find
nothing better. Leaving the anchor still is what keeps that 0.375 an *enumerated*
answer rather than a bracket, which is the difference between reporting a fact
and reporting a search.

The round-varying tasks are the ones this mattered most for. Comparing 3x132
against 8x48 asks whether many small rounds beat few large ones, and with a
fixed anchor neither shape can move at all -- so the comparison was between two
identically stranded campaigns.

Regret is against what is attainable, not against the nominal optimum
---------------------------------------------------------------------

Even with the anchor moving, the reachable set is not the landscape. Each task
therefore declares what [evogfn.benchmark.attainable][] measured it to contain,
and `run_task` stores regret against *that*. The declarations are constants
rather than computations because the audit costs minutes per task, and
``tests/benchmark/test_suite_tasks.py`` re-derives them rather than trusting
them.

The numbers this replaces were not small. On ``large-space`` 95% of the
published regret was a floor no method could have cleared, and on
``feasibility`` a genetic algorithm that had *solved* the task on 99 of 100
seeds was reported at a regret of 0.626.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import is_deterministic
from evogfn.benchmark.protocol import PLATE, Protocol, round_sweep
from evogfn.benchmark.tasks import Attainable, Task
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape
from evogfn.metrics.diversity import diversity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.store import ResultStore, RunRecord
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.loop.ledger import CampaignResult

#: GB1's four measured sites. Equal to the sequence length, so every variant in
#: the published table is reachable and the anchor exercises no search radius.
GB1_MUTATIONS = 4

#: Per-round radius on the flagship task, at L=256. Four re-anchored rounds of
#: this reach 248 substitutions -- the distance to the planted optimum -- while
#: any single round still searches a ball a lab would recognise.
LARGE_SPACE_MUTATIONS = 62

#: Per-round radius on the feasibility task at L=64, and the whole radius: this
#: task keeps a fixed anchor. Four substitutions of a sparse chain reach 26,580
#: constructible designs out of a Hamming ball of 8e10, which is the ratio the
#: task exists to measure. Widening it would measure something else.
FEASIBILITY_MUTATIONS = 4

#: Per-round radius for ALDE's three rounds. The audit measured 21 to be the
#: point at which a re-anchored chain pins 1.0 exactly; at 4 it reaches 0.75.
ALDE_MUTATIONS = 21

#: Per-round radius for EVOLVEpro's eight rounds. Four is enough here and is not
#: enough at three rounds, which is the whole content of the protocol
#: comparison: a shape with more rounds buys reach at the same per-round cost.
EVOLVEPRO_MUTATIONS = 4

#: Per-round radius on the shared diagnostic landscape at L=32. The original
#: shared constant, kept because re-anchoring already makes it sufficient --
#: four rounds of it were measured to pin 1.0 exactly.
DIAGNOSTIC_MUTATIONS = 4

#: A task whose per-round radius is deliberately smaller than the distance to
#: its own planted optimum has to say so, in these words, so
#: ``experiments/audit_optima.py`` reports the gap as a property of the design
#: rather than as a defect in it. Every Ehrlich task here is in that position by
#: intent: the radius is a round of mutagenesis and the reach is cumulative.
CAPPED = (
    "The per-round mutation budget is deliberately capped below the distance to "
    "the planted optimum."
)

#: What the capped radius buys back, on a task whose anchor moves.
CUMULATIVE = f"{CAPPED} The campaign re-anchors, so its reach is cumulative."

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


def _task(  # noqa: PLR0913 - a task is defined by what it declares
    name: str,
    purpose: str,
    build: Callable[[], FitnessLandscape],
    protocol: Protocol,
    *,
    reanchor: bool,
    attainable: Attainable,
) -> Task:
    """A task whose search radius, anchor rule and reachable optimum are all stated.

    Taking the radius from the protocol rather than accepting it separately is
    what stops the two disagreeing. They are read by different code -- the
    environment uses the task's, `Protocol.constrains_search` reports the
    protocol's -- and a task that searched at one radius while reporting another
    would be undetectable from a stored record.

    ``reanchor`` and ``attainable`` are keyword-only and have **no defaults**,
    which is the enforcement this module exists to apply. Both were previously
    absent rather than false: every task searched one fixed ball and every
    regret was measured against a nominal optimum, and nothing in the suite's
    definition said so. A default would let the next task added inherit the same
    silence.

    Args:
        name: Short identifier.
        purpose: What this task decides that the others cannot.
        build: Makes the landscape.
        protocol: Rounds, batch size and the per-round mutation budget.
        reanchor: Whether the anchor follows the best design measured so far.
        attainable: What an audit measured this task's search space to contain.

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
        reanchor=reanchor,
        attainable=attainable,
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
        # Nothing to move towards: the ball of radius four over four sites is
        # the entire published table, so the first round already sees every
        # design a later one could be anchored at.
        reanchor=False,
        attainable=Attainable.whole_optimum(
            "exact: four mutations over four sites reach every measured variant"
        ),
    ),
    _task(
        "large-space",
        "Can the method search a space it cannot enumerate? Stanton et al.'s "
        "base configuration, L=256 with four motifs of length eight. The budget "
        "is 384 designs against a reachable set with no useful upper digit -- "
        "62 substitutions a round of 256 positions over an alphabet of 20. "
        f"{CUMULATIVE}",
        STANTON_BASE,
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=LARGE_SPACE_MUTATIONS, label="four plates"
        ),
        reanchor=True,
        # The one task whose bracket the audit could not close, and the reason
        # the interval is carried rather than a point: 0.2812 is witnessed by a
        # design the beam actually built, 1.0 is what the reward's structure
        # permits at 248 cumulative substitutions, and nothing measured says
        # which. A comparison here is read against the interval.
        attainable=Attainable.between(
            0.2812,
            1.0,
            "bounded: 4 re-anchored rounds of 62, beam search below the budget-split bound",
        ),
    ),
    _task(
        "feasibility",
        "Can the method stay inside the constructible set? A sparse transition "
        "matrix makes most sequences unbuildable, so rejection sampling spends "
        "the budget on designs that cannot be made while masking cannot. "
        f"{CAPPED} The anchor is held fixed: what binds here is the transition "
        "matrix rather than the radius, and a re-anchored chain outward from "
        "this ball was measured to find nothing better.",
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
        reanchor=False,
        # Enumerated, not searched: the ball holds 8e10 designs and 26,580 of
        # them are constructible, so the maximum over the reachable set is a
        # measurement. It is also the number that turned a 0.626 regret into a
        # solved task.
        attainable=Attainable.exactly(
            0.375, "exact: enumerated the 26,580 reachable terminal states at 4 mutations"
        ),
    ),
    _task(
        "protocol-alde",
        "Does the ranking survive the shape a real campaign takes? Three rounds "
        "of 132, after ALDE's six 96-well plates over three rounds. "
        f"{CUMULATIVE} Three rounds is the fewest here, so it needs the widest "
        "radius: 21 pins the optimum where 4 reaches 0.75.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(rounds=3, batch_size=132, max_mutations=ALDE_MUTATIONS, label="ALDE"),
        reanchor=True,
        attainable=Attainable.exactly(
            1.0, "pinned: 3 re-anchored rounds of 21 reach the budget-split bound"
        ),
    ),
    _task(
        "protocol-evolvepro",
        "The opposite shape at a comparable budget: eight rounds of 48, after "
        "EVOLVEpro. Many small rounds against few large ones, on the same "
        "landscape as protocol-alde so only the shape differs. "
        f"{CUMULATIVE} Eight rounds buy the same reach from a radius of 4 that "
        "three rounds need 21 for, which is the point of running both.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(
            rounds=8, batch_size=48, max_mutations=EVOLVEPRO_MUTATIONS, label="EVOLVEpro-like"
        ),
        reanchor=True,
        attainable=Attainable.exactly(
            1.0, "pinned: 8 re-anchored rounds of 4 reach the budget-split bound"
        ),
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

#: What every diagnostic can reach. One landscape and one per-round radius, so
#: one audited answer -- measured at the fewest rounds any diagnostic runs, and
#: therefore valid for all of them, since rounds only add reach.
DIAGNOSTIC_ATTAINABLE = Attainable.exactly(
    1.0, "pinned: 4 re-anchored rounds of 4 reach the budget-split bound at L=32"
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
            f"Budget gradient at {rounds * batch} calls. {CUMULATIVE}",
            DIAGNOSTIC_LANDSCAPE,
            Protocol(rounds=rounds, batch_size=batch, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            attainable=DIAGNOSTIC_ATTAINABLE,
        )
        for rounds, batch in ((8, 12), (4, PLATE), (10, 100), (10, 1000))
    )


def rounds_curve(budget: int = 384) -> tuple[Task, ...]:
    """Tasks splitting one budget across different numbers of rounds.

    The diagnostic re-anchoring matters most for: with the anchor fixed, every
    shape in this sweep searches the identical ball and the curve is flat by
    construction, so whatever it showed was not about rounds.

    Args:
        budget: Total oracle calls to hold fixed.

    Returns:
        One task per split, on the shared diagnostic landscape.
    """
    return tuple(
        _task(
            f"rounds-{protocol.rounds}x{protocol.batch_size}",
            f"Response curve: {protocol.rounds} rounds of {protocol.batch_size}. {CUMULATIVE}",
            DIAGNOSTIC_LANDSCAPE,
            # `round_sweep` varies the shape and says nothing about the search
            # radius, so the diagnostic's own is filled in here rather than
            # letting the sweep decide an axis it is not varying.
            replace(protocol, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            attainable=DIAGNOSTIC_ATTAINABLE,
        )
        for protocol in round_sweep(budget)
    )


def objective_task() -> Task:
    """The single task the GFlowNet objectives are compared on."""
    return _task(
        "objectives",
        "Which training objective, at equal budget? GFlowNet-only, since a "
        f"classical baseline has no objective to vary. {CUMULATIVE}",
        DIAGNOSTIC_LANDSCAPE,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=DIAGNOSTIC_MUTATIONS),
        reanchor=True,
        attainable=DIAGNOSTIC_ATTAINABLE,
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


def _scores(
    task: Task, result: CampaignResult, attainable: AttainableOptimum | None
) -> dict[str, object]:
    """The two numbers a stored record is indexed by, whatever it measured.

    `RunRecord` has one larger-is-better field and one smaller-is-better field,
    and every table built off the store reads them positionally. So the only
    safe thing to put in them is a pair that means the same thing down the whole
    column, and what that pair *is* differs by objective count:

    * **One objective.** Best value measured, and the distance from it to the
      **attainable** optimum -- what the audit measured this task's search space
      to contain, conservatively its searched lower bound. Not the landscape's
      own optimum: on ``large-space`` 95% of the regret computed that way was a
      floor no method could clear, and on ``feasibility`` an arm sitting exactly
      on the reachable maximum was reported at a regret of 0.626.
    * **More than one.** Hypervolume above the campaign's reference point, and
      IGD+ against its reference front. These are the multi-objective
      counterparts with the same orientation -- hypervolume rises as the set
      improves, IGD+ falls to zero when the front is covered -- so a column
      built from them is at least internally coherent.
      [CampaignResult.best_value][evogfn.loop.ledger.CampaignResult.best_value]
      raises on a multi-objective result rather than returning the maximum over
      designs *and* objectives, which is why this branch exists at all.

    A regret can come out **negative**, and that is deliberate. Where the audit
    could only bracket the attainable optimum, an arm beating the searched lower
    bound is evidence about the audit rather than about the arm, and clamping it
    at zero would erase the only signal that a bound needs re-deriving.

    Args:
        task: The task being run, named in any error.
        result: The completed campaign.
        attainable: What this task's search space was audited to contain, or
            ``None`` where no audit covers it -- in which case no regret is
            stored at all. An absent number is recoverable; a number measured
            against an unreachable target is not distinguishable from a real one
            once it is in the column.

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
        best = result.best_value
        return {
            "best": best,
            "regret": None if attainable is None else attainable.lower - best,
        }
    volume = _indicator(result)
    if volume is None:
        raise ValueError(
            f"{task.name} measured {result.n_objectives} objectives but supplied no reference "
            f"point, so its result has no indicator to be stored under; give the campaign a "
            f"reference_point, or score it single-objective"
        )
    return {"best": volume, "regret": result.igd_plus}


def _indicator(result: CampaignResult) -> float | None:
    """Hypervolume for a multi-objective run, or ``nan`` where it is not exact.

    The exact hypervolume in [evogfn.metrics.pareto][] is inclusion-exclusion
    over the front, which it caps at 16 points for three or more objectives. A
    384-design campaign can easily carry a larger front than that, and the
    honest answer is then that the indicator was not computed -- an approximation
    written into the same column as an exact value would be indistinguishable
    from one. ``nan`` propagates; the measurements survive on the result for
    anyone who wants to score them with a dedicated implementation.

    Args:
        result: The completed campaign.

    Returns:
        The dominated volume, ``nan`` where the exact method cannot run, or
        ``None`` when no reference point was supplied.
    """
    try:
        return result.hypervolume
    except NotImplementedError:
        return float("nan")


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

    Raises:
        ValueError: If the task declares an attainable optimum its landscape
            contradicts -- an upper bound above the landscape's own maximum.
            Refused here rather than folded into every stored regret.
    """
    landscape = task.landscape()
    optimum = landscape.optimum
    # A multi-objective landscape's optimum is an ideal point, and the maximum
    # over its components is not a target anything could reach. The attainable
    # declaration is single-objective by construction, so it is simply absent
    # there and the multi-objective branch of `_scores` never consults it.
    attainable = (
        task.attainable_optimum(float(np.max(optimum)))
        if optimum is not None and landscape.n_objectives == 1
        else None
    )
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
                    # The task's repr, not the protocol's: rounds and batch size
                    # alone do not say what a run could have reached, and two
                    # records at 4x96=384 that differ in search radius or in
                    # whether the anchor moved are not comparable.
                    protocol=repr(task),
                    **_scores(task, result, attainable),
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
