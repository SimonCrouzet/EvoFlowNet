"""Run the benchmark suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run. Every campaign is written the moment it
finishes, and a second invocation runs only what is missing -- so raising a
tier's seed count from 30 to 50 costs twenty campaigns per arm, not fifty.

    uv run python experiments/run_suite.py                  # everything
    uv run python experiments/run_suite.py --tier main      # headline only
    uv run python experiments/run_suite.py --seeds 50       # raise the count
    uv run python experiments/run_suite.py --report         # no runs, just read

Results land under ``results/`` as one JSONL file per task and method.

What the regret column is, and what it is not
---------------------------------------------

Regret here is against the **attainable** optimum -- what
[evogfn.benchmark.attainable][] measured a task's search space to contain --
rather than against the landscape's own. The two are not close. On
``large-space`` 95% of the regret computed the old way was a floor no method
could clear, so the part of the column that varied between arms, which is the
only part a comparison reads, was whatever fraction was left.

Where the audit could not close the bracket, the interval is printed and the
regret is against its conservative end. A regret at or below zero therefore does
not mean an arm was perfect; it means the arm matched everything the audit could
construct, and the task has no demonstrated headroom left to separate it from a
better method with. Those arms are named as **solved**, and comparisons drawn on
them are marked vacuous rather than quietly reported -- a p-value against an arm
sitting on the ceiling is a statement about the ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import BASELINES, OBJECTIVES, flow_objectives, sensitivity
from evogfn.benchmark.selection import _build_objective
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    MAIN,
    Purpose,
    Tier,
    budget_gradient,
    objective_task,
    records_to_metric,
    rounds_curve,
    run_tier,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.store import RunRecord
    from evogfn.benchmark.tasks import Task

#: Seeds per tier. The main tiers differ because campaign cost differs by an
#: order of magnitude between L=4 and L=256, not because the claims differ.
MAIN_SEEDS = 100
LARGE_SPACE_SEEDS = 30
DIAGNOSTIC_SEEDS = 50

#: Everything except the flow objectives, which need a policy with a flow head
#: and are compared separately in the objectives diagnostic.
MAIN_METHODS = {**BASELINES, **OBJECTIVES}

#: Share of an arm's seeds sitting on the attainable optimum before the task is
#: called solved for that arm. Half is already fatal: a comparison against an arm
#: that hits the ceiling on half its seeds is measuring the ceiling on those, and
#: the p-value it reports is about the remainder.
VACUOUS_SHARE = 0.5

#: How close to the attainable optimum an arm has to sit to count as on it.
SOLVED_TOLERANCE = 1e-9


def tiers(main_seeds: int, diagnostic_seeds: int) -> list[Tier]:
    """The suite, split by what each tier is for.

    Args:
        main_seeds: Seeds for the headline tiers.
        diagnostic_seeds: Seeds for the diagnostics.

    Returns:
        Tiers in the order they should run: cheap and decisive first, so an
        interrupted night still yields something readable.
    """
    cheap = tuple(t for t in MAIN if t.name != "large-space")
    expensive = tuple(t for t in MAIN if t.name == "large-space")
    return [
        Tier("objectives", (objective_task(),), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC),
        # Shares the objectives task deliberately: same landscape, same
        # protocol, same seeds, so a setting's effect and an objective's are
        # measured against each other rather than across two configurations.
        Tier("sensitivity", (objective_task(),), tuple(range(diagnostic_seeds)), Purpose.SELECTION),
        Tier("main", cheap, tuple(range(main_seeds)), Purpose.BENCHMARK),
        Tier("rounds-curve", rounds_curve(), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC),
        Tier(
            "budget-gradient", budget_gradient(), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC
        ),
        # Last: ~200s a campaign at L=256, and fewer seeds for the same reason.
        Tier("large-space", expensive, tuple(range(LARGE_SPACE_SEEDS)), Purpose.BENCHMARK),
    ]


#: Where the selection phase records the configuration it chose.
CHOICE_FILE = Path("results/selected.json")


def selected_gflownet() -> dict[str, object]:
    """The GFlowNet arm the selection phase chose, if it has run.

    Reading the decision rather than re-deriving it matters: re-deriving would
    silently pick a different arm the moment a seed count or an arm list moved,
    and the table would then report a configuration no selection ever chose.

    Returns:
        A single-entry mapping, or an empty one when no selection has been
        recorded -- in which case the caller falls back to the untuned defaults
        and says so, rather than quietly benchmarking them as though they had
        been chosen.
    """
    if not CHOICE_FILE.exists():
        return {}
    choice = json.loads(CHOICE_FILE.read_text())
    arm = str(choice["arm"])
    objective = str(choice["objective"])
    beta = float(arm.rsplit("-", 1)[1])
    return {arm: _build_objective(objective, beta)}


def methods_for(tier: Tier) -> dict[str, object]:
    """Which methodologies a tier runs.

    The objectives diagnostic is GFlowNet-only, since a classical baseline has
    no training objective to vary; the sensitivity tier is narrower still, being
    one GFlowNet with one setting moved at a time; everything else compares
    methods.
    """
    if tier.name == "objectives":
        return {**OBJECTIVES, **flow_objectives()}
    if tier.name == "sensitivity":
        return dict(sensitivity())
    chosen = selected_gflownet()
    if not chosen:
        return dict(MAIN_METHODS)
    # The selected arm replaces the untuned GFlowNet arms rather than joining
    # them: keeping both would put two configurations of the same method in one
    # table, and the better of the two would be the one the selection was run to
    # avoid reporting.
    return {**BASELINES, **chosen}


def attainable_for(task: Task) -> AttainableOptimum | None:
    """What this task's search space was audited to contain.

    Args:
        task: The task being reported on.

    Returns:
        The audited optimum, or ``None`` for a task carrying no declaration or
        measuring more than one objective -- where the landscape's optimum is an
        ideal point and the gap to it is not a regret.
    """
    landscape = task.landscape()
    optimum = landscape.optimum
    if optimum is None or landscape.n_objectives != 1:
        return None
    return task.attainable_optimum(float(np.max(optimum)))


def _attainable_line(attainable: AttainableOptimum | None) -> str:
    """State the attainable optimum so a bound cannot be read as a measurement."""
    if attainable is None:
        return "  attainable: not audited, so no regret is reported for this task"
    if attainable.exact is not None:
        value = f"{attainable.exact:.4f} exact"
    else:
        value = f"[{attainable.lower:.4f}, {attainable.upper:.4f}]"
    floor = attainable.regret_floor
    gap = f"{floor[0]:.4f}" if np.isclose(*floor) else f"[{floor[0]:.4f}, {floor[1]:.4f}]"
    return (
        f"  attainable {value} of a nominal {attainable.nominal:.4f} "
        f"at {attainable.budget} cumulative mutations; regret against the nominal "
        f"would carry a floor of {gap}\n    ({attainable.method})"
    )


def _at_optimum(records: Mapping[int, RunRecord], attainable: AttainableOptimum | None) -> float:
    """Share of an arm's seeds sitting on the attainable optimum.

    Args:
        records: The arm's stored records, by seed.
        attainable: What the task can reach, or ``None`` when unaudited.

    Returns:
        A fraction in ``[0, 1]``, or ``nan`` when there is nothing to compare
        against. ``nan`` rather than zero: "no audit" is a different statement
        from "no seed reached it".
    """
    if attainable is None or not records:
        return float("nan")
    best = np.asarray([record.best for record in records.values()], dtype=np.float64)
    usable = best[np.isfinite(best)]
    if not usable.size:
        return float("nan")
    return float(np.mean(usable >= attainable.lower - SOLVED_TOLERANCE))


#: What each tier's arms are compared against. A tier that does not contain the
#: default reference gets its own, because `report` skips the paired section
#: silently when the reference is absent -- which reads as "nothing separated
#: these arms" when what happened is that nothing was tested.
REFERENCES = {
    # The shipped configuration, so each swept value is read as a change from
    # what the headline rows were produced at.
    "sensitivity": "steps-300",
    # Trajectory balance, the objective the others are alternatives to.
    "objectives": "gfn-tb",
}

#: The strongest control: a genetic algorithm with the same proxy access the
#: GFlowNet gets. Comparing against a method that only meets the model as a
#: filter would not be a comparison of methods.
DEFAULT_REFERENCE = "genetic+proxy"


def reference_for(tier: Tier, methods: dict[str, object]) -> str | None:
    """Which arm a tier's comparisons are drawn against.

    Args:
        tier: The tier being reported on.
        methods: The arms it runs, so an absent reference is caught here rather
            than becoming a missing section further down.

    Returns:
        The reference arm's name, or ``None`` when the tier has no arm to serve
        as one -- said explicitly so the report can name the omission.
    """
    chosen = REFERENCES.get(tier.name, DEFAULT_REFERENCE)
    return chosen if chosen in methods else None


def report(store: ResultStore, tier: Tier, reference: str | None = None) -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Regret is read straight from the records, where it is already against the
    attainable optimum -- see `evogfn.benchmark.suite._scores`. What this adds is
    the context that makes it readable: the interval the task can reach, how many
    of an arm's seeds are sitting on it, and a refusal to present a paired
    comparison drawn on a solved task as though it ranked anything.

    Args:
        store: Where results live.
        tier: The tier to report on.
        reference: Arm to compare against. Defaults to whatever
            `reference_for` picks for this tier. ``None`` where the tier has no
            arm that can serve, which is reported rather than passed over.

    Returns:
        A multi-line report.
    """
    lines = []
    names = list(methods_for(tier))
    if reference is None:
        reference = reference_for(tier, methods_for(tier))
    for task in tier.tasks:
        attainable = attainable_for(task)
        lines.append(f"\n{task!r}")
        lines.append(_attainable_line(attainable))
        held = {name: store.usable(task.name, name) for name in names}
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        solved = set()
        for name in names:
            records = held[name]
            if not records:
                continue
            regret = records_to_metric(records, tier.seeds, "regret")
            feasible = records_to_metric(records, tier.seeds, "feasible_fraction")
            spread = records_to_metric(records, tier.seeds, "diversity")
            spent = records_to_metric(records, tier.seeds, "oracle_calls")
            error = regret.std(ddof=1) / len(regret) ** 0.5 if len(regret) > 1 else 0.0
            share = _at_optimum(records, attainable)
            if share >= VACUOUS_SHARE:
                solved.add(name)
            lines.append(
                f"  {name:<18} regret {regret.mean():>7.3f} +/- {error:<6.3f} "
                f"at-opt {share:>5.2f}  feas {feasible.mean():>5.3f}  "
                f"div {spread.mean():>5.2f}  spent {spent.mean():>6.0f}  n={len(regret)}"
            )
        for name in sorted(solved):
            lines.append(
                f"  SOLVED  {name} sits on the attainable optimum "
                f"{attainable.lower:.4f} on {_at_optimum(held[name], attainable):.0%} of its "  # type: ignore[union-attr]
                f"seeds; this task cannot rank it against anything better, and a "
                f"comparison naming it is not a comparison of methods"
            )

        lines.extend(_paired(held, names, seeds, reference, solved))
    return "\n".join(lines)


def _paired(
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: list[int],
    reference: str | None,
    solved: set[str],
) -> list[str]:
    """Compare every arm against the reference, paired across shared seeds.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: Seeds every arm has, so the comparison is genuinely paired.
        reference: The arm to compare against, or ``None`` if the tier has none.
        solved: Arms already sitting on the attainable optimum.

    Returns:
        Report lines, including a line naming the omission when there is
        nothing to compare against -- an absent section reads as "nothing
        separated these arms", which is a different statement from "nothing
        was tested".
    """
    if reference is None:
        return ["  no reference arm in this tier, so nothing is paired"]
    base = held.get(reference)
    if not base or not seeds:
        return [f"  reference {reference} has no usable seeds here, so nothing is paired"]
    lines = [f"  paired vs {reference} (positive favours the first):"]
    for name in names:
        if name == reference or not held[name]:
            continue
        mine = records_to_metric(held[name], seeds, "regret")
        theirs = records_to_metric(base, seeds, "regret")
        if len(mine) != len(theirs) or len(mine) < 2:  # noqa: PLR2004
            continue
        outcome = compare(name, mine, theirs, higher_is_better=False)
        lines.append(f"    {outcome!r}")
        # Said before the p-value is read rather than after: an arm on the
        # ceiling makes the difference a measurement of the ceiling, and
        # significance there is significance about the task.
        if vacuous := solved.intersection({name, reference}):
            lines.append(
                f"        vacuous: {', '.join(sorted(vacuous))} already reached "
                f"everything this task was audited to contain"
            )
        elif not outcome.significant and (needed := seeds_needed(outcome)):
            lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run or report on the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", help="Run only these tiers.")
    parser.add_argument(
        "--task",
        action="append",
        help="Run only these tasks. Sharding by task is race-free -- the store "
        "keeps one file per task and method -- so a process per task uses the "
        "cores far better than threads do, most of the work being serial Python.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Run only these arms. The classical baselines do not depend on which "
        "GFlowNet configuration the selection phase picks, so they can be banked "
        "while it is still running.",
    )
    parser.add_argument("--seeds", type=int, default=MAIN_SEEDS, help="Seeds for main tiers.")
    parser.add_argument(
        "--diagnostic-seeds", type=int, default=DIAGNOSTIC_SEEDS, help="Seeds for diagnostics."
    )
    parser.add_argument("--results", default="results", help="Where to store results.")
    parser.add_argument("--report", action="store_true", help="Report without running.")
    args = parser.parse_args(argv)

    # Before any tensor work: a multithreaded matmul sums in thread-completion
    # order, and a few hundred gradient steps turn that into a different design.
    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    selected = tiers(args.seeds, args.diagnostic_seeds)
    if args.tier:
        selected = [t for t in selected if t.name in set(args.tier)]
    wanted_methods = set(args.method) if args.method else set()
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(
                t.name,
                tuple(task for task in t.tasks if task.name in wanted),
                t.seeds,
                t.purpose,
            )
            for t in selected
        ]
        selected = [t for t in selected if t.tasks]
    if not selected:
        print(f"nothing matched tier={args.tier} task={args.task}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    for tier in selected:
        if not args.report:
            arms = methods_for(tier)
            if args.method:
                arms = {k: v for k, v in arms.items() if k in wanted_methods}
                if not arms:
                    _flush(f"{tier.name}: no arm matched --method, skipping")
                    continue
            ran = run_tier(tier, arms, store, report=_flush)  # type: ignore[arg-type]
            _flush(f"{tier.name}: ran {ran} campaigns")
        _flush(report(store, tier))
    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    _flush(store.summarise())
    return 0


def _flush(message: str) -> None:
    """Print immediately, so an overnight run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
