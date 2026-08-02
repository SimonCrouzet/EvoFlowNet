"""Audit what each benchmark task can reach, and rescore what it already stored.

Regret is only a measure of a method to the extent that the optimum it is
measured against was available to that method. This report answers, per task,
whether it was -- and then re-reads ``results/`` and puts the regret against the
nominal optimum beside the regret against the attainable one.

    uv run python experiments/audit_optima.py
    uv run python experiments/audit_optima.py --main-only
    uv run python experiments/audit_optima.py --stored-budget 4

Nothing under ``results/`` is written. The store is opened read-only in the only
sense that matters here: nothing calls ``append`` or ``bless``.

Two budgets, and the difference matters
---------------------------------------

Stored records do not carry the mutation budget they ran under -- a record keeps
`Protocol.__repr__`, which is rounds, batch size and total. So rescoring a stored
number against today's budget would compare it against a search space it never
had. ``--stored-budget`` names the budget the results on disk were produced at,
and the rescoring uses the attainable optimum computed there. The
forward-looking table uses each task's current, per-task budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.attainable import (
    AttainableOptimum,
    attainable_optimum,
    per_round_budget,
    planted_distance,
    planted_optimum_reachable,
    reanchored_attainable,
)
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import MAIN, budget_gradient, objective_task, rounds_curve

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evogfn.benchmark.tasks import Task

#: The budget every task shared when the stored results were produced. Named
#: rather than inferred because no stored record carries it.
LEGACY_MUTATIONS = 4

#: A task whose budget deliberately excludes its own planted optimum must say so
#: in its purpose, using this phrase. Then the audit reports the floor as a
#: property of the experiment rather than as a defect in it -- and a task that
#: has the floor without the sentence is reported as the defect it is.
INTENTIONAL_CAP = "mutation budget is deliberately capped"

#: How close to the attainable optimum an arm has to sit before the task stops
#: being able to rank it against anything.
SOLVED_TOLERANCE = 1e-9

#: Share of an arm's seeds sitting exactly on the attainable optimum before the
#: task is called solved. Half is already fatal: a comparison against an arm
#: that hits the ceiling on half its seeds is measuring the ceiling on those,
#: and the p-value it produces is about the remainder.
VACUOUS_SHARE = 0.5


def audited_tasks(*, main_only: bool) -> tuple[Task, ...]:
    """Every task the suite can run, headline tasks first.

    Args:
        main_only: Skip the diagnostics, which are seven tasks over one shared
            landscape and so add six repetitions of the same audit.

    Returns:
        The tasks to report on.
    """
    if main_only:
        return MAIN
    return (*MAIN, *budget_gradient(), *rounds_curve(), objective_task())


def cached_optimum(
    cache: dict[tuple[object, int], AttainableOptimum],
) -> Callable[[Task, int], AttainableOptimum]:
    """An `attainable_optimum` that computes each landscape and budget once.

    Every diagnostic shares one landscape factory, so the diagnostics alone
    would otherwise pay for the same bounded walk and the same beam search seven
    times over.

    Args:
        cache: Where to keep results, keyed by landscape factory and budget.

    Returns:
        A callable taking a task and a budget.
    """

    def lookup(task: Task, budget: int) -> AttainableOptimum:
        key = (task.build, budget)
        if key not in cache:
            cache[key] = attainable_optimum(task, budget=budget)
        return cache[key]

    return lookup


def cached_reachability(
    cache: dict[tuple[object, int], bool | None],
) -> Callable[[Task], bool | None]:
    """A `planted_optimum_reachable` that walks each landscape and budget once.

    Args:
        cache: Where to keep answers, keyed by landscape factory and budget.

    Returns:
        A callable taking a task.
    """

    def lookup(task: Task) -> bool | None:
        key = (task.build, task.max_mutations)
        if key not in cache:
            cache[key] = planted_optimum_reachable(task)
        return cache[key]

    return lookup


def _value(result: AttainableOptimum) -> str:
    """The attainable optimum written so a bound cannot read as a measurement."""
    if result.is_exact:
        assert result.exact is not None  # noqa: S101 - narrowing for the formatter
        return f"{result.exact:.4f} exact"
    return f"[{result.lower:.4f},{result.upper:.4f}]"


def _floor(result: AttainableOptimum) -> str:
    """The regret floor, as a number when it is one and an interval otherwise."""
    low, high = result.regret_floor
    return f"{low:.4f}" if np.isclose(low, high) else f"[{low:.4f},{high:.4f}]"


def _verdict(task: Task, result: AttainableOptimum, distance: int | None) -> str:
    """Whether a regret floor is a property of the design or a defect in it."""
    if distance is None or distance <= result.budget:
        return "ok" if result.regret_floor[0] <= SOLVED_TOLERANCE else "unreachable"
    return "capped" if INTENTIONAL_CAP in task.purpose else "DEFECT"


def optima_report(
    tasks: Sequence[Task],
    lookup: Callable[[Task, int], AttainableOptimum],
    reachable: Callable[[Task], bool | None],
    *,
    report: Callable[[str], None],
) -> None:
    """What each task's search space actually contains, at its current budget.

    Args:
        tasks: Tasks to audit.
        lookup: Supplies the attainable optimum for a task and budget.
        reachable: Says whether the planted optimum has a construction order.
        report: Where lines go.
    """
    report("\n=== attainable optima, at each task's current mutation budget ===\n")
    report(
        f"{'task':<20} {'L':>4} {'budget':>6} {'planted':>7} {'reach':>5} "
        f"{'nominal':>8} {'attainable':>18} {'regret floor':>18}  verdict"
    )
    for task in tasks:
        result = lookup(task, task.max_mutations)
        distance = planted_distance(task)
        walks = reachable(task)
        report(
            f"{task.name:<20} {task.landscape().sequence_length:>4} {task.max_mutations:>6} "
            f"{'-' if distance is None else distance:>7} "
            f"{'-' if walks is None else ('yes' if walks else 'no'):>5} "
            f"{result.nominal:>8.4f} {_value(result):>18} {_floor(result):>18}  "
            f"{_verdict(task, result, distance)}"
        )
    report("")
    for task in tasks:
        report(f"  {task.name}: {lookup(task, task.max_mutations).method}")


def reanchor_report(
    tasks: Sequence[Task],
    legacy: int,
    *,
    report: Callable[[str], None],
) -> None:
    r"""What it would take to make each optimum reachable, two different ways.

    Raising the fixed budget to the planted distance is one way and it is the
    blunt one: it makes the search radius so wide that the mutation constraint
    stops being an experiment. Re-anchoring is the other -- a campaign that
    moves its anchor to the round's best design travels cumulatively, so $R$
    rounds of $b$ reach $R \cdot b$ substitutions. This prices both, and
    measures what the re-anchored chain actually attains rather than asserting
    that the counting bound is enough.

    Nothing here turns re-anchoring on. It reports what doing so would buy.

    Args:
        tasks: Tasks to size.
        legacy: The per-round budget to price as the wet-lab-sized option --
            the suite's original shared constant.
        report: Where lines go.
    """
    report("\n\n=== what makes the optimum reachable: fixed budget vs re-anchoring ===\n")
    report(
        "fixed is the budget a single-anchor campaign needs -- the planted distance itself.\n"
        f"per-round is the counting bound ceil(distance / rounds) for a campaign that\n"
        f"re-anchors. The last two columns are what a re-anchored chain was measured to\n"
        f"attain, at the original shared budget of {legacy} per round and at that bound.\n"
    )
    report(
        f"{'task':<20} {'rounds':>6} {'fixed':>6} {'per-round':>9} "
        f"{f'attained @{legacy}':>16} {'attained @bound':>16}"
    )
    cache: dict[tuple[object, int, int], AttainableOptimum] = {}

    def chain(task: Task, per_round: int) -> AttainableOptimum:
        key = (task.build, per_round, task.protocol.rounds)
        if key not in cache:
            cache[key] = reanchored_attainable(task, per_round=per_round)
        return cache[key]

    for task in tasks:
        distance = planted_distance(task)
        bound = per_round_budget(task)
        if distance is None or bound is None:
            report(
                f"{task.name:<20} {task.protocol.rounds:>6} {'-':>6} {'-':>9}  no planted optimum"
            )
            continue
        cheap, sized = chain(task, legacy), chain(task, bound)
        report(
            f"{task.name:<20} {task.protocol.rounds:>6} {distance:>6} {bound:>9} "
            f"{_value(cheap):>16} {_value(sized):>16}"
        )


def _finite(values: np.ndarray) -> np.ndarray:
    """Drop the non-finite entries a diverged arm leaves behind.

    An arm that proposed an infeasible design scores minus infinity, and a
    single one of those turns every mean in a column into minus infinity. They
    are dropped and counted rather than silently averaged.

    Args:
        values: Measured values, one per seed.

    Returns:
        Only the finite ones.
    """
    return values[np.isfinite(values)]


def rescore_report(
    tasks: Sequence[Task],
    lookup: Callable[[Task, int], AttainableOptimum],
    store: ResultStore,
    budget: int,
    *,
    report: Callable[[str], None],
) -> tuple[list[str], list[str]]:
    """Put published regret beside regret against what was actually reachable.

    Args:
        tasks: Tasks to rescore.
        lookup: Supplies the attainable optimum for a task and budget.
        store: Where stored results live. Read only.
        budget: The mutation budget the stored results were produced under.
        report: Where lines go.

    Returns:
        Lines for arms that exhausted their task, and lines for arms whose best
        result sits above a bound -- which is evidence about this audit rather
        than about the arm, and has to be reported in whichever direction it
        points.
    """
    report(f"\n\n=== stored results rescored, at the budget they ran under ({budget}) ===")
    report(
        "\nregret(nom) is what was published; regret(att) is against the attainable\n"
        "optimum, conservatively its searched lower bound. at-opt is the share of\n"
        "seeds sitting on that value: a task where it is 1.00 for any arm cannot\n"
        "rank that arm against anything.\n"
    )

    held = set(store.tasks())
    vacuous: list[str] = []
    anomalies: list[str] = []
    for task in tasks:
        if task.name not in held:
            continue
        result = lookup(task, budget)
        report(
            f"\n{task.name}  [{task.protocol!r}]  nominal {result.nominal:.4f}, "
            f"attainable {_value(result)} at {budget} mutations"
        )
        report(
            f"  {'arm':<18} {'n':>4} {'best':>9} {'regret(nom)':>12} "
            f"{'regret(att)':>12} {'at-opt':>7} {'lost':>5}"
        )
        for method in store.methods(task.name):
            records = store.load(task.name, method)
            if not records:
                continue
            best = np.asarray([record.best for record in records.values()], dtype=np.float64)
            usable = _finite(best)
            if not usable.size:
                report(f"  {method:<18} {len(best):>4}   all seeds diverged to -inf")
                continue
            at_optimum = float(np.mean(usable >= result.lower - SOLVED_TOLERANCE))
            report(
                f"  {method:<18} {len(best):>4} {usable.mean():>9.4f} "
                f"{result.nominal - usable.mean():>12.4f} "
                f"{result.lower - usable.mean():>12.4f} {at_optimum:>7.2f} "
                f"{len(best) - usable.size:>5}"
            )
            if at_optimum >= VACUOUS_SHARE or result.lower - usable.mean() <= SOLVED_TOLERANCE:
                vacuous.append(
                    f"{task.name}/{method}: {at_optimum:.0%} of seeds sit on the attainable "
                    f"optimum {result.lower:.4f}, so its published regret of "
                    f"{result.nominal - usable.mean():.4f} is almost all floor and the task "
                    f"cannot separate this arm from a better one"
                )
            anomalies.extend(_anomalies(task.name, method, result, float(usable.max())))
    return vacuous, anomalies


def _anomalies(task: str, method: str, result: AttainableOptimum, achieved: float) -> list[str]:
    """What a stored result says about the bounds, when it contradicts one.

    A stored run is a witness. Above the searched lower bound it says the search
    here was weak; above the *certified* upper bound it says something far more
    interesting, because that bound holds over the whole reachable set -- so the
    arm must have measured a design the construction graph cannot build, which a
    method free to propose sequences the environment never had to construct is
    entirely able to do.

    Args:
        task: Task name.
        method: Arm name.
        result: The attainable optimum for that task and budget.
        achieved: Best value the arm reached on any seed.

    Returns:
        Zero, one or two lines.
    """
    found = []
    if achieved > result.upper + SOLVED_TOLERANCE:
        found.append(
            f"{task}/{method}: measured {achieved:.4f}, above the certified upper bound "
            f"{result.upper:.4f} -- this arm scored a design the environment cannot construct"
        )
    elif not result.is_exact and achieved > result.lower + SOLVED_TOLERANCE:
        found.append(
            f"{task}/{method}: measured {achieved:.4f}, above this audit's searched lower "
            f"bound {result.lower:.4f} -- the bracket's lower end is the weaker number"
        )
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the audit.

    Args:
        argv: Command line, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` when no task carries an unexplained regret floor, ``1`` when one
        does. Non-zero is the point: this is the kind of thing that should be
        able to fail a pipeline rather than be read and forgotten.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="Where stored results live.")
    parser.add_argument("--main-only", action="store_true", help="Skip the diagnostics.")
    parser.add_argument(
        "--stored-budget",
        type=int,
        default=LEGACY_MUTATIONS,
        help="Mutation budget the stored results were produced at.",
    )
    parser.add_argument("--no-rescore", action="store_true", help="Skip reading results/.")
    parser.add_argument(
        "--no-reanchor",
        action="store_true",
        help="Skip the re-anchoring section, which is most of the runtime.",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    tasks = audited_tasks(main_only=args.main_only)
    lookup = cached_optimum({})
    reachable = cached_reachability({})

    optima_report(tasks, lookup, reachable, report=_flush)
    if not args.no_reanchor:
        reanchor_report(tasks, args.stored_budget, report=_flush)

    vacuous: list[str] = []
    anomalies: list[str] = []
    if not args.no_rescore:
        root = Path(args.results)
        if not root.is_dir():
            _flush(f"\nno results under {root}; skipping the rescore")
        else:
            vacuous, anomalies = rescore_report(
                tasks, lookup, ResultStore(root), args.stored_budget, report=_flush
            )

    defects = [
        task.name
        for task in tasks
        if _verdict(task, lookup(task, task.max_mutations), planted_distance(task)) == "DEFECT"
    ]
    stranded = [task.name for task in tasks if reachable(task) is False]
    _flush("\n\n=== findings ===\n")
    for line in vacuous:
        _flush(f"  SOLVED    {line}")
    for line in anomalies:
        _flush(f"  BOUND     {line}")
    for name in defects:
        _flush(f"  DEFECT    {name}: planted optimum outside the budget, and nothing says so")
    if stranded:
        _flush(
            f"  STRANDED  {', '.join(stranded)}: the planted optimum is inside the mutation\n"
            f"            budget and still has no legal construction order, because mutations "
            f"are\n            applied one at a time and every intermediate must satisfy the "
            f"transition\n            matrix on its own. Budget is necessary and not sufficient."
        )
    if not (vacuous or anomalies or defects or stranded):
        _flush("  none: every task's budget reaches its own optimum, and no arm has exhausted one")
    _flush(f"\n{time.perf_counter() - started:.0f}s")
    return 1 if defects else 0


def _flush(message: str) -> None:
    """Print immediately, so a long audit can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
