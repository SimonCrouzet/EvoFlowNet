"""Run the multi-objective suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run, on the same terms as
``experiments/run_suite.py``: every campaign is written the moment it finishes
and a second invocation runs only what is missing.

    uv run python experiments/run_multi_objective.py                 # everything
    uv run python experiments/run_multi_objective.py --tier main     # headline only
    uv run python experiments/run_multi_objective.py --seeds 50      # raise the count
    uv run python experiments/run_multi_objective.py --report        # no runs, just read

Results land under ``results-mo/`` -- a different root from the single-objective
suite on purpose. The two score different columns: there ``best`` is a fitness
and ``regret`` is a distance to an audited optimum, here ``best`` is a
hypervolume and ``regret`` is IGD+. Sharing a directory would let one table be
built across both, and nothing in a record would say the columns meant different
things.

The tiers differ in role, not just in seed count
------------------------------------------------

Only **main** carries results. ``conflict`` and ``objectives`` are *explanatory*:
they say under what conditions the main table's ranking would change, and a win
that appears at one conflict level and vanishes at another is a finding about the
sweep rather than an extra headline row. ``preferences`` is the only diagnostic
that decides anything -- how many preference vectors the main-table GFlowNet arm
should get -- and it is measured at fixed total budget, so eight preferences buys
48 assays each rather than eight full campaigns.

The report prints that role next to every tier, because a reader who takes an
explanatory sweep for a result has been misled by the layout rather than by the
numbers.

Two limitations the report states rather than hides
---------------------------------------------------

**Hypervolume goes missing exactly where it matters.** The exact method in
[evogfn.metrics.pareto][] raises past 16 front points in three or more
objectives, and on ``ch65-real`` a converged arm's 384 measurements carry a front
of up to 19 while a random arm's carries 2--9. So the column is present for the
arms that did badly and ``nan`` for the arms that did well, which is worse than
useless if read as a ranking. ``ch65-real`` is read on IGD+, and the report says
how many seeds lost their hypervolume so the gap cannot be mistaken for a run
that failed.

**The Ehrlich reference fronts are constructed, not enumerated.** A 20-letter
alphabet is enumerable only up to L=5, so no instance in this suite has an exact
front available; what stands in is the exact front over a declared set of
recombinations of the objectives' planted optima. Every point in it is attained
by a sequence that exists, so IGD+ = 0 is reachable -- but it is a *subset* of
the true front, and an arm saturating it has covered what the construction found
rather than the front. The report marks which tasks are in that position.

What it costs, measured rather than guessed
--------------------------------------------

One campaign per arm, timed on one core at the protocols the tasks declare:

| task | random | nsga2 | genetic+proxy | gfn-tb |
| --- | --- | --- | --- | --- |
| `ch65-real` | 2s | 1s | 11s | 35s |
| `mo-ehrlich-hard` (L=64) | 3s | <1s | 5s | 261s |
| explanatory (L=32) | <1s | <1s | 4s | 39s |

At the default seed counts that is roughly **4.4 CPU-hours** for the main tier,
1.8 for the conflict sweep and 1.1 for the objective-count sweep.

The preference diagnostic is the rest of the budget and then some: **~31
CPU-hours**, four fifths of the suite. Training is not divided when the plate is,
so eight preferences means eight policies each trained for the full 300 steps a
round on an eighth of the assays -- 2,363s a seed, against 1,040s at four
preferences and 264s at one. That asymmetry is the arm's design rather than an
accident: the comparison is held at equal *oracle budget*, which is the
constrained resource, and not at equal wall clock. It is also why the diagnostic
runs at the explanatory seed count rather than the main one.

Total, at the defaults: **~38 CPU-hours**. Sharding is per task and arm, since
the store keeps one file per pair, so the critical path is the eight-preference
arm at roughly 20 hours; ``--explanatory-seeds 10`` brings that under seven if
the answer only needs to be directional.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.multi_objective import (
    EXACT_FRONT_LIMIT,
    MultiObjectiveTask,
    arms_for_tier,
    multi_objective_tiers,
    run_multi_objective_tier,
)
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import Tier, records_to_metric

if TYPE_CHECKING:
    from evogfn.benchmark.tasks import Task

#: Seeds for the headline tier. Fewer than the single-objective suite's 100: a
#: multi-objective campaign carries the same oracle budget but a GFlowNet arm at
#: eight preferences trains eight policies for it, and CH65 rebuilds a 62,926-row
#: table per campaign.
MAIN_SEEDS = 50

#: Seeds for the explanatory sweeps and the preference diagnostic.
EXPLANATORY_SEEDS = 30

#: Where results go. Deliberately not ``results/``: see the module docstring.
DEFAULT_RESULTS = "results-mo"

#: The arm every other one is compared against -- a genetic algorithm with the
#: same proxy access and the same stated trade-off the GFlowNet gets, so a
#: difference against it is a difference between search methods rather than
#: between one method that optimises the model and one that does not.
REFERENCE_ARM = "genetic+proxy"

#: What a tier is for, printed next to it. Kept here rather than on
#: [Tier][evogfn.benchmark.suite.Tier] because it is a property of this suite's
#: reading, and `Tier.headline` already carries the part that is structural.
TIER_ROLES = {
    "main": "carries results",
    "conflict": "explanatory: says when the ranking changes, not what it is",
    "objectives": "explanatory: says what the objective count itself costs",
    "preferences": "decides the main table's preference count, and nothing else",
}

# Two observations are the fewest a paired comparison can be drawn from.
_MIN_PAIRED = 2


def _task_note(task: Task) -> str:
    """State what this task's indicators can and cannot say.

    Args:
        task: The task being reported on.

    Returns:
        One or two lines naming the reference point, whether the reference front
        is exact, and where hypervolume is at risk of not being computable.
    """
    if not isinstance(task, MultiObjectiveTask):
        return "  not a multi-objective task; nothing here applies to it"
    point = ", ".join(f"{value:g}" for value in task.reference_point)
    front = task.reference_front()
    if front is None:
        against = "no reference front, so IGD+ is unreported"
    else:
        quality = "an exact" if task.front_is_exact else "a constructed subset of the true"
        against = f"IGD+ against {quality} front of {front.shape[0]} point(s)"
    lines = [f"  hypervolume from ({point}); {against}"]
    if task.n_objectives > 2 and front is not None:  # noqa: PLR2004 - two objectives sweep exactly
        lines.append(
            f"  note: exact hypervolume needs a measured front of at most "
            f"{EXACT_FRONT_LIMIT} points at {task.n_objectives} objectives; "
            f"seeds above it record nan and are counted below"
        )
    if not task.front_is_exact:
        lines.append(
            "  note: IGD+ = 0 here means the constructed front was covered, which is "
            "weaker than covering the true front"
        )
    return "\n".join(lines)


def report(store: ResultStore, tier: Tier, reference: str = REFERENCE_ARM) -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Hypervolume and IGD+ rather than best and regret, because that is what the
    records hold -- see
    [set_indicators][evogfn.benchmark.multi_objective.set_indicators]. The
    comparison is drawn on **IGD+**, and on IGD+ alone: hypervolume is missing
    wherever the exact method could not run, and a paired test over a column
    whose absences correlate with an arm's quality would be reading the absences.

    Args:
        store: Where results live.
        tier: The tier to report on.
        reference: Arm to compare against.

    Returns:
        A multi-line report.
    """
    role = TIER_ROLES.get(tier.name, "unstated")
    lines = [f"\n=== {tier!r} -- {role}"]
    names = list(arms_for_tier(tier))
    for task in tier.tasks:
        lines.append(f"\n{task!r}")
        lines.append(_task_note(task))
        held = {name: store.usable(task.name, name) for name in names}
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        for name in names:
            records = held[name]
            if not records:
                continue
            volume = records_to_metric(records, tier.seeds, "best")
            coverage = records_to_metric(records, tier.seeds, "regret")
            spread = records_to_metric(records, tier.seeds, "diversity")
            spent = records_to_metric(records, tier.seeds, "oracle_calls")
            uncomputed = int(np.isnan(volume).sum())
            finite = volume[np.isfinite(volume)]
            error = coverage.std(ddof=1) / len(coverage) ** 0.5 if len(coverage) > 1 else 0.0
            lines.append(
                f"  {name:<18} igd+ {np.nanmean(coverage):>7.4f} +/- {error:<7.4f} "
                f"hv {(finite.mean() if finite.size else float('nan')):>9.4f} "
                f"(nan on {uncomputed}/{len(volume)})  "
                f"div {np.nanmean(spread):>5.2f}  spent {np.nanmean(spent):>6.0f}  "
                f"n={len(coverage)}"
            )

        base = held.get(reference)
        if not base or not seeds:
            continue
        lines.append(f"  paired on IGD+ vs {reference} (positive favours the first):")
        for name in names:
            if name == reference or not held[name]:
                continue
            mine = records_to_metric(held[name], seeds, "regret")
            theirs = records_to_metric(base, seeds, "regret")
            if len(mine) != len(theirs) or len(mine) < _MIN_PAIRED:
                continue
            # Lower IGD+ is better, so this is a loss like regret is.
            outcome = compare(name, mine, theirs, higher_is_better=False)
            lines.append(f"    {outcome!r}")
            if not outcome.significant and (needed := seeds_needed(outcome)):
                lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run or report on the multi-objective suite.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: 0 on success, 2 when nothing matched the selection,
        3 when threading is not pinned and a run was asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", help="Run only these tiers.")
    parser.add_argument(
        "--task",
        action="append",
        help="Run only these tasks. Sharding by task is race-free -- the store "
        "keeps one file per task and method -- so a process per task uses the "
        "cores far better than threads do, most of the work being serial Python.",
    )
    parser.add_argument("--seeds", type=int, default=MAIN_SEEDS, help="Seeds for the main tier.")
    parser.add_argument(
        "--explanatory-seeds",
        type=int,
        default=EXPLANATORY_SEEDS,
        help="Seeds for the sweeps and the preference diagnostic.",
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Where to store results.")
    parser.add_argument("--report", action="store_true", help="Report without running.")
    args = parser.parse_args(argv)

    # Before any tensor work: a multithreaded matmul sums in thread-completion
    # order, and a few hundred gradient steps turn that into a different design.
    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    selected = multi_objective_tiers(args.seeds, args.explanatory_seeds)
    if args.tier:
        selected = [t for t in selected if t.name in set(args.tier)]
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(t.name, tuple(k for k in t.tasks if k.name in wanted), t.seeds, t.headline)
            for t in selected
        ]
        selected = [t for t in selected if t.tasks]
    if not selected:
        print(f"nothing matched tier={args.tier} task={args.task}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    for tier in selected:
        if not args.report:
            ran = run_multi_objective_tier(tier, arms_for_tier(tier), store, report=_flush)
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
