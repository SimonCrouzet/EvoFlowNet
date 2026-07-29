"""Run the benchmark suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run. Every campaign is written the moment it
finishes, and a second invocation runs only what is missing -- so raising a
tier's seed count from 30 to 50 costs twenty campaigns per arm, not fifty.

    uv run python experiments/run_suite.py                  # everything
    uv run python experiments/run_suite.py --tier main      # headline only
    uv run python experiments/run_suite.py --seeds 50       # raise the count
    uv run python experiments/run_suite.py --report         # no runs, just read

Results land under ``results/`` as one JSONL file per task and method.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import BASELINES, OBJECTIVES, flow_objectives
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    MAIN,
    Tier,
    budget_gradient,
    objective_task,
    records_to_metric,
    rounds_curve,
    run_tier,
)

#: Seeds per tier. The main tiers differ because campaign cost differs by an
#: order of magnitude between L=4 and L=256, not because the claims differ.
MAIN_SEEDS = 100
LARGE_SPACE_SEEDS = 30
DIAGNOSTIC_SEEDS = 50

#: Everything except the flow objectives, which need a policy with a flow head
#: and are compared separately in the objectives diagnostic.
MAIN_METHODS = {**BASELINES, **OBJECTIVES}


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
        Tier("objectives", (objective_task(),), tuple(range(diagnostic_seeds)), headline=False),
        Tier("main", cheap, tuple(range(main_seeds)), headline=True),
        Tier("rounds-curve", rounds_curve(), tuple(range(diagnostic_seeds)), headline=False),
        Tier("budget-gradient", budget_gradient(), tuple(range(diagnostic_seeds)), headline=False),
        # Last: ~200s a campaign at L=256, and fewer seeds for the same reason.
        Tier("large-space", expensive, tuple(range(LARGE_SPACE_SEEDS)), headline=True),
    ]


def methods_for(tier: Tier) -> dict[str, object]:
    """Which methodologies a tier runs.

    The objectives diagnostic is GFlowNet-only, since a classical baseline has
    no training objective to vary; everything else compares methods.
    """
    if tier.name == "objectives":
        return {**OBJECTIVES, **flow_objectives()}
    return dict(MAIN_METHODS)


def report(store: ResultStore, tier: Tier, reference: str = "genetic+proxy") -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Args:
        store: Where results live.
        tier: The tier to report on.
        reference: Arm to compare against. Defaults to the strongest control --
            a genetic algorithm with the same proxy access the GFlowNet gets.

    Returns:
        A multi-line report.
    """
    lines = []
    names = list(methods_for(tier))
    for task in tier.tasks:
        lines.append(f"\n{task.name}  [{task.protocol!r}]")
        held = {name: store.usable(task.name, name) for name in names}
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        for name in names:
            records = held[name]
            if not records:
                continue
            regret = records_to_metric(records, tier.seeds, "regret")
            feasible = records_to_metric(records, tier.seeds, "feasible_fraction")
            spread = records_to_metric(records, tier.seeds, "diversity")
            spent = records_to_metric(records, tier.seeds, "oracle_calls")
            error = regret.std(ddof=1) / len(regret) ** 0.5 if len(regret) > 1 else 0.0
            lines.append(
                f"  {name:<18} regret {regret.mean():>7.3f} +/- {error:<6.3f} "
                f"feas {feasible.mean():>5.3f}  div {spread.mean():>5.2f}  "
                f"spent {spent.mean():>6.0f}  n={len(regret)}"
            )

        base = held.get(reference)
        if not base or not seeds:
            continue
        lines.append(f"  paired vs {reference} (positive favours the first):")
        for name in names:
            if name == reference or not held[name]:
                continue
            mine = records_to_metric(held[name], seeds, "regret")
            theirs = records_to_metric(base, seeds, "regret")
            if len(mine) != len(theirs) or len(mine) < 2:  # noqa: PLR2004
                continue
            outcome = compare(name, mine, theirs, higher_is_better=False)
            lines.append(f"    {outcome!r}")
            if not outcome.significant and (needed := seeds_needed(outcome)):
                lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    return "\n".join(lines)


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
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(
                t.name,
                tuple(task for task in t.tasks if task.name in wanted),
                t.seeds,
                t.headline,
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
            ran = run_tier(tier, methods_for(tier), store, report=_flush)  # type: ignore[arg-type]
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
