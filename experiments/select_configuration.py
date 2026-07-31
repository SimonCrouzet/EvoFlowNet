"""Fix the GFlowNet's configuration before the benchmark measures it.

Every classical baseline in the suite runs at hyperparameters its own authors
tuned. A GFlowNet run at inherited defaults against that field is not being
compared to it, so this phase runs first and its answer is an input to the
headline tables rather than one of their rows.

Two stages, because the full cross of objectives and reward exponents is far
more compute than the question needs:

**Stage A** compares the six training objectives at the default exponent. The
30-seed diagnostic put four of them within 0.02 regret of each other and asked
for thousands of seeds to separate them, so this runs at 100 -- enough to
resolve the one gap that looked real (sub-trajectory balance against trajectory
balance, at roughly 97 seeds) and enough to state honestly that the rest are
tied.

**Stage B** scans the reward exponent for whichever objective stage A chose. It
runs second because the winner is not known until stage A finishes, and it scans
only the winner because scanning all six would cost five more nights. The cost of
that economy is real and worth naming: an objective that loses at the default
exponent and would have won at another is invisible to this design.

The rule is in [evogfn.benchmark.selection][], written down before the numbers
arrived. Both stages run on the diagnostic landscape, which no headline task
uses, so nothing here is tuning on the test set.

    uv run python experiments/select_configuration.py            # both stages
    uv run python experiments/select_configuration.py --report   # read, no runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import OBJECTIVES, flow_objectives
from evogfn.benchmark.selection import Scored, Selection, beta_arms, select
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import Purpose, Tier, objective_task, run_tier

#: Seeds per arm. Set by what the 30-seed diagnostic said it would take to
#: resolve sub-trajectory balance against trajectory balance, rounded up.
SELECTION_SEEDS = 100

#: Where the chosen configuration is written, so the benchmark reads a decision
#: rather than re-deriving one and possibly re-deriving it differently.
CHOICE_FILE = Path("results/selected.json")


def stage_a_methods() -> dict[str, object]:
    """The six training objectives, at the default reward exponent."""
    return {**OBJECTIVES, **flow_objectives()}


def run_stage(name: str, methods: dict[str, object], store: ResultStore, *, report: bool) -> None:
    """Run one stage's arms on the diagnostic landscape."""
    tier = Tier(name, (objective_task(),), tuple(range(SELECTION_SEEDS)), Purpose.SELECTION)
    if not report:
        ran = run_tier(tier, methods, store, report=_flush)  # type: ignore[arg-type]
        _flush(f"{name}: ran {ran} campaigns")


def held(store: ResultStore, methods: dict[str, object]) -> dict[str, dict[int, Scored]]:
    """Usable records per arm, dropping arms with nothing stored yet."""
    task = objective_task().name
    found = {name: store.usable(task, name) for name in methods}
    kept = {name: records for name, records in found.items() if records}
    # RunRecord satisfies Scored structurally; the store is typed concretely.
    return cast("dict[str, dict[int, Scored]]", kept)


def describe(stage: str, choice: Selection) -> str:
    """Lay out a stage's table and the decision drawn from it."""
    lines = [f"\n=== {stage} ===="]
    for name in sorted(choice.regret, key=lambda n: choice.regret[n]):
        mark = "<-- chosen" if name == choice.chosen else ("tied" if name in choice.tied else "")
        lines.append(
            f"  {name:<24} regret {choice.regret[name]:7.4f}  "
            f"div {choice.diversity[name]:6.2f}  {mark}"
        )
    lines.append(f"  decision: {choice.chosen}")
    lines.append(f"  because:  {choice.reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run both stages and record the configuration they select."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Read the store without running.")
    parser.add_argument("--results", default="results", help="Where results live.")
    parser.add_argument("--seeds", type=int, default=SELECTION_SEEDS, help="Seeds per arm.")
    args = parser.parse_args(argv)

    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    started = time.perf_counter()

    objectives = stage_a_methods()
    run_stage("select-objective", objectives, store, report=args.report)
    stored = held(store, objectives)
    if not stored:
        print("stage A has no usable results yet", file=sys.stderr)
        return 2
    objective = select(stored)
    _flush(describe("stage A: training objective", objective))

    exponents = beta_arms(objective.chosen)
    run_stage("select-beta", exponents, store, report=args.report)  # type: ignore[arg-type]
    stored = held(store, exponents)  # type: ignore[arg-type]
    if not stored:
        print("stage B has no usable results yet", file=sys.stderr)
        return 2
    exponent = select(stored)
    _flush(describe("stage B: reward exponent", exponent))

    choice = {
        "objective": objective.chosen,
        "objective_reason": objective.reason,
        "objective_tied": list(objective.tied),
        "arm": exponent.chosen,
        "arm_reason": exponent.reason,
        "seeds": args.seeds,
        "task": objective_task().name,
    }
    if not args.report:
        CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHOICE_FILE.write_text(json.dumps(choice, indent=2) + "\n")
    _flush(f"\nselected {exponent.chosen}")
    _flush(f"total {time.perf_counter() - started:.0f}s")
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long phase can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
