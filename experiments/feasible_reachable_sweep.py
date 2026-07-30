r"""What action masking excludes: feasible designs with no feasible construction order.

Enforcing a constraint by masking actions does not restrict a sampler to the
feasible set. It restricts it to the feasible states reachable *through feasible
intermediates*, which is smaller, and on a tightly constrained instance it can
exclude the best feasible design outright.

    uv run python experiments/feasible_reachable_sweep.py
    uv run python experiments/feasible_reachable_sweep.py --seeds 200
    uv run python experiments/feasible_reachable_sweep.py --densities 0.9 0.5

This is the measurement behind that claim. It was previously an uncommitted
script, so the numbers in the manuscript could not be re-derived by anyone --
including their author. Everything the sweep needs is a constructor argument
here, and the default geometry reproduces the original run exactly (see below).

Why the two sets differ
-----------------------

Mutations are applied one at a time, so a design carrying $k$ of them is
constructible only if some ordering exists along which all $k$ intermediates are
feasible too. When every ordering passes through a forbidden adjacency, the
forward mask refuses that step in all of them and no path to the destination
exists -- even though the destination is itself perfectly feasible. Feasibility
of a design is a property of the design; reachability is a property of the
*graph*, and the graph is what a masked policy samples from.

The consequence is not a rounding error. Where the excluded set contains the
best feasible design, a masked sampler is optimising against a ceiling below the
one every report of it assumes, and no amount of training reaches the difference.

Why many seeds
--------------

The original run used three. That is not enough to say anything: at a fixed
density the exclusion measured here ranges from 0% to over 75% across seeds,
because it depends on which adjacencies the instance forbade and where the
parent happens to sit relative to them. A single seed at density 0.5 showed
68.8% and another showed 12.2%. So this reports the mean, the spread and the
range over `--seeds` instances, and counts the instances whose optimum was
excluded rather than reporting whether one particular instance's was.

The default instance
--------------------

`L = 8`, `v = 4`, a budget of 3 substitutions -- a Hamming ball of 1,789
sequences, small enough that the reachable set can be found by exhaustive
forward search at every seed and every density. Two motifs of length two at
quantization two, which is the smallest motif geometry that puts a partial
reward level between 0 and 1 and so lets "the optimum is excluded" show up as a
number rather than as a flag.

At `--seeds 1` this reproduces the original ad-hoc table cell for cell:

| density | ball | feasible | reachable | excluded | best feasible | best reachable |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | 1789 | 1662 | 1662 | 0.0% | 1.000 | 1.000 |
| 0.70 | 1789 | 380 | 324 | 14.7% | 1.000 | 1.000 |
| 0.50 | 1789 | 77 | 24 | 68.8% | 1.000 | 0.500 |
| 0.15 | 1789 | 18 | 4 | 77.8% | 1.000 | 0.500 |
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evogfn.core.types import Tokens

#: Transition densities swept by default, from "barely a constraint" to
#: "almost nothing is legal". The last two are where the exclusion appears.
DEFAULT_DENSITIES = (0.90, 0.70, 0.50, 0.30, 0.15)

#: How much worse the best reachable design has to be before it counts as
#: excluded. Rewards here are quantised, so any real gap is far above this; the
#: tolerance exists only so floating-point equality is not asked for.
_GAP_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Instance:
    """One measured (density, seed) pair.

    Attributes:
        density: Transition density the instance was drawn at.
        seed: Seed for the landscape and the parent.
        ball: Size of the Hamming ball, the set the budget alone allows.
        feasible: Designs in the ball satisfying the transition constraint.
        reachable: Designs a masked trajectory can actually terminate in.
        best_feasible: Best fitness over the feasible set -- the ceiling a
            report implicitly assumes when it says the constraint is enforced.
        best_reachable: Best fitness over the reachable set -- the ceiling that
            actually applies.
    """

    density: float
    seed: int
    ball: int
    feasible: int
    reachable: int
    best_feasible: float
    best_reachable: float

    @property
    def excluded(self) -> float:
        """Share of the feasible set that no trajectory can reach.

        Returns:
            A fraction in ``[0, 1]``, or ``0.0`` where the feasible set is empty
            and there is nothing to exclude.
        """
        if self.feasible == 0:
            return 0.0
        return 1.0 - self.reachable / self.feasible

    @property
    def optimum_excluded(self) -> bool:
        """Whether the best feasible design is one a masked sampler cannot emit."""
        return self.best_reachable < self.best_feasible - _GAP_TOLERANCE


def measure(  # noqa: PLR0913 - the instance is defined by its parameters
    *,
    density: float,
    seed: int,
    sequence_length: int,
    vocab_size: int,
    max_mutations: int,
    n_motifs: int,
    motif_length: int,
    quantization: int,
    max_spacing: int,
) -> Instance:
    """Measure one instance: how much of its feasible set masking can reach.

    The reachable set is found by walking the environment's own forward masks
    rather than by re-deriving reachability from the transition matrix. That is
    deliberate: a second implementation could disagree with the masks, and then
    this measurement would be about the second implementation rather than about
    the sampler.

    Args:
        density: Fraction of token pairs allowed to be adjacent.
        seed: Seeds the landscape and, separately, the parent draw.
        sequence_length: Length of every sequence.
        vocab_size: Alphabet size.
        max_mutations: Substitutions a trajectory may accumulate.
        n_motifs: Motifs that must be satisfied simultaneously.
        motif_length: Tokens per motif.
        quantization: Reward levels per motif.
        max_spacing: Largest gap between consecutive motif positions.

    Returns:
        The counts and the two ceilings.
    """
    landscape = EhrlichLandscape(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        n_motifs=n_motifs,
        motif_length=motif_length,
        quantization=quantization,
        max_spacing=max_spacing,
        transition_density=density,
        seed=seed,
    )
    env = MutationEnvironment(
        landscape.feasible_sequence(seed),
        landscape.alphabet,
        max_mutations=max_mutations,
        transitions=landscape.transition_matrix,
    )

    ball = env.enumerate_terminal_states()
    feasible = ball[landscape.is_feasible(ball)]
    reachable = env.reachable_terminal_states()

    return Instance(
        density=density,
        seed=seed,
        ball=int(ball.shape[0]),
        feasible=int(feasible.shape[0]),
        reachable=int(reachable.shape[0]),
        best_feasible=_best(landscape, feasible),
        best_reachable=_best(landscape, reachable),
    )


def _best(landscape: EhrlichLandscape, designs: Tokens) -> float:
    """Best fitness over a set of designs.

    Args:
        landscape: What to score against.
        designs: An ``(n, length)`` array, possibly empty.

    Returns:
        The maximum fitness, or ``-inf`` for an empty set -- which is the honest
        answer for "the best design you can build" when there are none.
    """
    if designs.shape[0] == 0:
        return -float("inf")
    return float(landscape.evaluate(designs)[:, 0].max())


@dataclass(frozen=True, slots=True)
class Summary:
    """What a column of seeds at one density says.

    Attributes:
        density: The density these instances were drawn at.
        instances: Every measurement at this density.
    """

    density: float
    instances: tuple[Instance, ...]

    @property
    def exclusions(self) -> list[float]:
        """Excluded share, one per seed."""
        return [instance.excluded for instance in self.instances]

    @property
    def mean_excluded(self) -> float:
        """Mean excluded share over the seeds."""
        return float(statistics.fmean(self.exclusions))

    @property
    def spread(self) -> float:
        """Standard deviation of the excluded share, ``0`` for a single seed."""
        values = self.exclusions
        return float(statistics.stdev(values)) if len(values) > 1 else 0.0

    @property
    def stranded(self) -> tuple[Instance, ...]:
        """Instances whose best feasible design no trajectory can construct."""
        return tuple(instance for instance in self.instances if instance.optimum_excluded)


def sweep(
    densities: Sequence[float],
    seeds: int,
    *,
    report: Callable[[str], None],
    **geometry: int,
) -> tuple[Summary, ...]:
    """Measure every density at every seed and report as it goes.

    Args:
        densities: Transition densities to sweep.
        seeds: How many instances to draw per density.
        report: Where lines go. Called per density so a long sweep can be
            watched rather than waited on.
        **geometry: Instance parameters, forwarded to `measure`.

    Returns:
        One summary per density, in the order given.
    """
    report(
        f"{'density':>7} {'ball':>6} {'feasible':>18} {'reachable':>18} "
        f"{'excluded (mean+-sd)':>21} {'range':>15} {'optimum lost':>12}"
    )
    summaries = []
    for density in densities:
        instances = tuple(measure(density=density, seed=seed, **geometry) for seed in range(seeds))
        summary = Summary(density=density, instances=instances)
        summaries.append(summary)

        feasible = [instance.feasible for instance in instances]
        reachable = [instance.reachable for instance in instances]
        exclusions = summary.exclusions
        report(
            f"{density:>7.2f} {instances[0].ball:>6} "
            f"{statistics.fmean(feasible):>9.1f} [{min(feasible):>3},{max(feasible):>4}] "
            f"{statistics.fmean(reachable):>9.1f} [{min(reachable):>3},{max(reachable):>4}] "
            f"{summary.mean_excluded:>13.1%} +-{summary.spread:>5.1%} "
            f"{min(exclusions):>6.1%}-{max(exclusions):>6.1%} "
            f"{len(summary.stranded):>6}/{seeds:<5}"
        )
    return tuple(summaries)


def stranded_report(summaries: Sequence[Summary], *, report: Callable[[str], None]) -> int:
    """Name every instance whose optimum masking put out of reach.

    Args:
        summaries: What `sweep` measured.
        report: Where lines go.

    Returns:
        How many instances were stranded, across all densities.
    """
    report("\n\n=== instances whose best feasible design is unreachable ===\n")
    report(
        "On these, a masked sampler is optimising against a ceiling below the one the\n"
        "feasible set implies. Training cannot close the gap: the design has no legal\n"
        "construction order, so the policy's support does not contain it.\n"
    )
    report(
        f"  {'density':>7} {'seed':>5} {'feasible':>9} {'reachable':>10} "
        f"{'best feas':>10} {'best reach':>11}"
    )
    total = 0
    for summary in summaries:
        for instance in summary.stranded:
            total += 1
            report(
                f"  {instance.density:>7.2f} {instance.seed:>5} {instance.feasible:>9} "
                f"{instance.reachable:>10} {instance.best_feasible:>10.3f} "
                f"{instance.best_reachable:>11.3f}"
            )
    if total == 0:
        report("  none at these densities and seeds")
    return total


def monotonicity_report(summaries: Sequence[Summary], *, report: Callable[[str], None]) -> None:
    """Check the qualitative claim: exclusion grows as the constraint tightens.

    Reported rather than asserted, because it is a claim about a mean over
    instances and any single pair of adjacent densities can invert on a small
    sample. The test in ``tests/env`` pins the qualitative property; this says
    whether the sweep just run agrees with it.

    Args:
        summaries: What `sweep` measured, in the order the densities were given.
        report: Where lines go.
    """
    report("\n\n=== does exclusion grow as density falls? ===\n")
    ordered = sorted(summaries, key=lambda summary: summary.density, reverse=True)
    previous: Summary | None = None
    for summary in ordered:
        arrow = ""
        if previous is not None:
            change = summary.mean_excluded - previous.mean_excluded
            arrow = f"  {'+' if change >= 0 else ''}{change:.1%} vs density {previous.density:.2f}"
            if change < 0:
                arrow += "   <- inverted"
        report(f"  density {summary.density:.2f}: mean excluded {summary.mean_excluded:.1%}{arrow}")
        previous = summary


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Args:
        argv: Command line, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` always. Exclusion is the finding, not a failure -- a sweep that
        found none would be reporting a property of the geometry it was given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=list(DEFAULT_DENSITIES),
        help="Transition densities to sweep.",
    )
    parser.add_argument("--seeds", type=int, default=50, help="Instances per density.")
    parser.add_argument("--length", type=int, default=8, help="Sequence length.")
    parser.add_argument("--vocab", type=int, default=4, help="Alphabet size.")
    parser.add_argument("--max-mutations", type=int, default=3, help="Per-trajectory budget.")
    parser.add_argument("--n-motifs", type=int, default=2, help="Motifs to satisfy at once.")
    parser.add_argument("--motif-length", type=int, default=2, help="Tokens per motif.")
    parser.add_argument("--quantization", type=int, default=2, help="Reward levels per motif.")
    parser.add_argument("--max-spacing", type=int, default=3, help="Largest gap within a motif.")
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error(f"--seeds must be at least 1, got {args.seeds}")

    _flush(
        f"L={args.length} v={args.vocab} budget={args.max_mutations}, "
        f"{args.n_motifs} motifs of length {args.motif_length} at q={args.quantization}, "
        f"{args.seeds} seeds per density\n"
    )
    summaries = sweep(
        args.densities,
        args.seeds,
        report=_flush,
        sequence_length=args.length,
        vocab_size=args.vocab,
        max_mutations=args.max_mutations,
        n_motifs=args.n_motifs,
        motif_length=args.motif_length,
        quantization=args.quantization,
        max_spacing=args.max_spacing,
    )
    monotonicity_report(summaries, report=_flush)
    stranded = stranded_report(summaries, report=_flush)

    measured = sum(len(summary.instances) for summary in summaries)
    _flush(
        f"\n\n{stranded} of {measured} instances ({stranded / measured:.1%}) have a best "
        f"feasible design that action masking cannot construct."
    )
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long sweep can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
