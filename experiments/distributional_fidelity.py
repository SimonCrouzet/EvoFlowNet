r"""Does the policy sample $p^*(x) \propto R(x)^\beta$, or merely climb well?

Every other metric this project reports -- best found, top-K, regret, even
diversity -- is satisfiable by an optimiser that never samples anything. The
exact L1 against the enumerated target is the one measurement that is not. The
same condition asserted as a pass/fail in a test suite cannot be compared across
objectives and carries no spread, which is why it is reported as a number here.

    uv run python experiments/distributional_fidelity.py
    uv run python experiments/distributional_fidelity.py --seeds 5 --steps 4000
    uv run python experiments/distributional_fidelity.py --objectives tb db

This reports it as a table: L1 against the exact target and against the
**sampling-noise floor**, for five training objectives over several seeds.

Why the floor, and not the L1 alone
-----------------------------------

Drawing $m$ samples from a distribution does not reproduce it. A *perfect*
sampler shows a non-zero L1 at any finite $m$, so a small L1 is uninterpretable
on its own -- it could be a well-fitted policy or a badly measured one. Every
number here is therefore printed as a multiple of what a sampler drawing from
the target itself would show at the same sample count. A ratio near 1 means
"indistinguishable from exact"; the raw L1 alone means nothing.

Which support the L1 is normalised over
---------------------------------------

The target must be normalised over the set the policy can actually construct.
Under a transition constraint that is **not** the Hamming ball: mutations are
applied one at a time, so a feasible design whose every construction order
passes through an infeasible intermediate is excluded from the policy's support
entirely (this is what ``experiments/feasible_reachable_sweep.py`` measures).

Normalising over the ball instead puts target mass on designs of probability
zero, and the L1 that comes back measures the mis-specified support rather than
the sampler. A correct policy is penalised twice for each such design, once for
the mass it cannot place and once for the mass it places elsewhere, so the
inflation is structural rather than incidental. The script computes both and
prints the ratio, because the failure mode is invisible unless you look for it:
the ball is the cheap closed-form answer, it is what `enumerate_terminal_states`
returns, and it is wrong here.

The instance
------------

`L = 8` over 4 tokens within 3 substitutions, small enough that both the ball
and the reachable set can be enumerated exactly. Two motifs of length four at
quantization four give a graded target rather than a spike; matching a spike
would prove nothing. The transition density is set tight enough that a
non-trivial share of the ball's target mass sits on feasible designs no
trajectory can build, which is the condition the support comparison needs in
order to have anything to show.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.gflownet import (
    ContrastiveBalance,
    DetailedBalance,
    FlowObjective,
    ForwardLookingDetailedBalance,
    SubTrajectoryBalance,
    TrainingConfig,
    TrajectoryBalance,
    sample_trajectories,
    train_trajectory_balance,
)
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.metrics import (
    empirical_distribution,
    expected_l1_from_sampling_noise,
    l1_distance,
    target_distribution,
)
from evogfn.models import SequencePolicy
from evogfn.rewards import TemperedReward

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from evogfn.algorithms.gflownet import GFlowNetObjective
    from evogfn.core.types import Tokens

#: The default instance. Chosen so the reachable set is a strict subset of the
#: feasible one -- without that, the support comparison has nothing to show --
#: and small enough to enumerate exactly.
INSTANCE = {
    "sequence_length": 8,
    "vocab_size": 4,
    "n_motifs": 2,
    "motif_length": 4,
    "quantization": 4,
    "max_spacing": 1,
    "transition_density": 0.7,
    "seed": 1,
}

#: Substitutions a trajectory may accumulate.
BUDGET = 3

#: Objectives compared, by the name the command line uses. Every one enforces
#: the same condition and differs only in how it measures the violation, so a
#: difference in the L1 column is a difference in what the objective can fit.
OBJECTIVES: dict[str, Callable[[], GFlowNetObjective]] = {
    "tb": TrajectoryBalance,
    "contrastive": ContrastiveBalance,
    "db": DetailedBalance,
    "subtb": SubTrajectoryBalance,
    "fl-db": ForwardLookingDetailedBalance,
}


@dataclass(frozen=True, slots=True)
class Fidelity:
    """What one trained policy was measured to sample.

    Attributes:
        objective: Which objective trained it.
        seed: Seed for the policy and its rollouts.
        l1: L1 against the exact target over the reachable support.
        l1_ball: L1 against the target normalised over the Hamming ball
            instead -- the number a report gets by using the cheap enumeration.
        log_z: The learned partition function, or ``None`` for the objectives
            that never learn one.
        seconds: Wall time for training and measurement.
    """

    objective: str
    seed: int
    l1: float
    l1_ball: float
    log_z: float | None
    seconds: float


@dataclass(frozen=True, slots=True)
class Space:
    """The enumerated instance, and both targets defined over it.

    Two supports are carried deliberately. `reachable` is what the policy can
    construct and therefore what the L1 must be normalised over; `ball` is what
    the mutation budget alone allows, and is here only so the cost of using it
    can be measured rather than argued about.

    Attributes:
        landscape: The instance.
        env: The construction graph.
        ball: Every sequence within the budget.
        reachable: Every sequence a trajectory can terminate in.
        target: The exact target over `reachable`.
        target_ball: The exact target over `ball`.
        log_z: ``log Σ R(x)`` over the reachable support, which is what an
            objective that learns a partition function should converge to.
    """

    landscape: EhrlichLandscape
    env: MutationEnvironment
    ball: Tokens
    reachable: Tokens
    target: npt.NDArray[np.float64]
    target_ball: npt.NDArray[np.float64]
    log_z: float


def build_space(reward: TemperedReward, *, budget: int, **instance: object) -> Space:
    """Enumerate the instance and compute both targets over it.

    Args:
        reward: The reward transform the policy will be trained against. The
            target has to use the same one, or the L1 measures the disagreement
            between two reward definitions rather than the sampler.
        budget: Substitutions a trajectory may accumulate.
        **instance: Landscape parameters.

    Returns:
        The enumerated space and its targets.
    """
    landscape = EhrlichLandscape(**instance)  # type: ignore[arg-type]
    env = MutationEnvironment(
        landscape.feasible_sequence(int(instance["seed"])),  # type: ignore[call-overload]
        landscape.alphabet,
        max_mutations=budget,
        transitions=landscape.transition_matrix,
    )
    ball = env.enumerate_terminal_states()
    reachable = env.reachable_terminal_states()

    values = landscape.evaluate(reachable)[:, 0]
    log_rewards = reward.log_reward(values)
    return Space(
        landscape=landscape,
        env=env,
        ball=ball,
        reachable=reachable,
        target=target_distribution(values, beta=reward.beta, min_reward=reward.floor),
        target_ball=target_distribution(
            landscape.evaluate(ball)[:, 0], beta=reward.beta, min_reward=reward.floor
        ),
        log_z=float(np.logaddexp.reduce(log_rewards)),
    )


def train_and_measure(  # noqa: PLR0913 - the measurement is defined by its parts
    name: str,
    space: Space,
    reward: TemperedReward,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    samples: int,
) -> Fidelity:
    """Train one policy and measure what it samples.

    Args:
        name: Which objective to train with, a key of `OBJECTIVES`.
        space: The enumerated instance.
        reward: The reward transform, shared with the target.
        seed: Seeds the policy's weights and its rollouts.
        steps: Optimisation steps.
        batch_size: Trajectories per step.
        samples: Draws used for the empirical distribution.

    Returns:
        The measured fidelity.
    """
    objective = OBJECTIVES[name]()
    started = time.perf_counter()
    policy = SequencePolicy(
        n_tokens=space.landscape.alphabet.size,
        sequence_length=space.landscape.sequence_length,
        n_actions=space.env.n_actions,
        hidden_dim=128,
        embedding_dim=32,
        # The detailed-balance family constrains flow through each state and
        # cannot be trained without a flow head; trajectory balance and
        # contrastive balance never read one, and giving them one would add
        # parameters that receive no gradient.
        learn_flow=isinstance(objective, FlowObjective),
        seed=seed,
    )
    train_trajectory_balance(
        space.env,
        policy,
        space.landscape,
        reward,
        TrainingConfig(steps=steps, batch_size=batch_size, seed=seed),
        objective=objective,
    )

    drawn = sample_trajectories(
        space.env, policy, samples, generator=torch.Generator().manual_seed(1000 + seed)
    ).terminal
    return Fidelity(
        objective=name,
        seed=seed,
        l1=l1_distance(empirical_distribution(drawn, space.reachable), space.target),
        l1_ball=l1_distance(empirical_distribution(drawn, space.ball), space.target_ball),
        log_z=float(policy.log_z.detach().item()) if objective.uses_log_z else None,
        seconds=time.perf_counter() - started,
    )


def fidelity_report(
    results: Sequence[Fidelity],
    space: Space,
    floor: float,
    *,
    report: Callable[[str], None],
) -> None:
    """The headline table: L1 per objective, against the target and the floor.

    Args:
        results: Every measured run.
        space: The enumerated instance, for the exact ``log Z``.
        floor: L1 a perfect sampler would show at this sample count.
        report: Where lines go.
    """
    report("\n\n=== L1 against the exact target, over the reachable support ===\n")
    report(
        "x floor is the L1 as a multiple of what a sampler drawing from the target itself\n"
        "would show at this sample count. Near 1 is indistinguishable from exact; the raw\n"
        "L1 on its own says nothing without it.\n"
    )
    report(
        f"  {'objective':<12} {'n':>3} {'L1 (mean+-sd)':>20} {'x floor':>8} "
        f"{'best':>7} {'worst':>7} {'log Z err':>10} {'s/seed':>7}"
    )
    for name in dict.fromkeys(result.objective for result in results):
        rows = [result for result in results if result.objective == name]
        values = [result.l1 for result in rows]
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        mean = statistics.fmean(values)
        errors = [result.log_z - space.log_z for result in rows if result.log_z is not None]
        report(
            f"  {name:<12} {len(rows):>3} {mean:>13.4f} +-{spread:>5.4f} {mean / floor:>8.2f} "
            f"{min(values):>7.4f} {max(values):>7.4f} "
            f"{(f'{statistics.fmean(errors):+.3f}' if errors else '-'):>10} "
            f"{statistics.fmean([result.seconds for result in rows]):>7.1f}"
        )
    report(
        f"\n  exact log Z over the reachable support: {space.log_z:.3f}. Only trajectory\n"
        f"  balance learns one; contrastive balance cancels it and the flow objectives\n"
        f"  learn F(s_0) instead, so their column is empty by construction rather than\n"
        f"  by omission."
    )


def reference_report(
    space: Space, floor: float, samples: int, *, report: Callable[[str], None]
) -> None:
    """What the L1 column has to beat before it means anything.

    A distributional number is only informative against the degenerate samplers
    it is supposed to separate a GFlowNet from. The greedy row is the one that
    matters: it has zero regret and wins every performance metric in this
    repository.

    Args:
        space: The enumerated instance.
        floor: L1 a perfect sampler would show.
        samples: Sample count the floor was computed at.
        report: Where lines go.
    """
    size = space.reachable.shape[0]
    uniform = l1_distance(np.full(size, 1.0 / size), space.target)
    values = space.landscape.evaluate(space.reachable)[:, 0]
    best = space.reachable[int(np.argmax(values))]
    greedy = l1_distance(
        empirical_distribution(np.tile(best, (samples, 1)), space.reachable), space.target
    )
    report("\n\n=== what the column has to beat ===\n")
    report(f"  perfect sampler (noise floor at {samples:,} draws)  L1 = {floor:.4f}")
    report(f"  uniform over the reachable support                 L1 = {uniform:.4f}")
    report(f"  greedy: every draw is the best design              L1 = {greedy:.4f}")
    report(
        "\n  The greedy row has zero regret and the best top-K in this repository. It is\n"
        "  the sampler every other metric here cannot tell a GFlowNet apart from."
    )


def support_report(
    results: Sequence[Fidelity], space: Space, *, report: Callable[[str], None]
) -> None:
    """Compare the L1 over the reachable support against the L1 over the ball.

    Args:
        results: Every measured run.
        space: The enumerated instance.
        report: Where lines go.
    """
    report("\n\n=== a result about evaluation: the ball is the wrong support ===\n")
    excluded = 1.0 - space.reachable.shape[0] / int(space.landscape.is_feasible(space.ball).sum())
    report(
        f"  Hamming ball within {BUDGET} substitutions   {space.ball.shape[0]:>6,} sequences\n"
        f"  of which feasible                     "
        f"{int(space.landscape.is_feasible(space.ball).sum()):>6,}\n"
        f"  of which a trajectory can construct   {space.reachable.shape[0]:>6,}"
        f"   ({excluded:.1%} of the feasible set excluded)\n"
    )
    report(
        f"  target mass on designs the policy cannot emit: "
        f"{float(space.target_ball.sum() - space.target_ball[_ball_index(space)].sum()):.3f}\n"
    )
    report(f"  {'objective':<12} {'L1 (reachable)':>15} {'L1 (ball)':>11} {'inflation':>10}")
    for name in dict.fromkeys(result.objective for result in results):
        rows = [result for result in results if result.objective == name]
        reachable = statistics.fmean([result.l1 for result in rows])
        ball = statistics.fmean([result.l1_ball for result in rows])
        report(
            f"  {name:<12} {reachable:>15.4f} {ball:>11.4f} "
            f"{(ball / reachable if reachable > 0 else float('nan')):>9.1f}x"
        )
    report(
        "\n  Nothing about the sampler changes between the two L1 columns above. The ball\n"
        "  simply contains feasible designs that carry target mass and have no legal\n"
        "  construction order, so a correct policy is penalised for every one of them,\n"
        "  and a report using `enumerate_terminal_states` -- the cheap, closed-form,\n"
        "  obvious choice -- concludes the sampler is broken."
    )


def _ball_index(space: Space) -> npt.NDArray[np.intp]:
    """Positions of the reachable designs within the ball enumeration.

    Args:
        space: The enumerated instance.

    Returns:
        An index array, so the ball target can be split into the mass the policy
        can emit and the mass it cannot.
    """
    lookup = {row.tobytes(): position for position, row in enumerate(space.ball.astype(np.int64))}
    return np.asarray(
        [lookup[row.tobytes()] for row in space.reachable.astype(np.int64)], dtype=np.intp
    )


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Args:
        argv: Command line, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` always. Which objective wins is the finding; there is nothing here
        that can fail a pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=sorted(OBJECTIVES),
        default=sorted(OBJECTIVES),
        help="Which objectives to compare.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Policies trained per objective.")
    parser.add_argument("--steps", type=int, default=3000, help="Optimisation steps per policy.")
    parser.add_argument("--batch-size", type=int, default=64, help="Trajectories per step.")
    parser.add_argument("--samples", type=int, default=20000, help="Draws for the empirical.")
    parser.add_argument("--beta", type=float, default=1.0, help="Reward exponent.")
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error(f"--seeds must be at least 1, got {args.seeds}")

    reward = TemperedReward(beta=args.beta)
    space = build_space(reward, budget=BUDGET, **INSTANCE)
    floor = expected_l1_from_sampling_noise(space.target, n_samples=args.samples)

    _flush(
        f"Ehrlich L={INSTANCE['sequence_length']} v={INSTANCE['vocab_size']} "
        f"density={INSTANCE['transition_density']} seed={INSTANCE['seed']}, budget {BUDGET}, "
        f"beta={args.beta}\n"
        f"ball {space.ball.shape[0]:,}  reachable {space.reachable.shape[0]:,}  "
        f"exact log Z {space.log_z:.3f}  noise floor {floor:.4f} at {args.samples:,} draws\n"
        f"{args.steps} steps x {args.batch_size} trajectories, {args.seeds} seeds per objective\n"
    )

    results: list[Fidelity] = []
    for name in args.objectives:
        for seed in range(args.seeds):
            result = train_and_measure(
                name,
                space,
                reward,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                samples=args.samples,
            )
            results.append(result)
            _flush(
                f"  {name:<12} seed {seed}  L1 {result.l1:.4f} ({result.l1 / floor:.2f}x floor)"
                f"  {result.seconds:.0f}s"
            )

    fidelity_report(results, space, floor, report=_flush)
    reference_report(space, floor, args.samples, report=_flush)
    support_report(results, space, report=_flush)
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long comparison can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    sys.exit(main())
