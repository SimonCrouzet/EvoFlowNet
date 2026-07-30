"""The methodologies under test: samplers crossed with training objectives.

A methodology is whatever turns a task and a seed into a campaign. Keeping that
one callable means a GFlowNet variant, a classical baseline, and a baseline
given surrogate access are all the same kind of thing to the harness, and no arm
can quietly receive a different budget, surrogate or starting point than
another.

Everything a methodology varies is varied *inside* the campaign it returns. The
task fixes the landscape, the protocol and the wild type; the seed fixes the
surrogate initialisation and the sampler's randomness. What is left is the
method, which is the only thing a comparison should be measuring.

The GFlowNet variants train against a proxy, never the oracle
--------------------------------------------------------------

Each builds a [ProxyLandscape][evogfn.surrogate.proxy.ProxyLandscape] over the
same surrogate instance the campaign refits, so training costs proxy evaluations
and never oracle calls. The classical baselines are offered both blind and with
the same proxy access, because comparing a method that optimises the model
against one that only meets it as a filter is not a comparison of methods.

Every methodology can follow a moved anchor
-------------------------------------------

A task that re-anchors moves its
[MutationEnvironment][evogfn.env.mutation.MutationEnvironment] to the best design
measured so far at the end of every round, and the campaign refuses at
construction if the sampler cannot follow. There are two ways to follow, and the
campaign prefers the first:

* the sampler implements
  [reanchored][evogfn.loop.campaign.ReanchorableSampler.reanchored] and says what
  should happen to its own state -- a trained policy survives, a CMA-ES mean
  decoded relative to the old parent does not;
* the campaign rebuilds it from a factory, which is always correct and always
  forgetful.

Every methodology here supplies a factory, so no task can be configured to
re-anchor and then fail at construction. The factories are written to close over
whatever the rebuild must not lose -- the *same* `SequencePolicy` object, so a
GFlowNet keeps its trained weights, and the *same* `ProxyLandscape`, so it keeps
its link to the surrogate the campaign refits. What a rebuild does lose is the
sampler's own accounting: `proxy_calls` and the round count restart, and
[Campaign.sampler][evogfn.loop.campaign.Campaign.sampler] returns the rebuilt
object, so a stored ``proxy_calls`` under re-anchoring counts the last anchor's
rounds rather than the campaign's. That is the reason a sampler with state worth
keeping should grow a `reanchored` hook rather than rely on the factory -- and
because the campaign resolves the hook first, doing so takes effect here without
any change to this module.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from typing import TYPE_CHECKING

import numpy as np

from evogfn.acquisition.rules import Greedy, TopK
from evogfn.algorithms.baselines.annealing import SimulatedAnnealing
from evogfn.algorithms.baselines.cmaes import CMAES
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.mlde import MLDE
from evogfn.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis
from evogfn.algorithms.gflownet.genetic_gfn import GeneticConfig
from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.env.mutation import MutationEnvironment
from evogfn.loop.campaign import Campaign
from evogfn.models.policy import SequencePolicy
from evogfn.rewards.base import TemperedReward
from evogfn.surrogate.ensemble import DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.algorithms.base import Sampler
    from evogfn.algorithms.gflownet.objectives import GFlowNetObjective
    from evogfn.benchmark.tasks import Task

#: A methodology turns a task and a seed into a runnable campaign.
Methodology = Callable[["Task", int], Campaign]

#: Reward exponent. Jain et al. and the MOGFN papers use 3.
DEFAULT_BETA = 3.0

#: Gradient steps per campaign round. Free in oracle terms, not in wall clock.
DEFAULT_TRAINING_STEPS = 300

#: Candidates generated per round before selection.
DEFAULT_POOL = 2048

#: Genetic-GFN's offline mixing ratio, after Jain et al. (2022).
DEFAULT_MIX = 0.5


def _anchor_seed(seed: int, generation: int) -> int:
    """A distinct, reproducible seed for each anchor a campaign moves to.

    A sampler rebuilt from a factory starts from its constructor, which means it
    starts from its seed -- and a sampler re-seeded identically every round
    proposes the identical pool every round. The campaign then deduplicates
    almost all of it and the campaign stalls: measured directly, a random
    mutagenesis arm re-proposed its first pool at every anchor and spent rounds
    two onward on the tail of a batch it had already generated.

    Reproducibility is not given up to fix it. The stream is a pure function of
    the campaign's seed and how many times its anchor has moved, both of which
    are fixed by the run, so a re-run reproduces it exactly.

    Args:
        seed: The campaign's seed.
        generation: How many times the anchor has moved, from zero.

    Returns:
        The seed to build the sampler for that anchor with. Generation zero
        returns `seed` unchanged, so a task that never re-anchors is bit-for-bit
        what it was before the mechanism existed.
    """
    if generation == 0:
        return seed
    return int(np.random.SeedSequence([seed, generation]).generate_state(1)[0])


def _parts(task: Task, seed: int) -> tuple[object, MutationEnvironment, DeepEnsemble]:
    """Everything a campaign needs that is not the method itself.

    Built identically for every methodology on a given task and seed, which is
    what makes the comparison paired rather than merely simultaneous.

    The landscape's feasibility rule is handed to the environment, which is what
    makes masked sampling possible at all. Omitting it does not raise: it
    silently switches feasibility-by-construction off, so every proposal scores
    minus infinity and the surrogate has nothing finite to fit. That is how this
    was wrong in its first version, and it is the whole of claim C1.
    """
    landscape = task.landscape()
    env = MutationEnvironment(
        task.parent(landscape),
        landscape.alphabet,
        max_mutations=task.max_mutations,
        transitions=_feasibility_of(landscape),
    )
    surrogate = DeepEnsemble(
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        epochs=150,
        seed=seed,
    )
    return landscape, env, surrogate


def _feasibility_of(landscape: object) -> npt.NDArray[np.floating] | None:
    """The landscape's transition matrix, when it has one.

    Returns:
        The matrix whose zeros mark forbidden adjacent token pairs, or ``None``
        for a landscape with no constructibility constraint.
    """
    matrix = getattr(landscape, "transition_matrix", None)
    return None if matrix is None else np.asarray(matrix)


def _campaign(
    task: Task,
    landscape: object,
    env: MutationEnvironment,
    build: Callable[[MutationEnvironment], Sampler],
    surrogate: DeepEnsemble | None,
) -> Campaign:
    """Assemble a campaign under the task's protocol, anchored where it says.

    Args:
        task: Fixes the protocol, the search radius and whether the anchor moves.
        landscape: The oracle.
        env: The environment the sampler was built against, handed to the
            campaign so the ledger records which design each round searched from
            and so the anchor has something to move.
        build: Rebuilds the sampler for a moved anchor. Called once here for the
            opening round, so the sampler the campaign starts with and the ones
            it rebuilds come from one place and cannot drift apart.
        surrogate: Model fitted to the measurements, or ``None`` for the
            unassisted ablation.

    Returns:
        The campaign, which refuses at construction if the task asks to
        re-anchor and anything needed for it is missing.
    """
    return Campaign(
        landscape=landscape,  # type: ignore[arg-type]
        sampler=build(env),
        surrogate=surrogate,
        acquisition=Greedy(),
        selector=TopK(),
        rounds=task.protocol.rounds,
        batch_size=task.protocol.batch_size,
        pool_size=max(DEFAULT_POOL, task.protocol.batch_size * 4),
        environment=env,
        reanchor=task.reanchor,
        sampler_factory=build,
    )


def classical(
    build: Callable[[MutationEnvironment, int], Sampler],
    *,
    surrogate: bool = True,
    proxy_access: bool = False,
) -> Methodology:
    """A classical baseline, optionally given the same proxy the GFlowNet gets.

    Args:
        build: Makes the sampler from an environment and a seed.
        surrogate: Whether a surrogate screens the proposal pool.
        proxy_access: Whether the sampler may also *optimise* against the
            surrogate, as the GFlowNet does. Without this the comparison is
            between a method that uses the model and one that does not.

    Returns:
        A methodology.
    """

    def methodology(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed)
        # One proxy for the whole campaign, closed over rather than rebuilt: it
        # wraps the surrogate instance the campaign refits in place, and a fresh
        # one per anchor would still see the same model but would make that
        # dependence look accidental. Its shape does not change with the anchor.
        proxy = (
            ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
            if proxy_access
            else None
        )

        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the baseline against whichever anchor the campaign is at."""
            sampler = build(anchored, _anchor_seed(seed, next(generation)))
            return sampler if proxy is None else ProxyOptimising(sampler, proxy=proxy)

        return _campaign(task, landscape, env, make, ensemble if surrogate else None)

    return methodology


def gflownet(
    objective: GFlowNetObjective | None = None,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
    learn_flow: bool = False,
) -> Methodology:
    """A GFlowNet trained against the surrogate proxy.

    Args:
        objective: How balance violation is measured. Defaults to trajectory
            balance.
        steps: Gradient steps per round.
        beta: Reward exponent.
        learn_flow: Whether to build a flow head. Required by the
            detailed-balance family and useless to the others, so it is set by
            the caller alongside the objective rather than guessed.

    Returns:
        A methodology.
    """

    def methodology(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed)
        # The policy is built once and closed over, so a rebuild for a moved
        # anchor keeps the trained weights. It survives the move because its
        # action space -- length * |alphabet| + 1 indices -- and its input, the
        # state sequence, are both properties of the space rather than of the
        # anchor. Only the masks move.
        policy = SequencePolicy(
            n_actions=env.n_actions,
            sequence_length=env.sequence_length,
            n_tokens=env.alphabet.size,
            hidden_dim=128,
            learn_flow=learn_flow,
            seed=seed,
        )
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)

        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the sampler against whichever anchor the campaign is at."""
            stream = _anchor_seed(seed, next(generation))
            return GFlowNetSampler(
                anchored,
                policy,
                proxy=proxy,
                reward=TemperedReward(beta=beta),
                config=TrainingConfig(steps=steps, batch_size=64, seed=stream),
                objective=objective,
                seed=stream,
            )

        return _campaign(task, landscape, env, make, ensemble)

    return methodology


def genetic_gflownet(
    objective: GFlowNetObjective | None = None,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
    mix: float = DEFAULT_MIX,
) -> Methodology:
    """A GFlowNet taught by a genetic algorithm, after Kim et al. (2024).

    The variant most likely to matter here: directed evolution is a genetic
    algorithm, the Ehrlich benchmark's own baseline is one, and a vanilla
    GFlowNet trails Mol GA by 58% on PMO. Genetic-GFN closes that by absorbing
    the GA rather than competing with it.

    Args:
        objective: How balance violation is measured.
        steps: Gradient steps per round.
        beta: Reward exponent.
        mix: Share of each training batch bred by the genetic teacher rather
            than sampled from the policy. This is the knob the method is *about*
            -- at zero it is an ordinary GFlowNet and at one the policy only
            ever sees the GA's offspring -- so it is exposed rather than left at
            the config default.

    Returns:
        A methodology.
    """

    def methodology(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed)
        policy = SequencePolicy(
            n_actions=env.n_actions,
            sequence_length=env.sequence_length,
            n_tokens=env.alphabet.size,
            hidden_dim=128,
            seed=seed,
        )
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the sampler and a fresh teacher for the current anchor.

            The policy carries over and the genetic teacher does not, which is
            the right split rather than an oversight: a population bred around
            the old parent can sit wholly outside the new mutation budget, so
            re-seeding it at the new anchor is what keeps the teacher's
            offspring inside the space the policy is being taught to sample.
            """
            stream = _anchor_seed(seed, next(generation))
            return GFlowNetSampler(
                anchored,
                policy,
                proxy=proxy,
                reward=TemperedReward(beta=beta),
                config=TrainingConfig(steps=steps, batch_size=64, seed=stream),
                objective=objective,
                genetic=GeneticAlgorithm(anchored, seed=stream),
                genetic_config=GeneticConfig(offspring=64, mix=mix, warmup=max(steps // 10, 1)),
                seed=stream,
            )

        return _campaign(task, landscape, env, make, ensemble)

    return methodology


def _random(env: MutationEnvironment, seed: int) -> Sampler:
    return RandomMutagenesis(env, seed=seed)


def _hill_climb(env: MutationEnvironment, seed: int) -> Sampler:
    return HillClimbing(env, seed=seed)


def _genetic(env: MutationEnvironment, seed: int) -> Sampler:
    return GeneticAlgorithm(env, seed=seed)


def _annealing(env: MutationEnvironment, seed: int) -> Sampler:
    return SimulatedAnnealing(env, seed=seed)


def _cmaes(env: MutationEnvironment, seed: int) -> Sampler:
    return CMAES(env, seed=seed)


def _mlde(env: MutationEnvironment, seed: int) -> Sampler:
    """Machine-learning-directed evolution, the method protein engineers run.

    The most important baseline here after the genetic algorithm: it is what
    Wittmann et al. actually do, at almost exactly this budget, and its whole
    claim is reaching the answer in hundreds of assays rather than thousands.
    """
    return MLDE(env, seed=seed)


def _feasible_genetic(env: MutationEnvironment, seed: int) -> Sampler:
    """A genetic algorithm that rejection-samples until its offspring are legal.

    The control for the feasibility claim. Where masking is free, rejection
    sampling costs proposals, and on a sparse feasible set it becomes
    impractical -- which is itself the result.
    """
    return GeneticAlgorithm(env, seed=seed, feasible_only=True, max_attempts=200)


#: The classical baselines. Directed evolution *is* a genetic algorithm, so
#: these are the incumbents rather than strawmen to be cleared.
BASELINES: dict[str, Methodology] = {
    "random": classical(_random, surrogate=False),
    "random+surrogate": classical(_random),
    "hill-climb": classical(_hill_climb),
    "genetic": classical(_genetic),
    "genetic+proxy": classical(_genetic, proxy_access=True),
    "genetic-feasible": classical(_feasible_genetic, proxy_access=True),
    "annealing": classical(_annealing, proxy_access=True),
    "cmaes": classical(_cmaes, proxy_access=True),
    "mlde": classical(_mlde, proxy_access=True),
}

#: GFlowNet objectives, each behind the same interface. Comparing them is a
#: configuration change rather than a rewrite, which is the point of the seam.
OBJECTIVES: dict[str, Methodology] = {
    "gfn-tb": gflownet(TrajectoryBalance()),
    "gfn-contrastive": gflownet(ContrastiveBalance(prune_threshold=0.1)),
    "genetic-gfn": genetic_gflownet(TrajectoryBalance()),
}


def flow_objectives() -> dict[str, Methodology]:
    """The detailed-balance family, which needs a policy with a flow head.

    Separate from `OBJECTIVES` because they require ``learn_flow`` and
    would raise rather than silently degrade if handed a policy without one.

    Returns:
        Methodologies by name.
    """
    from evogfn.algorithms.gflownet.flow_objectives import (  # noqa: PLC0415
        DetailedBalance,
        ForwardLookingDetailedBalance,
        SubTrajectoryBalance,
    )

    return {
        "gfn-db": gflownet(DetailedBalance(), learn_flow=True),
        "gfn-subtb": gflownet(SubTrajectoryBalance(lam=0.9), learn_flow=True),
        "gfn-fldb": gflownet(ForwardLookingDetailedBalance(), learn_flow=True),
    }


#: The GFlowNet settings this project has never measured, and the values to
#: measure them at. Each was inherited rather than chosen: ``steps`` because 300
#: ran in acceptable time, ``beta`` from Jain et al. (2022), ``mix`` from
#: Kim et al.'s (2024) offline ratio. A headline comparison against baselines
#: tuned to their own papers, run at settings nobody tuned, measures the
#: settings -- so what this grid is for is establishing that the reported
#: configuration is not a bad one, and saying by how much it could be beaten.
#:
#: Values bracket each default above and below rather than extending in one
#: direction, so a monotone column is legible as "the grid is too narrow" rather
#: than being mistaken for an optimum.
SENSITIVITY_GRID: dict[str, tuple[float, ...]] = {
    "steps": (100.0, float(DEFAULT_TRAINING_STEPS), 900.0),
    "beta": (1.0, DEFAULT_BETA, 10.0),
    "mix": (0.0, DEFAULT_MIX, 1.0),
}


def sensitivity() -> dict[str, Methodology]:
    """One arm per hyperparameter value, varying one axis at a time.

    One at a time rather than a full grid: the full cross of
    `SENSITIVITY_GRID` is 27 arms where this is 9, and the question being asked
    is whether any single setting is badly chosen -- not where the joint optimum
    sits, which this benchmark has nothing like the seed count to locate.

    The arm at each default duplicates a configuration the objectives
    diagnostic already runs -- ``steps-300`` and ``beta-3`` are ``gfn-tb``,
    ``mix-0.5`` is ``genetic-gfn`` -- and is re-run anyway. It costs two extra
    arms on the cheapest landscape in the suite, and it buys an axis that reads
    as a curve on its own terms rather than as two measured points plus a
    cross-reference to another tier's table.

    Returns:
        Methodologies by name, named ``<axis>-<value>`` so a report groups them.
    """
    arms: dict[str, Methodology] = {}
    for axis, values in SENSITIVITY_GRID.items():
        for value in values:
            name = f"{axis}-{value:g}"
            if axis == "steps":
                arms[name] = gflownet(TrajectoryBalance(), steps=int(value))
            elif axis == "beta":
                arms[name] = gflownet(TrajectoryBalance(), beta=value)
            else:
                # `mix` belongs to the genetic teacher, so it varies the arm
                # that has one. Its endpoints are the method's own limits: at 0
                # this is plain `gfn-tb` and at 1 the policy never samples for
                # itself, which brackets the claim that the hybrid beats both.
                arms[name] = genetic_gflownet(TrajectoryBalance(), mix=value)
    return arms


def default_methodologies() -> dict[str, Methodology]:
    """Every methodology, for a full sweep.

    Returns:
        Baselines, then GFlowNet objectives, in a stable order so a report
        reads the same way each run.
    """
    return {**BASELINES, **OBJECTIVES, **flow_objectives()}


def rng_for(seed: int) -> np.random.Generator:
    """A generator for anything a methodology needs beyond its components."""
    return np.random.default_rng(seed)
