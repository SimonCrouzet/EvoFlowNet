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

Each builds a :class:`~evogfn.surrogate.proxy.ProxyLandscape` over the same
surrogate instance the campaign refits, so training costs proxy evaluations and
never oracle calls. The classical baselines are offered both blind and with the
same proxy access, because comparing a method that optimises the model against
one that only meets it as a filter is not a comparison of methods.
"""

from __future__ import annotations

from collections.abc import Callable
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
    sampler: Sampler,
    surrogate: DeepEnsemble | None,
) -> Campaign:
    """Assemble a campaign under the task's protocol."""
    return Campaign(
        landscape=landscape,  # type: ignore[arg-type]
        sampler=sampler,
        surrogate=surrogate,
        acquisition=Greedy(),
        selector=TopK(),
        rounds=task.protocol.rounds,
        batch_size=task.protocol.batch_size,
        pool_size=max(DEFAULT_POOL, task.protocol.batch_size * 4),
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
        sampler = build(env, seed)
        if proxy_access:
            proxy = ProxyLandscape(
                ensemble,
                alphabet=env.alphabet,
                sequence_length=env.sequence_length,
            )
            sampler = ProxyOptimising(sampler, proxy=proxy)
        return _campaign(task, landscape, sampler, ensemble if surrogate else None)

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
        policy = SequencePolicy(
            n_actions=env.n_actions,
            sequence_length=env.sequence_length,
            n_tokens=env.alphabet.size,
            hidden_dim=128,
            learn_flow=learn_flow,
            seed=seed,
        )
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
        sampler = GFlowNetSampler(
            env,
            policy,
            proxy=proxy,
            reward=TemperedReward(beta=beta),
            config=TrainingConfig(steps=steps, batch_size=64, seed=seed),
            objective=objective,
            seed=seed,
        )
        return _campaign(task, landscape, sampler, ensemble)

    return methodology


def genetic_gflownet(
    objective: GFlowNetObjective | None = None,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
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
        sampler = GFlowNetSampler(
            env,
            policy,
            proxy=proxy,
            reward=TemperedReward(beta=beta),
            config=TrainingConfig(steps=steps, batch_size=64, seed=seed),
            objective=objective,
            genetic=GeneticAlgorithm(env, seed=seed),
            genetic_config=GeneticConfig(offspring=64, warmup=max(steps // 10, 1)),
            seed=seed,
        )
        return _campaign(task, landscape, sampler, ensemble)

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

    Separate from :data:`OBJECTIVES` because they require ``learn_flow`` and
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
