"""Genetic-GFN: a genetic algorithm used as the GFlowNet's teacher.

Kim et al. (2024) report the single most useful negative-to-positive result in
this literature. On PMO a vanilla GFlowNet scores 9.93 against Mol GA's 15.69 --
a 58% deficit -- and Genetic-GFN reaches 16.21 by *absorbing* the GA rather than
competing with it. Directed evolution is itself a genetic algorithm, and the
Ehrlich benchmark's own baseline is one, so shipping vanilla trajectory balance
and reporting a loss would be a result about our implementation rather than
about GFlowNets.

How it works
------------

Each step:

#. The policy samples a batch, which is scored.
#. Everything seen goes into a rank-based buffer.
#. A genetic algorithm recombines and mutates the buffer's best, producing
   offspring that are usually better than anything the policy would have drawn.
#. Those offspring are *replayed* under the policy -- a backward path is drawn
   and scored forward -- so they become trajectories the objective can train on.
#. The loss mixes on-policy trajectories with these off-policy ones.

The GA is a proposal mechanism, not the thing being trained. What the GFlowNet
keeps is what a GA cannot give: a distribution to sample from rather than a
population to read off, so a second batch is not a near-copy of the first.

Rank-based rather than reward-based sampling
--------------------------------------------

The buffer samples with probability proportional to ``1 / (rank + k)`` rather
than to reward. Fitness scales differ by orders of magnitude between landscapes
and across a campaign's rounds, so a reward-proportional buffer is implicitly
re-tuned by the landscape; rank is invariant to any monotone rescaling, which
means one setting works everywhere.

Known limitation
----------------

Replayed trajectories carry no per-step record, so the detailed-balance family
cannot train on them. This trainer refuses those objectives up front rather than
failing partway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from evoflownet.algorithms.gflownet.flow_objectives import FlowObjective
from evoflownet.algorithms.gflownet.objectives import (
    TrajectoryBalance,
    parameter_groups,
)
from evoflownet.algorithms.gflownet.replay import replay_trajectories
from evoflownet.algorithms.gflownet.sampling import sample_trajectories
from evoflownet.algorithms.gflownet.training import TrainingConfig, TrainingResult
from evoflownet.tracking.base import NoOpTracker

if TYPE_CHECKING:
    from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
    from evoflownet.algorithms.gflownet.objectives import GFlowNetObjective
    from evoflownet.core.types import Tokens
    from evoflownet.env.base import SequenceEnvironment
    from evoflownet.landscapes.base import FitnessLandscape
    from evoflownet.models.policy import SequencePolicy
    from evoflownet.rewards.base import Reward
    from evoflownet.tracking.base import Tracker

#: Smoothing constant in the rank weighting, following prioritised replay.
RANK_OFFSET = 10.0


@dataclass(frozen=True, slots=True)
class GeneticConfig:
    """How much genetic guidance to apply.

    Attributes:
        buffer_size: Sequences retained, best-first.
        offspring: Candidates the GA breeds per step. The best of them fill
            the genetic share of the training batch, so this is a pool size
            rather than the number trained on.
        generations: GA generations run against the buffer per step.
        mix: Fraction of the training batch drawn from genetic offspring rather
            than from the policy. At ``0`` this is ordinary training; at ``1``
            the policy only ever learns from the GA. Jain et al.'s offline
            mixing ratio of 0.5 is the default.
        warmup: Steps of pure on-policy training before guidance starts, so the
            buffer is not seeded entirely from an untrained policy.
    """

    buffer_size: int = 1000
    offspring: int = 64
    generations: int = 2
    mix: float = 0.5
    warmup: int = 10

    def __post_init__(self) -> None:
        """Reject configurations that cannot describe a run."""
        if self.buffer_size < 1 or self.offspring < 1 or self.generations < 1:
            raise ValueError("buffer_size, offspring and generations must all be at least 1")
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError(f"mix must lie in [0, 1], got {self.mix}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")


class RankedBuffer:
    """Keeps the best sequences seen, sampled by rank rather than by reward.

    Args:
        capacity: How many sequences to retain.
        offset: Smoothing constant ``k`` in ``1 / (rank + k)``. Larger values
            flatten the distribution toward uniform.

    Raises:
        ValueError: If the capacity is not positive or the offset is negative.
    """

    def __init__(self, capacity: int, *, offset: float = RANK_OFFSET) -> None:
        """Start empty."""
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        self._capacity = capacity
        self._offset = offset
        self._sequences: Tokens | None = None
        self._log_rewards = np.zeros(0, dtype=np.float64)

    def __len__(self) -> int:
        """How many sequences are held."""
        return int(self._log_rewards.shape[0])

    @property
    def sequences(self) -> Tokens:
        """The retained sequences, best first."""
        if self._sequences is None:
            raise ValueError("the buffer is empty")
        return self._sequences

    @property
    def log_rewards(self) -> np.ndarray:
        """Their log rewards, descending."""
        return self._log_rewards

    def add(self, sequences: Tokens, log_rewards: np.ndarray) -> None:
        """Insert scored sequences, keeping the best and dropping duplicates.

        Args:
            sequences: An ``(n, length)`` array.
            log_rewards: An ``(n,)`` array of their log rewards.
        """
        incoming = np.ascontiguousarray(sequences)
        values = np.asarray(log_rewards, dtype=np.float64).reshape(-1)
        finite = np.isfinite(values)
        incoming, values = incoming[finite], values[finite]
        if incoming.shape[0] == 0:
            return

        if self._sequences is not None:
            incoming = np.concatenate([self._sequences, incoming])
            values = np.concatenate([self._log_rewards, values])

        # Sort by reward before deduplicating, so the survivor of a repeated
        # sequence is its best measurement rather than whichever np.unique
        # happened to index first -- which under a noisy oracle would discard
        # the better reading at random.
        ranked = np.argsort(-values, kind="stable")
        incoming, values = incoming[ranked], values[ranked]

        # Duplicates would let one lucky design dominate the rank distribution
        # in proportion to how often it was resampled rather than how good it is.
        _, unique = np.unique(incoming, axis=0, return_index=True)
        keep = np.sort(unique)[: self._capacity]
        self._sequences = incoming[keep]
        self._log_rewards = values[keep]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[Tokens, np.ndarray]:
        """Draw ``n`` sequences with probability falling off by rank.

        Args:
            n: How many to draw.
            rng: Source of randomness.

        Returns:
            An ``(n, length)`` array and their ``(n,)`` log rewards, drawn with
            replacement. The rewards come back because a genetic algorithm
            selects on fitness -- handing it the sequences alone would reduce
            its teacher role to random recombination.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._sequences is None:
            raise ValueError("cannot sample from an empty buffer")
        ranks = np.arange(len(self), dtype=np.float64)
        weights = 1.0 / (ranks + self._offset)
        weights /= weights.sum()
        drawn = rng.choice(len(self), size=n, p=weights)
        return self._sequences[drawn], self._log_rewards[drawn]


def train_genetic_gfn(  # noqa: PLR0913 - the run is defined by its parts
    env: SequenceEnvironment,
    policy: SequencePolicy,
    landscape: FitnessLandscape,
    reward: Reward,
    config: TrainingConfig,
    *,
    genetic: GeneticAlgorithm,
    genetic_config: GeneticConfig | None = None,
    objective: GFlowNetObjective | None = None,
    tracker: Tracker | None = None,
) -> TrainingResult:
    """Train a policy with a genetic algorithm supplying off-policy targets.

    Args:
        env: The construction graph.
        policy: The policy to train, modified in place.
        landscape: What terminal sequences are scored against. In a campaign
            this is the surrogate proxy, never the assay.
        reward: Transforms objective values into log rewards.
        config: Training settings.
        genetic: The genetic algorithm used as the teacher. Its own
            hyperparameters are its own; this trainer only drives it.
        genetic_config: How much guidance to apply.
        objective: How balance violation is measured. Defaults to trajectory
            balance.
        tracker: Where to report.

    Returns:
        The losses, the learned ``log Z``, and the evaluations consumed.

    Raises:
        ValueError: If the objective needs a per-step record. Replayed
            trajectories carry none, so such an objective would fail partway
            through a run rather than at the start.
    """
    settings = genetic_config or GeneticConfig()
    loss_fn = objective or TrajectoryBalance()
    if isinstance(loss_fn, FlowObjective):
        raise ValueError(
            f"{type(loss_fn).__name__} needs a per-step record, which replayed "
            f"trajectories do not carry; use trajectory or contrastive balance "
            f"with genetic guidance"
        )

    recorder = tracker or NoOpTracker()
    optimiser = torch.optim.Adam(
        parameter_groups(
            policy,
            learning_rate=config.learning_rate,
            log_z_multiplier=config.log_z_multiplier,
        )
    )
    generator = torch.Generator().manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    buffer = RankedBuffer(settings.buffer_size)
    result = TrainingResult()

    for step in range(config.steps):
        fraction = step / max(config.steps - 1, 1)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)

        n_genetic = 0 if step < settings.warmup else int(config.batch_size * settings.mix)
        n_policy = config.batch_size - n_genetic

        batches = []
        if n_policy:
            on_policy = sample_trajectories(
                env, policy, n_policy, epsilon=epsilon, generator=generator
            )
            values = landscape.evaluate(on_policy.terminal)
            result.oracle_calls += int(on_policy.terminal.shape[0])
            log_rewards = reward.log_reward(values)
            buffer.add(on_policy.terminal, log_rewards)
            batches.append((on_policy, torch.as_tensor(log_rewards, dtype=torch.float32)))

        if n_genetic and len(buffer):
            offspring = _breed(env, genetic, buffer, settings, rng)
            if offspring.shape[0]:
                bred_values = landscape.evaluate(offspring)
                result.oracle_calls += int(offspring.shape[0])
                bred_rewards = np.asarray(reward.log_reward(bred_values)).reshape(-1)
                buffer.add(offspring, bred_rewards)
                # The GA breeds a pool; its best fill the genetic share of the
                # training batch, so `mix` is the ratio the caller actually gets.
                finite = np.flatnonzero(np.isfinite(bred_rewards))
                chosen = finite[np.argsort(-bred_rewards[finite], kind="stable")][:n_genetic]
                if chosen.size:
                    replayed = replay_trajectories(
                        env, policy, offspring[chosen], generator=generator
                    )
                    batches.append(
                        (
                            replayed,
                            torch.as_tensor(bred_rewards[chosen], dtype=torch.float32),
                        )
                    )

        loss = torch.stack([loss_fn.loss(t, r, policy) for t, r in batches]).mean()
        optimiser.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()

        loss_value = float(loss.detach().item())
        result.losses.append(loss_value)
        if step % config.log_every == 0 or step == config.steps - 1:
            recorder.log_metrics(
                {
                    "loss": loss_value,
                    "buffer": float(len(buffer)),
                    "best_log_reward": float(buffer.log_rewards[0]) if len(buffer) else 0.0,
                    "epsilon": epsilon,
                },
                step=step,
            )

    if loss_fn.uses_log_z:
        result.final_log_z = float(policy.log_z.detach().item())
    return result


def _breed(
    env: SequenceEnvironment,
    genetic: GeneticAlgorithm,
    buffer: RankedBuffer,
    settings: GeneticConfig,
    rng: np.random.Generator,
) -> Tokens:
    """Run the genetic algorithm against the buffer and return its offspring.

    The GA is seeded from the buffer each step rather than carrying its own
    population, so it always recombines what the policy has actually found
    rather than drifting away on its own trajectory.
    """
    parents, parent_rewards = buffer.sample(settings.offspring, rng)
    genetic.observe(parents, parent_rewards[:, None])
    offspring = parents
    for _ in range(settings.generations):
        offspring = genetic.propose(settings.offspring)
    # A GA under a mutation budget can produce designs outside this environment;
    # scoring those would be meaningless rather than merely inaccurate.
    return np.asarray(offspring)[env.is_reachable(offspring)]
