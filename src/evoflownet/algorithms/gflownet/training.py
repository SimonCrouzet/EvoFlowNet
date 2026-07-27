"""The GFlowNet training loop.

Deliberately small. Everything it needs -- the graph, the reward, the policy, the
objective -- is passed in, so swapping any of them is a configuration change
rather than a code change. What lives here is only the bookkeeping that would
otherwise be duplicated at every call site: exploration annealing, the optimiser
groups, and the reporting.

Exploration is annealed rather than fixed. Early on, a policy that follows its
own weak preferences visits a narrow slice of the space and the balance
condition is fitted to that slice. Late on, uniform noise is just variance. A
linear decay to zero is the simplest schedule that does both jobs and has no
tuning of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from evoflownet.algorithms.gflownet.sampling import sample_trajectories
from evoflownet.algorithms.gflownet.trajectory_balance import (
    parameter_groups,
    trajectory_balance_loss,
)
from evoflownet.tracking.base import NoOpTracker

if TYPE_CHECKING:
    from evoflownet.env.base import SequenceEnvironment
    from evoflownet.landscapes.base import FitnessLandscape
    from evoflownet.models.policy import SequencePolicy
    from evoflownet.rewards.base import Reward
    from evoflownet.tracking.base import Tracker


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """How to run trajectory balance training.

    Attributes:
        steps: Number of optimisation steps.
        batch_size: Trajectories sampled per step.
        learning_rate: Rate for the policy parameters.
        log_z_multiplier: How much faster ``log Z`` learns. It is one scalar
            that must travel to ``log sum R(x)``; at the policy's rate it
            becomes the bottleneck.
        epsilon_start: Initial probability of choosing uniformly at random.
        epsilon_end: Final exploration probability.
        log_every: Report metrics every this many steps.
        seed: Seeds the policy's sampling. The landscape and policy carry their
            own.
    """

    steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 3e-3
    log_z_multiplier: float = 10.0
    epsilon_start: float = 0.3
    epsilon_end: float = 0.0
    log_every: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject configurations that cannot describe a run."""
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must both be at least 1")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                f"need 0 <= epsilon_end <= epsilon_start <= 1, got "
                f"{self.epsilon_end} and {self.epsilon_start}"
            )


@dataclass(slots=True)
class TrainingResult:
    """What a completed run produced.

    Attributes:
        losses: Loss at every step.
        final_log_z: The learned ``log Z``.
        oracle_calls: How many times the landscape was evaluated. Reported
            because sample efficiency is the axis on which GFlowNets are
            usually challenged, and a result without its budget cannot be
            compared to one.
    """

    losses: list[float] = field(default_factory=list)
    final_log_z: float = 0.0
    oracle_calls: int = 0


def train_trajectory_balance(  # noqa: PLR0913 - the run is defined by its parts
    env: SequenceEnvironment,
    policy: SequencePolicy,
    landscape: FitnessLandscape,
    reward: Reward,
    config: TrainingConfig,
    *,
    tracker: Tracker | None = None,
) -> TrainingResult:
    """Train a policy to sample proportionally to reward.

    Args:
        env: The construction graph.
        policy: The policy to train, modified in place.
        landscape: What terminal sequences are scored against.
        reward: Transforms objective values into log rewards.
        config: Run settings.
        tracker: Where to report. Defaults to discarding.

    Returns:
        The losses, the learned ``log Z``, and the oracle budget consumed.
    """
    recorder = tracker if tracker is not None else NoOpTracker()
    optimiser = torch.optim.Adam(
        parameter_groups(
            policy,
            learning_rate=config.learning_rate,
            log_z_multiplier=config.log_z_multiplier,
        )
    )
    generator = torch.Generator().manual_seed(config.seed)
    result = TrainingResult()

    for step in range(config.steps):
        fraction = step / max(config.steps - 1, 1)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)

        trajectories = sample_trajectories(
            env, policy, config.batch_size, epsilon=epsilon, generator=generator
        )
        values = landscape.evaluate(trajectories.terminal)
        result.oracle_calls += int(trajectories.terminal.shape[0])
        log_rewards = torch.as_tensor(reward.log_reward(values), dtype=torch.float32)

        loss = trajectory_balance_loss(trajectories, log_rewards, policy.log_z)
        optimiser.zero_grad()
        # torch ships stubs but leaves Tensor.backward unannotated.
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()

        loss_value = float(loss.detach().item())
        result.losses.append(loss_value)
        if step % config.log_every == 0 or step == config.steps - 1:
            recorder.log_metrics(
                {
                    "loss": loss_value,
                    "log_z": float(policy.log_z.detach().item()),
                    "epsilon": epsilon,
                    "mean_reward": float(np.exp(log_rewards.numpy()).mean()),
                    "oracle_calls": float(result.oracle_calls),
                },
                step=step,
            )

    result.final_log_z = float(policy.log_z.detach().item())
    return result
