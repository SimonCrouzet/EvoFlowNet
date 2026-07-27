"""Trajectory balance under the name the paper gives it.

The objective now lives in :mod:`evoflownet.algorithms.gflownet.objectives`
alongside the alternatives, behind a shared interface. This module keeps
``trajectory_balance_loss`` because it is the form Malkin et al. state, and the
form a reader checking the code against the paper will look for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evoflownet.algorithms.gflownet.objectives import (
    LOG_Z_LR_MULTIPLIER,
    balance_violation,
    parameter_groups,
)

if TYPE_CHECKING:
    import torch

    from evoflownet.algorithms.gflownet.sampling import Trajectories

__all__ = ["LOG_Z_LR_MULTIPLIER", "parameter_groups", "trajectory_balance_loss"]


def trajectory_balance_loss(
    trajectories: Trajectories,
    log_rewards: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
    """Mean squared trajectory-balance violation over a batch.

    Args:
        trajectories: Completed trajectories carrying summed forward and
            backward log probabilities.
        log_rewards: An ``(n,)`` tensor of ``log R(x)``. Must be finite.
        log_z: The scalar ``log Z``.

    Returns:
        A scalar loss.

    Raises:
        ValueError: If the batch sizes disagree, or a log reward is not finite.
    """
    violation = log_z + balance_violation(trajectories, log_rewards)
    return violation.pow(2).mean()
