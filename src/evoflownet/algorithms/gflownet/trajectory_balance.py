r"""The trajectory balance objective.

Malkin et al. (NeurIPS 2022):

.. math::

    \mathcal{L}(\tau) = \left(
        \log \frac{Z_\theta \prod_{t} P_F(s_t \mid s_{t-1})}
                  {R(x) \prod_{t} P_B(s_{t-1} \mid s_t)}
    \right)^2

At a global minimum the forward policy samples terminal states in proportion to
their reward. The loss is a squared *log-ratio*, which is why every quantity
feeding it is carried in log space: the ratio spans many orders of magnitude on
any interesting landscape, and computing it directly loses the difference before
the square ever happens.

Two practical points.

``log Z`` is a single scalar shared by every trajectory, and it has to travel
from its initialisation to :math:`\log \sum_x R(x)` -- which on GB1 at
:math:`\beta = 3` is a distance of tens of nats. Malkin et al. give it a learning
rate roughly an order of magnitude above the policy's, and
:func:`parameter_groups` builds that.

The loss is *not* an estimate of anything that should reach zero on a finite
sample. It reaches zero only when the balance condition holds on the trajectories
seen, so a small loss is necessary but nowhere near sufficient for correctness --
which is why the real check is the distributional comparison in
:mod:`evoflownet.metrics.distribution`, not this number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from evoflownet.algorithms.gflownet.sampling import Trajectories
    from evoflownet.models.policy import SequencePolicy

#: Default multiplier on the ``log Z`` learning rate, following Malkin et al.
LOG_Z_LR_MULTIPLIER = 10.0


def trajectory_balance_loss(
    trajectories: Trajectories,
    log_rewards: torch.Tensor,
    log_z: torch.Tensor,
) -> torch.Tensor:
    """Mean squared trajectory-balance violation over a batch.

    Args:
        trajectories: Completed trajectories carrying summed forward and
            backward log probabilities.
        log_rewards: An ``(n,)`` tensor of ``log R(x)`` for the terminal states.
            Must be finite; the reward transform is responsible for flooring
            dead designs, since ``-inf`` here becomes ``nan`` at the subtraction.
        log_z: The scalar ``log Z`` parameter.

    Returns:
        A scalar loss.

    Raises:
        ValueError: If the batch sizes disagree, or a log reward is not finite.
    """
    if log_rewards.shape != trajectories.log_forward.shape:
        raise ValueError(
            f"got {tuple(log_rewards.shape)} log rewards for "
            f"{tuple(trajectories.log_forward.shape)} trajectories"
        )
    if not torch.isfinite(log_rewards).all():
        raise ValueError(
            "log rewards must be finite; floor dead designs in the reward transform "
            "rather than letting -inf reach the loss, where it becomes nan"
        )

    violation = log_z + trajectories.log_forward - log_rewards - trajectories.log_backward
    return violation.pow(2).mean()


def parameter_groups(
    policy: SequencePolicy,
    *,
    learning_rate: float = 1e-3,
    log_z_multiplier: float = LOG_Z_LR_MULTIPLIER,
) -> list[dict[str, Any]]:
    """Optimiser parameter groups giving ``log Z`` its own learning rate.

    ``log Z`` is one scalar that must travel a long way, while the policy is many
    parameters that should move carefully. Training both at one rate makes
    ``log Z`` the bottleneck and the loss plateaus with the policy still wrong --
    a failure that looks like slow convergence rather than a misconfiguration.

    Args:
        policy: The policy whose parameters to group.
        learning_rate: Rate for the policy parameters.
        log_z_multiplier: How much faster ``log Z`` should learn.

    Returns:
        Groups suitable for a torch optimiser.

    Raises:
        ValueError: If either rate is not positive.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if log_z_multiplier <= 0:
        raise ValueError(f"log_z_multiplier must be positive, got {log_z_multiplier}")
    return [
        {"params": policy.policy_parameters(), "lr": learning_rate},
        {"params": [policy.log_z], "lr": learning_rate * log_z_multiplier},
    ]
