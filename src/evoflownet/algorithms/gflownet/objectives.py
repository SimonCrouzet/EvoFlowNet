r"""Training objectives, behind one interface so they can be swapped.

Every objective here enforces the same thing -- that the forward policy samples
terminal states in proportion to their reward -- and differs only in how it
measures the violation. Keeping them behind :class:`GFlowNetObjective` means
choosing between them is a configuration change, and means a training strategy
(replay, genetic guidance, local search) composes with any of them rather than
being written against one.

Trajectory balance versus contrastive balance
---------------------------------------------

:class:`TrajectoryBalance` measures each trajectory against a learned scalar
``log Z``:

.. math:: \left(\log Z + \log P_F(\tau) - \log R(x) - \log P_B(\tau|x)\right)^2

:class:`ContrastiveBalance` measures *pairs* of trajectories against each other.
Writing :math:`v(\tau) = \log P_F(\tau) - \log R(x) - \log P_B(\tau|x)`, the
condition is that ``v`` is the same constant (namely :math:`-\log Z`) for every
trajectory, so any two must agree:

.. math:: \left(v(\tau_1) - v(\tau_2)\right)^2

``Z`` cancels. This is Deleu et al.'s contrastive balance condition (UAI 2024)
and is the VarGrad estimator noted by Malkin et al.; what is recent is the
empirical case for preferring it, from Stable-GFN (ICML 2026) and GFlowRL, both
reporting the learned ``log Z`` to be the dominant source of instability.

The reason to care here specifically: trajectory length varies from 1 to
``max_mutations + 1``, and the backward term contributes :math:`-\log k!`, which
spans tens of nats across that range. A single global scalar fitted by gradient
descent against that spread is the pathology those papers describe. Whether it
actually bites at our scale is an empirical question -- which is the point of
having both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from evoflownet.algorithms.gflownet.sampling import Trajectories
    from evoflownet.models.policy import SequencePolicy

#: Default multiplier on the ``log Z`` learning rate, following Malkin et al.
LOG_Z_LR_MULTIPLIER = 10.0

#: Default threshold for noisy gradient pruning, from Stable-GFN.
DEFAULT_PRUNE_THRESHOLD = 0.1

#: A contrastive pair needs two trajectories.
_PAIR = 2


class GFlowNetObjective(ABC):
    """Measures how far a batch of trajectories is from balance."""

    @abstractmethod
    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Compute the objective for a batch.

        Args:
            trajectories: Completed trajectories with summed log probabilities.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``, finite.
            policy: The policy being trained. Objectives that need a learned
                partition function read it from here.

        Returns:
            A scalar loss.
        """

    @property
    def uses_log_z(self) -> bool:
        """Whether this objective trains a learned partition function."""
        return True

    @property
    def needs_state_rewards(self) -> bool:
        """Whether the trainer must score every visited state, not just terminals.

        Only the forward-looking parameterisation needs this. It is asked
        rather than assumed so that scoring intermediate states -- which costs
        proxy evaluations -- happens only when an objective will use them.
        """
        return False

    def __repr__(self) -> str:
        """Name the objective."""
        return f"{type(self).__name__}()"


def balance_violation(trajectories: Trajectories, log_rewards: torch.Tensor) -> torch.Tensor:
    """The per-trajectory quantity every objective is built from.

    Returns ``log P_F(tau) - log R(x) - log P_B(tau|x)``. At balance this equals
    ``-log Z`` for every trajectory, which is what trajectory balance compares
    against a learned scalar and what contrastive balance compares between
    pairs.

    Args:
        trajectories: Completed trajectories.
        log_rewards: An ``(n,)`` tensor of ``log R(x)``.

    Returns:
        An ``(n,)`` tensor.

    Raises:
        ValueError: If the batch sizes disagree, or a log reward is not finite.
            ``-inf`` here becomes ``nan`` at the subtraction and destroys the
            batch silently; flooring dead designs belongs in the reward.
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
    return trajectories.log_forward - log_rewards - trajectories.log_backward


class TrajectoryBalance(GFlowNetObjective):
    """Malkin et al. (NeurIPS 2022), measured against a learned ``log Z``."""

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Mean squared deviation of the violation from ``-log Z``.

        Args:
            trajectories: Completed trajectories.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``.
            policy: Supplies the learned ``log Z``.

        Returns:
            A scalar loss.
        """
        violation = policy.log_z + balance_violation(trajectories, log_rewards)
        return violation.pow(2).mean()


class ContrastiveBalance(GFlowNetObjective):
    """Deleu et al. (UAI 2024), comparing pairs so that ``Z`` cancels.

    Trajectories are paired by splitting the batch in half, so a batch of ``n``
    yields ``n // 2`` pairs. Pairing consecutive halves rather than adjacent
    items keeps each pair independent when the batch was drawn in one go.

    Args:
        prune_threshold: Drop pairs whose log rewards differ by less than this,
            following Stable-GFN's noisy gradient pruning. Two designs of
            near-identical reward carry almost no signal about which the policy
            should prefer, and when the reward is a noisy fitness oracle the
            difference between them is mostly measurement error. ``0`` disables
            pruning.

    Raises:
        ValueError: If ``prune_threshold`` is negative.
    """

    def __init__(self, *, prune_threshold: float = 0.0) -> None:
        """Configure pruning."""
        if prune_threshold < 0:
            raise ValueError(f"prune_threshold must be non-negative, got {prune_threshold}")
        self._prune_threshold = prune_threshold

    @property
    def prune_threshold(self) -> float:
        """Minimum log-reward gap for a pair to contribute."""
        return self._prune_threshold

    @property
    def uses_log_z(self) -> bool:
        """``False`` -- the partition function cancels and is never learned."""
        return False

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,  # noqa: ARG002 - unused: Z cancels, by design
    ) -> torch.Tensor:
        """Mean squared disagreement between paired trajectories.

        Args:
            trajectories: Completed trajectories. At least two are needed.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``.
            policy: Unused. Accepted so objectives share one signature.

        Returns:
            A scalar loss. Zero if every pair was pruned, which is a real
            answer -- the batch carried no usable signal.

        Raises:
            ValueError: If fewer than two trajectories are supplied.
        """
        violation = balance_violation(trajectories, log_rewards)
        n = violation.shape[0]
        if n < _PAIR:
            raise ValueError(
                f"contrastive balance compares pairs and needs at least {_PAIR} "
                f"trajectories, got {n}"
            )

        half = n // 2
        first, second = violation[:half], violation[half : 2 * half]
        differences = first - second

        if self._prune_threshold > 0:
            reward_gap = (log_rewards[:half] - log_rewards[half : 2 * half]).abs()
            keep = reward_gap >= self._prune_threshold
            if not keep.any():
                # Nothing to learn from. Returning a real zero connected to the
                # graph keeps the optimiser step well defined.
                return differences.sum() * 0.0
            differences = differences[keep]

        return differences.pow(2).mean()


def parameter_groups(
    policy: SequencePolicy,
    *,
    learning_rate: float = 1e-3,
    log_z_multiplier: float = LOG_Z_LR_MULTIPLIER,
) -> list[dict[str, Any]]:
    """Optimiser parameter groups giving ``log Z`` its own learning rate.

    ``log Z`` is one scalar that must travel a long way while the policy is many
    parameters that should move carefully. Training both at one rate makes
    ``log Z`` the bottleneck and the loss plateaus with the policy still wrong --
    a failure that reads as slow convergence rather than a misconfiguration.

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
