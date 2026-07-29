"""Turning objective values into the reward a GFlowNet is trained against.

A landscape reports whatever its assay reports: possibly negative, possibly
unbounded, and ``-inf`` where a design cannot be built. A GFlowNet needs a
strictly positive reward, and needs it in log space, because trajectory balance
is a difference of logs and every alternative loses precision.

Keeping this a separate layer rather than folding it into the landscape means
the landscape stays a faithful transcription of its source -- checkable against
the paper it came from -- while the parts that are modelling choices, the
exponent and the floor, live where they can be swept.

The exponent
------------

``R(x) = f(x)^β``. Raising the reward to a power sharpens the target
distribution ``p*(x) ∝ R(x)^β`` without changing which designs are best, and is
the standard knob for trading diversity against quality: Jain et al. use
``β = 3`` throughout. ``β = 0`` gives a uniform target, ``β → ∞`` a greedy one.

The floor
---------

A design with zero fitness has ``log R = -∞``, which propagates through the loss
as ``nan`` the moment it meets a subtraction. Since roughly a fifth of measured
GB1 variants are exactly dead, this is the common case rather than an edge case.
A small positive floor keeps the loss finite. It is a real modelling choice --
it puts a little probability mass on designs that deserve none -- so it is
explicit and configurable rather than hidden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness


class Reward(ABC):
    """Maps objective values to log rewards."""

    @abstractmethod
    def log_reward(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Compute ``log R(x)`` for a batch of designs.

        Args:
            values: An ``(n, n_objectives)`` array of objective values.

        Returns:
            An ``(n,)`` array of log rewards, finite everywhere.
        """

    def reward(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Compute ``R(x)`` directly.

        Provided for metrics and inspection. Training should use
        :meth:`log_reward`, which is what the loss is expressed in.

        Args:
            values: An ``(n, n_objectives)`` array of objective values.

        Returns:
            An ``(n,)`` array of rewards.
        """
        return np.exp(self.log_reward(values))


class TemperedReward(Reward):
    """``R(x) = max(f(x), floor)^β`` for a single objective.

    Args:
        beta: Reward exponent. Higher values concentrate the target distribution
            on the best designs. ``0`` makes the target uniform.
        floor: Reward assigned to designs whose objective is non-positive or
            infeasible. Must be positive, since ``log 0`` is not finite.

    Raises:
        ValueError: If ``beta`` is negative or ``floor`` is not positive.
    """

    def __init__(self, *, beta: float = 1.0, floor: float = 1e-8) -> None:
        """Store the exponent and floor."""
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if floor <= 0:
            raise ValueError(
                f"floor must be positive, got {floor}; a zero floor makes log R "
                f"negatively infinite and the loss nan"
            )
        self._beta = beta
        self._floor = floor

    @property
    def beta(self) -> float:
        """The reward exponent."""
        return self._beta

    @property
    def floor(self) -> float:
        """The reward given to dead or infeasible designs."""
        return self._floor

    def log_reward(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Compute ``β · log(max(f(x), floor))``.

        Args:
            values: An ``(n, 1)`` or ``(n,)`` array of objective values.

        Returns:
            An ``(n,)`` array of finite log rewards.

        Raises:
            ValueError: If the input carries more than one objective, which
                needs a scalarisation this class does not provide.
        """
        flat = _single_objective(values)
        # nan is not a small value, it is a missing one; treating it as the
        # floor would quietly present an unmeasured design as a dead one, so it
        # is rejected instead.
        if np.isnan(flat).any():
            raise ValueError(
                "objective values contain nan; a missing measurement is not a low "
                "reward, so decide explicitly what it should count as"
            )
        clamped = np.maximum(flat, self._floor)
        return np.asarray(self._beta * np.log(clamped), dtype=np.float64)


def _single_objective(values: Fitness) -> npt.NDArray[np.float64]:
    """Accept ``(n,)`` or ``(n, 1)`` and return ``(n,)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:  # noqa: PLR2004 - a single objective
        return array[:, 0]
    raise ValueError(
        f"expected shape (n,) or (n, 1), got {array.shape}; multi-objective values "
        f"need a scalarisation before a scalar reward applies"
    )
