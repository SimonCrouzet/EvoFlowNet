"""Collapsing an objective vector into the single number a GFlowNet trains on.

A GFlowNet needs one scalar reward per design. A real campaign has several
objectives -- binding, stability, expression, immunogenicity -- and no scalar
ordering of them is given by the biology. Something has to combine them, and the
only question is *where*.

Why the combination happens here, and not earlier
-------------------------------------------------

It would be simpler to have each landscape return an already-combined number.
That is the choice this package deliberately does not make: :meth:`evaluate`
returns ``(n, n_objectives)`` everywhere, including for single-objective
landscapes, and the vector survives until this module.

The reason is that a scalarised landscape has thrown the vector away, and three
separate layers need it:

* The **reward** layer, for MOGFN-PC (Jain et al., 2023). That method trains one
  model on ``p(x|ω) ∝ R(x|ω)^β`` with the preference ``ω`` resampled every batch,
  so a single trained model covers the whole Pareto front instead of one model
  per trade-off. It can only do that if the objective vector is still present at
  reward time, where ``ω`` is applied.
* The **metrics** layer, where hypervolume, R2 and IGD+ are defined on objective
  *vectors*. There is no way to recover them from a scalarised value.
* The **acquisition** layer, which ranks candidates and may want to rank them by
  a different trade-off than the one used for training.

So scalarising early would not save work, it would move it: retrofitting
multi-objective support later would mean rewriting the reward, metric and
acquisition layers in one change. Scalarising late costs one extra dimension on
an array and keeps every option open.

What this buys the training loop
--------------------------------

:class:`ScalarizedReward` is a :class:`~evogfn.rewards.base.Reward` like any
other. The sampler, the objective and the training loop see ``log_reward`` and
nothing else, so multi-objective training needs no change anywhere in them.

The three scalarisations
------------------------

All three are the ones compared in Jain et al. (2023), Multi-Objective GFlowNets
(ICML), Section 4:

* :class:`WeightedSum` -- ``R(x|ω) = Σ_i ω_i R_i(x)``. Cannot reach points on a
  non-convex part of the Pareto front, whatever ``ω`` is (Miettinen, 1999,
  Thm 3.1.4); this is a property of the scalarisation, not of the sampler.
* :class:`Tchebycheff` -- ``R(x|ω) = min_i ω_i |R_i(x) - z_i|``. Can reach every
  Pareto-optimal point, including on non-convex fronts, which is the reason to
  prefer it despite being non-smooth.
* :class:`WeightedLogSum` -- ``R(x|ω) = Π_i R_i(x)^{ω_i}``. A weighted geometric
  mean: multiplicative rather than additive, so one objective at zero cannot be
  compensated by another being large.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from evogfn.rewards.base import Reward, TemperedReward

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness

#: Tolerance on ``Σ_i ω_i = 1``. Preferences are drawn from a Dirichlet or typed
#: into a config file, so exact equality would reject vectors that are right to
#: every digit anyone wrote down.
PREFERENCE_SUM_TOLERANCE = 1e-6

# An objective matrix is two-dimensional by definition; named so the shape checks
# below do not read as magic numbers.
_MATRIX_NDIM = 2


class Scalarization(ABC):
    """Combines an objective vector and a preference into one value per design.

    Subclasses implement :meth:`_combine`. The public :meth:`scalarize` validates
    the objective matrix and the preference first, so no subclass repeats those
    checks and none can forget them.
    """

    @abstractmethod
    def _combine(
        self,
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Combine validated objectives and preferences.

        Args:
            values: An ``(n, n_objectives)`` array of finite-or-infinite
                objective values, already checked for shape and for ``nan``.
            weights: An ``(n, n_objectives)`` array of preferences, one row per
                design, each row non-negative and summing to one.

        Returns:
            An ``(n,)`` array of scalarised values.
        """

    def scalarize(self, values: Fitness, preference: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Reduce objective vectors to one value per design.

        Args:
            values: An ``(n, n_objectives)`` array of objective values, as
                returned by :meth:`~evogfn.landscapes.base.FitnessLandscape.evaluate`.
            preference: An ``(n_objectives,)`` preference vector applied to every
                design, or an ``(n, n_objectives)`` array giving each design its
                own -- which is what MOGFN-PC does, one preference per sampled
                trajectory.

        Returns:
            An ``(n,)`` array of scalarised values.

        Raises:
            ValueError: If ``values`` is not a two-dimensional objective matrix,
                contains ``nan``, or if the preference is negative, does not sum
                to one, or has the wrong width.
        """
        objectives = _as_objective_matrix(values)
        weights = _as_preference_matrix(preference, objectives.shape)
        return self._combine(objectives, weights)


class WeightedSum(Scalarization):
    """``R(x|ω) = Σ_i ω_i R_i(x)``.

    The linear scalarisation, and the one to reach for first because it is the
    only one with no free parameter beyond ``ω``. Its limitation is structural
    rather than numerical: sweeping ``ω`` over the whole simplex recovers only
    the convex hull of the Pareto front (Miettinen, 1999, Thm 3.1.4), so points
    in a concave dent are unreachable at every preference. If a front is expected
    to be concave, use :class:`Tchebycheff` instead.
    """

    def _combine(
        self,
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Sum the weighted objectives."""
        # ``where`` rather than a plain product: an infeasible design scores
        # ``-inf``, and ``0 * -inf`` is ``nan``, which would turn a zero
        # preference on an infeasible objective into a missing value.
        weighted = np.zeros_like(values)
        np.multiply(values, weights, out=weighted, where=weights > 0.0)
        return np.asarray(weighted.sum(axis=1), dtype=np.float64)


class Tchebycheff(Scalarization):
    """``R(x|ω) = min_i ω_i |R_i(x) - z_i|``.

    The weighted Chebyshev scalarisation, in the maximisation-facing form used by
    Jain et al. (2023). ``z`` is a reference point placed *below* the achievable
    set, so ``|R_i(x) - z_i|`` is the improvement of design ``x`` on objective
    ``i`` over the reference, and the scalarised value is the worst weighted
    improvement across objectives. Maximising it means improving the objective
    that is currently doing worst, which is what makes the whole Pareto front --
    concave parts included -- reachable by sweeping ``ω`` (Miettinen, 1999,
    Thm 3.4.5).

    The absolute value is as printed in the source. It only behaves as intended
    while ``z`` lies below every objective value: a reference above some values
    makes the transform fold back on itself and stop being monotone in those
    objectives. The default of zeros is below the achievable set for the
    non-negative fitness scales used here.

    Args:
        reference: The ``(n_objectives,)`` reference point ``z``. Defaults to
            zeros, which is correct when objectives are non-negative.

    Raises:
        ValueError: If ``reference`` is not one-dimensional or is not finite.
    """

    def __init__(self, reference: npt.ArrayLike | None = None) -> None:
        """Store the reference point."""
        if reference is None:
            self._reference: npt.NDArray[np.float64] | None = None
            return
        point = np.asarray(reference, dtype=np.float64)
        if point.ndim != 1:
            raise ValueError(f"reference must be a single point with ndim 1, got ndim {point.ndim}")
        if not np.isfinite(point).all():
            raise ValueError(
                "reference must be finite; an infinite reference makes every "
                "weighted improvement infinite and the scalarisation constant"
            )
        self._reference = point

    @property
    def reference(self) -> npt.NDArray[np.float64] | None:
        """The reference point ``z``, or ``None`` when it defaults to zeros."""
        return self._reference

    def _combine(
        self,
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Take the smallest weighted deviation from the reference.

        Raises:
            ValueError: If the reference does not have one entry per objective.
        """
        n_objectives = values.shape[1]
        if self._reference is None:
            reference = np.zeros(n_objectives, dtype=np.float64)
        elif self._reference.shape[0] != n_objectives:
            raise ValueError(
                f"reference has {self._reference.shape[0]} entries but the values carry "
                f"{n_objectives} objectives"
            )
        else:
            reference = self._reference

        deviations = np.abs(values - reference)
        # Objectives with zero preference are dropped from the minimum rather
        # than contributing a zero. A weight of zero declares an objective
        # irrelevant; letting it cap the scalarised value at zero would make
        # every design equally worthless and destroy the ordering entirely.
        active = weights > 0.0
        weighted = np.full_like(values, np.inf)
        np.multiply(weights, deviations, out=weighted, where=active)
        # Preferences sum to one, so at least one entry per row is active and the
        # minimum is never the placeholder infinity.
        return np.asarray(weighted.min(axis=1), dtype=np.float64)


class WeightedLogSum(Scalarization):
    """``R(x|ω) = Π_i R_i(x)^{ω_i}``, a weighted geometric mean.

    Computed as ``exp(Σ_i ω_i log R_i(x))``, which is where the name comes from
    and how it stays in range when objectives differ by orders of magnitude.

    Being multiplicative, it cannot trade one objective away entirely: a design
    that is dead on any objective with non-zero weight scores at the floor no
    matter how good the rest are. That is usually the honest model of a protein
    -- an unfoldable variant with excellent predicted binding is not a partial
    success -- and it is the substantive difference from :class:`WeightedSum`.

    Args:
        floor: Value substituted for non-positive or infeasible objectives before
            taking the log. Must be positive, since ``log 0`` is not finite and
            would make the whole product ``0`` with no way to rank the designs
            that reached it.

    Raises:
        ValueError: If ``floor`` is not positive.
    """

    def __init__(self, *, floor: float = 1e-8) -> None:
        """Store the floor applied before the logarithm."""
        if floor <= 0:
            raise ValueError(
                f"floor must be positive, got {floor}; a zero floor makes log R_i "
                f"negatively infinite for every dead design"
            )
        self._floor = floor

    @property
    def floor(self) -> float:
        """The value substituted for dead or infeasible objectives."""
        return self._floor

    def _combine(
        self,
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Exponentiate the weighted sum of log objectives."""
        # Clamping first also removes the -inf of an infeasible design, so no
        # `0 * -inf` can arise from a zero preference.
        clamped = np.maximum(values, self._floor)
        log_values = np.log(clamped)
        return np.asarray(np.exp((weights * log_values).sum(axis=1)), dtype=np.float64)


class ScalarizedReward(Reward):
    """A multi-objective reward that behaves exactly like a single-objective one.

    Composes a :class:`Scalarization` with an ordinary scalar reward, so the
    sampler, the loss and the training loop see only ``log_reward`` and need no
    knowledge that there was ever more than one objective.

    The preference is held on the instance rather than passed to
    :meth:`log_reward`, because :class:`~evogfn.rewards.base.Reward` takes
    objective values and nothing else. For MOGFN-PC, where the preference is
    resampled every batch, build a new reward per batch with
    :meth:`with_preference` -- it is a cheap object, and keeping the preference
    immutable means a reward and the log rewards it produced can never disagree.

    Args:
        scalarization: How to combine the objectives.
        preference: The ``(n_objectives,)`` preference vector ``ω``. Must be
            non-negative and sum to one.
        reward: The scalar reward applied to the combined value. Defaults to
            :class:`~evogfn.rewards.base.TemperedReward`, which supplies the
            exponent ``β`` and the floor that keeps ``log R`` finite.

    Raises:
        ValueError: If the preference is not a one-dimensional, non-negative
            vector summing to one.
    """

    def __init__(
        self,
        scalarization: Scalarization,
        preference: npt.ArrayLike,
        *,
        reward: Reward | None = None,
    ) -> None:
        """Store the scalarisation, the preference and the scalar reward."""
        self._scalarization = scalarization
        self._preference = _as_preference_vector(preference)
        self._reward = reward if reward is not None else TemperedReward()

    @property
    def scalarization(self) -> Scalarization:
        """How the objectives are combined."""
        return self._scalarization

    @property
    def preference(self) -> npt.NDArray[np.float64]:
        """The preference vector ``ω``, as a copy so it cannot be mutated."""
        return self._preference.copy()

    @property
    def scalar_reward(self) -> Reward:
        """The scalar reward applied after scalarisation.

        Named ``scalar_reward`` rather than ``reward`` because
        :meth:`~evogfn.rewards.base.Reward.reward` is already the method that
        computes ``R(x)``; a property of that name would shadow it and turn every
        call into an attribute access on a reward object.
        """
        return self._reward

    def with_preference(self, preference: npt.ArrayLike) -> ScalarizedReward:
        """Return the same reward at a different point on the simplex.

        Args:
            preference: The new ``(n_objectives,)`` preference vector.

        Returns:
            A new :class:`ScalarizedReward` sharing this one's scalarisation and
            scalar reward.

        Raises:
            ValueError: If the preference is not a one-dimensional, non-negative
                vector summing to one.
        """
        return ScalarizedReward(self._scalarization, preference, reward=self._reward)

    def log_reward(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Compute ``log R(x|ω)`` for a batch of designs.

        Args:
            values: An ``(n, n_objectives)`` array of objective values.

        Returns:
            An ``(n,)`` array of log rewards, finite everywhere.

        Raises:
            ValueError: If ``values`` is not a two-dimensional objective matrix,
                contains ``nan``, or has a width the preference does not match.
        """
        scalarized = self._scalarization.scalarize(values, self._preference)
        return self._reward.log_reward(scalarized)


def _as_objective_matrix(values: Fitness) -> npt.NDArray[np.float64]:
    """Validate an ``(n, n_objectives)`` objective matrix.

    Raises:
        ValueError: If the input is not two-dimensional, carries no objectives,
            or contains ``nan``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != _MATRIX_NDIM:
        raise ValueError(
            f"expected shape (n, n_objectives), got {array.shape}; a one-dimensional "
            f"array is ambiguous between n designs and n objectives, so landscapes "
            f"always return the matrix form"
        )
    if array.shape[1] == 0:
        raise ValueError("expected at least one objective, got a matrix of width 0")
    # nan is not a low value, it is a missing one; scalarising it would present
    # an unmeasured objective as a measured bad one.
    if np.isnan(array).any():
        raise ValueError(
            "objective values contain nan; a missing measurement is not a low value, "
            "so decide explicitly what it should count as"
        )
    return array


def _as_preference_vector(preference: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validate a single preference vector on the simplex.

    Raises:
        ValueError: If the input is not one-dimensional, is empty, contains a
            negative or non-finite entry, or does not sum to one.
    """
    weights = np.asarray(preference, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"expected a preference vector with ndim 1, got ndim {weights.ndim}")
    if weights.size == 0:
        raise ValueError("preference must cover at least one objective")
    _check_simplex(weights.reshape(1, -1))
    return weights


def _as_preference_matrix(
    preference: npt.ArrayLike, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Broadcast a preference to one row per design and validate it.

    Args:
        preference: An ``(n_objectives,)`` or ``(n, n_objectives)`` array.
        shape: The ``(n, n_objectives)`` shape of the objective matrix.

    Returns:
        An ``(n, n_objectives)`` array of preferences.

    Raises:
        ValueError: If the preference does not match the number of objectives or
            of designs, or is not on the simplex.
    """
    n, n_objectives = shape
    weights = np.asarray(preference, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n, weights.shape[0]))
    elif weights.ndim != _MATRIX_NDIM:
        raise ValueError(
            f"expected a preference of shape (n_objectives,) or (n, n_objectives), "
            f"got ndim {weights.ndim}"
        )
    if weights.shape[1] != n_objectives:
        raise ValueError(
            f"preference covers {weights.shape[1]} objectives but the values carry {n_objectives}"
        )
    if weights.shape[0] != n:
        raise ValueError(
            f"got {weights.shape[0]} preferences for {n} designs; pass one preference "
            f"for all designs or one per design"
        )
    _check_simplex(weights)
    return weights


def _check_simplex(weights: npt.NDArray[np.float64]) -> None:
    """Raise unless every row is non-negative, finite and sums to one."""
    if not np.isfinite(weights).all():
        raise ValueError("preference must be finite")
    if (weights < 0.0).any():
        raise ValueError(
            f"preference must be non-negative, got a minimum of {weights.min()}; a "
            f"negative weight turns an objective into something to be minimised"
        )
    sums = weights.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=PREFERENCE_SUM_TOLERANCE, rtol=0.0):
        worst = sums[int(np.argmax(np.abs(sums - 1.0)))]
        raise ValueError(
            f"preference must sum to 1, got {worst}; an unnormalised preference "
            f"rescales the reward as well as tilting it, which changes beta by stealth"
        )
