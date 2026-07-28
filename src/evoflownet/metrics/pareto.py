"""How good a *set* of trade-offs is: dominance, hypervolume, R2, IGD+.

With one objective, "how well did this run do" has an obvious answer. With
several it does not: two runs can each be better on one objective, and no
ordering of the objectives is given by the biology. These indicators answer the
question that is actually well-posed -- how good is the *set* of designs
returned, as a whole -- and they are the only way to compare multi-objective
methods without silently picking a trade-off and calling it the truth.

Everything here maximises
-------------------------

The rest of the package maximises fitness, so a design ``a`` dominates ``b``
when it is at least as good on every objective and strictly better on one. This
matters because most of the multi-objective literature is written for
minimisation, and the sign convention is the single easiest thing to get
backwards: a hypervolume computed with the wrong convention is a plausible
positive number that ranks methods in reverse. Each function below states the
form it implements and the reference it came from.

Refusing rather than approximating
----------------------------------

:func:`hypervolume` is exact in every dimension it accepts and raises past the
point where its exact method becomes intractable. An approximate hypervolume
reported next to an exact one is worse than no number at all, because nothing in
the output says which one it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Fitness

#: Largest number of front points :func:`hypervolume` will accept in three or
#: more objectives. Beyond two objectives it uses inclusion--exclusion, which is
#: exact but sums ``2^k - 1`` terms, so the limit is where that stops being
#: seconds and starts being hours. Two objectives use a sweep and have no limit.
MAX_INCLUSION_EXCLUSION_POINTS = 16

# Dimensions handled by a closed form rather than by inclusion--exclusion.
_ONE_OBJECTIVE = 1
_TWO_OBJECTIVES = 2

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


def non_dominated(values: Fitness) -> npt.NDArray[np.bool_]:
    """Mark the designs no other design beats outright.

    A design ``a`` is dominated when some ``b`` satisfies ``b_i ≥ a_i`` for every
    objective and ``b_j > a_j`` for at least one -- the standard Pareto
    dominance relation (Miettinen, 1999, Def. 2.2.1), written for maximisation.
    Everything not dominated is returned.

    Duplicated points are all kept: neither copy strictly beats the other, so
    neither is dominated. Deduplicate first if the count matters.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.

    Returns:
        An ``(n,)`` boolean array, ``True`` where the design is non-dominated.

    Raises:
        ValueError: If ``values`` is not a two-dimensional objective matrix or
            contains ``nan``, where dominance is undefined.
    """
    points = _as_objective_matrix(values)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.bool_)
    # O(n^2 · d) and fully vectorised. Row i, column j asks whether j beats i.
    at_least_as_good = (points[None, :, :] >= points[:, None, :]).all(axis=2)
    strictly_better_somewhere = (points[None, :, :] > points[:, None, :]).any(axis=2)
    dominated = (at_least_as_good & strictly_better_somewhere).any(axis=1)
    return ~dominated


def pareto_front(values: Fitness) -> npt.NDArray[np.float64]:
    """The non-dominated subset of a set of objective vectors.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.

    Returns:
        A ``(k, n_objectives)`` array of the non-dominated points, in the order
        they appeared in the input.

    Raises:
        ValueError: If ``values`` is not a two-dimensional objective matrix or
            contains ``nan``.
    """
    points = _as_objective_matrix(values)
    return points[non_dominated(points)]


def hypervolume(values: Fitness, reference: npt.ArrayLike) -> float:
    """Volume dominated by a set of designs, measured from a reference point.

    Zitzler & Thiele (1998): the measure of the region dominated by at least one
    point of the set and bounded by the reference point. It is the only common
    indicator that is strictly monotone with respect to Pareto dominance -- a set
    that dominates another always scores higher -- which is why it is the
    headline number in multi-objective GFlowNet work (Jain et al., 2023).

    The reference point is the worst value considered acceptable on each
    objective, and it is part of the measurement: hypervolumes computed against
    different reference points are not comparable. Designs that fail to beat the
    reference on every objective contribute nothing.

    Exactness by dimension:

    * one or two objectives -- a sweep, exact, any number of points;
    * three or more -- inclusion--exclusion over the front, exact, but limited to
      :data:`MAX_INCLUSION_EXCLUSION_POINTS` non-dominated points.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.
        reference: An ``(n_objectives,)`` point that every contributing design
            must beat on every objective. Must be finite.

    Returns:
        The dominated volume. Zero when no design beats the reference.

    Raises:
        ValueError: If ``values`` is not a two-dimensional objective matrix,
            contains ``nan``, or if the reference is not a finite point of
            matching width.
        NotImplementedError: If three or more objectives are combined with more
            than :data:`MAX_INCLUSION_EXCLUSION_POINTS` non-dominated points,
            where the exact method here becomes intractable. The alternative
            would be an approximation reported as if it were exact.
    """
    points = _as_objective_matrix(values)
    point = _as_reference_point(reference, points.shape[1])

    # Only points that beat the reference everywhere enclose any volume, and only
    # non-dominated ones can add to it.
    contributing = points[(points > point).all(axis=1)]
    if contributing.shape[0] == 0:
        return 0.0
    front = contributing[non_dominated(contributing)]
    n_objectives = front.shape[1]

    if n_objectives == _ONE_OBJECTIVE:
        return float(front.max() - point[0])
    if n_objectives == _TWO_OBJECTIVES:
        return _hypervolume_2d(front, point)
    return _hypervolume_inclusion_exclusion(front, point)


def r2_indicator(
    values: Fitness,
    weights: npt.ArrayLike,
    *,
    ideal: npt.ArrayLike | None = None,
) -> float:
    """Average distance from an ideal point, over a set of trade-offs.

    Hansen & Jaszkiewicz (1998), in the unary form standardised by Brockhoff et
    al. (2012)::

        R2(A, Λ, z*) = (1/|Λ|) Σ_{λ∈Λ} min_{a∈A} max_i λ_i |z*_i - a_i|

    Each weight vector ``λ`` stands for one decision-maker's trade-off, scored by
    the best design the set offers *them*; the indicator is the average over
    decision-makers. Unlike hypervolume it needs no reference point below the
    set and costs ``O(|Λ|·|A|·d)``, which is why Jain et al. (2023) report it
    alongside hypervolume on problems with more objectives.

    **Lower is better**, and zero means some design sits exactly at the ideal for
    every weight vector. This is the opposite direction to hypervolume, and it is
    the standard convention for R2.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.
        weights: An ``(m, n_objectives)`` array of weight vectors, each
            non-negative and summing to one, or a single ``(n_objectives,)``
            vector. A uniform grid or a Dirichlet sample of the simplex is usual.
        ideal: The ``(n_objectives,)`` point ``z*``, the best value considered
            attainable on each objective. Defaults to the componentwise maximum
            of ``values``, which makes the indicator relative to the set being
            measured -- fine for tracking one run over time, but two runs can
            only be compared against the *same* fixed ideal.

    Returns:
        The R2 indicator. Lower is better.

    Raises:
        ValueError: If ``values`` is empty or not a two-dimensional objective
            matrix, if the weights are not on the simplex or of matching width,
            or if ``ideal`` is not a finite point of matching width.
    """
    points = _as_objective_matrix(values)
    if points.shape[0] == 0:
        raise ValueError("R2 is undefined for an empty set: there is no design to score")
    n_objectives = points.shape[1]

    grid = np.asarray(weights, dtype=np.float64)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.ndim != _MATRIX_NDIM or grid.shape[1] != n_objectives:
        raise ValueError(f"expected weights of shape (m, {n_objectives}), got {grid.shape}")
    if grid.shape[0] == 0:
        raise ValueError("R2 needs at least one weight vector")
    _check_simplex(grid)

    z = points.max(axis=0) if ideal is None else _as_reference_point(ideal, n_objectives)

    deviations = np.abs(z[None, :] - points)
    # ``where`` rather than a plain product: an infeasible design deviates by
    # infinity, and ``0 * inf`` is ``nan``. A zero weight contributes nothing to
    # the maximum, an infinite deviation dominates it -- both are correct here,
    # and only the explicit form avoids a missing value in between.
    utilities = np.zeros((grid.shape[0], points.shape[0], n_objectives), dtype=np.float64)
    np.multiply(
        grid[:, None, :],
        deviations[None, :, :],
        out=utilities,
        where=grid[:, None, :] > 0.0,
    )
    best_per_weight = utilities.max(axis=2).min(axis=1)
    return float(best_per_weight.mean())


def igd_plus(values: Fitness, reference_front: Fitness) -> float:
    """How far a reference front is from being covered by the designs found.

    Ishibuchi et al. (2015), *Modified Distance Calculation in Generational
    Distance and Inverted Generational Distance*::

        IGD+(A, Z) = (1/|Z|) Σ_{z∈Z} min_{a∈A} d+(a, z)

    where, for maximisation, ``d+(a, z) = ‖max(z - a, 0)‖₂`` counts only the
    objectives on which ``a`` falls short of ``z``. Plain IGD uses the Euclidean
    distance instead and is famously not Pareto-compliant: it can rank a set
    above another that dominates it, because a point that *overshoots* the
    reference is charged for the distance. The ``+`` modification is what fixes
    that, and is why this package reports IGD+ rather than IGD.

    **Lower is better**, and zero means every reference point is matched or
    beaten by some design.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.
        reference_front: A ``(m, n_objectives)`` array of reference points --
            the true Pareto front where it is known, or the non-dominated union
            of all methods being compared where it is not.

    Returns:
        The IGD+ indicator. Lower is better.

    Raises:
        ValueError: If either input is not a two-dimensional objective matrix,
            if either is empty, or if the reference front is not finite.
    """
    points, front = _as_indicator_pair(values, reference_front, name="IGD+")
    return float(_modified_distances(points, front).min(axis=0).mean())


def gd_plus(values: Fitness, reference_front: Fitness) -> float:
    """How far the designs found are from a reference front.

    The companion of :func:`igd_plus` from the same paper (Ishibuchi et al.,
    2015), averaging over the designs instead of over the reference points::

        GD+(A, Z) = (1/|A|) Σ_{a∈A} min_{z∈Z} d+(a, z)

    It measures convergence only. A set of one excellent design scores perfectly
    here while covering none of the front, which is exactly what :func:`igd_plus`
    catches and this does not -- report the pair, not either alone.

    **Lower is better**.

    Args:
        values: An ``(n, n_objectives)`` array of objective values.
        reference_front: A ``(m, n_objectives)`` array of reference points.

    Returns:
        The GD+ indicator. Lower is better.

    Raises:
        ValueError: If either input is not a two-dimensional objective matrix,
            if either is empty, or if the reference front is not finite.
    """
    points, front = _as_indicator_pair(values, reference_front, name="GD+")
    return float(_modified_distances(points, front).min(axis=1).mean())


def _hypervolume_2d(front: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]) -> float:
    """Exact hypervolume of a two-objective front by sweeping the first axis.

    Sorting the non-dominated points by the first objective descending makes the
    second objective increase, so each point adds one rectangle whose height is
    the gain over the previous point.
    """
    order = np.argsort(-front[:, 0], kind="stable")
    ordered = front[order]
    volume = 0.0
    previous_y = reference[1]
    for x, y in ordered:
        if y > previous_y:
            volume += (x - reference[0]) * (y - previous_y)
            previous_y = y
    return float(volume)


def _hypervolume_inclusion_exclusion(
    front: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> float:
    """Exact hypervolume in any dimension, by inclusion--exclusion over the front.

    The region dominated by point ``p`` is the box ``[reference, p]``, and the
    intersection of several such boxes is the box up to their componentwise
    minimum. So the volume of the union is the alternating sum over all non-empty
    subsets -- exact, dimension-agnostic, and exponential in the number of
    points, which is what the guard is for.

    Raises:
        NotImplementedError: If the front is larger than
            :data:`MAX_INCLUSION_EXCLUSION_POINTS`.
    """
    k = front.shape[0]
    if k > MAX_INCLUSION_EXCLUSION_POINTS:
        raise NotImplementedError(
            f"exact hypervolume in {front.shape[1]} objectives is computed here by "
            f"inclusion-exclusion over the front, which sums 2^k terms and is limited "
            f"to k = {MAX_INCLUSION_EXCLUSION_POINTS} points; this front has {k}. Use a "
            f"dedicated implementation (pymoo, BoTorch) for larger fronts rather than "
            f"an approximation that would not announce itself as one."
        )
    codes = np.arange(1, 1 << k, dtype=np.int64)
    membership = ((codes[:, None] >> np.arange(k, dtype=np.int64)) & 1).astype(np.bool_)
    # Componentwise minimum over each subset: the corner of the intersected box.
    selected = np.where(membership[:, :, None], front[None, :, :], np.inf)
    corners = selected.min(axis=1)
    volumes = np.prod(corners - reference, axis=1)
    signs = np.where(membership.sum(axis=1) % 2 == 1, 1.0, -1.0)
    # Alternating sums lose precision as the front grows; at the k allowed here
    # the loss is far below anything that changes a comparison.
    return float((signs * volumes).sum())


def _modified_distances(
    points: npt.NDArray[np.float64], front: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Pairwise ``d+`` between designs and reference points, shaped ``(n, m)``.

    For maximisation, ``d+(a, z) = ‖max(z - a, 0)‖₂``: only the objectives on
    which the design falls short of the reference point are charged.
    """
    shortfall = np.maximum(front[None, :, :] - points[:, None, :], 0.0)
    return np.asarray(np.sqrt((shortfall**2).sum(axis=2)), dtype=np.float64)


def _as_indicator_pair(
    values: Fitness, reference_front: Fitness, *, name: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate the (designs, reference front) pair shared by GD+ and IGD+.

    Raises:
        ValueError: If either side is empty or malformed, if their widths
            disagree, or if the reference front is not finite.
    """
    points = _as_objective_matrix(values)
    front = _as_objective_matrix(reference_front)
    if points.shape[0] == 0 or front.shape[0] == 0:
        raise ValueError(f"{name} needs a non-empty design set and a non-empty reference front")
    if points.shape[1] != front.shape[1]:
        raise ValueError(
            f"designs carry {points.shape[1]} objectives and the reference front "
            f"{front.shape[1]}; they must describe the same objectives"
        )
    if not np.isfinite(front).all():
        raise ValueError(
            "the reference front must be finite; an infinite reference point makes "
            "every distance to it infinite or undefined"
        )
    return points, front


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
    if np.isnan(array).any():
        raise ValueError(
            "objective values contain nan; dominance between a measured design and an "
            "unmeasured one is undefined, so decide what a missing value counts as"
        )
    return array


def _as_reference_point(point: npt.ArrayLike, n_objectives: int) -> npt.NDArray[np.float64]:
    """Validate a single finite point with one entry per objective.

    Raises:
        ValueError: If the point is not one-dimensional, is not finite, or does
            not have one entry per objective.
    """
    array = np.asarray(point, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n_objectives:
        raise ValueError(f"expected a point of shape ({n_objectives},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("the point must be finite")
    return array


def _check_simplex(weights: npt.NDArray[np.float64]) -> None:
    """Raise unless every row is non-negative, finite and sums to one."""
    if not np.isfinite(weights).all():
        raise ValueError("weight vectors must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"weight vectors must be non-negative, got a minimum of {weights.min()}")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("each weight vector must sum to 1")
