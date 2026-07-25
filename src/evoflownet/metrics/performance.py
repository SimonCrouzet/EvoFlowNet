"""How good the designs are: regret, top-K performance, feasibility.

Definitions here are transcribed from the papers that introduced them rather
than reinvented, so numbers produced by this package can be placed beside
published ones. Each function names its source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Fitness


def simple_regret(values: Fitness, optimum: float) -> float:
    """Gap between the best design found and the best attainable.

    Stanton et al. (2024): ``r_t = f* - f(x̂*_t)``, where ``x̂*_t`` is the best
    design produced so far. Reported rather than "best found" because a raw best
    value is not comparable across landscapes -- regret is.

    Non-finite values (infeasible designs scoring ``-inf``) are ignored. If
    nothing finite was produced, regret is the full gap from the optimum down to
    negative infinity, which is reported as ``inf`` rather than as a number that
    would average misleadingly.

    Args:
        values: An ``(n,)`` or ``(n, 1)`` array of observed objective values.
        optimum: The best attainable value, ``f*``.

    Returns:
        The simple regret. Zero means the optimum was found.
    """
    flat = _as_flat(values)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("inf")
    return float(optimum - finite.max())


def cumulative_regret(values: Fitness, optimum: float) -> float:
    """Total regret accumulated over a run.

    Stanton et al. (2024): ``R_t = Σ_j r_j``. Where simple regret asks only how
    good the best design was, this charges for every evaluation spent on a poor
    one, so a method that finds the optimum quickly scores better than one that
    finds it at the end of the same budget.

    Args:
        values: An ``(n,)`` or ``(n, 1)`` array of observed values, in the order
            they were evaluated.
        optimum: The best attainable value, ``f*``.

    Returns:
        The cumulative regret of the running best.

    Raises:
        ValueError: If no observed value is finite, since the running best is
            then undefined at every step.
    """
    flat = _as_flat(values)
    if not np.isfinite(flat).any():
        raise ValueError("cumulative regret is undefined when no value is finite")
    # Ignore infeasible designs when tracking the best, but still charge for the
    # evaluation they consumed.
    usable = np.where(np.isfinite(flat), flat, -np.inf)
    running_best = np.maximum.accumulate(usable)
    return float(np.sum(optimum - running_best[np.isfinite(running_best)]))


def top_k_performance(values: Fitness, k: int) -> float:
    """Mean objective value of the best ``k`` designs.

    Jain et al. (2022) define performance as the mean over the selected set
    rather than its maximum, because a batch is delivered whole: one excellent
    design among 95 poor ones is a worse round than 96 good ones.

    Args:
        values: An ``(n,)`` or ``(n, 1)`` array of observed values.
        k: How many of the best designs to average over. Values beyond ``n`` are
            clamped to ``n``.

    Returns:
        The mean of the top ``k`` finite values.

    Raises:
        ValueError: If ``k`` is not positive, or no value is finite.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    flat = _as_flat(values)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("top-k performance is undefined when no value is finite")
    top = np.sort(finite)[::-1][: min(k, finite.size)]
    return float(top.mean())


def feasible_fraction(feasible: npt.NDArray[np.bool_]) -> float:
    """Share of designs that are actually constructible.

    Stanton et al. (2024) track this alongside regret, and it is the metric that
    separates search which respects a constraint from search which ignores one.
    A method that reaches a good design while 80% of its evaluations were spent
    on sequences that could never be made has not used the budget it was given.

    Args:
        feasible: An ``(n,)`` boolean array.

    Returns:
        The fraction in ``[0, 1]``. An empty input gives ``0.0``.
    """
    array = np.asarray(feasible, dtype=np.bool_)
    if array.size == 0:
        return 0.0
    return float(array.mean())


def _as_flat(values: Fitness) -> npt.NDArray[np.float64]:
    """Accept ``(n,)`` or single-objective ``(n, 1)`` and return ``(n,)``.

    Raises:
        ValueError: If the input has more than one objective, where "the best
            value" is not defined without a scalarisation.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:  # noqa: PLR2004 - a single objective
        return array[:, 0]
    raise ValueError(
        f"expected shape (n,) or (n, 1), got {array.shape}; multi-objective values "
        f"must be scalarised before a scalar performance metric applies"
    )
