"""Comparing a sampler's output against the exact target distribution.

A GFlowNet is supposed to sample ``p*(x) ∝ R(x)^β``. Every other metric in this
package is satisfied by a good hill-climber: high top-K performance, low regret,
even respectable diversity are all achievable by an optimiser that never samples
anything. This module contains the one check that is not.

It requires enumerating the space, which is why the benchmarks here were chosen
to be enumerable. Malkin et al. (2022) and Jain et al. (2023) report exactly this
quantity, on synthetic hypergrids; the landscapes in this package make it
computable on an empirical protein fitness landscape as well.

The comparison is only meaningful with enough samples. An empirical distribution
over 100 draws differs from any target simply by sampling noise, so
:func:`expected_l1_from_sampling_noise` is provided to say what "close" should
mean for a given sample count before reading anything into the number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Fitness, Tokens


def target_distribution(
    values: Fitness, *, beta: float = 1.0, min_reward: float = 0.0
) -> npt.NDArray[np.float64]:
    """The exact distribution a GFlowNet should sample, over an enumerated space.

    Computes ``p*(x) = R(x)^β / Z`` with ``Z = Σ_x R(x)^β``, in log space so that
    a large ``β`` on a landscape with a wide dynamic range does not overflow
    before normalising.

    Args:
        values: An ``(n,)`` or ``(n, 1)`` array of objective values for **every**
            sequence in the space, in a fixed enumeration order.
        beta: Reward exponent. Higher values concentrate the target on the best
            designs; ``β = 3`` is the default in Jain et al. (2022).
        min_reward: Value assigned to designs whose objective is non-positive or
            non-finite. Rewards must be non-negative for the distribution to
            exist, and infeasible designs score ``-inf`` here.

    Returns:
        An ``(n,)`` array summing to 1.

    Raises:
        ValueError: If no design has positive reward, in which case the target
            is undefined, or if the values carry more than one objective, in
            which case ``R(x)`` is not yet a number.
    """
    flat = _single_objective(values)
    rewards = np.where(np.isfinite(flat) & (flat > 0.0), flat, min_reward)
    if not (rewards > 0.0).any():
        raise ValueError(
            "no design has positive reward, so p*(x) proportional to R(x)^beta is undefined"
        )
    with np.errstate(divide="ignore"):
        log_rewards = beta * np.log(rewards)
    log_rewards[~np.isfinite(log_rewards)] = -np.inf
    shifted = log_rewards - log_rewards.max()
    weights = np.exp(shifted)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def empirical_distribution(samples: Tokens, enumeration: Tokens) -> npt.NDArray[np.float64]:
    """Frequency of each enumerated sequence among the samples.

    Args:
        samples: An ``(m, length)`` array of sampled sequences.
        enumeration: An ``(n, length)`` array listing every sequence in the
            space, in the same order used for the target.

    Returns:
        An ``(n,)`` array summing to 1, or to 0 if no samples were given.

    Raises:
        ValueError: If a sample does not appear in the enumeration, which means
            the two were built from different spaces and any comparison between
            them would be meaningless.
    """
    space = np.asarray(enumeration)
    drawn = np.asarray(samples)
    counts = np.zeros(space.shape[0], dtype=np.float64)
    if drawn.shape[0] == 0:
        return counts

    index = {row.tobytes(): position for position, row in enumerate(space.astype(np.int64))}
    for row in drawn.astype(np.int64):
        position = index.get(row.tobytes())
        if position is None:
            raise ValueError(
                "a sampled sequence is absent from the enumeration; the samples and "
                "the enumerated space do not describe the same landscape"
            )
        counts[position] += 1.0
    return counts / counts.sum()


def l1_distance(empirical: npt.NDArray[np.floating], target: npt.NDArray[np.floating]) -> float:
    """Total variation-style distance between two distributions.

    ``Σ_x |p(x) - q(x)|``, as reported by Malkin et al. (2022). Ranges from 0
    (identical) to 2 (disjoint support).

    Args:
        empirical: An ``(n,)`` distribution over the enumerated space.
        target: An ``(n,)`` distribution over the same space, in the same order.

    Returns:
        The L1 distance.

    Raises:
        ValueError: If the two distributions have different lengths.
    """
    left = np.asarray(empirical, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"distributions have shapes {left.shape} and {right.shape}")
    return float(np.abs(left - right).sum())


def expected_l1_from_sampling_noise(
    target: npt.NDArray[np.floating], n_samples: int, *, seed: int = 0, repeats: int = 32
) -> float:
    """L1 distance a *perfect* sampler would show at this sample count.

    Drawing ``m`` times from a distribution does not reproduce it exactly, so a
    non-zero L1 is expected even from a sampler that is exactly right. This
    estimates that floor by sampling from the target itself.

    Without it, an L1 of 0.3 is uninterpretable: it may be a badly-fitted policy,
    or a perfect one measured with too few samples.

    Args:
        target: The exact target distribution.
        n_samples: How many samples the measurement used.
        seed: Seed for the reference draws.
        repeats: How many reference draws to average over.

    Returns:
        The mean L1 distance between the target and draws from itself.

    Raises:
        ValueError: If ``n_samples`` or ``repeats`` is not positive.
    """
    if n_samples < 1 or repeats < 1:
        raise ValueError(
            f"n_samples and repeats must both be at least 1, got {n_samples} and {repeats}"
        )
    probabilities = np.asarray(target, dtype=np.float64)
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(repeats):
        counts = rng.multinomial(n_samples, probabilities) / n_samples
        distances.append(np.abs(counts - probabilities).sum())
    return float(np.mean(distances))


def _single_objective(values: Fitness) -> npt.NDArray[np.float64]:
    """Accept ``(n,)`` or single-objective ``(n, 1)`` and return ``(n,)``.

    Args:
        values: Objective values for every sequence in the space.

    Returns:
        An ``(n,)`` array, one value per enumerated sequence.

    Raises:
        ValueError: If the input has more than one objective. Flattening would
            return ``n * n_objectives`` probabilities against an ``n``-sequence
            enumeration, and every comparison downstream would misalign.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:  # noqa: PLR2004 - a single objective
        return array[:, 0]
    raise ValueError(
        f"expected shape (n,) or (n, 1), got {array.shape}; multi-objective values "
        f"must be scalarised before a single target distribution exists"
    )
