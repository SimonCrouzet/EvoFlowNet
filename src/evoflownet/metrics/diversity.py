"""How varied the designs are: diversity, novelty, and distinct modes.

These are the metrics the project exists to move. A batch of 96 near-identical
sequences is a single bet dressed as ninety-six, and no performance metric
notices -- which is exactly why diversity has to be reported alongside, never
instead of, performance.

Definitions follow Jain et al. (2022) so that numbers remain comparable to the
GFlowNet sequence-design literature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Tokens

#: Below this many sequences, mean pairwise distance is undefined: there are no
#: pairs.
_MIN_FOR_PAIRS = 2


def hamming_distances(a: Tokens, b: Tokens) -> npt.NDArray[np.int64]:
    """All pairwise Hamming distances between two sets of equal-length sequences.

    Args:
        a: An ``(n, length)`` array of token indices.
        b: An ``(m, length)`` array of token indices.

    Returns:
        An ``(n, m)`` array of distances.

    Raises:
        ValueError: If the two sets have different sequence lengths.
    """
    left = np.asarray(a)
    right = np.asarray(b)
    if left.ndim != 2 or right.ndim != 2:  # noqa: PLR2004 - both must be batches
        raise ValueError(f"expected two batches, got ndim {left.ndim} and {right.ndim}")
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"sequence lengths differ: {left.shape[1]} and {right.shape[1]}; Hamming "
            f"distance is only defined between sequences of equal length"
        )
    distances: npt.NDArray[np.int64] = (left[:, None, :] != right[None, :, :]).sum(axis=2)
    return distances.astype(np.int64)


def diversity(sequences: Tokens) -> float:
    """Mean pairwise distance within a set of designs.

    Jain et al. (2022) define diversity as the mean Hamming distance over all
    ordered pairs of distinct designs in the set.

    Args:
        sequences: An ``(n, length)`` array of token indices.

    Returns:
        The mean pairwise distance. A set with fewer than two designs has no
        pairs, and returns ``0.0``.
    """
    array = np.asarray(sequences)
    if array.shape[0] < _MIN_FOR_PAIRS:
        return 0.0
    distances = hamming_distances(array, array)
    n = array.shape[0]
    # Exclude the diagonal: a sequence's distance to itself is not a pair.
    return float(distances.sum() / (n * (n - 1)))


def novelty(sequences: Tokens, reference: Tokens) -> float:
    """How far the designs are from what was already known.

    Jain et al. (2022): the mean, over designs, of the distance to the nearest
    sequence in a reference set. High diversity with low novelty means a method
    is exploring a region it was already given.

    Args:
        sequences: An ``(n, length)`` array of proposed designs.
        reference: An ``(m, length)`` array of already-known sequences, such as
            the initial dataset or everything measured so far.

    Returns:
        The mean nearest-neighbour distance.

    Raises:
        ValueError: If the reference set is empty, where "distance to the
            nearest known sequence" has no value.
    """
    array = np.asarray(sequences)
    known = np.asarray(reference)
    if known.shape[0] == 0:
        raise ValueError("novelty needs a non-empty reference set")
    if array.shape[0] == 0:
        return 0.0
    return float(hamming_distances(array, known).min(axis=1).mean())


def distinct_modes(
    sequences: Tokens,
    values: npt.NDArray[np.floating],
    *,
    threshold: float,
    min_distance: int = 1,
) -> int:
    """Count separated high-scoring designs.

    A mode is a design scoring at or above ``threshold`` that is at least
    ``min_distance`` away from every mode already counted. Counting modes rather
    than high-scoring designs is what distinguishes finding one peak repeatedly
    from finding several: an optimiser that returns a thousand copies of one good
    variant scores 1 here.

    Designs are considered in descending order of value, so the best
    representative of each region is the one that counts.

    Args:
        sequences: An ``(n, length)`` array of token indices.
        values: An ``(n,)`` array of their objective values.
        threshold: Minimum value for a design to count at all.
        min_distance: How far apart two designs must be to count separately.
            ``1`` counts every distinct sequence; larger values require genuinely
            separated regions.

    Returns:
        The number of distinct modes found.

    Raises:
        ValueError: If the inputs disagree in length, or ``min_distance`` is not
            positive.
    """
    array = np.asarray(sequences)
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] != scores.shape[0]:
        raise ValueError(
            f"got {array.shape[0]} sequences and {scores.shape[0]} values; they must match"
        )
    if min_distance < 1:
        raise ValueError(f"min_distance must be at least 1, got {min_distance}")

    qualifying = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
    if qualifying.size == 0:
        return 0
    # Best first, so each region is represented by its strongest design.
    ordered = qualifying[np.argsort(-scores[qualifying], kind="stable")]

    modes = array[ordered[:1]]
    for index in ordered[1:]:
        candidate = array[index][None, :]
        if hamming_distances(candidate, modes).min() >= min_distance:
            modes = np.concatenate([modes, candidate])
    return int(modes.shape[0])
