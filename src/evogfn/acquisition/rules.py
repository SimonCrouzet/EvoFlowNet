"""Acquisition rules and batch selectors.

The rules span the exploitation-exploration axis so the comparison can be run
rather than assumed: :class:`Greedy` ignores uncertainty entirely,
:class:`UpperConfidenceBound` and :class:`ExpectedImprovement` weight it, and
:class:`Thompson` samples from the predictive distribution.

Defaults follow Jain et al. (ICML 2022): UCB with ``kappa = 0.1``. That is a
notably small weight -- close to greedy -- and worth knowing when reading their
results.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from evogfn.acquisition.base import Acquisition, BatchSelector
from evogfn.metrics.diversity import hamming_distances

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Tokens

#: Exploration weight used by Jain et al.
DEFAULT_KAPPA = 0.1


def _normal_cdf(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The standard normal CDF, without taking a SciPy dependency for one call."""
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return np.asarray(0.5 * (1.0 + erf(z / math.sqrt(2.0))), dtype=np.float64)


class Greedy(Acquisition):
    """Take the predicted value and ignore the uncertainty.

    The baseline that, on the published evidence, is harder to beat than it
    looks -- and the one whose weakness should show at small budgets, where the
    surrogate is uninformed almost everywhere and its argmax is largely an
    artefact of where it happened to extrapolate high.
    """

    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],  # noqa: ARG002 - ignored by definition
        *,
        best_observed: float,  # noqa: ARG002 - ignored by definition
    ) -> npt.NDArray[np.floating]:
        """Return the predicted mean unchanged."""
        return np.asarray(mean, dtype=np.float64)


class UpperConfidenceBound(Acquisition):
    """``mean + kappa * std``.

    Args:
        kappa: Weight on uncertainty. Jain et al. use 0.1.

    Raises:
        ValueError: If ``kappa`` is negative.
    """

    def __init__(self, *, kappa: float = DEFAULT_KAPPA) -> None:
        """Store the exploration weight."""
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        self._kappa = kappa

    @property
    def kappa(self) -> float:
        """Weight on uncertainty."""
        return self._kappa

    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],
        *,
        best_observed: float,  # noqa: ARG002 - not an improvement rule
    ) -> npt.NDArray[np.floating]:
        """Return ``mean + kappa * std``."""
        return np.asarray(mean, dtype=np.float64) + self._kappa * np.asarray(std, dtype=np.float64)

    def __repr__(self) -> str:
        """Name the rule and its weight."""
        return f"UpperConfidenceBound(kappa={self._kappa})"


class ExpectedImprovement(Acquisition):
    """Expected gain over the best value measured so far, under a normal model.

    Unlike UCB this has no free weight: the trade between mean and spread falls
    out of how far the candidate sits above the incumbent.
    """

    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],
        *,
        best_observed: float,
    ) -> npt.NDArray[np.floating]:
        """Return the expected improvement over ``best_observed``."""
        mu = np.asarray(mean, dtype=np.float64)
        sigma = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
        z = (mu - best_observed) / sigma
        cdf = _normal_cdf(z)
        pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        improvement: npt.NDArray[np.float64] = (mu - best_observed) * cdf + sigma * pdf
        return improvement


class Thompson(Acquisition):
    """One draw from the predictive distribution per candidate.

    Exploration comes from the draw rather than from a bonus term, so the
    exploration rate is set by the surrogate's own confidence instead of a
    hyperparameter. This is what ALDE uses.

    Args:
        seed: Seeds the draws.
    """

    def __init__(self, *, seed: int = 0) -> None:
        """Seed the sampler."""
        self._rng = np.random.default_rng(seed)

    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],
        *,
        best_observed: float,  # noqa: ARG002 - not an improvement rule
    ) -> npt.NDArray[np.floating]:
        """Draw one sample per candidate."""
        return self._rng.normal(
            np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
        )


class TopK(BatchSelector):
    """Take the ``k`` highest-scoring candidates.

    The conventional choice, and the one that produces a batch of near-duplicates
    when the top of the surrogate is a single peak.
    """

    def select(
        self,
        candidates: Tokens,  # noqa: ARG002 - selection ignores the sequences
        scores: npt.NDArray[np.floating],
        k: int,
    ) -> npt.NDArray[np.intp]:
        """Return the indices of the ``k`` best scores."""
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        return np.asarray(order[:k], dtype=np.intp)


class DiverseTopK(BatchSelector):
    """Greedy selection with a penalty for closeness to what is already chosen.

    Each pick maximises ``score - lambda * max_similarity_to_selected``, where
    similarity is one minus normalised Hamming distance. At ``lambda = 0`` this
    is exactly :class:`TopK`, which makes the comparison a single knob rather
    than two implementations.

    This is the component experiment E4 turns on: whether the value of spreading
    a batch grows with its size, which is the question the two published nulls
    left open by testing only single-point acquisition.

    Args:
        penalty: Weight on similarity to the already-selected set.

    Raises:
        ValueError: If ``penalty`` is negative.
    """

    def __init__(self, *, penalty: float = 1.0) -> None:
        """Store the diversity weight."""
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative, got {penalty}")
        self._penalty = penalty

    @property
    def penalty(self) -> float:
        """Weight on similarity to already-selected candidates."""
        return self._penalty

    def select(
        self,
        candidates: Tokens,
        scores: npt.NDArray[np.floating],
        k: int,
    ) -> npt.NDArray[np.intp]:
        """Pick ``k`` candidates, trading score against distance to the batch.

        Args:
            candidates: An ``(n, length)`` array of proposals.
            scores: An ``(n,)`` array of acquisition scores.
            k: How many to select.

        Returns:
            Indices of the chosen candidates, best first.
        """
        array = np.asarray(candidates)
        values = np.asarray(scores, dtype=np.float64)
        n, length = array.shape
        k = min(k, n)
        if k == 0:
            return np.zeros(0, dtype=np.intp)

        chosen = [int(np.argmax(values))]
        # Similarity to the nearest already-chosen candidate, updated
        # incrementally so each pick costs one distance computation rather than
        # a rescan of the whole selected set.
        closeness = 1.0 - hamming_distances(array, array[chosen]).min(axis=1) / length

        while len(chosen) < k:
            adjusted = values - self._penalty * closeness
            adjusted[chosen] = -np.inf
            pick = int(np.argmax(adjusted))
            chosen.append(pick)
            fresh = 1.0 - hamming_distances(array, array[[pick]])[:, 0] / length
            closeness = np.maximum(closeness, fresh)
        return np.asarray(chosen, dtype=np.intp)

    def __repr__(self) -> str:
        """Name the selector and its penalty."""
        return f"DiverseTopK(penalty={self._penalty})"
