"""Acquisition rules and batch selectors.

The rules span the exploitation-exploration axis so the comparison can be run
rather than assumed: [Greedy][evogfn.acquisition.rules.Greedy] ignores
uncertainty entirely,
[UpperConfidenceBound][evogfn.acquisition.rules.UpperConfidenceBound] and
[ExpectedImprovement][evogfn.acquisition.rules.ExpectedImprovement] weight it,
and [Thompson][evogfn.acquisition.rules.Thompson] samples from the predictive
distribution.

Defaults follow Jain et al. (ICML 2022): UCB with ``kappa = 0.1``. That is a
notably small weight -- close to greedy -- and worth knowing when reading their
results.

All four rank a scalar, and
[ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition] is what
lets them be used at all when the landscape returns a vector. It is the only
rule here that declares `Acquisition.supports_multi_objective`; every other one
refuses an objective vector rather than ranking on a component of it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from evogfn.acquisition.base import Acquisition, BatchSelector
from evogfn.metrics.diversity import hamming_distances
from evogfn.rewards.scalarization import PREFERENCE_SUM_TOLERANCE

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.rewards.scalarization import Scalarization

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

    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],  # noqa: ARG002 - ignored by definition
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

    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
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

    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
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

    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
        *,
        best_observed: float,  # noqa: ARG002 - not an improvement rule
    ) -> npt.NDArray[np.floating]:
        """Draw one sample per candidate."""
        return self._rng.normal(
            np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
        )


class ScalarizedAcquisition(Acquisition):
    r"""Rank a multi-objective problem by one stated trade-off.

    Wraps any scalar rule and gives it the two things it is missing when the
    landscape returns a vector: a way to turn measured objective vectors into
    the incumbent it improves on, and a way to turn a vector-valued prediction
    into the number it ranks. Both go through the same
    [Scalarization][evogfn.rewards.scalarization.Scalarization] and the same
    preference $\omega$, which is the point -- the surrogate is fitted to
    $s(R(x) \mid \omega)$, the incumbent is the best measured $s$, and the
    ranking is over predicted $s$. A campaign that scalarised in one place and
    ranked in another would be optimising a trade-off it never reports.

    ## Why this route and not expected hypervolume improvement

    EHVI (Emmerich et al., 2006; Daulton et al., 2020) needs no preference at
    all: it scores a candidate by how much volume it would add to the incumbent
    Pareto front, so it targets the whole front rather than one point on it, and
    it is the honest answer to "which batch improves the *set*". Two things
    stand between this package and it.

    * It needs a predictive distribution **per objective**. Every
      [Surrogate][evogfn.surrogate.base.Surrogate] here returns one mean and one
      spread per sequence, so there is no joint predictive to integrate.
    * Its cost is the cost of a hypervolume, once per candidate. The exact
      method in [evogfn.metrics.pareto][] is inclusion--exclusion over the
      front, $2^k$ terms for a $k$-point front, and the pool is 2,048 candidates
      *every round*. A front of 20 measured designs -- ordinary on CH65 -- puts
      that at two billion box intersections per round, which is why the box
      decompositions in the literature exist and why bolting an approximation
      into this loop would produce a number nothing downstream could tell apart
      from an exact one.

    What EHVI would have bought, concretely, is coverage: this rule can only
    ever find the part of the front its $\omega$ points at, and with
    [WeightedSum][evogfn.rewards.scalarization.WeightedSum] it cannot reach a
    concave part of the front at any $\omega$ (Miettinen, 1999, Thm 3.1.4).
    Sweeping $\omega$ across seeds recovers some of that; it is not the same
    thing, and results should say which was run.

    ## The spread it hands the inner rule

    A scalarisation of the means says nothing about the spread of the
    scalarised value, and improvement rules need one. This takes

    $$\sigma_s(x) = \max\bigl(s(\mu(x) + \sigma(x) \mid \omega)
    - s(\mu(x) \mid \omega),\ 0\bigr)$$

    -- the scalarised value of a one-standard-deviation upside, less the
    scalarised mean. For [WeightedSum][evogfn.rewards.scalarization.WeightedSum]
    that is exactly $\sum_i \omega_i \sigma_i$, which is the *perfectly
    correlated* case rather than the independent $\sqrt{\sum_i \omega_i^2
    \sigma_i^2}$: it overstates the spread when objectives are independent, and
    overstating a surrogate's uncertainty errs towards exploration rather than
    towards confident nonsense. For a non-linear scalarisation it is a
    first-order estimate, and it is non-negative for any scalarisation monotone
    in each objective, which all three here are.

    Args:
        rule: The scalar rule that does the ranking.
        scalarization: How objectives are combined.
        preference: The ``(n_objectives,)`` preference vector, non-negative and
            summing to one.

    Raises:
        ValueError: If the preference is not a one-dimensional, finite,
            non-negative vector summing to one.
    """

    supports_multi_objective: ClassVar[bool] = True

    def __init__(
        self,
        rule: Acquisition,
        scalarization: Scalarization,
        preference: npt.ArrayLike,
    ) -> None:
        """Store the inner rule and the trade-off it will rank under."""
        weights = np.asarray(preference, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(
                f"expected a preference vector of shape (n_objectives,), got {weights.shape}"
            )
        if not np.isfinite(weights).all():
            raise ValueError("preference must be finite")
        if (weights < 0.0).any():
            raise ValueError(
                f"preference must be non-negative, got a minimum of {weights.min()}; a "
                f"negative weight turns an objective into something to be minimised"
            )
        if abs(float(weights.sum()) - 1.0) > PREFERENCE_SUM_TOLERANCE:
            raise ValueError(
                f"preference must sum to 1, got {weights.sum()}; an unnormalised "
                f"preference rescales the ranking as well as tilting it"
            )
        self._rule = rule
        self._scalarization = scalarization
        self._preference = weights

    @property
    def rule(self) -> Acquisition:
        """The scalar rule doing the ranking."""
        return self._rule

    @property
    def scalarization(self) -> Scalarization:
        """How the objectives are combined."""
        return self._scalarization

    @property
    def preference(self) -> npt.NDArray[np.float64]:
        """The preference vector, as a copy so it cannot be mutated."""
        return self._preference.copy()

    def reduce_objectives(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Scalarise measured objective vectors under the stored preference.

        Args:
            values: An ``(n, n_objectives)`` array of measurements.

        Returns:
            An ``(n,)`` array of scalarised values.

        Raises:
            ValueError: If ``values`` is not a two-dimensional objective matrix,
                contains ``nan``, or has a width the preference does not match.
        """
        return self._scalarization.scalarize(values, self._preference)

    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
        *,
        best_observed: float,
    ) -> npt.NDArray[np.floating]:
        """Scalarise the prediction, then hand it to the inner rule.

        A prediction that is already one value per candidate is passed straight
        through: a single-output surrogate has, in effect, already been fitted
        to the scalarised target, and scalarising it twice would apply the
        preference to a number that is not an objective vector.
        """
        if mean.ndim == 1:
            return self._rule.score(mean, std, best_observed=best_observed)
        value = self._scalarization.scalarize(mean, self._preference)
        upside = self._scalarization.scalarize(mean + std, self._preference)
        spread = np.maximum(upside - value, 0.0)
        return self._rule.score(value, spread, best_observed=best_observed)

    def __repr__(self) -> str:
        """Name the inner rule, the scalarisation and the preference."""
        return (
            f"ScalarizedAcquisition({self._rule!r}, {self._scalarization!r}, "
            f"{np.array2string(self._preference, precision=3)})"
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
    is exactly [TopK][evogfn.acquisition.rules.TopK], which makes the comparison
    a single knob rather than two implementations.

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
