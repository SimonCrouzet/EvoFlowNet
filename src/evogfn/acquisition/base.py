"""Choosing which candidates to spend the budget on.

An acquisition function turns a surrogate's prediction and uncertainty into a
single score, and a batch selector turns those scores into the set actually
sent for measurement. Both matter, and the second is the one usually skipped.

Why the batch step is separate
------------------------------

Taking the top ``k`` by score is not the same as choosing a good batch of ``k``.
The top ``k`` are typically near-identical -- neighbouring points on one peak --
so a round buys one bet repeated ``k`` times. Whether that is bad is an
empirical question and a contested one, which is precisely why selection is a
replaceable component here rather than a hard-coded ``argsort``.

The evidence, honestly
----------------------

Greenman et al. (PLOS Comput Biol 2025) report that uncertainty-based methods
*"never outperform greedily sampling the sequences with the highest predicted
values"* -- but with sequential, single-point acquisition. GameOpt (ICLR 2025)
sweeps batch size on GB1 and reports a crossover: greedy wins at batch 1, the
diverse method wins increasingly as the batch grows. Both cannot be generalised
past their own regimes, so both selectors ship and the comparison is run.

More than one objective
-----------------------

Every rule below scores a *scalar* prediction against a *scalar* incumbent.
Neither survives contact with three objectives: expected improvement over "the
best value so far" is undefined when there is no such value, and UCB handed an
``(n, 3)`` prediction would happily rank on whichever component NumPy
broadcasting left it holding. That is the failure this module is built to make
impossible -- a plausible ranking computed on one arbitrary objective is worse
than a traceback, because nothing in the output says which objective it was.

So an acquisition rule declares what it can rank, in two places:

* `Acquisition.supports_multi_objective` -- whether
  [score][evogfn.acquisition.base.Acquisition.score] may be handed one *vector*
  per candidate. `False` everywhere except
  [ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition], and
  the base class raises rather than letting a scalar rule guess.
* [reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives] --
  how *measurements* collapse to the one number this rule ranks. The campaign
  asks the rule rather than deciding for itself, so the trade-off the surrogate
  is fitted to, the incumbent improvement is measured against, and the ledger
  reports best-so-far on are the same trade-off by construction. Two of them
  disagreeing is not an error anything would catch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


class Acquisition(ABC):
    """Scores candidates from a surrogate's mean and uncertainty.

    Subclasses implement `_score`. The public
    [score][evogfn.acquisition.base.Acquisition.score] checks the shapes first,
    so no rule repeats those checks and none can forget the one that matters:
    a scalar rule handed a prediction per *objective* must refuse rather than
    rank on whichever component it happens to see.
    """

    #: Whether [score][evogfn.acquisition.base.Acquisition.score] may be handed
    #: an ``(n, n_objectives)`` prediction. Left `False` unless a rule genuinely
    #: defines an ordering over objective vectors.
    supports_multi_objective: ClassVar[bool] = False

    def score(
        self,
        mean: npt.NDArray[np.floating],
        std: npt.NDArray[np.floating],
        *,
        best_observed: float,
    ) -> npt.NDArray[np.floating]:
        """Rank candidates. Higher is better.

        Args:
            mean: ``(n,)`` predicted values, or ``(n, n_objectives)`` for a rule
                that declares `supports_multi_objective`.
            std: Predictive standard deviations, the same shape as ``mean``.
            best_observed: Best value measured so far, on the scale
                [reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives]
                puts measurements on. Used by improvement-based rules.

        Returns:
            An ``(n,)`` array of scores.

        Raises:
            ValueError: If ``mean`` and ``std`` disagree in shape, if either has
                more than two dimensions, or if a scalar-only rule is handed one
                prediction per objective -- where its ranking would be a
                well-formed number computed on an arbitrary component.
        """
        mu = np.asarray(mean, dtype=np.float64)
        sigma = np.asarray(std, dtype=np.float64)
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mean has shape {mu.shape} and std {sigma.shape}; every candidate needs "
                f"exactly one spread per predicted value"
            )
        if mu.ndim > _MATRIX_NDIM:
            raise ValueError(f"expected shape (n,) or (n, n_objectives), got {mu.shape}")
        n_objectives = 1 if mu.ndim < _MATRIX_NDIM else mu.shape[1]
        if n_objectives > 1 and not self.supports_multi_objective:
            raise ValueError(
                f"{type(self).__name__} ranks one predicted value per candidate, but it was "
                f"handed {n_objectives} objectives per candidate. Ranking on one of them "
                f"would be arbitrary and would not announce itself; wrap the rule in "
                f"ScalarizedAcquisition to state the trade-off explicitly."
            )
        return self._score(mu, sigma, best_observed=best_observed)

    @abstractmethod
    def _score(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
        *,
        best_observed: float,
    ) -> npt.NDArray[np.floating]:
        """Rank candidates whose shapes have already been checked.

        Args:
            mean: ``(n,)`` predicted values, or ``(n, n_objectives)`` if the rule
                declares `supports_multi_objective`.
            std: Predictive standard deviations, the same shape as ``mean``.
            best_observed: Best measured value so far.

        Returns:
            An ``(n,)`` array of scores.
        """

    def reduce_objectives(self, values: Fitness) -> npt.NDArray[np.float64]:
        """Collapse measured objective vectors to the value this rule ranks.

        The campaign calls this on everything measured so far, and uses the
        result for three things at once: the surrogate's training target, the
        incumbent an improvement rule is scored against, and the ledger's
        best-so-far. Asking the rule is what keeps those three on the same
        trade-off -- if the campaign chose its own, a run could optimise one
        weighting, rank by a second and report a third, and nothing in the
        output would say so.

        The default accepts a single objective and refuses anything wider.

        Args:
            values: An ``(n,)`` or ``(n, n_objectives)`` array of measurements.

        Returns:
            An ``(n,)`` array, one value per measured design.

        Raises:
            ValueError: If the measurements carry more than one objective.
                Flattening them would hand an improvement rule an incumbent
                taken across objectives on different scales, which no expected
                improvement is defined against.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            return array
        if array.ndim == _MATRIX_NDIM and array.shape[1] == 1:
            return np.asarray(array[:, 0], dtype=np.float64)
        raise ValueError(
            f"{type(self).__name__} expects measurements of shape (n,) or (n, 1), got "
            f"{array.shape}; multi-objective measurements must be scalarised before an "
            f"improvement-based acquisition rule has an incumbent to improve on"
        )

    def __repr__(self) -> str:
        """Name the acquisition function."""
        return f"{type(self).__name__}()"


class BatchSelector(ABC):
    """Chooses which scored candidates to actually measure."""

    @abstractmethod
    def select(
        self,
        candidates: Tokens,
        scores: npt.NDArray[np.floating],
        k: int,
    ) -> npt.NDArray[np.intp]:
        """Pick ``k`` candidates to send for measurement.

        Args:
            candidates: An ``(n, length)`` array of proposals.
            scores: An ``(n,)`` array of acquisition scores.
            k: How many to select.

        Returns:
            Indices of the chosen candidates.
        """

    def __repr__(self) -> str:
        """Name the selector."""
        return f"{type(self).__name__}()"
