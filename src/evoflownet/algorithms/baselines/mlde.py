"""MLDE: the supervised baseline protein engineers actually run.

Machine-learning-assisted directed evolution (Wittmann, Yue & Arnold, *Cell
Systems* 2021) is three steps and no more: screen a random sample of the library,
fit a regressor to it, order the variants it predicts best. There is no
acquisition function, no uncertainty, no policy and no second thought. On the
GB1 four-site landscape that protocol found a variant in the top 0.1% far more
often than a directed-evolution walk did, which is why it is the reference point
the field cites.

It is here because it is the cheapest way for this project's central claim to be
wrong. If one supervised fit on a random plate reaches the same designs a
GFlowNet reaches, then everything the GFlowNet adds -- the flow objective, the
masked policy, the multi-round loop -- has bought nothing, and the honest report
is that a regression was enough. Beating a genetic algorithm while losing to
ridge regression on a random sample would be a hollow result.

Single-shot by design, multi-round by this harness
--------------------------------------------------

MLDE is a *two-stage* method: one training sample, one prediction, done. This
repository's campaign loop instead calls ``propose`` and ``observe`` once per
round, so this implementation refits on the accumulated measurements every round
and proposes against the refreshed model. That is a deviation and must be
reported as one -- iterating the fit makes it closer to CLADE (Qiu & Wei, 2021)
or to ftMLDE than to MLDE as published, and it can only help the baseline, which
is the right direction for a deviation in a baseline to run.

The budget arithmetic is worth stating too. The published protocol is 384
training variants plus a top-96 plate, 480 assays in all, and 384 is *exactly*
the four-plate budget this repository defaults to. Using the published training
size verbatim would therefore spend the entire default campaign before the model
made a single design. The default here is one plate of random screening; pass
``training_size=PUBLISHED_TRAINING_SIZE`` to reproduce the paper's split under a
budget that can afford it.

The model, and what was traded away
-----------------------------------

Wittmann et al. train an ensemble spanning 22 model classes -- Keras MLPs and
CNNs, XGBoost, and a spread of scikit-learn linear and kernel models -- and
average the best few. Reproducing that would mean three new dependencies for one
baseline, so what runs here is kernel ridge regression with a degree-2
homogeneous polynomial kernel over one-hot features. That choice is ours, and the
reasoning is that the degree-2 feature space is exactly the space of
position-pair interactions: pairwise epistasis is what MLDE exists to capture,
and the dual form gets it without ever forming the ``(L·V)²`` features. The
ensemble's remaining benefit is variance reduction across model classes, which
matters least where all the members are fitting the same few hundred points.

The other adaptation is the candidate set. Wittmann's library is a four-site
combinatorial one -- 160,000 variants, exhaustively scorable. The environments
here reach ``10^13`` designs and upwards, so a random pool stands in for
exhaustive enumeration, and the model ranks that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines._values import single_objective
from evoflownet.algorithms.baselines.mutagenesis import RandomMutagenesis

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Fitness, Tokens
    from evoflownet.env.mutation import MutationEnvironment

#: Training variants in Wittmann et al.'s published protocol.
PUBLISHED_TRAINING_SIZE = 384

#: Variants in the plate their fitted model then designs.
PUBLISHED_BATCH_SIZE = 96

#: One 96-well plate, and this package's default round. Used as the training
#: sample size because the published 384 is the whole default campaign budget.
DEFAULT_TRAINING_SIZE = 96

#: Fewest measurements worth fitting. A kernel ridge on one point predicts that
#: point's value everywhere, which is a constant ranking and no ranking at all.
_MIN_TRAINING = 2

#: Rows of the candidate pool compared against the training set at a time. The
#: kernel is an agreement count over positions, so the intermediate is
#: ``(chunk, n_train, length)`` booleans; chunking bounds that at tens of MB
#: instead of letting it scale with the pool.
_KERNEL_CHUNK = 256


class MLDE(Sampler):
    """Fits a regressor to a random sample, then proposes its top predictions.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        training_size: Measurements gathered at random before the model takes
            over. Defaults to one plate rather than Wittmann et al.'s 384; see
            the module docstring for why, and pass
            :data:`PUBLISHED_TRAINING_SIZE` for the published split.
        pool_multiplier: Candidates generated per candidate returned. Our
            choice: the published method ranks an exhaustive library, and this
            is how much of an unenumerable one the model gets to rank instead.
        ridge_alpha: Ridge penalty. Our choice of 1.0, which is meaningful only
            because the kernel is normalised to ``[0, 1]``; with a few hundred
            points in a space this size, under-regularising is the failure mode.
        kernel_degree: Degree of the polynomial kernel. 2 is the pairwise
            epistasis model; 1 is purely additive, and is the ablation that says
            how much of MLDE's advantage needs interactions at all.
        feasible_only: Draw only constructible candidates.
        max_attempts: Draws before giving up on filling the pool.
        seed: Seeds the random training sample and the candidate pool.

    Raises:
        ValueError: If a size is not positive, the ridge penalty is negative, or
            the kernel degree is below 1.
    """

    def __init__(  # noqa: PLR0913 - the protocol is a training split plus a model
        self,
        env: MutationEnvironment,
        *,
        training_size: int = DEFAULT_TRAINING_SIZE,
        pool_multiplier: int = 4,
        ridge_alpha: float = 1.0,
        kernel_degree: int = 2,
        feasible_only: bool = False,
        max_attempts: int = 10,
        seed: int = 0,
    ) -> None:
        """Start in the random-screening stage, with nothing fitted."""
        super().__init__()
        for label, value in [
            ("training_size", training_size),
            ("pool_multiplier", pool_multiplier),
            ("max_attempts", max_attempts),
        ]:
            if value < 1:
                raise ValueError(f"{label} must be at least 1, got {value}")
        if ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must not be negative, got {ridge_alpha}")
        if kernel_degree < 1:
            raise ValueError(f"kernel_degree must be at least 1, got {kernel_degree}")

        self._env = env
        self._training_size = training_size
        self._pool_multiplier = pool_multiplier
        self._ridge_alpha = ridge_alpha
        self._kernel_degree = kernel_degree
        self._max_attempts = max_attempts
        self._feasible_only = feasible_only
        # The random stage *is* random mutagenesis -- "sample the library
        # uniformly" is what the protocol says -- so it is the same object
        # rather than a second copy of the same sampling code.
        self._explorer = RandomMutagenesis(env, feasible_only=feasible_only, seed=seed)

        self._sequences: list[Tokens] = []
        self._values: list[float] = []
        self._measured: set[bytes] = set()
        self._fitted: Tokens | None = None
        self._dual = np.zeros(0, dtype=np.float64)
        self._offset = 0.0
        self._stale = True

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "MLDE" + (" (feasible)" if self._feasible_only else "")

    @property
    def is_fitted(self) -> bool:
        """Whether the model has taken over from random screening."""
        return self._fitted is not None

    @property
    def training_examples(self) -> int:
        """Measurements gathered so far."""
        return len(self._values)

    def propose(self, n: int) -> Tokens:
        """Screen at random, or return the model's ``n`` best predictions.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array, ranked best-first once the model
            is fitted so that a caller taking a prefix takes the top designs.

        Raises:
            RuntimeError: If ``feasible_only`` and no feasible candidate can be
                drawn; raised by the underlying random draw rather than here.
        """
        if not self._ready():
            return self._draw(n)

        self._refit()
        pool = self._pool(n)
        predictions = self._predict(pool)
        order = np.argsort(-predictions, kind="stable")[:n]
        return pool[order]

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Add measurements to the training set and mark the model out of date.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which the
                single-output surrogate has no target to regress on.
        """
        flat = single_objective(values)
        rows = np.ascontiguousarray(np.asarray(sequences))
        for row, value in zip(rows, flat, strict=False):
            self._measured.add(row.tobytes())
            if not np.isfinite(value):
                # A failed or infeasible assay carries no fitness to regress on.
                # Keeping it in `_measured` still stops it being proposed again.
                continue
            self._sequences.append(row.copy())
            self._values.append(float(value))
        self._stale = True

    def _ready(self) -> bool:
        """Whether enough has been measured to hand over to the model."""
        return len(self._values) >= max(self._training_size, _MIN_TRAINING)

    def _draw(self, n: int) -> Tokens:
        """Random library members, charging their cost to this sampler.

        The delta is taken from the explorer's own counter so that rejection
        sampling under ``feasible_only`` is charged here too, rather than
        vanishing into a nested object nobody reports on.
        """
        before = self._explorer.proposals_made
        drawn = self._explorer.propose(n)
        self._count(self._explorer.proposals_made - before)
        return drawn

    def _pool(self, n: int) -> Tokens:
        """Candidates for the model to rank, excluding anything already assayed.

        Returns:
            At least ``n`` distinct unmeasured sequences where the environment
            can supply them, and a plain random draw where it cannot -- a short
            plate would be a worse answer than a repeated one.
        """
        wanted = n * self._pool_multiplier
        collected: list[Tokens] = []
        seen: set[bytes] = set()
        found = 0
        for _ in range(self._max_attempts):
            batch = np.ascontiguousarray(self._draw(wanted))
            keep = []
            for position, row in enumerate(batch):
                key = row.tobytes()
                if key in seen or key in self._measured:
                    continue
                seen.add(key)
                keep.append(position)
            if keep:
                collected.append(batch[keep])
                found += len(keep)
            if found >= n:
                break
        if found < n:
            # The reachable set is nearly exhausted, or nearly all of it has
            # been assayed. Ranking is meaningless at that point; fall back to
            # the null so the round still fills.
            return self._draw(n)
        return np.concatenate(collected)

    def _refit(self) -> None:
        """Solve the kernel ridge system on everything measured, if it changed."""
        if not self._stale:
            return
        X = np.stack(self._sequences)
        y = np.asarray(self._values, dtype=np.float64)
        # Centring removes the intercept, which must not be penalised: a ridge
        # that shrinks the mean pulls every prediction toward zero, which on a
        # landscape with a large offset is most of the signal.
        self._offset = float(y.mean())
        gram = self._kernel(X, X)
        gram[np.diag_indices_from(gram)] += self._ridge_alpha
        self._dual = np.linalg.solve(gram, y - self._offset)
        self._fitted = X
        self._stale = False

    def _predict(self, sequences: Tokens) -> npt.NDArray[np.float64]:
        """Predicted objective value for each sequence.

        Returns:
            An ``(n,)`` array.
        """
        if self._fitted is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("MLDE predicted before it was fitted")
        predictions: npt.NDArray[np.float64] = (
            self._kernel(sequences, self._fitted) @ self._dual + self._offset
        )
        return predictions

    def _kernel(self, left: Tokens, right: Tokens) -> npt.NDArray[np.float64]:
        """Polynomial kernel over one-hot encodings, computed without the encoding.

        The inner product of two one-hot sequences is the number of positions at
        which they agree, so the Gram matrix is an agreement count and the
        ``(L·V)``-dimensional features are never built. Normalising by the length
        keeps the kernel in ``[0, 1]``, which is what makes a single default
        ridge penalty meaningful across sequence lengths.

        Returns:
            An ``(len(left), len(right))`` array.
        """
        length = self._env.sequence_length
        gram = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
        for start in range(0, left.shape[0], _KERNEL_CHUNK):
            block = left[start : start + _KERNEL_CHUNK]
            agreement = (block[:, None, :] == right[None, :, :]).sum(axis=2)
            gram[start : start + block.shape[0]] = agreement / length
        return gram**self._kernel_degree
