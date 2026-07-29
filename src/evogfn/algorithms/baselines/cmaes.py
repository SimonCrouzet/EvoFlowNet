"""CMA-ES: the classical method that adapts a *distribution* rather than a set.

Every other baseline here carries points -- a population, a current design, a
training set. CMA-ES (Hansen & Ostermeier, *Evol. Comput.* 2001) carries a
probability distribution and updates its shape from the ranking of what it drew.
That makes it the closest thing in classical optimisation to what a GFlowNet
does, and therefore the baseline that tests the part of the claim the others
cannot reach: adapting a distribution over sequences is not itself novel, and a
method whose advantage is "it learns a sampler" has to beat the fifty-year-old
method that also learns a sampler.

What it does not do is sample proportionally to reward. CMA-ES contracts onto a
single optimum by construction -- that is the point of the step-size control --
so it should win on best-found and lose badly on diversity and on distributional
distance. If it does not lose on those, the sampling claim is in trouble.

Discrete sequences via a continuous relaxation
----------------------------------------------

CMA-ES optimises in ``R^d``. The standard adaptation to categorical variables is
to relax: carry a real matrix of shape ``(length, vocabulary)``, treat it as
per-position logits, and read a sequence off it by taking the argmax at each
position. The Gaussian that CMA-ES maintains supplies the exploration, so no
extra sampling temperature is needed -- a position whose logits are close
together flips between tokens under the noise, and one whose logits have
separated stops flipping. The relaxation is where the method's assumptions are
weakest and is worth stating plainly: rank information is fed back into a
Gaussian over logits, and a Gaussian over logits is not a natural model of an
epistatic landscape.

Why the covariance is diagonal
------------------------------

The relaxation has ``d = length * vocabulary`` dimensions, which is 5,120 for the
``L = 256`` protein sequences this package benchmarks on. A full covariance
matrix is then 26 million entries, and CMA-ES needs its eigendecomposition,
costing ``O(d^3) ~ 10^11`` operations per update. That is not a tuning
inconvenience, it is intractable, and it is intractable for the honest reason
that ``d`` is large rather than because of anything about this codebase.

So this is the **separable** variant, sep-CMA-ES (Ros & Hansen, *PPSN* 2008):
the covariance is constrained to be diagonal, its square root is then elementwise
and free, and the learning rates for the rank-one and rank-mu updates are
multiplied by ``(d + 2) / 3`` as that paper prescribes, since a diagonal matrix
has ``d`` rather than ``d(d+1)/2`` parameters to estimate and can afford to move
faster. The cost is real: sep-CMA-ES cannot learn correlations between
coordinates, so it cannot represent epistasis between two positions in its search
distribution. It compensates only in the sense that it reaches a good diagonal
much faster.

Everything else -- the recombination weights, the two evolution paths, the
cumulative step-size adaptation -- follows Hansen's tutorial (arXiv:1604.00772)
with its published default constants, so the baseline is the configuration its
author chose rather than one convenient to us.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._values import single_objective

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment

#: Fewest matched measurements that will move the distribution. With one sample
#: the recombination is a copy and the rank-mu update is empty, so the generation
#: carries no information and updating on it only inflates the step size.
_MIN_FOR_UPDATE = 2

#: Bounds on the step size. CMA-ES diverges or collapses on a pathological
#: ranking, and on a landscape returning -inf for infeasible designs pathological
#: rankings happen. Clamping keeps a bad round from producing NaNs that would
#: silently poison every later round of the campaign.
_SIGMA_FLOOR = 1e-12
_SIGMA_CEILING = 1e12

#: Smallest variance a coordinate may hold. At zero it could never move again,
#: and the elementwise whitening below would divide by it.
_VARIANCE_FLOOR = 1e-20


class CMAES(Sampler):
    """Separable CMA-ES over a continuous relaxation of the sequence space.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        initial_sigma: Starting step size. Hansen advises roughly a third of the
            search domain's width, but logits have no natural width, so 1.0 is
            our choice: it makes the initial argmax uniform over the alphabet,
            and the step-size adaptation corrects the scale within a few
            generations anyway.
        feasible_only: Resample until every proposal is constructible.
        max_attempts: Resampling rounds before giving up when ``feasible_only``.
        seed: Seeds the Gaussian.

    Raises:
        ValueError: If ``initial_sigma`` is not positive.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        initial_sigma: float = 1.0,
        feasible_only: bool = False,
        max_attempts: int = 50,
        seed: int = 0,
    ) -> None:
        """Start the distribution uniform over the alphabet at every position."""
        super().__init__()
        if initial_sigma <= 0.0:
            raise ValueError(f"initial_sigma must be positive, got {initial_sigma}")

        self._env = env
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts
        self._rng = np.random.default_rng(seed)

        self._dimension = env.sequence_length * env.alphabet.size
        # A zero mean is the uniform categorical at every position, which is the
        # least committed start available and needs no arbitrary bias toward the
        # parent -- the mutation budget already anchors every sample to it.
        self._mean = np.zeros(self._dimension, dtype=np.float64)
        self._diagonal = np.ones(self._dimension, dtype=np.float64)
        self._sigma = float(initial_sigma)
        self._path_sigma = np.zeros(self._dimension, dtype=np.float64)
        self._path_c = np.zeros(self._dimension, dtype=np.float64)
        self._generation = 0
        # E||N(0, I)||, Hansen's series approximation. Used as the reference the
        # step-size controller compares the evolution path's length against.
        self._expected_norm = math.sqrt(self._dimension) * (
            1.0 - 1.0 / (4.0 * self._dimension) + 1.0 / (21.0 * self._dimension**2)
        )

        # The last batch handed out, so `observe` can find the Gaussian draw a
        # returned sequence came from. Keyed by sequence rather than by row
        # index because the harness scores a *selected subset* of the proposals,
        # in its own order; assuming alignment would attribute each score to
        # some other candidate's draw and update the distribution with noise.
        self._samples = np.zeros((0, self._dimension), dtype=np.float64)
        self._index: dict[bytes, int] = {}

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "CMAES" + (" (feasible)" if self._feasible_only else "")

    @property
    def sigma(self) -> float:
        """Current step size."""
        return self._sigma

    @property
    def mean_logits(self) -> npt.NDArray[np.float64]:
        """The distribution's mean, shaped ``(sequence_length, vocabulary)``."""
        reshaped: npt.NDArray[np.float64] = self._mean.reshape(
            self._env.sequence_length, self._env.alphabet.size
        ).copy()
        return reshaped

    def propose(self, n: int) -> Tokens:
        """Draw ``n`` sequences from the current search distribution.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.

        Raises:
            RuntimeError: If ``feasible_only`` and the attempt budget is spent
                before enough feasible candidates are found. Returning
                infeasible designs silently would corrupt the comparison this
                exists for.
        """
        if not self._feasible_only:
            sequences, draws = self._sample(n)
            self._count(n)
            self._remember(sequences, draws)
            return sequences

        kept_sequences: list[Tokens] = []
        kept_draws: list[npt.NDArray[np.float64]] = []
        found = 0
        for _ in range(self._max_attempts):
            sequences, draws = self._sample(n)
            self._count(n)
            reachable = self._env.is_reachable(sequences)
            if reachable.any():
                kept_sequences.append(sequences[reachable])
                kept_draws.append(draws[reachable])
                found += int(reachable.sum())
            if found >= n:
                chosen = np.concatenate(kept_sequences)[:n]
                self._remember(chosen, np.concatenate(kept_draws)[:n])
                return chosen
        raise RuntimeError(
            f"could not draw {n} feasible candidates in {self._max_attempts} attempts "
            f"({found} found); rejection sampling has become impractical at this "
            f"feasible density, which is itself the result"
        )

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Move the mean, the evolution paths, the covariance and the step size.

        Only candidates traceable to the most recent :meth:`propose` contribute:
        CMA-ES updates from the *Gaussian draws* behind the ranking, not from the
        sequences, and a sequence from anywhere else has no draw behind it.
        A batch with fewer than two such candidates leaves the distribution
        untouched rather than updating it on a ranking of one.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                the rank-based update no single ordering to work from.
        """
        rows, scores = self._matched(sequences, values)
        if rows.size < _MIN_FOR_UPDATE:
            return

        order = np.argsort(-scores, kind="stable")
        selected = self._samples[rows[order]]
        weights, mu_eff = self._recombination_weights(rows.size)
        selected = selected[: weights.size]

        d = self._dimension
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        damping = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff),
        )
        # Ros & Hansen's separable speed-up: a diagonal covariance has d free
        # parameters rather than d(d+1)/2, so it may be learned this much faster.
        c_1, c_mu = self._separable_rates(c_1, c_mu)

        self._generation += 1
        step = weights @ selected  # the weighted recombination, in y-space
        self._mean = self._mean + self._sigma * step

        # With a diagonal covariance the whitening C^(-1/2) is elementwise, which
        # is the entire computational reason for the separable variant.
        whitened = step / np.sqrt(self._diagonal)
        self._path_sigma = (1.0 - c_sigma) * self._path_sigma + math.sqrt(
            c_sigma * (2.0 - c_sigma) * mu_eff
        ) * whitened

        path_norm = float(np.linalg.norm(self._path_sigma))
        correction = math.sqrt(max(1e-16, 1.0 - (1.0 - c_sigma) ** (2 * self._generation)))
        # Hansen's h_sigma: stall the rank-one update when the path has grown
        # unusually long, which is the signature of a step size still adapting
        # rather than of a genuine direction worth recording.
        h_sigma = path_norm / correction < (1.4 + 2.0 / (d + 1.0)) * self._expected_norm

        gain = math.sqrt(c_c * (2.0 - c_c) * mu_eff) if h_sigma else 0.0
        self._path_c = (1.0 - c_c) * self._path_c + gain * step
        leak = 0.0 if h_sigma else c_c * (2.0 - c_c)
        rank_mu = weights @ (selected**2)
        self._diagonal = (
            (1.0 - c_1 - c_mu + c_1 * leak) * self._diagonal
            + c_1 * self._path_c**2
            + c_mu * rank_mu
        )
        self._diagonal = np.maximum(self._diagonal, _VARIANCE_FLOOR)

        self._sigma = float(
            np.clip(
                self._sigma
                * math.exp((c_sigma / damping) * (path_norm / self._expected_norm - 1.0)),
                _SIGMA_FLOOR,
                _SIGMA_CEILING,
            )
        )

    def _sample(self, n: int) -> tuple[Tokens, npt.NDArray[np.float64]]:
        """Draw ``n`` points and decode them.

        Returns:
            The decoded sequences and the ``(n, dimension)`` array of draws in
            ``y``-space -- that is, ``(x - mean) / sigma``, which is the form
            every CMA-ES update equation is written in.
        """
        z = self._rng.standard_normal((n, self._dimension))
        y = z * np.sqrt(self._diagonal)[None, :]
        x = self._mean[None, :] + self._sigma * y
        return self._decode(x), y

    def _decode(self, x: npt.NDArray[np.float64]) -> Tokens:
        """Read sequences off the relaxation, projected onto the mutation budget.

        Returns:
            An ``(n, sequence_length)`` array inside the environment's graph.
        """
        parent = self._env.parent
        length = self._env.sequence_length
        size = self._env.alphabet.size
        logits = x.reshape(x.shape[0], length, size)

        chosen = np.asarray(logits.argmax(axis=2), dtype=parent.dtype)
        differing = chosen != parent[None, :]
        counts = differing.sum(axis=1)

        # Confidence that the substitution beats keeping the parent's token.
        # Projecting onto the budget by reverting the *least* confident
        # substitutions keeps the ones the distribution actually asked for; a
        # random projection would discard the search's own signal.
        preference = np.take_along_axis(logits, chosen[:, :, None], axis=2)[:, :, 0]
        margin = preference - logits[:, np.arange(length), parent]

        budget = self._env.max_mutations
        for row in np.flatnonzero(counts > budget):
            positions = np.flatnonzero(differing[row])
            weakest = positions[np.argsort(-margin[row, positions], kind="stable")[budget:]]
            chosen[row, weakest] = parent[weakest]
        return chosen

    def _remember(self, sequences: Tokens, draws: npt.NDArray[np.float64]) -> None:
        """Record which draw produced which sequence, for the next ``observe``."""
        self._samples = draws
        contiguous = np.ascontiguousarray(sequences)
        # Duplicates collapse onto one draw. Two draws decoding to the same
        # sequence are indistinguishable to the oracle, so attributing the score
        # to either is equally defensible and neither biases the ranking.
        self._index = {row.tobytes(): position for position, row in enumerate(contiguous)}

    def _matched(
        self, sequences: Tokens, values: Fitness
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """Line up scored sequences with the draws they came from.

        Returns:
            The sample indices and their finite objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                the rank-based update no single ordering to work from.
        """
        flat = single_objective(values)
        contiguous = np.ascontiguousarray(np.asarray(sequences))
        rows: list[int] = []
        scores: list[float] = []
        for row, value in zip(contiguous, flat, strict=False):
            position = self._index.get(row.tobytes())
            if position is None or not np.isfinite(value):
                continue
            rows.append(position)
            scores.append(float(value))
        return np.asarray(rows, dtype=np.intp), np.asarray(scores, dtype=np.float64)

    def _separable_rates(self, c_1: float, c_mu: float) -> tuple[float, float]:
        """Scale the covariance learning rates for a diagonal matrix.

        Returns:
            The scaled ``(c_1, c_mu)``, jointly capped at 1 so the covariance
            update stays a convex combination.
        """
        factor = (self._dimension + 2.0) / 3.0
        c_1, c_mu = c_1 * factor, c_mu * factor
        total = c_1 + c_mu
        if total > 1.0:
            c_1, c_mu = c_1 / total, c_mu / total
        return c_1, c_mu

    @staticmethod
    def _recombination_weights(n: int) -> tuple[npt.NDArray[np.float64], float]:
        """Hansen's default log-decreasing weights over the better half.

        Args:
            n: Candidates whose scores came back, which plays the role of the
                population size ``lambda``. It is read from the batch rather
                than fixed in the constructor because the harness decides how
                many designs a round screens.

        Returns:
            The normalised weights and the variance-effective selection mass
            ``mu_eff = 1 / sum(w^2)``.
        """
        mu = max(1, n // 2)
        raw = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        weights = raw / raw.sum()
        return weights, float(1.0 / np.sum(weights**2))
