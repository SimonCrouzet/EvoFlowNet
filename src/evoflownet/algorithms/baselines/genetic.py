"""The genetic algorithm: the baseline this project has to beat.

Directed evolution *is* a genetic algorithm, so this is not a strawman to clear
but the incumbent. Two results make that concrete. On PMO -- the field's own
sample-efficiency benchmark -- a vanilla GFlowNet scores 9.93 against Mol GA's
15.69, a ~58% deficit, and the GFlowNet only overtakes it (16.21) by absorbing a
GA. And the Ehrlich functions this package benchmarks on were introduced with a
tuned GA as their baseline and no GFlowNet evaluated at all.

Defaults follow Stanton et al.'s reported settings -- ``p_m = 1/L``,
``p_r = 1/L`` -- and holo-bench's ``DiscreteEvolution``, so the comparison is
against the configuration its authors chose rather than one convenient to us.

Rejection sampling and the feasibility claim
--------------------------------------------

:class:`GeneticAlgorithm` accepts ``feasible_only``, which resamples offspring
until they satisfy the environment's constraint. This exists to make the
feasibility claim falsifiable.

A masked policy is feasible by construction; the interesting question is whether
that is an *advantage*, and rejection sampling is the control that decides it. A
rejection GA wastes no oracle calls -- it discards before evaluating -- so at
equal oracle budget it may well match a masked policy. What it burns instead is
*proposals*, and that cost grows as the feasible fraction falls.

So the honest framing is: if rejection keeps up, the masking advantage is one of
proposal cost rather than sample efficiency, and should be reported that way.
:attr:`~evoflownet.algorithms.base.Sampler.proposals_made` is what makes the
difference visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines._values import single_objective

if TYPE_CHECKING:
    from evoflownet.core.types import Fitness, Tokens
    from evoflownet.env.mutation import MutationEnvironment


class GeneticAlgorithm(Sampler):
    """Population-based search by mutation and recombination.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        population_size: Individuals carried between generations.
        mutation_prob: Per-position probability of substitution. Defaults to
            ``1/L``, Stanton et al.'s setting.
        recombine_prob: Per-position probability of taking the other parent's
            token during crossover. Defaults to ``1/L``.
        survival_quantile: Fraction of the population retained as parents each
            generation. The default of 0.25 is our choice, not a published
            value -- holo-bench uses 0.01, which at our population sizes leaves
            too few parents to recombine. Stated because the argument beside it
            *does* carry its authors' value, and a reader is entitled to know
            which is which.
        feasible_only: Resample offspring until constructible. The control for
            the feasibility claim; see the module docstring.
        max_attempts: Resampling rounds before giving up when
            ``feasible_only``.
        seed: Seeds the population and the operators.

    Raises:
        ValueError: If a probability is outside ``[0, 1]`` or a size is not
            positive.
    """

    def __init__(  # noqa: PLR0913 - a GA is defined by its operators' rates
        self,
        env: MutationEnvironment,
        *,
        population_size: int = 256,
        mutation_prob: float | None = None,
        recombine_prob: float | None = None,
        survival_quantile: float = 0.25,
        feasible_only: bool = False,
        max_attempts: int = 50,
        seed: int = 0,
    ) -> None:
        """Seed the population from the parent."""
        super().__init__()
        length = env.sequence_length
        self._env = env
        self._population_size = population_size
        self._mutation_prob = 1.0 / length if mutation_prob is None else mutation_prob
        self._recombine_prob = 1.0 / length if recombine_prob is None else recombine_prob
        self._survival_quantile = survival_quantile
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts

        if population_size < 1:
            raise ValueError(f"population_size must be at least 1, got {population_size}")
        for label, value in [
            ("mutation_prob", self._mutation_prob),
            ("recombine_prob", self._recombine_prob),
            ("survival_quantile", survival_quantile),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")

        self._rng = np.random.default_rng(seed)
        self._population = np.tile(env.parent, (population_size, 1))
        self._fitness = np.full(population_size, -np.inf)

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "GeneticAlgorithm" + (" (rejection)" if self._feasible_only else "")

    @property
    def population(self) -> Tokens:
        """The current population."""
        return self._population.copy()

    def propose(self, n: int) -> Tokens:
        """Generate ``n`` offspring from the surviving population.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.

        Raises:
            RuntimeError: If ``feasible_only`` and the attempt budget is spent
                before enough feasible offspring are found. Returning infeasible
                designs silently would corrupt the comparison this exists for.
        """
        parents = self._survivors()
        if not self._feasible_only:
            offspring = self._breed(parents, n)
            self._count(n)
            return offspring

        collected: list[Tokens] = []
        found = 0
        for _ in range(self._max_attempts):
            batch = self._breed(parents, n)
            self._count(n)
            keep = batch[self._env.is_reachable(batch)]
            if keep.shape[0]:
                collected.append(keep)
                found += keep.shape[0]
            if found >= n:
                return np.concatenate(collected)[:n]
        raise RuntimeError(
            f"could not breed {n} feasible offspring in {self._max_attempts} attempts "
            f"({found} found); rejection sampling has become impractical at this "
            f"feasible density, which is itself the result"
        )

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Merge evaluated offspring into the population, keeping the best.

        Selection is over parents and offspring together, so a generation can
        never lose ground -- the elitism holo-bench's implementation uses.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                selection no single order to keep the best by.
        """
        flat = single_objective(values)
        combined = np.concatenate([self._population, np.asarray(sequences)])
        scores = np.concatenate([self._fitness, flat])
        # -inf sorts last, so unevaluated founders are displaced by anything real.
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[: self._population_size]
        self._population = combined[order]
        self._fitness = scores[order]

    def _survivors(self) -> Tokens:
        """The top fraction of the population, or all of it if none are scored."""
        count = max(2, int(self._population_size * self._survival_quantile))
        if not np.isfinite(self._fitness).any():
            return self._population[:count]
        order = np.argsort(-self._fitness)[:count]
        return self._population[order]

    def _breed(self, parents: Tokens, n: int) -> Tokens:
        """Produce ``n`` offspring by recombination then mutation."""
        length = self._env.sequence_length
        size = self._env.alphabet.size
        wild_type = self._env.parent

        first = parents[self._rng.integers(0, parents.shape[0], size=n)]
        second = parents[self._rng.integers(0, parents.shape[0], size=n)]
        take_second = self._rng.random((n, length)) < self._recombine_prob
        offspring = np.where(take_second, second, first)

        mutate = self._rng.random((n, length)) < self._mutation_prob
        # Draw from the other tokens rather than from all of them. Sampling
        # uniformly over the whole alphabet redraws the token already present
        # one time in V, so the realised substitution rate would be
        # p_m * (V - 1) / V -- a 5% shortfall at the protein alphabet, which
        # would mean running Stanton et al.'s hyperparameters at 95% of their
        # published value while reporting them as the published value.
        drawn = self._rng.integers(0, size - 1, size=(n, length))
        replacements = drawn + (drawn >= offspring)
        offspring = np.where(mutate, replacements, offspring)

        # The environment admits at most max_mutations differences from the
        # parent. Offspring beyond that are outside its graph, so revert the
        # excess rather than emit sequences no sampler could have produced.
        return self._enforce_budget(offspring, wild_type)

    def _enforce_budget(self, offspring: Tokens, wild_type: Tokens) -> Tokens:
        """Revert surplus mutations so every offspring is inside the graph."""
        budget = self._env.max_mutations
        differing = offspring != wild_type[None, :]
        counts = differing.sum(axis=1)
        for row in np.flatnonzero(counts > budget):
            positions = np.flatnonzero(differing[row])
            surplus = self._rng.choice(positions, size=int(counts[row] - budget), replace=False)
            offspring[row, surplus] = wild_type[surplus]
        return offspring
