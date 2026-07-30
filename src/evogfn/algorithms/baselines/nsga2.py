r"""NSGA-II: the multi-objective baseline a Pareto method has to beat.

Deb, Pratap, Agarwal and Meyarivan, *A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II*, IEEE Transactions on Evolutionary Computation 6(2),
182--197, 2002.

[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm] is the
incumbent for single-objective work, and this is its multi-objective
counterpart. Without it,
[CH65Landscape][evogfn.landscapes.ch65.CH65Landscape] and
[MultiEhrlichLandscape][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape]
have a hypervolume number and nothing to read it against: a preference-conditioned
GFlowNet that beats *no published multi-objective method* has not been shown to
beat anything. NSGA-II is the method the field actually reaches for, is still the
default in pymoo, DEAP and platypus, and has no learned component -- so a
difference against it is a difference attributable to learning.

## What makes it multi-objective rather than scalarised

Every other baseline in this package collapses the objective vector to one number
before it can rank anything, which is why they all route through
`evogfn.algorithms.baselines._values.single_objective` and refuse a
multi-objective array. NSGA-II does not: it ranks by *Pareto dominance*, so it
never picks a trade-off. Two designs that each win on one objective are equally
good to it, and the tie is broken by which of them sits in a lonelier part of
objective space. That is the whole content of the method, and it is delivered by
two pieces:

* **Fast non-dominated sorting** assigns every design the index of the front it
  belongs to -- rank 0 is non-dominated, rank 1 is non-dominated once rank 0 is
  removed, and so on.
* **Crowding distance** measures how much room a design has along each objective
  inside its own front, and breaks ties in favour of the sparser region.

Selection is a binary tournament on the pair `(rank, crowding)`, and survival is
elitist $(\mu + \lambda)$: parents and offspring are ranked together and the best
$N$ carried forward, so a generation can never lose ground.

## Two places implementations quietly go wrong

**Non-dominated sorting.** The obvious way to build the fronts is to call a
dominance filter, delete what it returns, and call it again. That gives the right
fronts, but at $O(MN^3)$ -- the cost the paper's title is about. Deb et al.'s
algorithm instead computes, in one pass, the set $S_p$ of designs each $p$
dominates and the count $n_p$ of designs that dominate it, then peels fronts by
*decrementing* $n_q$ as dominators retire. That is $O(MN^2)$, and it is what
[fast_non_dominated_sort][evogfn.algorithms.baselines.nsga2.fast_non_dominated_sort]
implements. On a combined parent-plus-offspring population this runs once per
round; the cubic version is what turns a large population from a parameter into a
budget.

**Crowding distance at the boundary.** The paper assigns *infinite* distance to
the best and worst design on **each objective separately**, so the extremes of a
front always survive truncation. Drop that -- or apply it only on the first
objective -- and the front erodes inwards from its tips over generations, which is
not a crash but a slow, plausible-looking loss of exactly the spread the
indicator you are reporting rewards. It is the failure hardest to notice and
easiest to introduce, so it is tested directly.

## Hyperparameters: whose is whose

Deb et al. report their settings in Section V-B. Where a value below is theirs it
says so; where it is ours it says that instead, because a reader is entitled to
know which numbers came from the paper.

**Theirs.** Population size $N = 100$, run for 250 generations to spend the
25,000 function evaluations every method in the paper is given. Selection by
*binary* tournament, which is part of the method rather than a knob. Crossover
probability $p_c = 0.9$ -- the chance a mating pair is recombined at all.
Mutation probability $p_m = 1/\ell$ for binary-coded GAs, which is $1/L$ here and
happens to coincide with Stanton et al.'s setting for
[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm], so the
two baselines mutate at the same rate without either borrowing from the other.

**Ours.** The per-position mixing rate of $1/L$, applied once a pair has been
selected for crossover. Deb et al. use single-point crossover on a bit string;
this package's recombination is the uniform per-position operator of
[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm], and the
rate is that class's default. Stated because everything beside it *is* the
paper's, and a reader is entitled to know which is which.

The variation operators are otherwise identical to
[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm] --
deliberately, and the duplication is the point. The two baselines then differ
*only* in selection and survival, so a gap measured between them is attributable
to the multi-objective ranking rather than to a different mutation rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._reanchor import reprojected

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment

#: Population size $N$ in Deb et al.'s own simulations: 100 individuals over 250
#: generations, for the 25,000 function evaluations every method in the paper is
#: given.
PUBLISHED_POPULATION_SIZE = 100

#: Crossover probability $p_c$ in the same simulations. This is the chance that a
#: mating *pair* is recombined at all, not a per-position rate.
PUBLISHED_CROSSOVER_PROB = 0.9

#: Contenders per tournament. NSGA-II is defined with a **binary** tournament on
#: the crowded-comparison operator, so this is part of the method rather than a
#: value to tune.
TOURNAMENT_SIZE = 2

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2

# Fronts of this size or smaller consist entirely of boundary solutions: with one
# or two members there is no interior to measure a gap across.
_ALL_BOUNDARY = 2


def fast_non_dominated_sort(values: Fitness) -> npt.NDArray[np.intp]:
    r"""Assign every design the index of the Pareto front it belongs to.

    Deb et al. (2002), Section III-A. Rank 0 is the non-dominated set; rank 1 is
    what would be non-dominated with rank 0 removed; and so on. Written for
    **maximisation**, matching the rest of this package: `a` dominates `b` when
    `a_i >= b_i` on every objective and `a_j > b_j` on at least one.

    The algorithm is the paper's, not a repeated dominance filter. Two passes:

    1. For every design `p`, the set `S_p` of designs it dominates and the count
       `n_p` of designs that dominate it. The paper writes this as a double loop
       over the population; the comparisons here are the same ones, evaluated as
       a single `(N, N, M)` broadcast.
    2. Front 0 is `{p : n_p == 0}`. Retiring a front decrements `n_q` for every
       `q` it dominated, and whatever reaches zero forms the next front.

    Step 2 is why this is $O(MN^2)$ rather than the $O(MN^3)$ of re-filtering the
    survivors once per front -- the distinction the paper's title is about. It is
    also what keeps the ranks consistent: a design enters a front exactly when its
    last remaining dominator has been retired, so no design can be ranked before
    something that beats it.

    Memory is $O(N^2)$ for the dominance matrix, which at the population sizes a
    campaign runs (hundreds) is kilobytes. It is not the routine to reach for on
    a set of $10^5$ designs.

    Args:
        values: An `(n, n_objectives)` array of objective values. Infinite values
            are allowed -- an infeasible design scoring `-inf` on every objective
            is dominated by anything finite, which is the correct behaviour.

    Returns:
        An `(n,)` array of zero-based front indices, aligned with the input rows.

    Raises:
        ValueError: If `values` is not a two-dimensional objective matrix or
            contains `nan`, where dominance is undefined.
    """
    points = _as_objective_matrix(values)
    n = points.shape[0]
    ranks = np.zeros(n, dtype=np.intp)
    if n == 0:
        return ranks

    # Row p, column q: does p dominate q?
    at_least_as_good = (points[:, None, :] >= points[None, :, :]).all(axis=2)
    strictly_better = (points[:, None, :] > points[None, :, :]).any(axis=2)
    dominates = at_least_as_good & strictly_better
    # n_p, the domination count: how many designs dominate p.
    remaining = dominates.sum(axis=0).astype(np.intp)

    current = np.flatnonzero(remaining == 0)
    rank = 0
    while current.size:
        ranks[current] = rank
        # Retire the front. A negative count can never return to zero, so a
        # design is ranked once and only once.
        remaining[current] = -1
        freed = dominates[current].sum(axis=0)
        remaining = np.where(remaining >= 0, remaining - freed, remaining)
        current = np.flatnonzero(remaining == 0)
        rank += 1
    return ranks


def crowding_distance(values: Fitness) -> npt.NDArray[np.float64]:
    r"""Measure how much room each design has inside its front.

    Deb et al. (2002), Section III-B. For each objective the front is sorted and
    every design is charged the normalised gap between its two neighbours; the
    contributions are summed over objectives. It estimates the perimeter of the
    largest cuboid enclosing a design and touching none of its neighbours, and it
    is what NSGA-II uses instead of a niche radius -- no sharing parameter, and
    the same $O(MN\log N)$ regardless of how the front is shaped.

    **The first and last design on each objective are set to infinity.** That is
    the paper's rule and it is load-bearing: the extremes of a front are what an
    indicator such as hypervolume or IGD+ is most sensitive to, and a truncation
    that can discard them erodes the front inwards from its tips over
    generations. The failure is gradual and produces no error, so it is asserted
    in the tests rather than trusted.

    Args:
        values: An `(n, n_objectives)` array holding **one front**. Passing a
            whole population instead is not an error here, but it is not the
            quantity NSGA-II selects on -- crowding is defined within a front.

    Returns:
        An `(n,)` array of distances. Larger means lonelier, therefore preferred.
        Every entry is `inf` when the front has two members or fewer, since then
        every member is a boundary on every objective.

    Raises:
        ValueError: If `values` is not a two-dimensional objective matrix or
            contains `nan`.
    """
    points = _as_objective_matrix(values)
    n = points.shape[0]
    if n <= _ALL_BOUNDARY:
        return np.full(n, np.inf, dtype=np.float64)

    distance = np.zeros(n, dtype=np.float64)
    for objective in range(points.shape[1]):
        order = np.argsort(points[:, objective], kind="stable")
        sorted_values = points[order, objective]

        # Per objective, and assigned rather than accumulated: a design that is
        # extreme on any one objective is a boundary solution and must survive.
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf

        # Read degeneracy off the endpoints rather than off their difference. A
        # constant objective separates nothing; an endpoint at -inf is what an
        # infeasible design scores, and normalising by the infinite spread that
        # produces would turn every interior gap into nan, which propagates into
        # the tournament and silently randomises selection. Both endpoints at
        # -inf -- a whole front of designs that scored nothing, which is one
        # round of infeasible proposals on a sparse landscape -- is worse still:
        # the *subtraction itself* is `-inf - -inf`, a numpy invalid-value
        # warning this project treats as an error, so the check has to come
        # before it rather than after. Either way this objective contributes
        # nothing to the interior, while the boundaries above stay infinite.
        low, high = float(sorted_values[0]), float(sorted_values[-1])
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            continue
        distance[order[1:-1]] += (sorted_values[2:] - sorted_values[:-2]) / (high - low)
    return distance


class NSGA2(Sampler):
    r"""Elitist multi-objective search by non-dominated sorting and crowding.

    Drives the same [Sampler][evogfn.algorithms.base.Sampler] interface as every
    other method here, so the campaign loop charges it for the same thing. One
    round is one generation: `propose` breeds offspring from the current
    population by binary tournament, `observe` merges parents and offspring and
    keeps the best `population_size` by `(rank, crowding)`.

    Unlike the single-objective baselines this class **requires** the objective
    matrix. It ranks by dominance, so it neither needs nor accepts a
    scalarisation -- handing it one would discard the trade-off it exists to
    explore. A single-objective array is accepted and degenerates to a
    dominance-ranked GA, which is the correct behaviour rather than a special
    case.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        population_size: Individuals carried between generations. Defaults to
            `PUBLISHED_POPULATION_SIZE`, Deb et al.'s $N = 100$.
        mutation_prob: Per-position probability of substitution. Defaults to
            $1/L$, which is both Deb et al.'s $p_m = 1/\ell$ for binary-coded
            GAs and Stanton et al.'s setting for
            [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm].
        crossover_prob: Probability that a mating pair is recombined at all.
            Defaults to `PUBLISHED_CROSSOVER_PROB`, Deb et al.'s $p_c = 0.9$.
        recombine_prob: Per-position probability of taking the second parent's
            token, given that the pair is recombined. Defaults to $1/L$. This
            one is **ours**, not the paper's: Deb et al. use single-point
            crossover on a bit string, and this is the uniform operator
            [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm]
            uses, at that class's rate, so the two baselines vary identically.
        carry_population: Keep the population, its objective vectors and its
            fronts when the campaign moves the anchor, rather than founding a
            fresh one on the new anchor. On by default; the trade is the one
            measured for
            [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.reanchored],
            and it is why the rebuild stays reachable through the hook rather
            than only through the campaign's factory fallback.
        feasible_only: Resample offspring until constructible, rather than
            projecting them back into the graph. The control for the feasibility
            claim; see
            [evogfn.algorithms.baselines.genetic][] for what it decides.
        max_attempts: Resampling rounds before giving up when `feasible_only`.
        seed: Seeds the population and the operators.

    Raises:
        ValueError: If a probability is outside `[0, 1]` or the population size
            is not positive.
    """

    def __init__(  # noqa: PLR0913 - an evolutionary algorithm is its operators' rates
        self,
        env: MutationEnvironment,
        *,
        population_size: int = PUBLISHED_POPULATION_SIZE,
        mutation_prob: float | None = None,
        crossover_prob: float = PUBLISHED_CROSSOVER_PROB,
        recombine_prob: float | None = None,
        carry_population: bool = True,
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
        self._crossover_prob = crossover_prob
        self._recombine_prob = 1.0 / length if recombine_prob is None else recombine_prob
        self._carry_population = carry_population
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts

        if population_size < 1:
            raise ValueError(f"population_size must be at least 1, got {population_size}")
        for label, value in [
            ("mutation_prob", self._mutation_prob),
            ("crossover_prob", self._crossover_prob),
            ("recombine_prob", self._recombine_prob),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")

        self._rng = np.random.default_rng(seed)
        self._population = np.tile(env.parent, (population_size, 1))
        # The objective count is not known until the first batch is scored, so
        # the values table cannot be allocated here. Until then selection has
        # nothing to rank on and draws parents uniformly, which is honest: the
        # founding population is the parent repeated and carries no information.
        self._values: npt.NDArray[np.float64] | None = None
        self._ranks = np.zeros(population_size, dtype=np.intp)
        self._crowding = np.full(population_size, np.inf, dtype=np.float64)

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "NSGA-II" + (" (rejection)" if self._feasible_only else "")

    @property
    def population(self) -> Tokens:
        """The current population."""
        return self._population.copy()

    def reanchored(self, env: MutationEnvironment) -> NSGA2:
        """Carry the population, its objective vectors and its fronts.

        The same argument as for
        [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.reanchored],
        and there is more at stake. A single-objective population carries one
        number per individual; an approximated Pareto front is a *shape* built
        up over generations, and discarding it on a rebuild throws away the
        spread as well as the quality -- which is precisely what the indicators
        this baseline exists to supply are measuring. A rebuilt NSGA-II reports
        the hypervolume of a population of identical copies of the anchor, and a
        method compared against that is compared against nothing.

        Individuals outside the new mutation budget are re-projected by
        [reprojected][evogfn.algorithms.baselines._reanchor.reprojected], and a
        re-projected individual is a different design, so its objective vector
        is dropped to ``-inf`` on every objective rather than carried onto a
        sequence it was not measured on. Ranks and crowding distances are then
        recomputed from what is left, because a front is a property of the
        population and both would otherwise describe the population that used to
        be there.

        Args:
            env: The re-anchored environment.

        Returns:
            An NSGA-II over ``env``, carrying this one's population and fronts.
            This one is not edited, though its random stream advances by the
            draws re-projection needed, as it does for
            [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.reanchored].
        """
        moved = NSGA2(
            env,
            population_size=self._population_size,
            mutation_prob=self._mutation_prob,
            crossover_prob=self._crossover_prob,
            recombine_prob=self._recombine_prob,
            carry_population=self._carry_population,
            feasible_only=self._feasible_only,
            max_attempts=self._max_attempts,
        )
        if not self._carry_population:
            moved._rng = self._rng
            moved._proposals_made = self._proposals_made
            return moved

        population, intact = reprojected(env, self._population, self._rng)
        moved._population = population
        values = None if self._values is None else self._values.copy()
        if values is not None:
            values[~intact] = -np.inf
        # A population with nothing measured left in it is the founding state,
        # and is left as the constructor made it: ranks over an all -inf table
        # are ties, and the crowding distance across a front whose every value
        # is -inf is `inf - inf`. Selection would then tournament on nan and
        # pick arbitrarily while looking like it was ranking.
        if values is not None and bool(np.isfinite(values).any()):
            moved._values = values
            moved._ranks = fast_non_dominated_sort(values)
            moved._crowding = moved._crowding_within_fronts(values, moved._ranks)
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made
        return moved

    @property
    def values(self) -> npt.NDArray[np.float64] | None:
        """Objective values of the current population, or `None` before scoring.

        Returns:
            An `(population_size, n_objectives)` array, or `None` if `observe`
            has not yet been called and the objective count is still unknown.
        """
        return None if self._values is None else self._values.copy()

    @property
    def ranks(self) -> npt.NDArray[np.intp]:
        """Front index of each population member, zero being non-dominated."""
        return self._ranks.copy()

    @property
    def crowding_distances(self) -> npt.NDArray[np.float64]:
        """Crowding distance of each population member, within its own front."""
        return self._crowding.copy()

    def front(self) -> Tokens:
        """The non-dominated members of the current population.

        This is what NSGA-II is for: the approximation of the Pareto set it has
        arrived at, rather than a single best design.

        Returns:
            An `(m, sequence_length)` array of the rank-zero members.
        """
        selected: Tokens = self._population[self._ranks == 0]
        return selected.copy()

    def propose(self, n: int) -> Tokens:
        """Breed `n` offspring by binary tournament, crossover and mutation.

        Args:
            n: How many candidates to return.

        Returns:
            An `(n, sequence_length)` array.

        Raises:
            RuntimeError: If `feasible_only` and the attempt budget is spent
                before enough feasible offspring are found. Returning infeasible
                designs silently would corrupt the comparison this exists for.
        """
        if not self._feasible_only:
            offspring = self._breed(n)
            self._count(n)
            return offspring

        collected: list[Tokens] = []
        found = 0
        for _ in range(self._max_attempts):
            batch = self._breed(n)
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
        r"""Merge scored offspring into the population by elitist survival.

        Deb et al.'s $(\mu + \lambda)$ step: parents and offspring are pooled,
        sorted into fronts, and whole fronts are admitted until the next one
        would overflow. That last front is admitted in order of *decreasing*
        crowding distance, which is where the boundary-at-infinity rule earns its
        keep -- the extremes of the front are the first admitted, never the first
        discarded.

        Ranking over the pool rather than over the offspring alone is what makes
        the algorithm elitist: a non-dominated design cannot be lost to a bad
        generation.

        Args:
            sequences: The evaluated candidates.
            values: An `(n, n_objectives)` array of their objective values.

        Raises:
            ValueError: If the values are not a two-dimensional objective matrix,
                contain `nan`, do not have one row per sequence, or carry a
                different number of objectives than earlier rounds -- which would
                mean the landscape changed underneath the population.
        """
        scored = _as_objective_matrix(values)
        designs = np.asarray(sequences)
        if designs.shape[0] != scored.shape[0]:
            raise ValueError(
                f"got {designs.shape[0]} sequences and {scored.shape[0]} value rows; "
                f"they must line up or every design is paired with somebody else's score"
            )

        if self._values is None:
            # Founders have not been scored. Minus infinity on every objective
            # makes them dominated by anything real, so the first generation
            # displaces them wholesale, and leaves them mutually non-dominated
            # among themselves rather than imposing an arbitrary order.
            self._values = np.full(
                (self._population.shape[0], scored.shape[1]), -np.inf, dtype=np.float64
            )
        elif scored.shape[1] != self._values.shape[1]:
            raise ValueError(
                f"population carries {self._values.shape[1]} objectives but the batch "
                f"carries {scored.shape[1]}; dominance between them is undefined"
            )

        pooled_designs = np.concatenate([self._population, designs])
        pooled_values = np.concatenate([self._values, scored])
        survivors = self._survive(pooled_values)

        self._population = pooled_designs[survivors]
        self._values = pooled_values[survivors]
        self._ranks = fast_non_dominated_sort(self._values)
        self._crowding = self._crowding_within_fronts(self._values, self._ranks)

    def _survive(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """Pick the surviving `population_size` indices from the pooled values.

        Args:
            values: The `(n, n_objectives)` pooled parent-and-offspring values.

        Returns:
            Indices into `values`, at most `population_size` of them, front by
            front and then by decreasing crowding distance.
        """
        ranks = fast_non_dominated_sort(values)
        chosen: list[npt.NDArray[np.intp]] = []
        room = self._population_size

        for rank in range(int(ranks.max()) + 1 if ranks.size else 0):
            front = np.flatnonzero(ranks == rank)
            if front.size <= room:
                chosen.append(front)
                room -= front.size
                if room == 0:
                    break
                continue
            # The front that overflows is truncated by the crowded-comparison
            # operator: sparser designs first, boundaries (at infinity) always.
            spacing = crowding_distance(values[front])
            chosen.append(front[np.argsort(-spacing, kind="stable")[:room]])
            break

        return (np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.intp)).astype(np.intp)

    @staticmethod
    def _crowding_within_fronts(
        values: npt.NDArray[np.float64], ranks: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Crowding distance computed per front, as NSGA-II defines it.

        Computing it over the whole population instead would compare a design's
        spacing against designs it does not compete with, and the tournament
        would then prefer loneliness across fronts over dominance within one.

        Args:
            values: The `(n, n_objectives)` population values.
            ranks: The front index of each row.

        Returns:
            An `(n,)` array of distances.
        """
        spacing = np.zeros(ranks.size, dtype=np.float64)
        for rank in np.unique(ranks):
            members = np.flatnonzero(ranks == rank)
            spacing[members] = crowding_distance(values[members])
        return spacing

    def _tournament(self, n: int) -> npt.NDArray[np.intp]:
        r"""Draw `n` parents by binary tournament on the crowded-comparison operator.

        Deb et al.'s $\prec_n$: `i` beats `j` when it sits in an earlier front,
        or in the same front with a larger crowding distance. Dominance first,
        spread only as a tie-break -- the reverse would trade convergence for
        diversity, which is the failure mode NSGA-II's predecessor had.

        Args:
            n: How many parents to draw.

        Returns:
            An `(n,)` array of population indices.
        """
        size = self._population.shape[0]
        if self._values is None:
            # Nothing has been scored, so there is no order to hold a tournament
            # on. Uniform selection says exactly that rather than inventing one.
            return self._rng.integers(0, size, size=n).astype(np.intp)

        left = self._rng.integers(0, size, size=n)
        right = self._rng.integers(0, size, size=n)
        left_wins = (self._ranks[left] < self._ranks[right]) | (
            (self._ranks[left] == self._ranks[right])
            & (self._crowding[left] > self._crowding[right])
        )
        return np.where(left_wins, left, right).astype(np.intp)

    def _breed(self, n: int) -> Tokens:
        """Produce `n` offspring by crossover then mutation.

        Deliberately identical to
        [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm]'s
        operator, apart from the pairwise `crossover_prob` gate that Deb et al.
        specify: the two baselines must vary in the same way for the difference
        between them to be attributable to selection.

        Args:
            n: How many offspring to produce.

        Returns:
            An `(n, sequence_length)` array, every row inside the environment's
            graph.
        """
        length = self._env.sequence_length
        size = self._env.alphabet.size
        wild_type = self._env.parent

        first = self._population[self._tournament(n)]
        second = self._population[self._tournament(n)]
        # p_c gates whether a pair is recombined at all; recombine_prob then sets
        # how much of the second parent is taken when it is.
        crossing = self._rng.random(n) < self._crossover_prob
        take_second = (self._rng.random((n, length)) < self._recombine_prob) & crossing[:, None]
        offspring = np.where(take_second, second, first)

        mutate = self._rng.random((n, length)) < self._mutation_prob
        # Draw from the other tokens rather than from all of them, so the
        # realised substitution rate is p_m and not p_m * (V - 1) / V. See
        # [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm].
        drawn = self._rng.integers(0, size - 1, size=(n, length))
        replacements = drawn + (drawn >= offspring)
        offspring = np.where(mutate, replacements, offspring)

        return self._enforce_budget(offspring, wild_type)

    def _enforce_budget(self, offspring: Tokens, wild_type: Tokens) -> Tokens:
        """Revert surplus mutations so every offspring is inside the graph.

        The environment admits at most `max_mutations` differences from the
        parent; offspring beyond that are outside its graph and no GFlowNet could
        have produced them, so emitting one would make the comparison meaningless.

        Args:
            offspring: The `(n, length)` candidates, modified in place.
            wild_type: The parent to revert towards.

        Returns:
            The same array, every row within the mutation budget.
        """
        budget = self._env.max_mutations
        differing = offspring != wild_type[None, :]
        counts = differing.sum(axis=1)
        for row in np.flatnonzero(counts > budget):
            positions = np.flatnonzero(differing[row])
            surplus = self._rng.choice(positions, size=int(counts[row] - budget), replace=False)
            offspring[row, surplus] = wild_type[surplus]
        return offspring


def _as_objective_matrix(values: Fitness) -> npt.NDArray[np.float64]:
    """Validate an `(n, n_objectives)` objective matrix.

    Mirrors `evogfn.metrics.pareto._as_objective_matrix`. Kept here rather than
    imported so that the sorting and crowding routines depend on nothing but
    numpy, and can therefore be checked against the paper in isolation.

    Args:
        values: The array to validate.

    Returns:
        The values as a float matrix.

    Raises:
        ValueError: If the input is not two-dimensional, carries no objectives,
            or contains `nan`, where dominance has no answer.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != _MATRIX_NDIM:
        raise ValueError(
            f"expected shape (n, n_objectives), got {array.shape}; NSGA-II ranks by "
            f"dominance over the objective vector, so a flat array is ambiguous "
            f"between n designs and n objectives"
        )
    if array.shape[1] == 0:
        raise ValueError("expected at least one objective, got a matrix of width 0")
    if np.isnan(array).any():
        raise ValueError(
            "objective values contain nan; dominance against a missing value is "
            "undefined, so decide what it counts as before ranking"
        )
    return array
