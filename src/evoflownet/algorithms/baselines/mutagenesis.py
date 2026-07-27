"""Random mutagenesis and hill climbing: the two floors any method must clear.

Random mutagenesis is the null. It adapts to nothing and is what a design round
looks like with no model at all, so any method that fails to beat it has
demonstrated nothing.

Hill climbing is the harder floor, and the more informative one. It is a pure
optimiser -- it finds a peak and stays there -- so it should score well on best
found and badly on diversity and on distributional distance. A method claiming
to sample rather than optimise has to separate from hill climbing on exactly
those metrics, and if it does not, the claim is empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evoflownet.algorithms.base import Sampler

if TYPE_CHECKING:
    from evoflownet.core.types import Fitness, Tokens
    from evoflownet.env.mutation import MutationEnvironment


class RandomMutagenesis(Sampler):
    """Draws variants uniformly from the mutational neighbourhood of the parent.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        feasible_only: Resample until every proposal is constructible. Off by
            default, because a method that ignores the constraint is exactly
            what the constraint experiment needs to compare against.
        max_attempts: Resampling rounds before giving up when
            ``feasible_only``. Exceeding it raises rather than silently
            returning infeasible designs.
        seed: Seeds proposals.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        feasible_only: bool = False,
        max_attempts: int = 100,
        seed: int = 0,
    ) -> None:
        """Store the environment and sampling policy."""
        super().__init__()
        self._env = env
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "RandomMutagenesis" + (" (feasible)" if self._feasible_only else "")

    def propose(self, n: int) -> Tokens:
        """Draw ``n`` variants with a uniformly random number of mutations.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.

        Raises:
            RuntimeError: If ``feasible_only`` and the attempt budget is spent
                before enough feasible candidates are found. Silently returning
                infeasible designs would corrupt the comparison it exists for.
        """
        if not self._feasible_only:
            drawn = self._draw(n)
            self._count(n)
            return drawn

        collected: list[Tokens] = []
        found = 0
        for _ in range(self._max_attempts):
            batch = self._draw(n)
            self._count(n)
            keep = batch[self._env.is_reachable(batch)]
            if keep.shape[0]:
                collected.append(keep)
                found += keep.shape[0]
            if found >= n:
                return np.concatenate(collected)[:n]
        raise RuntimeError(
            f"could not draw {n} feasible candidates in {self._max_attempts} attempts "
            f"({found} found); the feasible set is too sparse for rejection sampling"
        )

    def _draw(self, n: int) -> Tokens:
        """Uniform variants within the mutation budget."""
        parent = self._env.parent
        length = self._env.sequence_length
        size = self._env.alphabet.size
        sequences = np.tile(parent, (n, 1))

        counts = self._rng.integers(0, self._env.max_mutations + 1, size=n)
        for row, k in enumerate(counts):
            if k == 0:
                continue
            positions = self._rng.choice(length, size=int(k), replace=False)
            for position in positions:
                alternatives = [t for t in range(size) if t != parent[position]]
                sequences[row, position] = self._rng.choice(alternatives)
        return sequences


class HillClimbing(Sampler):
    """Proposes single-mutation neighbours of the best design seen so far.

    Restarts from the parent after ``patience`` rounds without improvement,
    which is what stops it reporting a single lucky peak as a search result.

    Args:
        env: Supplies the parent, alphabet and mutation budget.
        patience: Rounds without improvement before restarting.
        feasible_only: Discard neighbours that are not constructible.
        seed: Seeds proposals.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        patience: int = 5,
        feasible_only: bool = False,
        seed: int = 0,
    ) -> None:
        """Start at the parent."""
        super().__init__()
        self._env = env
        self._patience = patience
        self._feasible_only = feasible_only
        self._rng = np.random.default_rng(seed)
        self._current = env.parent
        self._best_value = -np.inf
        self._stale = 0

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "HillClimbing" + (" (feasible)" if self._feasible_only else "")

    @property
    def best_value(self) -> float:
        """Best objective value observed so far."""
        return self._best_value

    def propose(self, n: int) -> Tokens:
        """Return ``n`` neighbours of the current design.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.
        """
        parent = self._env.parent
        length = self._env.sequence_length
        size = self._env.alphabet.size
        proposals = np.tile(self._current, (n, 1))

        for row in range(n):
            # Only positions still at the parent token may be mutated, or the
            # candidate would leave the environment's graph.
            available = [p for p in range(length) if proposals[row, p] == parent[p]]
            if not available or (proposals[row] != parent).sum() >= self._env.max_mutations:
                proposals[row] = parent
                available = list(range(length))
            position = int(self._rng.choice(available))
            alternatives = [t for t in range(size) if t != proposals[row, position]]
            proposals[row, position] = self._rng.choice(alternatives)

        if self._feasible_only:
            reachable = self._env.is_reachable(proposals)
            proposals[~reachable] = self._current

        self._count(n)
        return proposals

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Move to the best design seen, or restart if stuck.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = np.isfinite(flat)
        if not finite.any():
            self._stale += 1
        else:
            best = int(np.flatnonzero(finite)[np.argmax(flat[finite])])
            if flat[best] > self._best_value:
                self._best_value = float(flat[best])
                self._current = np.asarray(sequences)[best].copy()
                self._stale = 0
            else:
                self._stale += 1

        if self._stale >= self._patience:
            self._current = self._env.parent
            self._stale = 0
