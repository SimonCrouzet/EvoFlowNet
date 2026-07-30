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

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._reanchor import carried_design
from evogfn.algorithms.baselines._values import single_objective

if TYPE_CHECKING:
    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment


def substituted(env: MutationEnvironment, current: Tokens, rng: np.random.Generator) -> Tokens:
    """One random single-substitution neighbour of ``current``.

    A position that already differs from the anchor may be changed again: the
    environment forbids mutating a position twice along one *trajectory*, but
    the sequence that results from revising an earlier substitution is a
    different point in the same Hamming ball, reached by a different path.
    Treating the trajectory constraint as a constraint on states would forbid a
    local search from ever undoing a move -- for an annealer, whose whole
    purpose is to accept moves it may later need to undo, that is fatal rather
    than merely limiting -- and would shrink the neighbourhood toward nothing as
    the search moved. At the budget only already-substituted positions may
    change, since touching a fresh one would push the design out of the graph.

    Args:
        env: Supplies the anchor, the alphabet and the mutation budget.
        current: The design to step away from.
        rng: Draws the position and the replacement token.

    Returns:
        A single sequence one substitution from ``current``.
    """
    proposal = np.asarray(current).copy()
    mutated = np.flatnonzero(proposal != env.parent)
    at_budget = mutated.size >= env.max_mutations
    available = mutated if at_budget else np.arange(env.sequence_length)
    position = int(rng.choice(available))
    alternatives = [t for t in range(env.alphabet.size) if t != proposal[position]]
    proposal[position] = rng.choice(alternatives)
    return proposal


def redrawn_until_buildable(
    env: MutationEnvironment,
    current: Tokens,
    proposals: Tokens,
    rng: np.random.Generator,
    attempts: int,
) -> tuple[Tokens, int]:
    """Redraw the neighbours the environment cannot build, and report the cost.

    A local searcher under a feasibility constraint has a third option between
    "emit something unbuildable" and "raise". The previous behaviour here was a
    fourth and worse one: hold the current design wherever the drawn neighbour
    was illegal. That never raises and never emits an infeasible design, so it
    reads as safe -- and on a sparse feasible set, where 96% of single
    substitutions are illegal, it quietly turns a plate of designs into 96
    copies of one. After the campaign deduplicates, the round measures a single
    variant. The search looks stalled and the diagnosis "annealing does not move
    on constrained landscapes" would be a fact about this function.

    Redrawing gives each illegal row fresh positions and tokens instead. The
    feasible neighbours of a feasible design are a small but non-empty set --
    the anchor is reachable and the graph is connected -- so a bounded number of
    draws finds one most of the time, and holding position remains the fallback
    for the rows it does not.

    Args:
        env: Supplies the anchor, alphabet, budget and feasibility rule.
        current: The design the neighbours were drawn from. Feasible by
            induction, since it was itself an accepted proposal.
        proposals: The ``(n, length)`` neighbours drawn so far, modified in
            place.
        rng: Draws the replacements.
        attempts: Redraw rounds before falling back to holding position.

    Returns:
        The proposals, every row now buildable, and how many extra draws that
        cost. The count is charged to
        [proposals_made][evogfn.algorithms.base.Sampler.proposals_made], because
        a search that needs twenty draws per design is not as cheap as one that
        needs one and the table has to be able to say so.
    """
    spent = 0
    for _ in range(attempts):
        unbuildable = np.flatnonzero(~env.is_reachable(proposals))
        if unbuildable.size == 0:
            return proposals, spent
        for row in unbuildable:
            proposals[row] = substituted(env, current, rng)
        spent += int(unbuildable.size)
    proposals[~env.is_reachable(proposals)] = current
    return proposals, spent


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

    def reanchored(self, env: MutationEnvironment) -> RandomMutagenesis:
        """Re-anchor, which for the null costs nothing because it holds nothing.

        This is the one baseline for which the campaign's rebuild path would
        have been adequate, and it implements the hook anyway so that the reason
        is on the record rather than inferred from an absence: random
        mutagenesis adapts to no measurement, so it has no learned state to
        lose. Its only carried state is the random stream, and continuing that
        stream rather than restarting it is what stops round three re-drawing
        round one's library.

        Args:
            env: The re-anchored environment.

        Returns:
            A sampler drawing uniformly from the new anchor's neighbourhood.
        """
        moved = RandomMutagenesis(
            env, feasible_only=self._feasible_only, max_attempts=self._max_attempts
        )
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made
        return moved

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
        feasible_only: Redraw neighbours that are not constructible, rather
            than emitting them.
        max_attempts: Redraw rounds before a row falls back to holding the
            current design. See
            [redrawn_until_buildable][evogfn.algorithms.baselines.mutagenesis.redrawn_until_buildable]
            for why holding position is a fallback rather than the remedy.
        seed: Seeds proposals.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        patience: int = 5,
        feasible_only: bool = False,
        max_attempts: int = 50,
        seed: int = 0,
    ) -> None:
        """Start at the parent."""
        super().__init__()
        self._env = env
        self._patience = patience
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts
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

    def reanchored(self, env: MutationEnvironment) -> HillClimbing:
        """Carry the whole climb across the move, because none of it is relative.

        A hill climber's state is a design, the value it scored, and how many
        rounds it has gone without improving. A design is a sequence and a value
        is a measurement; neither is expressed relative to an anchor, so all
        three cross the move intact and the climb continues from exactly where
        it stood. In the ordinary case it does not even move: the campaign
        anchors at the best design measured, which is usually the very design
        the climber is standing on, so it resumes at zero mutations from the new
        anchor with its whole neighbourhood ahead of it.

        The one thing worth guarding is the case where the two disagree. The
        campaign picks its anchor through the acquisition rule over the batch,
        the climber through its own record, so the incumbent can end up outside
        the new mutation budget -- at which point every neighbour it proposed
        would be a design the environment cannot build.
        [carried_design][evogfn.algorithms.baselines._reanchor.carried_design]
        restarts it at the anchor in that case, which is the better of the two
        points anyway.

        Args:
            env: The re-anchored environment.

        Returns:
            A climber over ``env``, standing where this one stood.
        """
        moved = HillClimbing(
            env,
            patience=self._patience,
            feasible_only=self._feasible_only,
            max_attempts=self._max_attempts,
        )
        moved._current = carried_design(env, self._current)
        moved._best_value = self._best_value
        moved._stale = self._stale
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made
        return moved

    def propose(self, n: int) -> Tokens:
        """Return ``n`` neighbours of the current design.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.
        """
        proposals = np.tile(self._current, (n, 1))
        for row in range(n):
            proposals[row] = substituted(self._env, self._current, self._rng)

        extra = 0
        if self._feasible_only:
            proposals, extra = redrawn_until_buildable(
                self._env, self._current, proposals, self._rng, self._max_attempts
            )

        self._count(n + extra)
        return proposals

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Move to the best design seen, or restart if stuck.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which has
                no single best design to move to without a scalarisation.
        """
        flat = single_objective(values)
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
