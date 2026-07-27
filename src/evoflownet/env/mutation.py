"""Constructing variants by accumulating mutations on a parent sequence.

This is directed evolution as a construction graph. A trajectory starts at the
parent, applies point mutations one at a time, and stops. It is the formulation
used by MOGFN-AL (Jain et al., ICML 2023), and unlike autoregressive generation
it models *evolution from something* rather than design from nothing.

Structure of the graph
----------------------

Each position may be mutated **at most once**. That restriction is what makes the
graph acyclic: without it, mutating a site and then reverting it would return to
a state already visited, and the flow equations would have no solution.

With it, a state is exactly the parent plus a *set* of applied mutations, and the
graph is the **subset lattice** over mutated positions, graded by how many have
been applied. Two consequences follow, and both matter:

* A variant carrying ``k`` mutations is reachable by exactly ``k!`` trajectories,
  one per order in which the mutations could have been applied.
* Its parents in the graph are the ``k`` states reached by undoing any single
  one, so the uniform backward policy is exactly ``1/k`` per parent.

The second point makes the uniform backward policy cheap to compute exactly:
``1/k`` per parent, with no model and no learning. Note carefully what this does
*not* mean. ``P_B`` is not a quantity that can be got wrong in a way that biases
the result -- Malkin et al. show that "for any choice of backward policy
``P_B``, there is a unique flow ... and thus a unique corresponding forward
policy", so every valid ``P_B`` still yields a ``P_F`` sampling proportional to
reward. Choosing uniform is a matter of cost and variance, not correctness, and
they report a *learned* ``P_B`` converging faster on some tasks.

MOGFN-AL makes the same observation about this graph -- "``P_B`` here is not
trivial as there are multiple ways (orders) of generating the set" -- and also
settles on uniform.

Action encoding
---------------

Action ``a < length * alphabet_size`` sets position ``a // alphabet_size`` to
token ``a % alphabet_size``. The final index is the stop action. A single integer
per action lets a policy emit one logit vector per state and apply the mask to it
without any reshaping.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from evoflownet.env.base import SequenceEnvironment, State

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Alphabet, Tokens


class MutationEnvironment(SequenceEnvironment):
    """Builds variants by applying point mutations to a fixed parent.

    Args:
        parent: The starting sequence, shape ``(length,)``.
        alphabet: The alphabet ``parent`` is written in.
        max_mutations: Most mutations a trajectory may accumulate. Defaults to
            the sequence length, meaning unrestricted. Capping it restricts the
            search to a mutational neighbourhood, which is what a real
            directed-evolution round does.
        transitions: Optional ``(v, v)`` matrix whose zeros mark token pairs that
            may not be adjacent. When given, mutations producing a forbidden
            adjacency are masked out, so every sequence the environment can reach
            is feasible by construction rather than filtered afterwards.
        allow_stop_before_max: Whether a trajectory may stop early. When
            ``False``, trajectories run until ``max_mutations`` and every
            terminal state carries exactly that many mutations.

    Raises:
        ValueError: If the parent is not one-dimensional, contains tokens outside
            the alphabet, or the arguments disagree in shape.
    """

    def __init__(
        self,
        parent: Tokens,
        alphabet: Alphabet,
        *,
        max_mutations: int | None = None,
        transitions: npt.NDArray[np.floating] | None = None,
        allow_stop_before_max: bool = True,
    ) -> None:
        """Validate the parent and prepare the action layout."""
        parent_array = np.asarray(parent)
        if parent_array.ndim != 1:
            raise ValueError(f"parent must be a single sequence, got ndim {parent_array.ndim}")
        if not np.issubdtype(parent_array.dtype, np.integer):
            raise ValueError(f"parent must hold token indices, got dtype {parent_array.dtype}")
        if parent_array.size and (parent_array.min() < 0 or parent_array.max() >= alphabet.size):
            raise ValueError(
                f"parent tokens must lie in [0, {alphabet.size}), got "
                f"[{parent_array.min()}, {parent_array.max()}]"
            )

        self._parent = parent_array.astype(np.int32)
        self._alphabet = alphabet
        self._length = int(parent_array.shape[0])
        self._max_mutations = self._length if max_mutations is None else max_mutations
        if not 0 <= self._max_mutations <= self._length:
            raise ValueError(
                f"max_mutations must lie in [0, {self._length}], got {self._max_mutations}"
            )

        if transitions is not None:
            expected = (alphabet.size, alphabet.size)
            if transitions.shape != expected:
                raise ValueError(
                    f"transitions must be {expected} to match the alphabet, got {transitions.shape}"
                )
        self._transitions = transitions
        self._allow_stop_before_max = allow_stop_before_max
        # log(k!) for every reachable k. A table rather than a call to lgamma
        # because k is bounded by the sequence length and this is read on every
        # balance computation.
        self._log_factorial = np.concatenate(
            [[0.0], np.cumsum(np.log(np.arange(1, self._length + 1, dtype=np.float64)))]
        )

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of the sequences this environment constructs."""
        return self._length

    @property
    def parent(self) -> Tokens:
        """The sequence every trajectory starts from."""
        return self._parent.copy()

    @property
    def max_mutations(self) -> int:
        """Most mutations a trajectory may accumulate."""
        return self._max_mutations

    @property
    def n_mutation_actions(self) -> int:
        """Number of substitution actions, excluding the stop action."""
        return self._length * self._alphabet.size

    @property
    def n_actions(self) -> int:
        """Size of the action space, including the stop action."""
        return self.n_mutation_actions + 1

    @property
    def stop_action(self) -> int:
        """Index of the action that terminates a trajectory."""
        return self.n_mutation_actions

    def initial(self, n: int) -> State:
        """Create ``n`` trajectories sitting at the parent.

        Args:
            n: Number of trajectories.

        Returns:
            A state of size ``n``, none of them stopped.
        """
        return State(
            sequences=np.tile(self._parent, (n, 1)),
            stopped=np.zeros(n, dtype=np.bool_),
        )

    def n_mutations(self, state: State) -> npt.NDArray[np.integer]:
        """How many positions differ from the parent, per trajectory.

        This is the grade of the state in the lattice, and doubles as the number
        of parents it has.

        Args:
            state: The current state.

        Returns:
            An ``(n,)`` integer array.
        """
        counts: npt.NDArray[np.integer] = (state.sequences != self._parent[None, :]).sum(axis=1)
        return counts

    def forward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which mutations, and whether stopping, are available.

        A substitution is available when the position is still unmutated, the
        token differs from the parent's (substituting the same token would be a
        no-op that never changes the state, so it cannot be an edge), and the
        result keeps every adjacency permitted.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array.
        """
        n = len(state)
        untouched = state.sequences == self._parent[None, :]
        capacity_left = self.n_mutations(state) < self._max_mutations

        # (n, length, v): allowed if the position is still untouched, the token
        # is a genuine change, and the trajectory has capacity left.
        differs = self._parent[None, :, None] != np.arange(self._alphabet.size)[None, None, :]
        allowed = untouched[:, :, None] & differs & capacity_left[:, None, None]

        if self._transitions is not None:
            allowed &= self._adjacency_allowed(state.sequences, self._transitions)

        mask = np.zeros((n, self.n_actions), dtype=np.bool_)
        mask[:, : self.n_mutation_actions] = allowed.reshape(n, -1)

        # Stopping is available once the trajectory is entitled to stop, and is
        # forced when nothing else is possible -- a state with no legal action
        # would leave the policy with nothing to normalise over.
        may_stop = np.ones(n, dtype=np.bool_) if self._allow_stop_before_max else ~capacity_left
        mask[:, self.stop_action] = may_stop | ~mask[:, : self.n_mutation_actions].any(axis=1)

        # A stopped trajectory takes no further actions at all.
        mask[state.stopped] = False
        return mask

    def backward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which actions could have produced this state.

        For an unstopped state carrying ``k`` mutations these are exactly the
        ``k`` substitutions that introduced them, so a uniform distribution over
        this mask is the exact backward policy the lattice induces.

        A *stopped* state is different, and getting it wrong is easy. Its only
        parent is the same state unstopped: the stop action is the sole edge
        into it. Also marking its mutations would give a terminal ``k + 1``
        parents instead of one, making the uniform backward policy ``1/(k+1)``
        where it should be ``1``, and admitting paths that undo a mutation while
        stopped -- which is not an edge of this graph.

        Under a transition constraint there is a further condition: undoing a
        mutation must land on a state that is itself feasible. A parent that
        violates an adjacency is not in the graph, so the edge into it does not
        exist either -- and a backward walk that ignored this would reconstruct
        paths the forward direction refuses.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array.
        """
        n = len(state)
        mask = np.zeros((n, self.n_actions), dtype=np.bool_)

        running = ~np.asarray(state.stopped, dtype=np.bool_)
        mutated = (state.sequences != self._parent[None, :]) & running[:, None]
        if self._transitions is not None:
            mutated &= self._parent_would_be_feasible(state.sequences)
        rows, positions = np.nonzero(mutated)
        tokens = state.sequences[rows, positions]
        mask[rows, positions * self._alphabet.size + tokens] = True

        mask[:, self.stop_action] = state.stopped
        return mask

    def is_reachable(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences this environment can actually construct.

        A sequence outside the mutation budget, or infeasible under the
        transition constraint, is not in the space the policy is defined over.
        Scoring one is meaningless rather than merely inaccurate, so callers
        that accept sequences from elsewhere -- a replay buffer, a genetic
        algorithm, an assay -- should check first.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.
        """
        array = np.asarray(sequences)
        within_budget = (array != self._parent[None, :]).sum(axis=1) <= self._max_mutations
        if self._transitions is None:
            return np.asarray(within_budget, dtype=np.bool_)
        permitted = self._transitions > 0
        feasible = np.all(permitted[array[:, :-1], array[:, 1:]], axis=1)
        return np.asarray(within_budget & feasible, dtype=np.bool_)

    def _parent_would_be_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which single-mutation reversions land on a feasible state.

        Returns:
            An ``(n, length)`` boolean array, ``True`` where reverting that
            position keeps every adjacency permitted.
        """
        if self._transitions is None:  # pragma: no cover - guarded by the caller
            return np.ones(sequences.shape, dtype=np.bool_)
        permitted = self._transitions > 0
        reverted = np.broadcast_to(self._parent[None, :], sequences.shape)

        allowed = np.ones(sequences.shape, dtype=np.bool_)
        if self._length > 1:
            # Reverting position p only disturbs the (p-1, p) and (p, p+1) pairs.
            allowed[:, 1:] &= permitted[sequences[:, :-1], reverted[:, 1:]]
            allowed[:, :-1] &= permitted[reverted[:, :-1], sequences[:, 1:]]
        return allowed

    def step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Apply one action per trajectory.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices.

        Returns:
            The resulting state.

        Raises:
            ValueError: If any action is masked out. Silently ignoring an
                illegal action would let a policy place probability on edges
                that do not exist, and the flow equations would be solved for the
                wrong graph.
        """
        actions = np.asarray(actions)
        self._check_permitted(self.forward_mask(state), actions, "forward")

        sequences = state.sequences.copy()
        stopped = state.stopped.copy()

        stopping = actions == self.stop_action
        stopped[stopping] = True

        mutating = ~stopping
        if mutating.any():
            rows = np.nonzero(mutating)[0]
            positions = actions[mutating] // self._alphabet.size
            tokens = actions[mutating] % self._alphabet.size
            sequences[rows, positions] = tokens

        return State(sequences=sequences, stopped=stopped)

    def backward_step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Undo one action per trajectory.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices to reverse.

        Returns:
            The preceding state.

        Raises:
            ValueError: If any action could not have produced this state.
        """
        actions = np.asarray(actions)
        self._check_permitted(self.backward_mask(state), actions, "backward")

        sequences = state.sequences.copy()
        stopped = state.stopped.copy()

        unstopping = actions == self.stop_action
        stopped[unstopping] = False

        reverting = ~unstopping
        if reverting.any():
            rows = np.nonzero(reverting)[0]
            positions = actions[reverting] // self._alphabet.size
            sequences[rows, positions] = self._parent[positions]

        return State(sequences=sequences, stopped=stopped)

    def enumerate_terminal_states(self) -> Tokens:
        """Every sequence reachable from the parent.

        The reachable set is the Hamming ball of radius ``max_mutations`` around
        the parent, not the whole sequence space -- which matters for any exact
        distributional comparison, since the target must be normalised over what
        the environment can actually produce rather than over everything that
        exists. Feasibility masking shrinks it further, so this is an upper
        bound when a transition matrix is in use.

        Returns:
            An ``(m, length)`` array of distinct reachable sequences, the parent
            first.

        Raises:
            ValueError: If the reachable set exceeds
            :data:`~evoflownet.landscapes.base.MAX_ENUMERABLE_SIZE`.
        """
        from itertools import combinations, product  # noqa: PLC0415 - only needed here

        from evoflownet.landscapes.base import MAX_ENUMERABLE_SIZE  # noqa: PLC0415

        alternatives = [
            [t for t in range(self._alphabet.size) if t != int(self._parent[position])]
            for position in range(self._length)
        ]
        size = sum(
            math.comb(self._length, k) * (self._alphabet.size - 1) ** k
            for k in range(self._max_mutations + 1)
        )
        if size > MAX_ENUMERABLE_SIZE:
            raise ValueError(
                f"{size:,} sequences are reachable, above the "
                f"{MAX_ENUMERABLE_SIZE:,} enumeration limit"
            )

        reachable = [self._parent.copy()]
        for k in range(1, self._max_mutations + 1):
            for positions in combinations(range(self._length), k):
                for tokens in product(*(alternatives[p] for p in positions)):
                    variant = self._parent.copy()
                    variant[list(positions)] = tokens
                    reachable.append(variant)
        return np.stack(reachable)

    def log_n_trajectories(self, state: State) -> npt.NDArray[np.float64]:
        """Log of how many distinct paths reach each state from the parent.

        A state with ``k`` mutations is reached by ``k!`` orderings. Returned in
        log space because ``k!`` overflows quickly, and because every use of it
        is inside a log-domain balance computation anyway.

        Args:
            state: The current state.

        Returns:
            An ``(n,)`` array of ``log(k!)``.
        """
        counts: npt.NDArray[np.float64] = self._log_factorial[self.n_mutations(state)]
        return counts

    def _adjacency_allowed(
        self, sequences: Tokens, transitions: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.bool_]:
        """Which substitutions keep every adjacency permitted.

        Returns:
            An ``(n, length, v)`` boolean array.
        """
        permitted = transitions > 0
        n = sequences.shape[0]
        candidates = np.arange(self._alphabet.size)

        allowed = np.ones((n, self._length, self._alphabet.size), dtype=np.bool_)
        if self._length > 1:
            # Substituting at position p constrains the (p-1, p) and (p, p+1)
            # pairs only; every other adjacency is untouched.
            left = sequences[:, :-1]  # token before positions 1..L-1
            allowed[:, 1:, :] &= permitted[left[:, :, None], candidates[None, None, :]]
            right = sequences[:, 1:]  # token after positions 0..L-2
            allowed[:, :-1, :] &= permitted[candidates[None, None, :], right[:, :, None]]
        return allowed

    def _check_permitted(
        self,
        mask: npt.NDArray[np.bool_],
        actions: npt.NDArray[np.integer],
        direction: str,
    ) -> None:
        """Raise if any action is not permitted by ``mask``."""
        if actions.shape != (mask.shape[0],):
            raise ValueError(
                f"expected one action per trajectory, got {actions.shape} for "
                f"{mask.shape[0]} trajectories"
            )
        if actions.size == 0:
            return
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(
                f"action indices must lie in [0, {self.n_actions}), got "
                f"[{actions.min()}, {actions.max()}]"
            )
        permitted = mask[np.arange(mask.shape[0]), actions]
        if not permitted.all():
            offenders = np.nonzero(~permitted)[0]
            raise ValueError(
                f"{direction} action not permitted for trajectories {offenders.tolist()}: "
                f"{actions[offenders].tolist()}"
            )
