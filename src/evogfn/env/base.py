"""The environment interface: the graph that variants are constructed through.

A GFlowNet does not score sequences directly -- it walks a directed acyclic graph
whose sink nodes are the sequences, and learns a policy over the edges. The
environment defines that graph: what a partial construction looks like, which
actions are available from it, and which actions could have led to it.

Three requirements are not negotiable, because the trajectory balance objective
is only valid when they hold:

* **Acyclicity.** Some quantity must strictly increase along every forward edge,
  or the "directed acyclic" in DAG fails and the flow equations have no solution.
* **Backward consistency.** Every forward edge must be recoverable as a backward
  edge from its destination, so ``P_B`` describes the same graph as ``P_F``.
* **Mask honesty.** An action the mask forbids must never be reachable, and an
  action it permits must always be applicable. A mask that lies produces a policy
  assigning probability to states that cannot exist.

These are properties of the graph rather than of any particular sampler, so they
are tested here rather than in the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Tokens


@dataclass(frozen=True, slots=True)
class State:
    """A batch of partial constructions.

    Batched rather than single because every consumer -- policy forward pass,
    masking, reward evaluation -- works on batches, and a per-item representation
    would be converted to this on entry every time.

    Attributes:
        sequences: ``(n, length)`` token indices for the current content.
        stopped: ``(n,)`` flags marking trajectories that have terminated. A
            stopped trajectory takes no further forward actions, so batches can
            hold a mixture of finished and unfinished work rather than needing
            to be split.
    """

    sequences: Tokens
    stopped: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        """Number of trajectories in the batch."""
        return int(self.sequences.shape[0])

    def with_(
        self,
        *,
        sequences: Tokens | None = None,
        stopped: npt.NDArray[np.bool_] | None = None,
    ) -> State:
        """Return a copy with some fields replaced.

        Args:
            sequences: Replacement token array, or ``None`` to keep.
            stopped: Replacement stop flags, or ``None`` to keep.

        Returns:
            A new state; the original is unchanged.
        """
        return replace(
            self,
            sequences=self.sequences if sequences is None else sequences,
            stopped=self.stopped if stopped is None else stopped,
        )


class SequenceEnvironment(ABC):
    """The construction graph over sequences.

    Actions are indexed by a single integer so that a policy can emit one logit
    vector per state and the mask can be applied to it directly. What an index
    means is the implementation's business.
    """

    @property
    @abstractmethod
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        """Length of the sequences this environment constructs."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Size of the action space, including any terminating action."""

    @abstractmethod
    def initial(self, n: int) -> State:
        """Create ``n`` trajectories at the source of the graph.

        Args:
            n: Number of trajectories.

        Returns:
            A state of size ``n``.
        """

    @abstractmethod
    def forward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which actions may be taken from each trajectory.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array. Every unstopped row must permit
            at least one action, or the trajectory has nowhere to go and the
            policy has nothing to normalise over.
        """

    @abstractmethod
    def backward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which actions could have produced each trajectory's current state.

        This is what ``P_B`` is a distribution over, so it must describe exactly
        the same edges as [forward_mask][evogfn.env.base.SequenceEnvironment.forward_mask] traversed
        in reverse.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array. The source state permits
            nothing, since it has no parents.
        """

    @abstractmethod
    def step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Apply one forward action per trajectory.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices.

        Returns:
            The resulting state.
        """

    @abstractmethod
    def backward_step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Undo one action per trajectory, moving towards the source.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices, as returned by the forward action
                that would produce ``state``.

        Returns:
            The preceding state.
        """

    def is_terminal(self, state: State) -> npt.NDArray[np.bool_]:
        """Which trajectories have finished.

        Args:
            state: The current state.

        Returns:
            An ``(n,)`` boolean array.
        """
        return np.asarray(state.stopped, dtype=np.bool_)

    def is_reachable(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences this environment can construct.

        Environments with no restriction accept everything. Where a restriction
        exists, callers that accept sequences from outside -- a replay buffer, a
        genetic algorithm, an assay -- should check before scoring them.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.
        """
        return np.ones(np.asarray(sequences).shape[0], dtype=np.bool_)

    def to_sequences(self, state: State) -> Tokens:
        """Extract the sequences a state represents.

        Args:
            state: The current state.

        Returns:
            An ``(n, sequence_length)`` array of token indices.
        """
        return state.sequences
