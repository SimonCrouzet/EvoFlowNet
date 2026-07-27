"""The policy network: forward and backward action distributions over a state.

Trajectory balance needs three things from a model -- ``P_F(s'|s)``,
``P_B(s|s')`` and a scalar ``log Z`` -- and each has a detail that matters more
than the architecture does.

**Masking is applied inside the model, not by the caller.** A caller who forgets
to mask gets a policy placing probability on edges that do not exist, and the
symptom is not a crash but a slightly wrong distribution. :meth:`log_probs`
therefore takes the mask and there is no way to obtain unmasked log
probabilities by accident.

**The heads share a trunk except for their final layer**, following Malkin et
al. The two policies describe the same graph from opposite directions, so most
of what they need to compute is the same.

**``log Z`` is a bare parameter, not a head.** It does not depend on the state --
it is the total flow through the whole DAG -- and Malkin et al. report it needs a
learning rate roughly an order of magnitude above the policy's, which is why it
is exposed separately for the optimiser to put in its own parameter group.

**The backward policy is uniform by default and not learned.** On the mutation
environment's subset lattice a state with ``k`` mutations has exactly ``k``
parents, so uniform ``P_B`` is ``1/k`` in closed form; it is also the maximum
entropy choice there, and ``log P_B(τ|x) = -log k!`` is constant across
trajectories reaching the same terminal state, so it adds no path-dependent
variance to the loss. Learning it is still supported, because any valid ``P_B``
induces the same optimal ``P_F`` and Malkin et al. report a learned one
converging faster on some tasks -- so which is better here is an empirical
question, not a settled one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from evoflownet.core.types import Tokens

#: Logit assigned to a masked action. Finite rather than ``-inf`` so that a row
#: which is entirely masked -- a stopped trajectory -- yields zeros instead of
#: ``nan`` and poisons nothing downstream. Large enough that a masked action's
#: probability is zero to within float32 precision.
MASKED_LOGIT = -1e9


class SequencePolicy(nn.Module):
    """Forward and backward action distributions over fixed-length sequences.

    An embedding of each position is concatenated and passed through an MLP
    trunk. That is adequate while sequences are short -- the benchmarks here run
    from 4 positions (GB1) to a few dozen -- and deliberately simple, since the
    claim being tested is about the training objective rather than the
    architecture. Longer sequences will want a different trunk; the interface
    does not change.

    Args:
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_actions: Size of the action space, including any stop action.
        embedding_dim: Width of the per-position embedding.
        hidden_dim: Width of the trunk.
        n_layers: Number of hidden layers in the trunk. Must be at least 1.
        learn_backward: Whether to learn ``P_B``. When ``False`` the backward
            policy is uniform over the parents permitted by the backward mask.
        learn_flow: Whether to estimate a state flow ``log F(s)``. Needed by
            the detailed-balance family, which constrains flow through each
            *state*; trajectory balance constrains whole trajectories and does
            not use it.

    Raises:
        ValueError: If any size is not positive.
    """

    def __init__(  # noqa: PLR0913 - the network's shape is its definition
        self,
        *,
        n_tokens: int,
        sequence_length: int,
        n_actions: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        learn_backward: bool = False,
        learn_flow: bool = False,
    ) -> None:
        """Build the trunk and heads."""
        super().__init__()
        for name, value in [
            ("n_tokens", n_tokens),
            ("sequence_length", sequence_length),
            ("n_actions", n_actions),
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
            ("n_layers", n_layers),
        ]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        self._n_actions = n_actions
        self._learn_backward = learn_backward
        self._learn_flow = learn_flow

        self.embedding = nn.Embedding(n_tokens, embedding_dim)

        layers: list[nn.Module] = []
        width = sequence_length * embedding_dim
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Separate final layers only, so the shared representation is genuinely
        # shared.
        self.forward_head = nn.Linear(hidden_dim, n_actions)
        self.backward_head = nn.Linear(hidden_dim, n_actions) if learn_backward else None
        # One scalar per state. Unlike log Z this is a genuine head: the flow
        # through a state depends on the state.
        self.flow_head = nn.Linear(hidden_dim, 1) if learn_flow else None

        # Not a head: log Z is state-independent, being the total flow through
        # the DAG. Initialised at 0 (Z = 1) and expected to move a long way, so
        # the trainer gives it its own learning rate.
        self.log_z = nn.Parameter(torch.zeros(()))

    @property
    def n_actions(self) -> int:
        """Size of the action space."""
        return self._n_actions

    @property
    def learns_backward(self) -> bool:
        """Whether ``P_B`` is learned rather than uniform."""
        return self._learn_backward

    @property
    def learns_flow(self) -> bool:
        """Whether a state flow estimate is available."""
        return self._learn_flow

    def log_flow(self, sequences: torch.Tensor) -> torch.Tensor:
        """Estimate ``log F(s)`` for a batch of states.

        Args:
            sequences: An ``(n, length)`` tensor of token indices.

        Returns:
            An ``(n,)`` tensor of log flows.

        Raises:
            RuntimeError: If the policy was built without a flow head. Falling
                back to a constant would silently turn detailed balance into a
                worse trajectory balance.
        """
        if self.flow_head is None:
            raise RuntimeError(
                "this policy has no flow head; build it with learn_flow=True to "
                "use a detailed-balance objective"
            )
        flow: torch.Tensor = self.flow_head(self.forward(sequences))
        return flow.squeeze(-1)

    def policy_parameters(self) -> list[nn.Parameter]:
        """Every parameter except ``log Z``.

        Returns:
            Parameters for the optimiser's main group.
        """
        return [p for name, p in self.named_parameters() if name != "log_z"]

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode a batch of states.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor.

        Returns:
            An ``(n, hidden_dim)`` representation.
        """
        embedded = self.embedding(sequences)
        hidden: torch.Tensor = self.trunk(embedded.flatten(start_dim=1))
        return hidden

    def log_probs(
        self,
        sequences: torch.Tensor,
        forward_mask: torch.Tensor,
        backward_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked log probabilities over forward and backward actions.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor of states.
            forward_mask: An ``(n, n_actions)`` boolean tensor of legal forward
                actions.
            backward_mask: An ``(n, n_actions)`` boolean tensor of actions that
                could have produced each state.

        Returns:
            Two ``(n, n_actions)`` tensors of log probabilities. Masked entries
            are ``-inf``. A fully masked row -- a stopped trajectory, which has
            no actions in either direction -- is all ``-inf`` rather than
            ``nan``.

        Raises:
            ValueError: If a mask does not match the action space.
        """
        self._check_mask(forward_mask, "forward_mask")
        self._check_mask(backward_mask, "backward_mask")

        hidden = self(sequences)
        forward = _masked_log_softmax(self.forward_head(hidden), forward_mask)

        if self.backward_head is not None:
            backward = _masked_log_softmax(self.backward_head(hidden), backward_mask)
        else:
            # Uniform over parents: exact on the subset lattice, and it needs no
            # network evaluation at all.
            backward = _uniform_log_probs(backward_mask)
        return forward, backward

    def _check_mask(self, mask: torch.Tensor, name: str) -> None:
        """Raise if a mask is not ``(n, n_actions)`` boolean."""
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean, got {mask.dtype}")
        if mask.ndim != 2 or mask.shape[1] != self._n_actions:  # noqa: PLR2004
            raise ValueError(
                f"{name} must have shape (n, {self._n_actions}), got {tuple(mask.shape)}"
            )


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax restricted to the permitted actions.

    Masked logits are set to a large negative constant rather than ``-inf``,
    because a row with every action masked would otherwise softmax to ``nan``
    and contaminate the whole batch through the loss. Stopped trajectories are
    exactly that case.
    """
    restricted = logits.masked_fill(~mask, MASKED_LOGIT)
    log_probabilities = torch.log_softmax(restricted, dim=-1)
    # Report masked actions as -inf: their probability really is zero, and any
    # arithmetic that touches one should be visibly wrong rather than merely
    # very small.
    return log_probabilities.masked_fill(~mask, float("-inf"))


def _uniform_log_probs(mask: torch.Tensor) -> torch.Tensor:
    """Uniform distribution over the permitted actions in each row."""
    counts = mask.sum(dim=-1, keepdim=True)
    # A row with no permitted action contributes nothing; guard the division so
    # it yields -inf everywhere rather than nan.
    safe = counts.clamp(min=1)
    uniform = -torch.log(safe.to(mask.device, dtype=torch.float32)).expand_as(mask)
    return uniform.masked_fill(~mask, float("-inf"))


def to_tensor(sequences: Tokens, device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert token indices to the integer tensor the policy expects.

    Args:
        sequences: An ``(n, sequence_length)`` array of token indices.
        device: Where to place the tensor.

    Returns:
        A ``(n, sequence_length)`` ``int64`` tensor.
    """
    return torch.as_tensor(sequences, dtype=torch.long, device=device)
