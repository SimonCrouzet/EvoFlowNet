"""Rolling out trajectories through the construction graph under a policy.

Two things here are easy to get wrong and expensive to notice.

**Trajectories in a batch finish at different times.** A stopped trajectory must
contribute nothing further -- not a zero, which would be a real term in the sum,
but nothing at all. Every accumulation is therefore guarded by a live mask.

**Exploration must not be scored.** Trajectories are drawn from a behaviour
policy that mixes in uniform noise, but the log-probabilities accumulated are
always the *model's*. Trajectory balance is an off-policy objective, so this is
allowed and is what makes exploration free; scoring the behaviour policy instead
would optimise the wrong thing while looking entirely reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from evoflownet.models.policy import to_tensor

if TYPE_CHECKING:
    from evoflownet.core.types import Tokens
    from evoflownet.env.base import SequenceEnvironment, State
    from evoflownet.models.policy import SequencePolicy


@dataclass(frozen=True, slots=True)
class Trajectories:
    """A batch of completed trajectories.

    Attributes:
        terminal: ``(n, length)`` final sequences.
        log_forward: ``(n,)`` summed ``log P_F`` along each trajectory, under the
            model policy, carrying gradients.
        log_backward: ``(n,)`` summed ``log P_B``, carrying gradients only when
            the backward policy is learned.
        lengths: ``(n,)`` number of actions taken, including the stop action.
        states: ``(n, T + 1, length)`` every state visited, padded on the right
            by repeating the terminal state. Recorded because the
            detailed-balance family constrains flow through *states*, which the
            summed quantities above cannot recover.
        step_log_forward: ``(n, T)`` per-step ``log P_F``, carrying gradients.
        step_log_backward: ``(n, T)`` per-step ``log P_B``.
        active: ``(n, T)`` whether each step was a real transition rather than
            padding after termination.
        stopping: ``(n, T)`` whether the step took the stop action, whose
            destination is terminal and whose flow is the reward rather than an
            estimate.
        state_log_rewards: ``(n, T + 1)`` ``log R(s)`` at every visited state,
            attached by the trainer when the objective asks for it. Available
            here because every state in a mutation lattice is a complete
            sequence; an autoregressive environment could not supply it.
    """

    terminal: Tokens
    log_forward: torch.Tensor
    log_backward: torch.Tensor
    lengths: npt_int
    states: npt_int | None = None
    step_log_forward: torch.Tensor | None = None
    step_log_backward: torch.Tensor | None = None
    active: npt_bool | None = None
    stopping: npt_bool | None = None
    state_log_rewards: torch.Tensor | None = None

    def __len__(self) -> int:
        """Number of trajectories."""
        return int(self.terminal.shape[0])

    def with_state_rewards(self, state_log_rewards: torch.Tensor) -> Trajectories:
        """Return a copy carrying rewards for every visited state.

        Args:
            state_log_rewards: An ``(n, T + 1)`` tensor of ``log R(s)``.

        Returns:
            The same trajectories with the rewards attached.
        """
        return replace(self, state_log_rewards=state_log_rewards)

    def require_steps(self) -> _Steps:
        """Return the per-step record, or explain why it is missing.

        Returns:
            States, per-step log probabilities and the two masks.

        Raises:
            ValueError: If this batch was built without per-step data --
                replayed trajectories, for instance. A detailed-balance
                objective cannot be computed from summed quantities, and
                silently falling back to trajectory balance would misreport
                which objective produced a result.
        """
        if (
            self.states is None
            or self.step_log_forward is None
            or self.step_log_backward is None
            or self.active is None
            or self.stopping is None
        ):
            raise ValueError(
                "these trajectories carry no per-step record, which the "
                "detailed-balance family requires; sample them with "
                "sample_trajectories rather than reconstructing them"
            )
        return _Steps(
            self.states,
            self.step_log_forward,
            self.step_log_backward,
            self.active,
            self.stopping,
        )


@dataclass(frozen=True, slots=True)
class _Steps:
    """The per-step record, once its presence has been checked."""

    states: npt_int
    log_forward: torch.Tensor
    log_backward: torch.Tensor
    active: npt_bool
    stopping: npt_bool


if TYPE_CHECKING:
    import numpy.typing as npt

    npt_int = npt.NDArray[np.integer]
    npt_bool = npt.NDArray[np.bool_]
else:  # pragma: no cover - only aliases for annotations
    npt_int = np.ndarray
    npt_bool = np.ndarray


def sample_trajectories(  # noqa: PLR0913 - each argument is part of the rollout contract
    env: SequenceEnvironment,
    policy: SequencePolicy,
    n: int,
    *,
    epsilon: float = 0.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Trajectories:
    """Roll out ``n`` trajectories from the source to termination.

    Args:
        env: The construction graph.
        policy: The policy to sample under and to score with.
        n: How many trajectories to sample.
        epsilon: Probability of choosing uniformly among legal actions instead
            of following the policy. Only affects which trajectories are drawn,
            never how they are scored.
        generator: Torch generator, for reproducible sampling.
        device: Where to run the policy.

    Returns:
        The completed trajectories.

    Raises:
        ValueError: If ``n`` is not positive or ``epsilon`` is outside ``[0, 1]``.
        RuntimeError: If a trajectory fails to terminate within the number of
            steps the environment can possibly need, which would mean the graph
            is not acyclic.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}")

    state = env.initial(n)
    log_forward = torch.zeros(n, device=device)
    log_backward = torch.zeros(n, device=device)
    lengths = np.zeros(n, dtype=np.int64)
    visited = [state.sequences.copy()]
    step_forward: list[torch.Tensor] = []
    step_backward: list[torch.Tensor] = []
    was_active: list[np.ndarray] = []
    was_stop: list[np.ndarray] = []

    # Every forward action strictly increases the mutation count, and the stop
    # action ends the trajectory, so termination is bounded. Exceeding this
    # means the acyclicity invariant is broken.
    max_steps = env.sequence_length + 1

    for _ in range(max_steps):
        live = ~env.is_terminal(state)
        if not live.any():
            break

        forward_mask = torch.as_tensor(env.forward_mask(state), device=device)
        backward_mask = torch.as_tensor(env.backward_mask(state), device=device)
        forward_log_probs, _ = policy.log_probs(
            to_tensor(state.sequences, device), forward_mask, backward_mask
        )

        actions = _choose(forward_log_probs, forward_mask, epsilon, generator)
        # A stopped trajectory is parked on the stop action so the batch shape
        # stays rectangular; its contribution is masked out below.
        actions = torch.where(
            torch.as_tensor(live, device=device),
            actions,
            torch.full_like(actions, env.n_actions - 1),
        )

        live_tensor = torch.as_tensor(live, device=device)
        step_log_pf = forward_log_probs.gather(1, actions[:, None]).squeeze(1)
        log_forward = log_forward + torch.where(
            live_tensor, step_log_pf, torch.zeros_like(step_log_pf)
        )
        lengths += live.astype(np.int64)

        next_state = _step_live(env, state, actions.cpu().numpy(), live)

        # P_B is read at the *destination*: it is the probability of undoing the
        # action just taken, given where it landed.
        next_backward_mask = torch.as_tensor(env.backward_mask(next_state), device=device)
        next_forward_mask = torch.as_tensor(env.forward_mask(next_state), device=device)
        _, backward_log_probs = policy.log_probs(
            to_tensor(next_state.sequences, device), next_forward_mask, next_backward_mask
        )
        step_log_pb = backward_log_probs.gather(1, actions[:, None]).squeeze(1)
        # The stop action's reverse is not a graph edge and carries no P_B term.
        contributes = live_tensor & (actions != env.n_actions - 1)
        log_backward = log_backward + torch.where(
            contributes, step_log_pb, torch.zeros_like(step_log_pb)
        )

        # Store the *masked* values. A row that has already stopped has a
        # fully masked action distribution, so its raw log prob is -inf; kept
        # unmasked, multiplying it by a zero weight later yields nan rather
        # than zero and destroys the batch.
        step_forward.append(torch.where(live_tensor, step_log_pf, torch.zeros_like(step_log_pf)))
        step_backward.append(torch.where(contributes, step_log_pb, torch.zeros_like(step_log_pb)))
        was_active.append(live.copy())
        was_stop.append((actions == env.n_actions - 1).cpu().numpy() & live)
        visited.append(next_state.sequences.copy())

        state = next_state
    else:
        if not env.is_terminal(state).all():
            raise RuntimeError(
                f"trajectories did not terminate within {max_steps} steps; the "
                f"environment's graph is not acyclic"
            )

    return Trajectories(
        terminal=env.to_sequences(state),
        log_forward=log_forward,
        log_backward=log_backward,
        lengths=lengths,
        states=np.stack(visited, axis=1),
        step_log_forward=torch.stack(step_forward, dim=1),
        step_log_backward=torch.stack(step_backward, dim=1),
        active=np.stack(was_active, axis=1),
        stopping=np.stack(was_stop, axis=1),
    )


def _choose(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample one action per row, mixing in uniform exploration.

    The returned actions are drawn from the behaviour policy; the caller scores
    them under ``log_probs``, which is the model's.
    """
    probabilities = log_probs.exp()
    if epsilon > 0.0:
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        uniform = mask.to(probabilities.dtype) / counts
        probabilities = (1.0 - epsilon) * probabilities + epsilon * uniform
    # A fully masked row (already stopped) sums to zero and would make
    # multinomial fail; give it an arbitrary valid row, since the action is
    # discarded by the live mask anyway.
    empty = probabilities.sum(dim=-1) <= 0.0
    if empty.any():
        probabilities = probabilities.clone()
        probabilities[empty, -1] = 1.0
    return _multinomial(probabilities, generator)


def _step_live(
    env: SequenceEnvironment,
    state: State,
    actions: npt_int,
    live: np.ndarray,
) -> State:
    """Apply actions only to trajectories that are still running."""
    if live.all():
        return env.step(state, actions)

    from evoflownet.env.base import State as StateType  # noqa: PLC0415 - avoids a cycle

    subset = StateType(sequences=state.sequences[live], stopped=state.stopped[live])
    stepped = env.step(subset, actions[live])

    sequences = state.sequences.copy()
    stopped = state.stopped.copy()
    sequences[live] = stepped.sequences
    stopped[live] = stepped.stopped
    return StateType(sequences=sequences, stopped=stopped)


def _multinomial(probabilities: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Draw one index per row, tolerating a generator on another device.

    ``torch.multinomial`` requires the generator and the tensor to share a
    device, so a run with ``device="cuda"`` and a CPU generator -- the obvious
    way to write a reproducible GPU experiment -- fails outright. Rather than
    make callers manage that, the draw is performed on the generator's device
    and the result moved back.

    Reproducibility is preserved exactly: the draw happens under the generator
    the caller supplied. The cost is one transfer of an ``(n, n_actions)``
    tensor per step, which is small beside the policy forward pass.
    """
    if generator is not None and generator.device != probabilities.device:
        drawn = torch.multinomial(
            probabilities.to(generator.device), num_samples=1, generator=generator
        )
        return drawn.to(probabilities.device).squeeze(1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)
