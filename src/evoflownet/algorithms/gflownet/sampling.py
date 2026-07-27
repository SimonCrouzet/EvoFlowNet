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

from dataclasses import dataclass
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
    """

    terminal: Tokens
    log_forward: torch.Tensor
    log_backward: torch.Tensor
    lengths: npt_int

    def __len__(self) -> int:
        """Number of trajectories."""
        return int(self.terminal.shape[0])


if TYPE_CHECKING:
    import numpy.typing as npt

    npt_int = npt.NDArray[np.integer]
else:  # pragma: no cover - only an alias for annotations
    npt_int = np.ndarray


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
