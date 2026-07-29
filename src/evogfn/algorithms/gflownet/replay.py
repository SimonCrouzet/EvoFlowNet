"""Scoring a sequence that the policy did not itself produce.

Trajectory balance is off-policy, so it can be trained on trajectories from any
source -- a replay buffer, a genetic algorithm, an existing assay. What it
cannot do is train on a bare *sequence*: the loss needs a path through the graph,
and a terminal state does not come with one.

Recovering a path is the awkward step in off-policy GFlowNet training generally.
For graph-structured objects it is an inverse problem, since you must work out
how the policy would have constructed the object. On a mutation lattice it is
almost free: any ordering of a variant's mutation set is a valid trajectory to
it.

The right way to draw one is **backward from the terminal under** ``P_B``, not by
permuting the mutation set uniformly. The two coincide only when nothing is
masked. Under a feasibility constraint some orderings pass through states the
environment forbids, so uniform permutation would propose paths that do not
exist, and quietly bias the estimator. Walking backward through the backward
mask can only ever traverse real edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.gflownet.sampling import Trajectories, _multinomial
from evogfn.env.base import State
from evogfn.models.policy import to_tensor

if TYPE_CHECKING:
    from evogfn.core.types import Tokens
    from evogfn.env.base import SequenceEnvironment
    from evogfn.models.policy import SequencePolicy


def replay_trajectories(
    env: SequenceEnvironment,
    policy: SequencePolicy,
    terminals: Tokens,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Trajectories:
    """Score externally supplied sequences under the policy.

    A path to each terminal is drawn backward under ``P_B``, then scored forward
    under ``P_F``. Both directions traverse the same edges, which is what makes
    the resulting loss term a valid trajectory balance contribution.

    Args:
        env: The construction graph.
        policy: The policy to score under.
        terminals: An ``(n, length)`` array of sequences to score.
        generator: Torch generator, for reproducible path selection.
        device: Where to run the policy.

    Returns:
        Trajectories carrying ``log P_F`` and ``log P_B`` for the supplied
        terminals, with gradients.

    Raises:
        ValueError: If ``terminals`` is not a batch of the right width, or a
            sequence cannot be reached from the environment's source -- which
            means it is not in the space the policy is defined over, and
            training on it would be meaningless rather than merely wrong.
    """
    sequences = np.asarray(terminals)
    if sequences.ndim != 2 or sequences.shape[1] != env.sequence_length:  # noqa: PLR2004
        raise ValueError(f"expected shape (n, {env.sequence_length}), got {tuple(sequences.shape)}")
    unreachable = ~env.is_reachable(sequences)
    if unreachable.any():
        raise ValueError(
            f"sequences {np.flatnonzero(unreachable).tolist()} are not constructible in "
            f"this environment, so scoring them is meaningless rather than inaccurate"
        )

    if sequences.shape[0] == 0:
        empty = torch.zeros(0, device=device)
        return Trajectories(
            terminal=sequences,
            log_forward=empty,
            log_backward=empty,
            lengths=np.zeros(0, dtype=np.int64),
        )

    paths = _backward_paths(env, policy, sequences, generator, device)
    return _score_forward(env, policy, sequences, paths, device)


def _backward_paths(
    env: SequenceEnvironment,
    policy: SequencePolicy,
    terminals: Tokens,
    generator: torch.Generator | None,
    device: torch.device | str,
) -> list[list[int]]:
    """Draw one path per terminal by walking backward under ``P_B``.

    Returns:
        For each terminal, the actions in forward order, ending with the stop
        action.
    """
    n = terminals.shape[0]
    state = State(sequences=terminals.copy(), stopped=np.ones(n, dtype=np.bool_))
    reversed_paths: list[list[int]] = [[] for _ in range(n)]
    active = np.ones(n, dtype=np.bool_)

    for _ in range(env.sequence_length + 1):
        if not active.any():
            break
        backward_mask = torch.as_tensor(env.backward_mask(state), device=device)
        forward_mask = torch.as_tensor(env.forward_mask(state), device=device)
        _, backward_log_probs = policy.log_probs(
            to_tensor(state.sequences, device), forward_mask, backward_mask
        )

        probabilities = backward_log_probs.exp()
        exhausted = probabilities.sum(dim=-1) <= 0.0
        active &= ~exhausted.cpu().numpy()
        if not active.any():
            break

        # Parked rows need a valid distribution for multinomial; their draw is
        # discarded by the active mask.
        probabilities = probabilities.clone()
        probabilities[exhausted, -1] = 1.0
        actions = _multinomial(probabilities, generator)

        for row in np.flatnonzero(active):
            reversed_paths[row].append(int(actions[row]))
        state = _backward_step_active(env, state, actions.cpu().numpy(), active)

    return [list(reversed(path)) for path in reversed_paths]


def _backward_step_active(
    env: SequenceEnvironment,
    state: State,
    actions: np.ndarray,
    active: np.ndarray,
) -> State:
    """Undo one action for the active rows only."""
    if active.all():
        return env.backward_step(state, actions)
    subset = State(sequences=state.sequences[active], stopped=state.stopped[active])
    stepped = env.backward_step(subset, actions[active])
    sequences = state.sequences.copy()
    stopped = state.stopped.copy()
    sequences[active] = stepped.sequences
    stopped[active] = stepped.stopped
    return State(sequences=sequences, stopped=stopped)


def _score_forward(
    env: SequenceEnvironment,
    policy: SequencePolicy,
    terminals: Tokens,
    paths: list[list[int]],
    device: torch.device | str,
) -> Trajectories:
    """Walk each path forward from the source, accumulating log probabilities."""
    n = terminals.shape[0]
    state = env.initial(n)
    log_forward = torch.zeros(n, device=device)
    log_backward = torch.zeros(n, device=device)
    lengths = np.array([len(path) for path in paths], dtype=np.int64)
    longest = int(lengths.max()) if n else 0

    for depth in range(longest):
        live = np.array([depth < len(path) for path in paths], dtype=np.bool_)
        if not live.any():
            break
        actions = np.array(
            [paths[i][depth] if live[i] else env.n_actions - 1 for i in range(n)],
            dtype=np.int64,
        )

        forward_mask = torch.as_tensor(env.forward_mask(state), device=device)
        backward_mask = torch.as_tensor(env.backward_mask(state), device=device)
        forward_log_probs, _ = policy.log_probs(
            to_tensor(state.sequences, device), forward_mask, backward_mask
        )
        action_tensor = torch.as_tensor(actions, device=device)
        live_tensor = torch.as_tensor(live, device=device)

        step_log_pf = forward_log_probs.gather(1, action_tensor[:, None]).squeeze(1)
        log_forward = log_forward + torch.where(
            live_tensor, step_log_pf, torch.zeros_like(step_log_pf)
        )

        next_state = _forward_step_active(env, state, actions, live)

        next_forward = torch.as_tensor(env.forward_mask(next_state), device=device)
        next_backward = torch.as_tensor(env.backward_mask(next_state), device=device)
        _, backward_log_probs = policy.log_probs(
            to_tensor(next_state.sequences, device), next_forward, next_backward
        )
        step_log_pb = backward_log_probs.gather(1, action_tensor[:, None]).squeeze(1)
        contributes = live_tensor & (action_tensor != env.n_actions - 1)
        log_backward = log_backward + torch.where(
            contributes, step_log_pb, torch.zeros_like(step_log_pb)
        )
        state = next_state

    return Trajectories(
        terminal=env.to_sequences(state),
        log_forward=log_forward,
        log_backward=log_backward,
        lengths=lengths,
    )


def _forward_step_active(
    env: SequenceEnvironment,
    state: State,
    actions: np.ndarray,
    live: np.ndarray,
) -> State:
    """Apply one action for the live rows only."""
    if live.all():
        return env.step(state, actions)
    subset = State(sequences=state.sequences[live], stopped=state.stopped[live])
    stepped = env.step(subset, actions[live])
    sequences = state.sequences.copy()
    stopped = state.stopped.copy()
    sequences[live] = stepped.sequences
    stopped[live] = stepped.stopped
    return State(sequences=sequences, stopped=stopped)
