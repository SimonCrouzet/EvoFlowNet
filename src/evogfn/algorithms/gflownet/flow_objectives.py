r"""The detailed-balance family: objectives that constrain flow through states.

Trajectory balance compares whole trajectories against one global scalar. That
gives an unbiased signal but a high-variance one: a single number has to carry
credit for every action in a rollout, so a trajectory that ends badly penalises
its good early steps equally. The objectives here instead constrain each
*transition*, using a learned per-state flow $F(s)$.

Detailed balance
----------------

For every edge $s \to s'$:

$$
F(s) P_F(s'|s) = F(s') P_B(s|s')
$$

with $F(x) = R(x)$ at terminal states, which is what anchors the whole
system to the reward. Credit is assigned locally, so the variance is lower --
but the flow estimate has to be learned everywhere, and early in training it is
wrong everywhere, which is the trade.

Sub-trajectory balance
----------------------

Malkin et al. and Madan et al. (ICML 2023) interpolate: constrain every
*sub-trajectory* $s_i \to \dots \to s_j$, weighting by
$\lambda^{j-i}$. At $\lambda \to 0$ only single transitions count and
this is detailed balance; as $\lambda$ grows, longer sub-trajectories
dominate and it approaches trajectory balance. One knob spans both, which is why
it is the better experiment than running the two endpoints separately.

Forward-looking detailed balance
--------------------------------

Pan et al. (2023) reparameterise $\log F(s) = \log R(s) + \log \tilde F(s)$,
so the network predicts a *correction* to a reward already known at $s$
rather than the flow from nothing. The telescoping intermediate rewards cancel
along a trajectory, so the objective is unchanged at optimum; what changes is
that the network starts from a good guess instead of zero.

**This environment is unusually well suited to it, and that is worth saying
plainly.** The usual objection to forward-looking rewards is that a partial
object has no reward -- half an autoregressively built sequence is not a
sequence, and there is nothing to evaluate. In the mutation lattice every
intermediate state *is* a complete, scorable sequence: it is the parent with
some subset of the mutations applied. So $R(s)$ is available at every
state for free, and the reparameterisation is exact rather than heuristic. A
method that is a workaround elsewhere is a natural fit here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.gflownet.objectives import GFlowNetObjective

if TYPE_CHECKING:
    from evogfn.algorithms.gflownet.sampling import Trajectories
    from evogfn.models.policy import SequencePolicy

#: Default weighting for sub-trajectory balance. One is uniform over lengths.
DEFAULT_LAMBDA = 0.9


class FlowObjective(GFlowNetObjective):
    """Shared machinery for objectives that need a per-state flow estimate."""

    @property
    def uses_log_z(self) -> bool:
        """These objectives learn ``F(s_0)`` instead of a free-standing ``log Z``."""
        return False

    @property
    def needs_state_rewards(self) -> bool:
        """Whether the trainer must supply a reward at every visited state."""
        return False

    @staticmethod
    def _flows(
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
        state_log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate ``log F`` at every visited state, anchored at the terminals.

        Terminal states take ``log R(x)`` rather than the network's estimate,
        which is the boundary condition that ties the flow system to the reward.
        Without it the objective is satisfied by any constant flow.
        """
        steps = trajectories.require_steps()
        states = np.asarray(steps.states)
        n, total, length = states.shape
        flat = torch.as_tensor(
            states.reshape(-1, length).astype(np.int64), device=log_rewards.device
        )
        log_flow = policy.log_flow(flat).reshape(n, total)
        if state_log_rewards is not None:
            # Forward-looking parameterisation: the head predicts a correction
            # to the reward already known at this state.
            log_flow = log_flow + state_log_rewards

        # Everything from the terminal state onward is the terminal state, so
        # clamping the whole tail to log R is both correct and simpler than
        # indexing the exact endpoint.
        reached = np.concatenate(
            [np.zeros((n, 1), dtype=bool), np.cumsum(steps.stopping, axis=1) > 0], axis=1
        )
        terminal = torch.as_tensor(reached, device=log_flow.device)
        return torch.where(terminal, log_rewards[:, None].expand_as(log_flow), log_flow)


class DetailedBalance(FlowObjective):
    """Constrains every transition, following Bengio et al. (JMLR 2023).

    Lower variance than trajectory balance because credit is assigned to the
    action that earned it, at the cost of having to learn a flow at every state.
    """

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Mean squared detailed-balance violation over real transitions.

        Args:
            trajectories: Completed trajectories, with their per-step record.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``.
            policy: Supplies both the policies and the flow estimate.

        Returns:
            A scalar loss.
        """
        steps = trajectories.require_steps()
        log_flow = self._flows(trajectories, log_rewards, policy)
        active = torch.as_tensor(steps.active, device=log_flow.device)

        # The stop action's reverse is not a graph edge, so it carries no P_B
        # term -- exactly as the sampler records it.
        violation = log_flow[:, :-1] + steps.log_forward - log_flow[:, 1:] - steps.log_backward
        stopping = torch.as_tensor(steps.stopping, device=log_flow.device)
        violation = torch.where(
            stopping,
            log_flow[:, :-1] + steps.log_forward - log_rewards[:, None],
            violation,
        )
        return _masked_mean(violation.pow(2), active)


class SubTrajectoryBalance(FlowObjective):
    r"""Constrains every sub-trajectory, weighted by $\lambda^{\text{length}}$.

    One knob spans detailed balance and trajectory balance, which makes the
    comparison between them a sweep rather than two implementations that might
    differ for incidental reasons.

    Args:
        lam: Weight per unit sub-trajectory length. Values below one favour
            short sub-trajectories (toward detailed balance); larger values
            favour long ones (toward trajectory balance).

    Raises:
        ValueError: If ``lam`` is not positive.
    """

    def __init__(self, *, lam: float = DEFAULT_LAMBDA) -> None:
        """Store the length weighting."""
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        self._lam = lam

    @property
    def lam(self) -> float:
        """Weight per unit sub-trajectory length."""
        return self._lam

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Weighted mean squared violation over all sub-trajectories.

        Args:
            trajectories: Completed trajectories, with their per-step record.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``.
            policy: Supplies both the policies and the flow estimate.

        Returns:
            A scalar loss.
        """
        steps = trajectories.require_steps()
        log_flow = self._flows(trajectories, log_rewards, policy)
        device = log_flow.device
        active = torch.as_tensor(steps.active, device=device)

        # Cumulative sums turn "sum over any sub-trajectory" into a difference,
        # so the whole O(T^2) family costs two prefix scans rather than a loop.
        zero = torch.zeros(len(trajectories), 1, device=device)
        forward = torch.cat([zero, torch.cumsum(steps.log_forward * active, dim=1)], dim=1)
        backward = torch.cat([zero, torch.cumsum(steps.log_backward * active, dim=1)], dim=1)

        total = log_flow.shape[1]
        losses = []
        weights = []
        for i in range(total):
            for j in range(i + 1, total):
                # A sub-trajectory is only real if every step it spans was.
                spans = active[:, i:j].all(dim=1)
                violation = (
                    log_flow[:, i]
                    + (forward[:, j] - forward[:, i])
                    - log_flow[:, j]
                    - (backward[:, j] - backward[:, i])
                )
                losses.append(violation.pow(2))
                weights.append(spans.to(violation.dtype) * self._lam ** (j - i))

        stacked = torch.stack(losses, dim=1)
        weighting = torch.stack(weights, dim=1)
        return (stacked * weighting).sum() / weighting.sum().clamp(min=1e-12)

    def __repr__(self) -> str:
        """Name the objective and its weighting."""
        return f"SubTrajectoryBalance(lam={self._lam})"


class ForwardLookingDetailedBalance(DetailedBalance):
    """Detailed balance with the flow reparameterised around known rewards.

    Pan et al. (2023). The network predicts ``log F(s) - log R(s)`` rather than
    ``log F(s)``, so it starts from a good guess instead of from zero.

    Applicable here because every state in the mutation lattice is a complete
    sequence and therefore scorable -- the reason this method is usually a
    workaround does not apply.
    """

    @property
    def needs_state_rewards(self) -> bool:
        """This objective requires a reward at every visited state."""
        return True

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Mean squared violation under the forward-looking parameterisation.

        Args:
            trajectories: Completed trajectories, carrying per-state rewards.
            log_rewards: An ``(n,)`` tensor of ``log R(x)``.
            policy: Supplies both the policies and the flow correction.

        Returns:
            A scalar loss.

        Raises:
            ValueError: If the state rewards are missing. Falling back to plain
                detailed balance would report a result under the wrong name.
        """
        state_log_rewards = trajectories.state_log_rewards
        if state_log_rewards is None:
            raise ValueError(
                "forward-looking detailed balance needs a reward at every visited "
                "state; the trainer supplies them when needs_state_rewards is set"
            )
        steps = trajectories.require_steps()
        log_flow = self._flows(trajectories, log_rewards, policy, state_log_rewards)
        device = log_flow.device
        active = torch.as_tensor(steps.active, device=device)
        stopping = torch.as_tensor(steps.stopping, device=device)

        violation = log_flow[:, :-1] + steps.log_forward - log_flow[:, 1:] - steps.log_backward
        violation = torch.where(
            stopping,
            log_flow[:, :-1] + steps.log_forward - log_rewards[:, None],
            violation,
        )
        return _masked_mean(violation.pow(2), active)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the entries the mask selects, safe when it selects none."""
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)
