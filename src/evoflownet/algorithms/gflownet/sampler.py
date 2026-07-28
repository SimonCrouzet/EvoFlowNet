"""A GFlowNet behind the same interface as the classical baselines.

This is what makes the comparison a config change rather than two harnesses.
The campaign asks for proposals; this sampler retrains its policy against the
current proxy and samples from it. Everything the loop knows about a genetic
algorithm it also knows about this.

Where the oracle budget does *not* go
-------------------------------------

Training happens against a :class:`~evoflownet.surrogate.proxy.ProxyLandscape`,
never the oracle. A thousand gradient steps at batch 64 is 64,000 reward
evaluations; charging those would exhaust a realistic campaign budget of a few
hundred before the first round returned. Because the proxy holds no oracle, the
separation is structural rather than a matter of remembering.

Retraining, not fine-tuning
---------------------------

The policy is retrained from its current parameters each round rather than reset.
The proxy changes between rounds -- it is refitted on strictly more data -- so
the target distribution moves, and a policy already near the previous target is
a better starting point than a fresh one. This is warm-starting, and it is why a
round costs a fraction of what training from scratch would.

The first round has no proxy to train against: nothing has been measured yet, so
the surrogate is unfitted. The policy samples from its initialisation, which
under a masked environment is close to uniform over the *feasible* set. That is
not a fallback -- it is the feasibility-by-construction property doing its work
before any model exists, and it is the fairest possible seed design.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.gflownet.genetic_gfn import train_genetic_gfn
from evoflownet.algorithms.gflownet.sampling import sample_trajectories
from evoflownet.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance

if TYPE_CHECKING:
    from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
    from evoflownet.algorithms.gflownet.genetic_gfn import GeneticConfig
    from evoflownet.algorithms.gflownet.objectives import GFlowNetObjective
    from evoflownet.core.types import Tokens
    from evoflownet.env.base import SequenceEnvironment
    from evoflownet.models.policy import SequencePolicy
    from evoflownet.rewards.base import Reward
    from evoflownet.surrogate.proxy import ProxyLandscape


class GFlowNetSampler(Sampler):
    """Trains a GFlowNet against a proxy and samples proportionally to it.

    Args:
        env: The construction graph. Feasibility lives here, as masks.
        policy: The policy to train, modified in place across rounds.
        proxy: The surrogate-backed landscape training optimises against.
            Never the oracle.
        reward: Transforms proxy values into log rewards.
        config: Training settings applied on every retrain.
        objective: How balance violation is measured. Defaults to trajectory
            balance.
        genetic: A genetic algorithm to use as the policy's teacher. With one,
            training runs through :func:`train_genetic_gfn` -- the GA
            recombines the best of a rank-based buffer and its offspring are
            replayed into the training batch. Kim et al. report this closing a
            58% deficit against Mol GA on PMO, and directed evolution *is* a
            genetic algorithm, so it is the variant most likely to matter here.
        genetic_config: How much guidance to apply. Ignored without ``genetic``.
        seed: Seeds proposal sampling, independently of training.
    """

    def __init__(  # noqa: PLR0913 - the sampler is defined by its parts
        self,
        env: SequenceEnvironment,
        policy: SequencePolicy,
        *,
        proxy: ProxyLandscape,
        reward: Reward,
        config: TrainingConfig | None = None,
        objective: GFlowNetObjective | None = None,
        genetic: GeneticAlgorithm | None = None,
        genetic_config: GeneticConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Store the training setup without running it."""
        super().__init__()
        self._env = env
        self._policy = policy
        self._proxy = proxy
        self._reward = reward
        self._config = config or TrainingConfig()
        self._objective = objective
        self._genetic = genetic
        self._genetic_config = genetic_config
        self._generator: torch.Generator | None = None
        self._seed = seed
        self._rounds_trained = 0
        self._proxy_calls = 0

    @property
    def name(self) -> str:
        """Short label naming the objective the policy was trained under."""
        objective = type(self._objective).__name__ if self._objective else "TrajectoryBalance"
        teacher = " + GA" if self._genetic is not None else ""
        return f"GFlowNet ({objective}){teacher}"

    @property
    def rounds_trained(self) -> int:
        """How many times the policy has been retrained."""
        return self._rounds_trained

    @property
    def proxy_calls(self) -> int:
        """Reward evaluations spent on the proxy.

        Free in budget terms and expensive in compute terms. Reported so the
        trade against a baseline that does no training is visible rather than
        implied.
        """
        return self._proxy_calls

    def propose(self, n: int) -> Tokens:
        """Retrain against the current proxy, then sample ``n`` designs.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array of terminal states.
        """
        if self._proxy.is_ready:
            # A distinct seed per round, or every round replays the same
            # trajectories and the later rounds teach nothing.
            config = replace(self._config, seed=self._config.seed + self._rounds_trained)
            if self._genetic is not None:
                result = train_genetic_gfn(
                    self._env,
                    self._policy,
                    self._proxy,
                    self._reward,
                    config,
                    genetic=self._genetic,
                    genetic_config=self._genetic_config,
                    objective=self._objective,
                )
            else:
                result = train_trajectory_balance(
                    self._env,
                    self._policy,
                    self._proxy,
                    self._reward,
                    config,
                    objective=self._objective,
                )
            self._proxy_calls += result.oracle_calls
            self._rounds_trained += 1

        trajectories = sample_trajectories(
            self._env, self._policy, n, epsilon=0.0, generator=self._sampling_generator()
        )
        self._count(n)
        return np.asarray(trajectories.terminal)

    def _sampling_generator(self) -> torch.Generator:
        """One generator across rounds, so proposals do not repeat themselves."""
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(self._seed)
        return self._generator

    def __repr__(self) -> str:
        """Name the sampler and how much training it has had."""
        return f"GFlowNetSampler(rounds_trained={self._rounds_trained})"
