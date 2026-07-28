"""Simulated annealing: the optimiser that is allowed to walk downhill.

Hill climbing is the floor for pure optimisation, and it is a floor with an
obvious defect: it cannot leave a local peak, and an epistatic fitness landscape
is very little except local peaks. Simulated annealing is the standard,
forty-year-old repair (Kirkpatrick, Gelatt & Vecchi, *Science* 1983) -- accept a
worse design with a probability that falls as the search cools, so early rounds
wander and late rounds refine.

It is here to close a specific way of overclaiming. A method that beats hill
climbing on a rugged landscape may be demonstrating only that hill climbing gets
stuck, which is a fact about hill climbing rather than evidence for anything
else. Annealing escapes local optima too, with no model and no learning, so an
advantage that survives comparison with it is an advantage of the model.

The comparison it sets up is sharper than that
----------------------------------------------

At a *fixed* temperature, Metropolis with acceptance ``min(1, exp(Δf / T))`` is a
sampler of ``exp(f / T)``, which is exactly the target a GFlowNet with reward
``R = exp(f / T)`` claims to sample from. The two methods therefore aim at the
same distribution, and differ only in how they get there: annealing runs a
sequential chain that needs one oracle call per step and mixes slowly on a rugged
landscape, while a GFlowNet amortises the same target into a policy that emits a
whole batch at once. A design round is a batch, not a chain, and that is the
asymmetry this baseline exists to expose.

Under a cooling schedule the chain instead concentrates on the optimum, so a
cooled annealer should look like hill climbing on diversity and on distributional
distance. Both regimes are reachable from the constructor: set ``cooling_rate``
to 1.0 for the fixed-temperature sampler, below 1.0 for the optimiser.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from evoflownet.algorithms.base import Sampler

if TYPE_CHECKING:
    from evoflownet.core.types import Fitness, Tokens
    from evoflownet.env.mutation import MutationEnvironment


class SimulatedAnnealing(Sampler):
    """Metropolis search over single substitutions, on a cooling schedule.

    Proposals are single-position substitutions of the current design; a round's
    scores are then fed through the Metropolis test in the order they arrive, and
    the temperature drops once per round.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        initial_temperature: Temperature the chain starts at. Sets what counts
            as a small loss in objective units, so no published value transfers
            across landscapes; 1.0 is our choice, and matches the reward
            temperature the GFlowNet rewards default to.
        cooling_rate: Multiplied into the temperature after every round --
            geometric cooling, the schedule Kirkpatrick et al. used. The ratio
            is our choice, and deliberately not the textbook 0.95: that figure
            assumes thousands of steps, whereas this schedule advances once per
            *round* and a campaign here is four rounds, so at 0.95 the chain
            would finish at 81% of its starting temperature having effectively
            never cooled. 0.5 takes it from 1.0 to 0.125 across four plates.
            Pass 1.0 for the fixed-temperature Metropolis sampler.
        min_temperature: Floor on the temperature, so the acceptance ratio never
            divides by zero and a long campaign degenerates gracefully into hill
            climbing rather than into a division error.
        feasible_only: Hold the current design rather than emit a proposal that
            is not constructible. Off by default, because a method that ignores
            the constraint is what the constraint experiment compares against.
        seed: Seeds proposals and the acceptance coin.

    Raises:
        ValueError: If a temperature is not positive, if ``cooling_rate`` is
            outside ``(0, 1]``, or if the floor exceeds the starting
            temperature.
    """

    def __init__(  # noqa: PLR0913 - an annealer is defined by its schedule
        self,
        env: MutationEnvironment,
        *,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.5,
        min_temperature: float = 1e-3,
        feasible_only: bool = False,
        seed: int = 0,
    ) -> None:
        """Start the chain at the parent, at the initial temperature."""
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError(f"initial_temperature must be positive, got {initial_temperature}")
        if min_temperature <= 0.0:
            raise ValueError(f"min_temperature must be positive, got {min_temperature}")
        if not 0.0 < cooling_rate <= 1.0:
            raise ValueError(f"cooling_rate must lie in (0, 1], got {cooling_rate}")
        if min_temperature > initial_temperature:
            raise ValueError(
                f"min_temperature {min_temperature} exceeds initial_temperature "
                f"{initial_temperature}; the chain would start below its own floor"
            )

        self._env = env
        self._cooling_rate = cooling_rate
        self._min_temperature = min_temperature
        self._feasible_only = feasible_only
        self._rng = np.random.default_rng(seed)
        self._temperature = initial_temperature
        self._current = env.parent
        # -inf, so the first finite measurement is always accepted: the chain has
        # to start somewhere, and refusing to move off an unscored parent would
        # freeze it for a whole round.
        self._current_value = -np.inf
        self._best_value = -np.inf

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "SimulatedAnnealing" + (" (feasible)" if self._feasible_only else "")

    @property
    def temperature(self) -> float:
        """Current temperature of the chain."""
        return self._temperature

    @property
    def current_value(self) -> float:
        """Objective value of the design the chain currently sits on."""
        return self._current_value

    @property
    def best_value(self) -> float:
        """Best objective value observed so far, accepted or not."""
        return self._best_value

    def propose(self, n: int) -> Tokens:
        """Return ``n`` single-substitution neighbours of the current design.

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
            # A position already differing from the parent may be changed again.
            # The environment forbids mutating a position twice along one
            # *trajectory*; the sequence that results from revising an earlier
            # substitution is a different point in the same Hamming ball,
            # reached by a different path, and is perfectly reachable. Treating
            # the trajectory constraint as a constraint on states would forbid
            # the chain from ever undoing a move -- which for an annealer, whose
            # whole purpose is to accept moves it may later need to undo, is
            # fatal rather than merely limiting.
            # At the budget only already-substituted positions may change, since
            # touching a fresh one would push the design out of the graph.
            mutated = np.flatnonzero(proposals[row] != parent)
            at_budget = mutated.size >= self._env.max_mutations
            available = mutated if at_budget else np.arange(length)
            position = int(self._rng.choice(available))
            alternatives = [t for t in range(size) if t != proposals[row, position]]
            proposals[row, position] = self._rng.choice(alternatives)

        if self._feasible_only:
            # Hold position rather than resample: the current design is feasible
            # by induction -- it was itself a feasible proposal -- so this cannot
            # emit anything outside the graph, and it costs no extra proposals.
            reachable = self._env.is_reachable(proposals)
            proposals[~reachable] = self._current

        self._count(n)
        return proposals

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Run the Metropolis test over the batch, then cool.

        The batch is tested sequentially, each candidate against wherever the
        chain has got to. This is an approximation and worth naming: textbook
        annealing draws each proposal from the *current* point, whereas all of
        these were drawn from wherever the chain stood at the start of the round.
        The approximation is forced by the setting rather than chosen -- a plate
        is designed in one go and assayed in one go, so a strictly sequential
        chain would need one round per step and would spend a four-plate budget
        on four moves.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        candidates = np.asarray(sequences)

        for row, value in zip(candidates, flat, strict=False):
            if not np.isfinite(value):
                # An infeasible or failed assay is not evidence about the
                # landscape, so it neither moves the chain nor counts against it.
                continue
            if value > self._best_value:
                self._best_value = float(value)
            if self._accepts(float(value)):
                self._current = row.copy()
                self._current_value = float(value)

        self._temperature = max(self._min_temperature, self._temperature * self._cooling_rate)

    def _accepts(self, value: float) -> bool:
        """Whether the Metropolis rule admits a move to a design scoring ``value``."""
        delta = value - self._current_value
        if delta >= 0.0:
            return True
        # exp underflows to 0.0 for a large enough loss, which is the intended
        # answer -- a catastrophic move is refused, not an error.
        return bool(self._rng.random() < math.exp(delta / self._temperature))
