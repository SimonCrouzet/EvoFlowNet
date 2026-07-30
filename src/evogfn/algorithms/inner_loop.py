"""Gives a classical sampler the same proxy access a GFlowNet gets.

This exists to close a confound that would otherwise invalidate the headline
comparison. In a surrogate-assisted campaign the GFlowNet *optimises* the proxy
-- hundreds of gradient steps against it every round -- while a genetic algorithm
proposing blindly only ever meets the proxy as a filter on its output. Comparing
those two is not a comparison of samplers; it is a comparison of one method that
uses the model and one that does not, and the winner is known in advance.

Wrapping the baseline restores the symmetry. The wrapped sampler runs its own
search loop against the proxy for a fixed number of generations, seeing proxy
values through the same ``observe`` call it would use for assay results, and only
then hands its population up. Both methods now spend real oracle calls only on
the measured batch and unlimited free calls on the model.

This is what AdaLead and CbAS do, and it is the honest baseline. A GFlowNet that
still wins against it is winning on how it searches rather than on being the only
method allowed to look at the model.

The compute is not free, and is reported
----------------------------------------

``proxy_calls`` counts what the inner loop spent, so the trade against a sampler
that does no inner search is visible rather than implied. Budget-wise the two are
identical; wall-clock they are not, and a paper that hides that is making the
same mistake in the other direction.

The wrapper has to forward re-anchoring, or it silently undoes it
------------------------------------------------------------------

Every classical arm in the benchmark reaches the campaign wrapped in this class,
so the campaign's ``isinstance`` check against
[ReanchorableSampler][evogfn.loop.campaign.ReanchorableSampler] is a check on the
*wrapper*, not on the sampler inside it. A wrapper that did not implement the
hook would send every baseline down the rebuild path however carefully each one
had implemented it -- MLDE's training set discarded, an annealer's schedule
reset -- and nothing would raise, because rebuilding is a legitimate outcome.
The comparison would then be between a GFlowNet that carried its policy and a
set of baselines that were reconstructed from nothing every round, which is the
result the whole exercise exists to avoid manufacturing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler

if TYPE_CHECKING:
    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.surrogate.proxy import ProxyLandscape

#: Inner generations per round. Enough for a population method to converge on
#: the proxy without the wall-clock becoming the reason nobody runs the control.
DEFAULT_GENERATIONS = 50


class ProxyOptimising(Sampler):
    """Runs a sampler's own search against the proxy before proposing.

    Args:
        sampler: The search method to wrap. Driven through its ordinary
            ``propose``/``observe`` interface, so anything implementing
            [Sampler][evogfn.algorithms.base.Sampler] works unmodified.
        proxy: The surrogate-backed landscape to search against.
        generations: Inner iterations per round.
        population: Candidates evaluated per inner generation.

    Raises:
        ValueError: If ``generations`` or ``population`` is not positive.
    """

    def __init__(
        self,
        sampler: Sampler,
        *,
        proxy: ProxyLandscape,
        generations: int = DEFAULT_GENERATIONS,
        population: int = 256,
    ) -> None:
        """Wrap the sampler without running it."""
        super().__init__()
        for name, value in [("generations", generations), ("population", population)]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        self._sampler = sampler
        self._proxy = proxy
        self._generations = generations
        self._population = population
        self._proxy_calls = 0

    @property
    def name(self) -> str:
        """Short label marking that the baseline was given proxy access."""
        return f"{self._sampler.name} + proxy"

    @property
    def inner(self) -> Sampler:
        """The wrapped sampler, for inspection."""
        return self._sampler

    @property
    def proxy_calls(self) -> int:
        """Reward evaluations the inner loop spent on the proxy."""
        return self._proxy_calls

    def reanchored(self, env: MutationEnvironment) -> ProxyOptimising:
        """Move the wrapped sampler to ``env`` and re-wrap it unchanged.

        The wrapper itself owns nothing anchored: the proxy is a surrogate over
        sequences, and the generation and population counts are budgets. So this
        forwards and keeps its own accounting -- including ``proxy_calls``,
        which is a running total for the campaign and would understate the
        method's compute if it restarted at each anchor.

        Args:
            env: The re-anchored environment.

        Returns:
            A wrapper around the re-anchored inner sampler.

        Raises:
            TypeError: If the wrapped sampler cannot re-anchor. The wrapper
                cannot invent a re-anchoring the sampler does not have, and it
                must not return itself either: it would then be a wrapper whose
                inner sampler searches the previous round's Hamming ball, which
                is wrong in a way that produces plausible designs. Raising names
                the sampler so the caller can supply a ``sampler_factory``
                instead.
        """
        from evogfn.loop.campaign import ReanchorableSampler  # noqa: PLC0415 - avoids a cycle

        if not isinstance(self._sampler, ReanchorableSampler):
            raise TypeError(
                f"{self._sampler.name} does not implement reanchored(env), so the proxy "
                f"wrapper around it cannot move either; pass a sampler_factory to the "
                f"campaign to rebuild it instead"
            )
        moved = ProxyOptimising(
            self._sampler.reanchored(env),
            proxy=self._proxy,
            generations=self._generations,
            population=self._population,
        )
        moved._proxy_calls = self._proxy_calls
        moved._proposals_made = self._proposals_made
        return moved

    def propose(self, n: int) -> Tokens:
        """Search the proxy, then return ``n`` candidates from the result.

        Before the surrogate is fitted there is nothing to search against, so
        this is the bare sampler -- which is the correct behaviour for a
        campaign's first round.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.
        """
        if self._proxy.is_ready:
            for _ in range(self._generations):
                candidates = self._sampler.propose(self._population)
                self._sampler.observe(candidates, self._proxy.evaluate(candidates))
                self._proxy_calls += int(candidates.shape[0])

        proposals = self._sampler.propose(n)
        self._count(n)
        return np.asarray(proposals)

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Pass assay results through to the wrapped sampler.

        Args:
            sequences: The candidates that were measured.
            values: Their measured objective values.
        """
        self._sampler.observe(sequences, values)

    def __repr__(self) -> str:
        """Name the wrapped sampler and the inner budget."""
        return f"ProxyOptimising({self._sampler!r}, generations={self._generations})"
