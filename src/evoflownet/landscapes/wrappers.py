"""Wrappers that add assay-like behaviour to any landscape.

A closed-form landscape is a perfect oracle: noiseless, instant, and free. A real
assay is none of those. These wrappers add the missing properties without any
landscape needing to know about them, and they compose:

    >>> from evoflownet.landscapes import EhrlichLandscape
    >>> assay = Budgeted(Noisy(EhrlichLandscape(seed=0), scale=0.05, seed=1), max_evaluations=500)

**Order matters, and expresses a real choice.** Wrapping is applied inside-out,
so the example above adds noise first and then counts. Two orderings in
particular say different things:

* ``Budgeted(Cached(landscape))`` -- a repeat measurement is free. This models
  looking up a result you already have.
* ``Cached(Budgeted(landscape))`` -- a repeat measurement costs budget. This
  models actually running the assay again.

The first is almost always what a benchmark wants; the second is what a wet lab
does. Neither is a default, so callers choose explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evoflownet.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Alphabet, Fitness, Tokens


class BudgetExhaustedError(RuntimeError):
    """Raised when a budgeted landscape is asked for more evaluations than it has.

    Deliberately an error rather than a silent clamp. A run that quietly stops
    measuring produces results that look like a completed experiment, and the
    comparison between methods -- which is only meaningful at equal budget --
    becomes wrong in a way nothing reveals.
    """


class LandscapeWrapper(FitnessLandscape):
    """Base class for wrappers that delegate to an inner landscape.

    Forwards everything structural -- alphabet, length, objectives, feasibility,
    ground truth -- so a subclass only overrides what it actually changes.

    Args:
        landscape: The landscape to wrap.
    """

    def __init__(self, landscape: FitnessLandscape) -> None:
        """Store the wrapped landscape."""
        self._inner = landscape

    @property
    def inner(self) -> FitnessLandscape:
        """The wrapped landscape."""
        return self._inner

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._inner.alphabet

    @property
    def sequence_length(self) -> int:
        """Length of every sequence this landscape scores."""
        return self._inner.sequence_length

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return self._inner.n_objectives

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Names of the objectives."""
        return self._inner.objective_names

    @property
    def optimum(self) -> Fitness | None:
        """Best attainable objective values, or ``None`` if unknown."""
        return self._inner.optimum

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Delegate feasibility to the wrapped landscape.

        Args:
            sequences: An ``(n, sequence_length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.
        """
        return self._inner.is_feasible(sequences)

    def enumerate(self) -> Tokens:
        """Delegate enumeration to the wrapped landscape.

        Returns:
            Every sequence in the search space.
        """
        return self._inner.enumerate()

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Delegate scoring to the wrapped landscape."""
        return self._inner.evaluate(sequences)


class Noisy(LandscapeWrapper):
    """Adds Gaussian measurement noise, as a real assay would.

    Noise is drawn fresh on every call, so evaluating the same sequence twice
    gives different answers -- which is the point. A method that treats one
    measurement as truth should degrade here, and one that averages or models
    uncertainty should not.

    The true values remain reachable through :attr:`inner`, so metrics can be
    computed against ground truth while the search only ever sees noise.

    Args:
        landscape: The landscape to wrap.
        scale: Standard deviation of the noise, per objective. A scalar applies
            to all objectives.
        seed: Seed for the noise stream.
        clip_to_optimum: If ``True``, clip noisy values so they never exceed the
            known optimum. Off by default: a measurement that overshoots is
            realistic, and silently clipping would make regret look better than
            it is.
    """

    def __init__(
        self,
        landscape: FitnessLandscape,
        *,
        scale: float = 0.1,
        seed: int = 0,
        clip_to_optimum: bool = False,
    ) -> None:
        """Wrap a landscape with additive Gaussian noise."""
        super().__init__(landscape)
        if scale < 0:
            raise ValueError(f"noise scale must be non-negative, got {scale}")
        self._scale = scale
        self._rng = np.random.default_rng(seed)
        self._clip_to_optimum = clip_to_optimum

    @property
    def scale(self) -> float:
        """Standard deviation of the added noise."""
        return self._scale

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score the sequences, then perturb the finite values."""
        values = np.asarray(self._inner.evaluate(sequences), dtype=np.float64).copy()
        # Leave -inf alone. An infeasible sequence is not a noisy measurement of
        # a feasible one, and -inf + noise is still -inf anyway.
        finite = np.isfinite(values)
        values[finite] += self._rng.normal(0.0, self._scale, size=int(finite.sum()))
        optimum = self._inner.optimum
        if self._clip_to_optimum and optimum is not None:
            values = np.minimum(values, optimum[None, :])
        return values


class Budgeted(LandscapeWrapper):
    """Enforces a hard limit on the number of sequence evaluations.

    The evaluation budget is what makes directed evolution a hard problem, and
    comparing two methods is only meaningful when both spent the same one. Making
    the limit a wrapper rather than a convention means it cannot be bypassed by a
    caller that forgets to count.

    Args:
        landscape: The landscape to wrap.
        max_evaluations: Total sequences that may be scored.

    Raises:
        ValueError: If ``max_evaluations`` is negative.
    """

    def __init__(self, landscape: FitnessLandscape, *, max_evaluations: int) -> None:
        """Wrap a landscape with an evaluation budget."""
        super().__init__(landscape)
        if max_evaluations < 0:
            raise ValueError(f"max_evaluations must be non-negative, got {max_evaluations}")
        self._max_evaluations = max_evaluations
        self._used = 0

    @property
    def max_evaluations(self) -> int:
        """Total evaluations permitted."""
        return self._max_evaluations

    @property
    def used(self) -> int:
        """Evaluations spent so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Evaluations still available."""
        return self._max_evaluations - self._used

    def reset(self) -> None:
        """Restore the full budget, for starting another run on this landscape."""
        self._used = 0

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Charge the batch against the budget, then score it.

        Raises:
            BudgetExhaustedError: If the batch does not fit in the remaining
                budget. The batch is rejected whole rather than partly served,
                so a caller cannot end up with fewer results than it asked for
                without noticing.
        """
        requested = sequences.shape[0]
        if requested > self.remaining:
            raise BudgetExhaustedError(
                f"requested {requested} evaluations with {self.remaining} remaining "
                f"of {self._max_evaluations}"
            )
        self._used += requested
        return self._inner.evaluate(sequences)


class Cached(LandscapeWrapper):
    """Remembers results so a repeated sequence is scored only once.

    Directed evolution revisits sequences constantly -- a mutation that undoes an
    earlier one lands back where it started -- so this is usually a large saving.

    Whether a cache hit costs budget is decided by wrapping order, not by this
    class: see the module docstring.

    Note:
        Caching a :class:`Noisy` landscape freezes the first measurement of each
        sequence and returns it forever, which removes exactly the repeated
        sampling that makes noise interesting. That combination is almost
        certainly a mistake, so the constructor rejects it.

    Args:
        landscape: The landscape to wrap.

    Raises:
        ValueError: If the wrapped landscape is stochastic.
    """

    def __init__(self, landscape: FitnessLandscape) -> None:
        """Wrap a landscape with a result cache."""
        if isinstance(landscape, Noisy):
            raise ValueError(
                "caching a Noisy landscape would freeze the first measurement of each "
                "sequence, defeating the noise; wrap the other way round if you meant "
                "to cache the underlying truth"
            )
        super().__init__(landscape)
        self._cache: dict[bytes, npt.NDArray[np.float64]] = {}
        self._hits = 0

    @property
    def hits(self) -> int:
        """Number of sequences answered from the cache."""
        return self._hits

    @property
    def size(self) -> int:
        """Number of distinct sequences remembered."""
        return len(self._cache)

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score only the sequences not already known, then reassemble the batch."""
        keys = [row.astype(np.int32).tobytes() for row in sequences]
        if not keys:
            return np.zeros((0, self.n_objectives), dtype=np.float64)

        unknown = [i for i, key in enumerate(keys) if key not in self._cache]
        self._hits += len(keys) - len(unknown)

        if unknown:
            fresh = np.asarray(self._inner.evaluate(sequences[unknown]), dtype=np.float64)
            for position, values in zip(unknown, fresh, strict=True):
                self._cache[keys[position]] = values

        return np.stack([self._cache[key] for key in keys])
