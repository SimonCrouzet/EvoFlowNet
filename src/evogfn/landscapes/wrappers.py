"""Wrappers that add assay-like behaviour to any landscape.

A closed-form landscape is a perfect oracle: noiseless, instant, and free. A real
assay is none of those. These wrappers add the missing properties without any
landscape needing to know about them, and they compose:

    >>> from evogfn.landscapes import EhrlichLandscape
    >>> assay = Budgeted(Noisy(EhrlichLandscape(seed=0), scale=0.05, seed=1), max_evaluations=500)

**Order matters.** A call enters the outermost wrapper first, so whichever of
:class:`Budgeted` and :class:`Cached` is on the outside decides what the budget
actually counts:

* ``Cached(Budgeted(landscape))`` -- the cache answers first, so the budget only
  ever sees sequences it has not scored before. **The budget counts distinct
  sequences.** Re-requesting a known sequence is free.
* ``Budgeted(Cached(landscape))`` -- the budget is charged before the cache is
  consulted. **The budget counts requests.** The cache still avoids recomputing,
  but a repeat costs budget just the same.

Which is right depends on what the budget represents. If it stands for a
screening capacity -- how many distinct variants can be made and assayed -- put
the cache outside. If it stands for the number of oracle calls a method is
permitted, which is the fairer basis for comparing methods that revisit
sequences at different rates, put the budget outside.

Neither is a default, because choosing silently would change what a reported
budget means.

**Two noise models, and they are not interchangeable.** :class:`Noisy` adds
homoscedastic Gaussian noise: the same uncertainty everywhere, so a sampler that
climbs to the top of the landscape finds the measurements there exactly as
trustworthy as the ones at the bottom. Real selection assays do not behave that
way, and :class:`SelectionNoisy` is the wrapper that reproduces what they
actually do -- see its docstring for the finding it exists to model. Use
:class:`Noisy` when you want a controlled, analytically simple perturbation; use
:class:`SelectionNoisy` when the claim being made is about robustness to *assay*
noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Fitness, Tokens


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

    #: Whether repeating a call can return a different answer. :class:`Cached`
    #: refuses anything that declares itself stochastic, since caching freezes
    #: the first measurement of each sequence and turns a noise model into a
    #: lookup table. Declared as a marker rather than checked with an isinstance
    #: tuple in :class:`Cached`, because a tuple has to be edited every time a
    #: noise model is added and nothing fails when it is not -- the new wrapper
    #: is simply cacheable, and the resulting numbers look ordinary. A class that
    #: draws randomly sets this to ``True`` beside the code that draws.
    stochastic: ClassVar[bool] = False

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

    stochastic: ClassVar[bool] = True

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


class SelectionNoisy(LandscapeWrapper):
    r"""Simulates a pooled selection assay read out by sequencing.

    **The finding this exists to reproduce.** Sundar, Tu, Guan and Esvelt
    (*FLIGHTED*, bioRxiv 2024; code MIT,
    https://github.com/vikram-sundar/FLIGHTED_public) fitted a generative noise
    model to the GB1 mRNA-display data and report that *"the highest-performing
    variants in a given single-step selection experiment show essentially 0
    correlation between measured and true fitness"* -- Pearson r ≈ 0 for the
    ~1,000 highest-enrichment variants. **The top of the measured landscape,
    precisely where a sampler concentrates, is where the measurement carries the
    least information.** :class:`Noisy` cannot produce this at any ``scale``: its
    error is the same size everywhere, so the correlation it leaves behind is
    flat across the fitness range.

    **Why the real assay does it.** The pathology is not additive noise. A
    pooled selection measures fitness only through a survival probability, and
    that probability saturates: once a variant survives 99% of the time, making
    it twice as good moves its read count almost not at all. Measured enrichment
    is then dominated by counting noise, and inverting the saturating link
    amplifies that noise without bound. So the model here is the mechanism, not
    a variance function bolted onto a Gaussian:

    .. math::

        p_i          &= \sigma(s \cdot (f_i - m)) \\
        \lambda_i    &\sim \mathrm{Gamma}(k,\ \bar{n}/k) \\
        n^{0}_i      &\sim \mathrm{Poisson}(\lambda_i) \\
        n^{1}_i      &\sim \mathrm{Binomial}(n^{0}_i,\ p_i) \\
        \hat{f}_i    &= m + \sigma^{-1}\!\left(\tfrac{n^{1}_i + 1/2}{n^{0}_i + 1}\right) / s

    The Gamma-Poisson draw is a negative binomial: library preparation does not
    deposit every variant at equal abundance, and ``dispersion`` is the
    over-dispersion relative to pure Poisson counting that DMS pipelines
    (DiMSum, Rosette/Rosace) fit from data. The binomial step is FLIGHTED's
    single-step selection. The final line inverts the forward model, so with
    unlimited reads the measurement is *exact* -- all of the error, and all of
    its fitness-dependence, comes from finite counts.

    The ``+1/2`` and ``+1`` are the Haldane-Anscombe correction. Without them a
    variant that lost every read would measure ``-inf``, and one that lost none
    would measure ``+inf``, which is an artefact of the estimator rather than of
    the assay.

    Note:
        Unlike :class:`Noisy`, this wrapper is not centred on the truth: the
        inverse link is convex at the top of the range, so measurements of the
        best variants are biased upward. That bias is a property of real
        enrichment assays and is deliberately not corrected.

    Note:
        :class:`Cached` rejects this class for the same reason it rejects
        :class:`Noisy`: caching freezes the first measurement of each sequence
        forever, and here that would remove precisely the fitness-dependent
        error the wrapper exists to produce.

    Args:
        landscape: The landscape to wrap.
        midpoint: Fitness at which a variant survives selection half the time.
            Measurements are most informative here and least informative far
            above it.
        slope: How sharply survival rises with fitness. Larger values saturate
            sooner, so the uninformative region starts lower.
        reads: Mean sequencing depth per variant before selection. This is the
            single knob that sets how noisy the assay is overall.
        dispersion: Shape of the Gamma-Poisson library abundance. Small values
            mean a badly skewed library; large values approach pure Poisson
            counting.
        seed: Seed for the assay's random stream.

    Raises:
        ValueError: If ``slope``, ``reads`` or ``dispersion`` is not positive.

    Example:
        Calibrate against the landscape's own range rather than guessing::

            >>> from evogfn.landscapes import EhrlichLandscape
            >>> truth = EhrlichLandscape(seed=0)
            >>> assay = SelectionNoisy.calibrated(truth, top_fitness=1.0, seed=1)
    """

    stochastic: ClassVar[bool] = True

    def __init__(  # noqa: PLR0913 - an assay is defined by its selection and its depth
        self,
        landscape: FitnessLandscape,
        *,
        midpoint: float = 0.0,
        slope: float = 1.0,
        reads: float = 100.0,
        dispersion: float = 3.0,
        seed: int = 0,
    ) -> None:
        """Wrap a landscape with count-based selection noise."""
        super().__init__(landscape)
        if slope <= 0:
            raise ValueError(f"slope must be positive so fitness raises survival, got {slope}")
        if reads <= 0:
            raise ValueError(f"reads must be positive, got {reads}")
        if dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {dispersion}")
        self._midpoint = midpoint
        self._slope = slope
        self._reads = reads
        self._dispersion = dispersion
        self._rng = np.random.default_rng(seed)

    @classmethod
    def calibrated(  # noqa: PLR0913 - mirrors the constructor, in fitness units
        cls,
        landscape: FitnessLandscape,
        *,
        neutral_fitness: float = 0.0,
        top_fitness: float | None = None,
        top_survival: float = 0.995,
        reads: float = 100.0,
        dispersion: float = 3.0,
        seed: int = 0,
    ) -> SelectionNoisy:
        """Build a wrapper whose saturation is placed against a landscape's range.

        ``midpoint`` and ``slope`` are in the units of whatever the landscape
        returns, so a default pair is meaningful for one landscape and absurd for
        the next -- and getting them wrong is silent: the assay just stops
        saturating, and the FLIGHTED signature disappears. This picks them from
        two fitness values instead, which is the calibration a real experiment
        does when it chooses selection stringency.

        Args:
            landscape: The landscape to wrap.
            neutral_fitness: Fitness that survives selection half the time.
                Usually wild-type, which is 1.0 on the GB1 and TrpB scales and
                0.0 on a log-enrichment scale.
            top_fitness: Fitness placed at ``top_survival``. Defaults to the
                landscape's known optimum.
            top_survival: Survival probability assigned to ``top_fitness``.
                Closer to 1 means a more stringent selection and a wider
                uninformative region at the top.
            reads: Mean sequencing depth per variant before selection.
            dispersion: Shape of the Gamma-Poisson library abundance.
            seed: Seed for the assay's random stream.

        Returns:
            A wrapper saturating across the requested range.

        Raises:
            ValueError: If ``top_survival`` is not in ``(0.5, 1)``, if
                ``top_fitness`` does not exceed ``neutral_fitness``, or if
                ``top_fitness`` is omitted and the landscape does not know its
                optimum.
        """
        if not 0.5 < top_survival < 1.0:  # noqa: PLR2004 - 0.5 is the midpoint by definition
            raise ValueError(
                f"top_survival must lie in (0.5, 1) so the top of the range sits above "
                f"the midpoint, got {top_survival}"
            )
        if top_fitness is None:
            optimum = landscape.optimum
            if optimum is None:
                raise ValueError(
                    f"{type(landscape).__name__} does not know its optimum, so top_fitness "
                    f"cannot be inferred; pass it explicitly"
                )
            top_fitness = float(np.max(optimum))
        if top_fitness <= neutral_fitness:
            raise ValueError(
                f"top_fitness must exceed neutral_fitness, got {top_fitness} <= {neutral_fitness}"
            )
        slope = float(np.log(top_survival / (1.0 - top_survival)) / (top_fitness - neutral_fitness))
        return cls(
            landscape,
            midpoint=neutral_fitness,
            slope=slope,
            reads=reads,
            dispersion=dispersion,
            seed=seed,
        )

    @property
    def midpoint(self) -> float:
        """Fitness at which a variant survives selection half the time."""
        return self._midpoint

    @property
    def slope(self) -> float:
        """How sharply survival rises with fitness."""
        return self._slope

    @property
    def reads(self) -> float:
        """Mean sequencing depth per variant before selection."""
        return self._reads

    @property
    def dispersion(self) -> float:
        """Shape of the Gamma-Poisson library abundance."""
        return self._dispersion

    def survival_probability(self, fitness: Fitness) -> npt.NDArray[np.float64]:
        """Probability that a variant of each fitness survives selection.

        Exposed because it is the diagnostic that says whether this wrapper is
        calibrated: if nothing in the landscape reaches a probability near 1,
        nothing saturates and the assay is merely noisy rather than
        uninformative at the top.

        Args:
            fitness: Objective values, of any shape.

        Returns:
            Survival probabilities of the same shape.
        """
        return _sigmoid(self._slope * (np.asarray(fitness, dtype=np.float64) - self._midpoint))

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Run the batch through one simulated selection and sequencing round."""
        values = np.asarray(self._inner.evaluate(sequences), dtype=np.float64).copy()
        # Leave -inf alone, as Noisy does: an infeasible sequence is not a noisy
        # measurement of a feasible one.
        finite = np.isfinite(values)
        truth = values[finite]

        probability = self.survival_probability(truth)
        abundance = self._rng.gamma(self._dispersion, self._reads / self._dispersion, truth.shape)
        initial = self._rng.poisson(abundance)
        surviving = self._rng.binomial(initial, probability)

        observed = (surviving + 0.5) / (initial + 1.0)
        values[finite] = self._midpoint + np.log(observed / (1.0 - observed)) / self._slope
        return values


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Logistic function, evaluated without overflowing on large magnitudes.

    Args:
        x: Any real values.

    Returns:
        Values in ``(0, 1)``, of the same shape.
    """
    # exp(-x) overflows for very negative x and exp(x) for very positive x, so
    # each half of the domain uses the branch that only ever exponentiates a
    # non-positive number. Tests run with warnings as errors, so an overflow
    # here would be a failure rather than a silently wrong number.
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    lower = np.exp(x[~positive])
    out[~positive] = lower / (1.0 + lower)
    return out


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
        Caching a stochastic landscape -- :class:`Noisy`, :class:`SelectionNoisy`
        or anything else that draws -- freezes the first measurement of each
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
        # Asked of the landscape rather than tested against a list of noise
        # classes: a list is a thing to forget, and forgetting it is silent --
        # see LandscapeWrapper.stochastic. getattr, so a landscape that is not a
        # wrapper can declare itself stochastic too.
        if bool(getattr(landscape, "stochastic", False)):
            raise ValueError(
                f"caching a {type(landscape).__name__} landscape would freeze the first "
                f"measurement of each sequence, defeating the noise; wrap the other way "
                f"round if you meant to cache the underlying truth"
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
