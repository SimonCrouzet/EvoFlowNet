"""Paired comparison across seeds, without taking a SciPy dependency.

Two methods run on the same seed share their surrogate initialisation, their
landscape, and their initial design. Pairing on that removes the seed-to-seed
variance, which on GB1 is the dominant term: an unpaired test at 15 seeds put
the GFlowNet's advantage over a genetic algorithm at t = 0.9, the paired test on
the same numbers at t = 1.45. Reporting the unpaired figure would have thrown
away most of the experiment.

What is reported, and why all of it
-----------------------------------

A mean difference alone hides whether one seed carried the result. So each
comparison reports the mean, a confidence interval, the paired ``t``, and the
**win rate** -- how many seeds the method actually won. A method that wins by
0.9 on average while winning 9 of 15 seeds is not the same as one that wins
0.9 on every seed, and only the second is a method you would use.

The interval is Student's ``t`` on the paired differences, which assumes those
differences are roughly normal. With 100 seeds that is unobjectionable; below
about 20 the win rate is the more trustworthy of the two figures, which is part
of why both are here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Two-sided 95% critical values of Student's t, indexed by degrees of freedom.
#: Tabulated rather than computed so this module needs no SciPy; beyond the
#: table the normal approximation is accurate to better than a percent.
_T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}
_T_ASYMPTOTIC = 1.960


def t_critical(degrees_of_freedom: int) -> float:
    """Two-sided 95% critical value, interpolating the table conservatively.

    Args:
        degrees_of_freedom: Sample size minus one.

    Returns:
        The critical value, or the normal approximation beyond the table.
    """
    if degrees_of_freedom < 1:
        return math.inf
    if degrees_of_freedom in _T_CRITICAL:
        return _T_CRITICAL[degrees_of_freedom]
    larger = [d for d in _T_CRITICAL if d > degrees_of_freedom]
    if not larger:
        return _T_ASYMPTOTIC
    # Round down to the next tabulated value, which overstates the interval
    # slightly rather than understating it.
    return _T_CRITICAL[min(larger)]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """One method against another, across shared seeds.

    Attributes:
        name: What was compared against what.
        mean: Mean paired difference, positive when the first method is better.
        low: Lower bound of the 95% interval.
        high: Upper bound.
        t: Paired t statistic.
        wins: Seeds on which the first method won.
        n: Seeds compared.
    """

    name: str
    mean: float
    low: float
    high: float
    t: float
    wins: int
    n: int

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero.

        A convenience, not a verdict. An interval that excludes zero on 15
        seeds and one that excludes it on 200 are different evidence.
        """
        return self.low > 0.0 or self.high < 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of seeds won. Half is a coin flip."""
        return self.wins / self.n if self.n else 0.0

    def __repr__(self) -> str:
        """One line carrying the effect, its interval, and the win rate."""
        mark = "*" if self.significant else " "
        return (
            f"{self.name}: {self.mean:+.3f} [{self.low:+.3f}, {self.high:+.3f}]{mark} "
            f"t={self.t:.2f}  wins {self.wins}/{self.n}"
        )


def compare(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> PairedComparison:
    """Compare two methods on matched seeds.

    Args:
        name: Label for the comparison.
        first: Per-seed metric for the method under test.
        second: Per-seed metric for the method compared against, in the same
            seed order.
        higher_is_better: Whether a larger metric is a better result. Regret
            and other losses should pass ``False`` so a positive difference
            always means the first method won.

    Returns:
        The paired comparison.

    Raises:
        ValueError: If the arrays differ in length, or hold fewer than two
            seeds -- a single seed has no variance and a statistic computed
            from it would be a division by zero dressed up as a result.
    """
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"cannot pair {a.shape[0]} seeds against {b.shape[0]}")
    if a.shape[0] < 2:  # noqa: PLR2004 - a variance needs two observations
        raise ValueError(f"need at least 2 seeds to compare, got {a.shape[0]}")

    differences = (a - b) if higher_is_better else (b - a)
    mean = float(differences.mean())
    spread = float(differences.std(ddof=1))
    n = differences.shape[0]
    error = spread / math.sqrt(n) if spread else 0.0
    half = t_critical(n - 1) * error
    return PairedComparison(
        name=name,
        mean=mean,
        low=mean - half,
        high=mean + half,
        t=mean / error if error else math.inf if mean else 0.0,
        wins=int((differences > 0).sum()),
        n=n,
    )


def seeds_needed(observed: PairedComparison, *, power: float = 0.8) -> int:
    """Seeds required to resolve an effect of the size just observed.

    Answers the question a non-significant result actually raises: is this a
    null, or an underpowered look at something real? Reported so an
    inconclusive comparison names its own price rather than inviting a guess.

    Args:
        observed: A comparison already run.
        power: Probability of detecting the effect if it is real.

    Returns:
        Approximate number of seeds, by the normal approximation. Returns 0
        when the observed effect is exactly zero, since no sample resolves it.
    """
    spread = observed.mean / observed.t if observed.t and math.isfinite(observed.t) else 0.0
    if not spread or not observed.mean:
        return 0
    # sigma of the paired differences, recovered from the standard error.
    sigma = spread * math.sqrt(observed.n)
    z_power = {0.8: 0.842, 0.9: 1.282, 0.95: 1.645}.get(power, 0.842)
    required = ((_T_ASYMPTOTIC + z_power) * sigma / observed.mean) ** 2
    return math.ceil(required)
