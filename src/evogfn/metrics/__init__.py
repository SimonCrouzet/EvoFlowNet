"""Evaluation metrics.

Split by what they answer. [performance][evogfn.metrics.performance] asks how
good the designs are, [diversity][evogfn.metrics.diversity] asks how varied,
[pareto][evogfn.metrics.pareto] asks how good the *set* of trade-offs is when
there is more than one objective, and
[distribution][evogfn.metrics.distribution] asks whether the sampler is sampling
at all -- the last being the only one a hill-climber cannot pass.
"""

from evogfn.metrics.distribution import (
    empirical_distribution,
    expected_l1_from_sampling_noise,
    l1_distance,
    target_distribution,
)
from evogfn.metrics.diversity import (
    distinct_modes,
    diversity,
    hamming_distances,
    novelty,
)
from evogfn.metrics.pareto import (
    gd_plus,
    hypervolume,
    igd_plus,
    non_dominated,
    pareto_front,
    r2_indicator,
)
from evogfn.metrics.performance import (
    cumulative_regret,
    feasible_fraction,
    simple_regret,
    top_k_performance,
)

__all__ = [
    "cumulative_regret",
    "distinct_modes",
    "diversity",
    "empirical_distribution",
    "expected_l1_from_sampling_noise",
    "feasible_fraction",
    "gd_plus",
    "hamming_distances",
    "hypervolume",
    "igd_plus",
    "l1_distance",
    "non_dominated",
    "novelty",
    "pareto_front",
    "r2_indicator",
    "simple_regret",
    "target_distribution",
    "top_k_performance",
]
