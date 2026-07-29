# Metrics

Evaluation, including the one check that a good hill climber cannot pass.

Best-found, top-K and diversity are all satisfied by an optimiser that never samples anything.
Comparing an empirical distribution against the exact target `p*(x) ∝ R(x)^β` is not — and it
is why the landscapes here were chosen to be enumerable.

::: evoflownet.metrics.performance

::: evoflownet.metrics.diversity

::: evoflownet.metrics.distribution

::: evoflownet.metrics.pareto
