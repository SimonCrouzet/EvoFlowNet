# Algorithms

One interface, deliberately. A benchmark in which the method under test and the methods it is
compared against run through different harnesses is not a comparison — the budget accounting
drifts, the stopping conditions differ, and the result measures the harness as much as the
method.

::: evoflownet.algorithms.base

## GFlowNets

::: evoflownet.algorithms.gflownet.sampler

::: evoflownet.algorithms.gflownet.training

::: evoflownet.algorithms.gflownet.sampling

::: evoflownet.algorithms.gflownet.objectives

::: evoflownet.algorithms.gflownet.flow_objectives

::: evoflownet.algorithms.gflownet.genetic_gfn

::: evoflownet.algorithms.gflownet.replay

::: evoflownet.algorithms.gflownet.trajectory_balance

## Classical baselines

Directed evolution *is* a genetic algorithm, so these are the incumbents rather than strawmen
to be cleared.

::: evoflownet.algorithms.baselines.mutagenesis

::: evoflownet.algorithms.baselines.genetic

::: evoflownet.algorithms.baselines.annealing

::: evoflownet.algorithms.baselines.cmaes

::: evoflownet.algorithms.baselines.mlde

## Giving a baseline the same model access

::: evoflownet.algorithms.inner_loop

## The policy network

::: evoflownet.models.policy

::: evoflownet.models.conditioning
