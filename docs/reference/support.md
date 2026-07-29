# Supporting pieces

## Acquisition

Turning a prediction and an uncertainty into one number per candidate, and choosing the batch
that number selects.

::: evoflownet.acquisition.base

::: evoflownet.acquisition.rules

## Surrogates

The model fitted to what has actually been measured, and the proxy that lets a sampler
optimise against it without touching the oracle.

::: evoflownet.surrogate.base

::: evoflownet.surrogate.ensemble

::: evoflownet.surrogate.proxy

## Rewards

::: evoflownet.rewards.base

::: evoflownet.rewards.scalarization

## Tracking

::: evoflownet.tracking.base

::: evoflownet.tracking.console

::: evoflownet.tracking.provenance

::: evoflownet.tracking.wandb

## Data

::: evoflownet.data.cache
