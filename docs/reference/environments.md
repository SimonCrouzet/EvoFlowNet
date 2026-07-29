# Environments

The state graph variants are built through. This is where directed evolution becomes a
**construction DAG**: a trajectory starts at the parent, applies point mutations one at a
time, and stops.

It is also where feasibility lives. A constructibility constraint handed to the environment
becomes an action mask, so infeasible designs are never generated rather than filtered
afterwards — see [notebook 3](../notebooks.md).

::: evoflownet.env.base

::: evoflownet.env.mutation
