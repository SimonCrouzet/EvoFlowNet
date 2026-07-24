# EvoFlowNet

**GFlowNets for in-silico directed evolution** — sampling diverse, feasible, high-fitness protein
variants instead of hill-climbing to one of them.

> [!NOTE]
> Early development. The public API is not yet stable.

## The idea

Directed evolution is usually posed as optimization: find the best variant. But a design round that
returns 96 near-identical sequences is a bad design round — any single candidate can fail for reasons
the model never saw, and screening budget spent on duplicates is wasted.

GFlowNets ([Bengio et al., NeurIPS 2021](https://arxiv.org/abs/2106.04399)) learn a policy that samples
objects with probability *proportional to their reward*, rather than maximizing it. Applied to
mutational trajectories, that turns directed evolution into a sampling problem and yields batches that
are simultaneously high-fitness and diverse.

## Status

| Milestone | Contents | State |
|---|---|---|
| M0 | Repository scaffold, tooling, CI | in progress |
| M1 | Fitness landscapes and sequence environments | |
| M2 | GFlowNet core (trajectory balance) | |
| M3 | Classical baselines and benchmark harness | |
| M4 | Multi-objective (Pareto) support | |
| M5 | Design-build-test-learn campaign loop | |
| M6 | Workshop notebooks and documentation | |

## License

Apache-2.0. See [LICENSE](LICENSE).
