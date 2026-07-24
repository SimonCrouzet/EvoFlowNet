# EvoFlowNet

**EvoFlowNet** applies Generative Flow Networks (GFlowNets) to in-silico directed evolution. It treats
variant design as a *sampling* problem rather than an optimisation one: instead of climbing to the
single best sequence, it learns a policy that samples variants in proportion to their fitness, so a
design round returns a diverse, feasible batch rather than many copies of one local optimum.

The library is built around replaceable components — fitness landscapes, sequence environments,
samplers, acquisition functions, metrics — so that a new landscape or a new baseline is an
implementation of one interface and a configuration change, not a fork.

> [!NOTE]
> Early development. The public API is not yet stable, and the milestones below are still landing.

---

## Why sampling rather than optimisation

A design round that returns 96 near-identical sequences is a bad design round. Any single candidate can
fail for reasons the model never saw — expression, aggregation, off-target effects — and screening
budget spent on duplicates buys no information.

Hill-climbing, genetic algorithms and greedy ML-assisted directed evolution all collapse onto one mode
by construction. GFlowNets ([Bengio et al., NeurIPS 2021](https://arxiv.org/abs/2106.04399)) instead
learn a policy whose sampling probability is proportional to reward, which yields batches that are
high-fitness *and* diverse. Applied to mutational trajectories from a parent sequence, this is a direct
model of directed evolution.

---

## Benchmarks with real ground truth

Most sequence-design benchmarks can only report "best score found", because the true optimum and the
true fitness distribution are unknown. EvoFlowNet is deliberately built on two landscapes where they
are not.

| Landscape | Kind | Why it was chosen |
|---|---|---|
| **Ehrlich functions** ([Stanton et al., ICML 2024 workshop](https://arxiv.org/abs/2407.00236)) | Closed-form, procedurally generated | Tunable epistasis, ruggedness and feasibility constraints, with a **provably attainable optimum** — so regret is exact, not relative |
| **GB1** ([Wu et al., eLife 2016](https://elifesciences.org/articles/16965)) | Empirical, 4-site combinatorial | 149,361 of 160,000 variants measured. **Combinatorially complete and fully enumerable**, so mode coverage and the exact target distribution are computable |

This makes three normally-unavailable measurements possible:

- **True simple regret** against the global optimum, rather than "best seen so far".
- **Mode coverage** — how many distinct peaks were found, not the height of one.
- **Distributional fidelity** — enumerate `p*(x) ∝ R(x)^β` exactly and measure L1 distance to the
  sampler's empirical distribution. This is the only real test that a GFlowNet is sampling rather
  than behaving as an expensive hill-climber.

Ehrlich functions also define feasibility through a Markov transition matrix, which maps directly onto
GFlowNet action masking. A masked policy is feasible *by construction*, where a genetic algorithm
spends much of its evaluation budget on infeasible sequences — a comparison the original authors
flagged as future work.

---

## Status

| Milestone | Contents | State |
|---|---|---|
| M0 | Repository scaffold, tooling, CI | done |
| M1 | Fitness landscapes and sequence environments | in progress |
| M2 | GFlowNet core (trajectory balance, detailed balance, SubTB) | |
| M3 | Classical baselines and benchmark harness | |
| M4 | Multi-objective rewards and Pareto metrics | |
| M5 | Design–build–test–learn campaign loop under a budget | |
| M6 | Workshop notebooks and documentation | |

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SimonCrouzet/EvoFlowNet
cd EvoFlowNet
uv sync                  # GPU-first: CUDA build of torch
uv sync --extra cpu      # CPU fallback: 1.1GB instead of 4.7GB
```

Both benchmark landscapes run on CPU. A GPU matters for Ehrlich instances at the sequence lengths used
in the literature, and for the optional protein-language-model oracles (`--extra plm`).

---

## License

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**.

You may freely use, modify, and distribute this software — including for commercial purposes —
provided that you preserve the copyright notice and license text in any distribution. See
[`LICENSE`](LICENSE) for the full terms.

---

## About

I'm Simon Crouzet, an independent researcher and consultant in AI/ML for molecular design and drug
discovery. EvoFlowNet came out of wanting an open, rigorous testbed for generative sequence design —
one where diversity and feasibility are measured against ground truth rather than asserted.

If you find this useful, have ideas, or are working on something in the same space and want to
exchange — feel free to reach out. I'm also available for project-based work in computational
molecular design and ML workflow development.

- **GitHub:** [@simoncrouzet](https://github.com/simoncrouzet)

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue to discuss
significant changes before submitting a pull request. All pull requests should include tests and pass
the existing suite. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup and the conventions this project
follows — in particular, that implementations keep the notation of the paper they come from, and that
tests check against known answers where one exists.

---

## Credit & Citation

EvoFlowNet is open source under the Apache 2.0 License. You are free to use it in research and
commercial work — please credit the original project and respect the license terms.

If you use EvoFlowNet in your work, please acknowledge it and feel free to get in touch.

---

## References

- Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. *NeurIPS* 34, 27381–27394.
- Bengio, Y. et al. (2023). GFlowNet Foundations. *JMLR* 24, 1–55.
- Malkin, N., Jain, M., Bengio, E., Sun, C. & Bengio, Y. (2022). Trajectory Balance: Improved Credit Assignment in GFlowNets. *NeurIPS* 35.
- Jain, M. et al. (2022). Biological Sequence Design with GFlowNets. *ICML*.
- Jain, M. et al. (2023). Multi-Objective GFlowNets. *ICML*.
- Stanton, S., Alberstein, R., Frey, N., Watkins, A. & Cho, K. (2024). Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms. *ICML Workshop on ML for Life and Material Sciences*.
- Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O. & Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife* 5, e16965.
- Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nature Methods* 16, 687–694.
