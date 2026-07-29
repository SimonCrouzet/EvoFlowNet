# EvoGFN

**EvoGFN** is a Python library for in-silico directed evolution with Generative Flow Networks
(GFlowNets). It generates batches of sequence variants that are diverse and high-fitness at the same
time, rather than many near-copies of a single best hit.

Use it to run directed-evolution campaigns against a fitness landscape — your own, or one of the
built-in benchmarks — and to compare a GFlowNet against classical baselines (genetic algorithm, hill
climbing, simulated annealing, CMA-ES, MLDE) on equal terms.

> [!NOTE]
> Early development. The API is not yet stable and the milestones below are still landing.

---

## Why you might want this

A design round has a fixed budget — you can synthesise and assay so many variants and no more. If your
proposal method returns 96 sequences that are minor variations on one design, you have spent the
budget on a single bet. Anything that kills that design kills the whole round.

Optimisers do this by construction: they climb toward one peak. A GFlowNet learns a policy that samples
variants *in proportion to* their predicted fitness, so a batch spreads across the high-fitness regions
of the landscape instead of piling onto one. Applied to mutation trajectories from a parent sequence,
that is a direct model of what directed evolution actually does.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SimonCrouzet/EvoGFN
cd EvoGFN
uv sync                  # GPU (CUDA build of torch)
uv sync --extra cpu      # CPU only — 1.1GB instead of 4.7GB
```

Everything in the default path runs on CPU. A GPU helps for long sequences and for the optional
protein-language-model oracles (`--extra plm`).

---

## Built-in landscapes

You can plug in any fitness function by implementing one interface. Two are included, chosen because
their correct answers are known — which means you can check whether a method actually worked, not just
whether it produced a plausible-looking number.

| Landscape | What it is | Why it is useful |
|---|---|---|
| **Ehrlich** | Closed-form, procedurally generated sequences with tunable epistasis, ruggedness and feasibility constraints | The optimum is guaranteed reachable, so you can measure true regret. No download; evaluation is instant |
| **GB1** | Real deep-mutational-scanning data: 149,361 measured variants across 4 positions | Combinatorially complete, so every sequence has a ground-truth fitness and the whole space can be enumerated |

Ehrlich landscapes also define which sequences are *constructible* at all. EvoGFN enforces this by
masking invalid actions during generation, so every proposed sequence is feasible by construction
rather than filtered out afterwards.

---

## Status

The library is being built in milestones. Each one leaves `main` green and usable.

| Milestone | Contents | State |
|---|---|---|
| M0 | Repository scaffold, tooling, CI | done |
| M1 | Fitness landscapes and sequence environments | in progress |
| M2 | GFlowNet core (trajectory balance, detailed balance, SubTB) | |
| M3 | Classical baselines and benchmark harness | |
| M4 | Multi-objective rewards and Pareto metrics | |
| M5 | Design–build–test–learn campaign loop under a budget | |
| M6 | Workshop notebooks and documentation | |

Usage examples and the CLI arrive with M2; this section will carry a worked example once there is
something to run.

---

## Design

Every major piece is an interface with swappable implementations, so adding a landscape, a sampler or
an acquisition function means implementing one class and pointing a config at it.

| Component | Replace it to... |
|---|---|
| `FitnessLandscape` | score sequences with your own assay data, model, or simulator |
| `SequenceEnvironment` | change how variants are built (append tokens, or mutate a parent) |
| `Sampler` | swap the search method — GFlowNets and baselines share one interface |
| `Acquisition` | change how a batch is chosen under uncertainty |
| `Tracker` | send metrics somewhere other than the console |

Landscapes compose: measurement noise, an evaluation budget and caching are wrappers you apply to any
of them, so a budget cannot be accidentally bypassed.

---

## License

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**.

You may freely use, modify, and distribute this software — including for commercial purposes —
provided that you preserve the copyright notice and license text in any distribution. See
[`LICENSE`](LICENSE) for the full terms.

---

## About

I'm Simon Crouzet, an independent researcher and consultant in AI/ML for molecular design and drug
discovery. EvoGFN came out of a long-standing interest in directed evolution, and in GFlowNets —
and in what happens when you stop treating the first as an optimisation problem and start treating it
as the sampling problem the second was built for.

If you find this useful, have ideas, or are working on something in the same space and want to
exchange — feel free to reach out. I'm also available for project-based work in computational molecular
design and ML workflow development.

- **GitHub:** [@simoncrouzet](https://github.com/simoncrouzet)

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue to discuss
significant changes before submitting a pull request. All pull requests should include tests and pass
the existing suite. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup and conventions.

---

## Credit & Citation

EvoGFN is open source under the Apache 2.0 License. You are free to use it in research and
commercial work — please credit the original project and respect the license terms.

If you use EvoGFN in your work, please acknowledge it and feel free to get in touch.

---

## References

- Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. *NeurIPS* 34, 27381–27394.
- Malkin, N., Jain, M., Bengio, E., Sun, C. & Bengio, Y. (2022). Trajectory Balance: Improved Credit Assignment in GFlowNets. *NeurIPS* 35.
- Jain, M. et al. (2022). Biological Sequence Design with GFlowNets. *ICML*.
- Jain, M. et al. (2023). Multi-Objective GFlowNets. *ICML*.
- Stanton, S., Alberstein, R., Frey, N., Watkins, A. & Cho, K. (2024). Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms. *ICML Workshop on ML for Life and Material Sciences*.
- Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O. & Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife* 5, e16965.
- Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nature Methods* 16, 687–694.
