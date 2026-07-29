# EvoGFN

**EvoGFN** is a Python library for in-silico directed evolution with Generative Flow
Networks (GFlowNets). It generates batches of sequence variants that are diverse and
high-fitness at the same time, rather than many near-copies of a single best hit.

Use it to run directed-evolution campaigns against a fitness landscape — your own, or one of
the built-in benchmarks — and to compare a GFlowNet against classical baselines (genetic
algorithm, hill climbing, simulated annealing, CMA-ES, MLDE) on equal terms.

!!! warning "Early development"
    The API is not stable. Results in these pages come from a benchmark suite that is still
    being extended, and several of its findings are negative. [What this does not
    show](limitations.md) is not an appendix — read it before quoting anything here.

---

## The argument

A design round has a fixed budget. You can synthesise and assay so many variants and no more.
If your proposal method returns 96 sequences that are minor variations on one design, you have
spent the budget on a single bet, and anything that kills that design kills the whole round.

Optimisers do this by construction: they climb toward one peak. A GFlowNet learns a policy
that samples variants *in proportion to* their predicted fitness, so a batch spreads across
the high-fitness regions of the landscape instead of piling onto one. Applied to mutation
trajectories from a parent sequence, that is a direct model of what directed evolution
actually does.

There is a second argument, and in this project's own measurements it is the stronger one.
Many sequence design problems have a **constructibility constraint**: most strings are not
things you could build. A method that proposes designs and filters them afterwards spends
its budget on wells that return nothing. A GFlowNet built on a masked construction graph
cannot generate an infeasible design at all — feasibility is a property of the graph, not a
post-hoc check.

---

## What the numbers actually say

Stated up front, because the rest of the documentation is easier to read honestly if the
headline is honest first.

| Claim | Status |
|---|---|
| Masking converts more of a fixed budget into usable measurements than rejection sampling | **Supported.** A rejection-sampling GA stalls at 82 ± 16 of 384 wells; a masked GFlowNet fills all 384 |
| Masking *searches better* than rejection sampling | **Not supported.** At matched budget the two tie exactly on 100 of 100 seeds |
| Masked generation restores the surrogate's learning signal | **Supported.** Rounds with a defined surrogate–oracle correlation: 189/400 masked, 0/400 for a proxy-optimising GA |
| Batches are more diverse at equal budget and equal feasibility | **Supported** in direction on every task; whether that diversity is *useful* is **not tested** |
| The GFlowNet beats a proxy-optimising GA on real GB1 measurements | **Supported in expectation** (+0.96 paired), but a single campaign wins 55% of the time, not always |
| Benchmarking above the wet-lab budget regime reverses conclusions | **Not supported.** Our own budget gradient shows no ranking flip between 96 and 10,000 calls |
| Any method solves an Ehrlich instance | **No.** Every method sits at 0.974–0.992 of a maximum regret of 1.0 on the large-space task |

The comparisons in this project are between **failure modes**, not between a working method
and broken ones.

---

## Where to start

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, run a campaign, train a policy.
- **[The benchmark suite](benchmark.md)** — what each task decides, and what a protocol is.
- **[Notebooks](notebooks.md)** — four runnable walkthroughs, landscape to campaign.
- **[What this does not show](limitations.md)** — the limits, in one place.
- **[API reference](reference/landscapes.md)** — generated from the source.

</div>

!!! note "One gap in the API reference"
    The reference pages are generated from the docstrings in `src/`. Module-level constants
    — `MAX_ENUMERABLE_SIZE`, `MIN_BINS`, `WET_LAB_PROTOCOLS`, the `PUBLISHED_*` values in
    `baselines.mlde` and the rest — are documented with `#:` comments, which the Markdown
    docstring parser does not pick up, so they do not appear on these pages and references to
    them read as plain names rather than links. Their values and the reasoning behind them are
    in the source.

---

## Design

Every major piece is an interface with swappable implementations, so adding a landscape, a
sampler or an acquisition function means implementing one class and pointing a config at it.

| Component | Replace it to... |
|---|---|
| [`FitnessLandscape`](reference/landscapes.md) | score sequences with your own assay data, model, or simulator |
| [`SequenceEnvironment`](reference/environments.md) | change how variants are built (append tokens, or mutate a parent) |
| [`Sampler`](reference/algorithms.md) | swap the search method — GFlowNets and baselines share one interface |
| [`Acquisition`](reference/support.md) | change how a batch is chosen under uncertainty |
| [`Tracker`](reference/support.md) | send metrics somewhere other than the console |

Landscapes compose: measurement noise, an evaluation budget and caching are wrappers you
apply to any of them, so a budget cannot be accidentally bypassed.

---

## Licence and citation

Copyright © 2026 Simon J. Crouzet. Licensed under the **Apache License 2.0**. You may use,
modify and distribute this software including for commercial purposes, provided you preserve
the copyright notice and licence text.

If you use EvoGFN in your work, please credit the project and feel free to get in touch —
[@simoncrouzet](https://github.com/simoncrouzet).

---

## References

- Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021). Flow Network based
  Generative Models for Non-Iterative Diverse Candidate Generation. *NeurIPS* 34, 27381–27394.
- Malkin, N., Jain, M., Bengio, E., Sun, C. & Bengio, Y. (2022). Trajectory Balance: Improved
  Credit Assignment in GFlowNets. *NeurIPS* 35.
- Jain, M. et al. (2022). Biological Sequence Design with GFlowNets. *ICML*.
- Jain, M. et al. (2023). Multi-Objective GFlowNets. *ICML*.
- Stanton, S., Alberstein, R., Frey, N., Watkins, A. & Cho, K. (2024). Closed-Form Test
  Functions for Biophysical Sequence Optimization Algorithms. *ICML Workshop on ML for Life
  and Material Sciences*.
- Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O. & Sun, R. (2016). Adaptation in protein
  fitness landscapes is facilitated by indirect paths. *eLife* 5, e16965.
- Yang, K.K., Wu, Z. & Arnold, F.H. (2019). Machine-learning-guided directed evolution for
  protein engineering. *Nature Methods* 16, 687–694.
