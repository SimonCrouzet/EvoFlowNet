# The benchmark suite

A benchmark is not a landscape and a number. It is a set of tests, each chosen because it can
settle a question the others cannot, run under a protocol a wet lab would recognise.

This page explains what a *task* is, what a *protocol* is, what each task in the suite
decides, and — the part that matters most — what the suite has and has not shown.

---

## A protocol is three numbers, and their product is the only one a claim can be indexed by

```python
Protocol(rounds=4, batch_size=96, max_mutations=4, label="four plates")
```

`rounds × batch_size` is the **oracle budget**: how many variants get measured. A result
reported without its budget cannot be compared to one that has it, and the surveyed literature
routinely compares across budgets differing by two orders of magnitude.

### The grounding

| Campaign | Per round | Rounds | Total |
|---|---|---|---|
| EVOLVEpro (*Science* 2025) | 11–12 | 4–8 | 50–90 |
| LaMBO-2 wet lab | ~125 | 3 | 374 |
| ALDE (Arnold lab, *Nat Commun* 2025) | 216 / 90 / 90 | 3 | 396 |
| MLDE / ftMLDE (Wittmann 2021) | 384 + 96 | 2 | 480 |
| CLADE (Qiu & Wei 2021) | 96 | 5 | 480 |
| TrpB (Buller, *PNAS* 2015), classical | 528 / 1408 / 1144 | 3 | ~3,080 |

Against which the machine-learning convention — 10 rounds of 100 or 128, and 10,000 for
GFN-AL on AMP — sits above even *classical* directed evolution. The sharp version: MLDE's
entire claim is reaching the answer in ~480 assays instead of ~3,000, and a benchmark run at
10,000 has given that back before the first comparison is made.

`WET_LAB_PROTOCOLS` and `ML_CONVENTION` name the real ones, so an experiment can cite a
campaign rather than a round number someone liked.

!!! danger "Our own budget experiment did not show what this argument predicts"
    The budget survey above is well sourced and stands as a survey finding. The obvious next
    step — that benchmarking above the wet-lab regime *reverses conclusions* — is the one
    thing we tested directly, and **it did not reproduce**. Across 96, 384, 1,000 and 10,000
    calls the ordering held; the gap to a proxy-optimising GA moved non-monotonically
    (+0.111 / +0.191 / +0.241 / +0.098) and never flipped. Report the budget gap as a reason
    to *measure at the wet-lab budget*, not as evidence that the ML budget lies.

### `max_mutations` sometimes does nothing, and it is worth checking

`Protocol.constrains_search(sequence_length)` returns `False` when the mutation budget reaches
every sequence anyway. On GB1 — four sites, four mutations — it is `False`. A result on GB1
therefore says nothing about search under a mutation constraint.

---

## A task is a landscape, a protocol, and a reason to run it

```python
Task(name=..., purpose=..., build=..., protocol=..., max_mutations=4)
```

The field a task cannot omit is `purpose`: what this task decides that the others do not. A
suite is only as good as its ability to distinguish methods, so a row that cannot say what it
settles should be deleted rather than kept for completeness.

`build` is a factory rather than an instance, so each seed can draw its own landscape where
that is meaningful. The mutation budget is **4 everywhere**, so a conclusion drawn on a cheap
diagnostic transfers to the main table instead of being confounded by a different search
radius.

---

## Main tests: the rows that carry claims

| Task | Landscape | Protocol | What it decides |
|---|---|---|---|
| `gb1-anchor` | GB1, 149,361 measured variants | 4 × 96 = 384 | Do the numbers hold on **real measurements**? The empirical anchor — and the easiest geometry here |
| `large-space` | Ehrlich `L=256, c=4, k=8, q=4` | 4 × 96 = 384 | Can the method search a space it **cannot enumerate**? ~10¹³ reachable designs against a budget of 384 |
| `feasibility` | Ehrlich `L=64`, transition density 0.15 | 4 × 96 = 384 | Can the method stay **inside the constructible set**? Rejection sampling burns the budget where masking cannot |
| `protocol-alde` | Ehrlich `L=64`, density 0.5 | 3 × 132 = 396 | Does the ranking survive the shape of a **real campaign**? After ALDE |
| `protocol-evolvepro` | same landscape as above | 8 × 48 = 384 | The **opposite shape** at a comparable budget, after EVOLVEpro. Many small rounds against few large ones |

Sequence lengths follow published practice rather than convenience. Stanton et al.'s own base
configuration is `L = 256`; HDBO uses `L = 5, 15, 64` and reports two published Bayesian
optimisation methods running out of memory at 64. So the flagship large-space task uses
Stanton's base configuration (directly comparable to the benchmark's authors), the mid-size
tasks use `L = 64` (where the published field degrades), and diagnostics use `L = 32` (cheap
enough to sweep an axis at 50 seeds).

!!! warning "`gb1-anchor` is the easiest geometry in the suite"
    Four sites, no feasibility constraint, and a mutation budget that reaches every sequence.
    `genetic-feasible` is bit-identical to `genetic+proxy` there, because there is nothing to
    reject. GB1 says the numbers are not an artefact of synthetic landscapes. It says nothing
    about constrained search, and an earlier version of this project claimed otherwise.

!!! warning "`large-space` is a comparison between degrees of stuck"
    Every method sits at **0.974 to 0.992** of a maximum regret of 1.0. The differences are
    real and statistically clean — 30/30 seeds — and they are differences between methods that
    all failed. "Marginally less stuck" is the honest phrasing.

---

## Diagnostics: the rows that inform choices

Diagnostics vary one axis on a fixed, cheap `L = 32` landscape at 50 seeds. They decide
things — which objective to carry into the main table, whether the ranking survives a change
of budget, whether rounds matter at fixed total. They are how choices get made, not what gets
claimed.

| Diagnostic | Axis varied | Outcome |
|---|---|---|
| `budget_gradient()` | 96 → 384 → 1,000 → 10,000 calls | **Negative.** No ranking flip. A blind GA improves most at 10,000 but not significantly |
| `rounds_curve(budget)` | many small rounds vs few large, fixed total | **Underpowered and method-dependent.** The better-powered replicate points the other way |
| `objective_task()` | trajectory balance vs the alternatives | **Suggestive only.** Total spread across five objectives ~5%, adjacent pairs within noise |

Two of the three are negative results. They are in the suite because that is what a diagnostic
is for, and running them is what stopped three claims from being made.

---

## The methodologies

A methodology is whatever turns a task and a seed into a runnable campaign. Keeping that one
callable is what makes a GFlowNet variant, a classical baseline and a baseline-with-model-
access the same kind of thing to the harness, so no arm can quietly receive a different budget,
surrogate or starting point than another.

| Arm | What it is |
|---|---|
| `random`, `random+surrogate` | the floor, with and without a surrogate screening the pool |
| `hill-climb` | neighbours of the incumbent, restarting after a patience window |
| `genetic`, `genetic+proxy` | a GA blind, and a GA given the same model access the GFlowNet gets |
| `genetic-feasible` | a GA that rejection-samples until its offspring are legal — the feasibility control |
| `annealing`, `cmaes` | simulated annealing and CMA-ES, both with proxy access |
| `mlde` | machine-learning-directed evolution — what protein engineers actually run, at almost exactly this budget |
| `gfn-tb`, `gfn-contrastive`, `genetic-gfn` | GFlowNet objectives |
| `gfn-db`, `gfn-subtb`, `gfn-fldb` | the detailed-balance family, which needs a policy with a flow head |

Two design decisions carry most of the fairness:

**GFlowNets train against a proxy, never the oracle.** Each builds a `ProxyLandscape` over the
same surrogate instance the campaign refits, so training costs proxy evaluations and never
oracle calls. Charging them would exhaust a 384-call campaign before the first round finished,
and no published method does it.

**Classical baselines are offered the same proxy access.** `ProxyOptimising` runs the
baseline's own search loop against the model before it hands its population up. Comparing a
method that optimises the model against one that only meets it as a filter is not a comparison
of methods, and the winner would be known in advance.

---

## Running it

```bash
uv run python experiments/run_suite.py                  # everything
uv run python experiments/run_suite.py --tier main      # headline only
uv run python experiments/run_suite.py --seeds 50       # raise the count
uv run python experiments/run_suite.py --report         # no runs, just read
```

Safe to interrupt and safe to re-run. Every campaign is written to `results/` the moment it
finishes, and a second invocation runs only what is missing — so raising a tier's seed count
from 30 to 50 costs twenty campaigns per arm, not fifty.

### Staleness

A stored result carries a fingerprint of the code that could have produced it. The entry
points are declared (`evoflownet.benchmark.methods` and `evoflownet.loop.campaign`) rather
than derived, and that declaration is what makes the mechanism pay: a record goes stale only
when something it can actually reach has changed. Hashing the whole package tree instead meant
that adding an unrelated file invalidated ~3,900 campaigns.

---

## How to read a result

The suite reports a **paired comparison**: the same seeds, the same task, the same protocol,
so a difference is a difference between methods. Three numbers, and all three are needed.

* **The mean advantage** and its confidence interval — what to expect on average.
* **The win/tie/loss count** — what a single laboratory campaign should expect. On GB1 the
  GFlowNet's mean advantage of +0.96 comes with **W/T/L = 55/19/26**. A lab running one
  campaign wins about 55% of the time. Reporting the mean alone would be misleading, and this
  is the case that taught us so.
* **The number of seeds.** 100 for the main tiers, 30 for `large-space` (campaign cost differs
  by an order of magnitude between `L = 4` and `L = 256`, not because the claims differ), 50
  for diagnostics.

One more caveat, and it is a defect rather than a nuance: the **GFlowNet arm is not
bit-reproducible**. At a fixed seed and configuration it returns identical results on only
30–32 of 50 seeds, with a per-seed standard deviation of 0.044. The classical baselines are
identical 50 of 50. Every GFlowNet number on this site carries that noise floor underneath it,
and effects smaller than it should not be read.

---

## API

- [`evoflownet.benchmark`](reference/benchmark.md) — tasks, protocols, suite, harness, store.
- [What this does not show](limitations.md) — the full ledger of claims and their status.
