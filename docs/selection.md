# Choosing the GFlowNet's configuration

Every classical baseline in this suite runs at hyperparameters its own authors tuned. The
genetic algorithm uses the mutation and recombination rates from the Ehrlich paper; MLDE runs
in the regime Wittmann et al. actually run it in. The GFlowNet had no such authority to appeal
to: gradient steps set to whatever was fast enough, a reward exponent carried over from a
different paper on a different problem, and a training objective that was never chosen at all.

A table built that way reports a comparison of *configurations* while claiming to compare
*methods* — and it does it in the direction that flatters the field, because the field is tuned
and we are not.

So the configuration is **selected**, by a rule written down before any of its numbers existed,
on a landscape no claim is ever drawn from. This page is that procedure: the rule, where it is
allowed to run, what each stage decides, what the design cannot see, and how to reproduce it.

---

## The rule

**Lowest mean regret, with top-K diversity breaking statistical ties.**

Both halves were fixed before the first campaign ran. A criterion chosen after seeing the table
is not a criterion, and "best regret, except on the axis where diversity looked better" is how a
sweep becomes a story.

Two things about the rule need to be precise, because loose versions of both are common.

**A tie is statistical, not numerical.** Two arms tie when a paired comparison over the seeds
they share cannot separate them. That is a statement about the evidence, not about decimal
places: an arm ahead by 0.006 with a confidence interval spanning zero has not won anything.
Arms with a *worse* mean stay in the running when the gap is inside the noise, because losing by
less than the noise is not losing.

**The tie-break is load-bearing, not tidying.** What this project claims is diverse, feasible,
high-fitness variants — not high-fitness ones. A rule reading regret alone would happily select
a configuration that optimises well and samples badly, and the diversity column of the headline
table would then have to live with whatever that produced. Ties are not a rare edge case here:
at 30 seeds on the diagnostic landscape, four of five objectives sat within 0.02 regret of each
other.

Two smaller decisions are part of the rule rather than implementation detail:

* Only seeds that **every** arm holds are used, so the paired comparisons deciding ties are
  genuinely paired.
* An arm that failed on some seeds is scored on the ones it survived; an arm that failed on all
  of them is not eligible to be chosen.

---

## Where it runs, and where it must not

Both the objective comparison and the scans run on the **diagnostic landscape** — Ehrlich at
`L = 32`, four rounds of 96, the same cheap instance the other diagnostics vary an axis on. No
headline task uses it.

That is the whole reason the phase is not tuning on the test set. The configuration is fixed
before a single claim-carrying campaign is scored, and it is fixed against a landscape that
carries no claim. The tier is tagged `Purpose.SELECTION` rather than `DIAGNOSTIC`, which keeps
the distinction in the type: a diagnostic measures how methods behave, a selection tier chooses
*our own* configuration, and the results table can refuse a tier that was never eligible to
appear in it.

---

## Three stages

| Stage | Decides | Arms | Seeds |
|---|---|---|---|
| **A** | the training objective | six candidates, at the default reward exponent | 100 |
| **B** | the reward exponent, for whichever objective A chose | `SELECTION_BETAS` | 100 |
| **C** | gradient steps per round — the GFlowNet's proxy budget | `SELECTION_STEPS` | 100 |

**Why 100 seeds.** The 30-seed diagnostic put four objectives within 0.02 regret of each other
and asked for thousands of seeds to separate the closest pairs. One hundred is what it said
would resolve the single gap that looked real — sub-trajectory balance against trajectory
balance — and it is enough to state honestly that the rest are tied.

**Why sequential.** The full cross of objectives, exponents and step counts is far more compute
than the question needs, and B and C cannot start before their input exists: the winning
objective is not known until A has finished. Running the scans only for the winner is what keeps
the phase affordable.

**Stage C is not an internal detail.** Gradient steps × batch size is the number of *proxy*
evaluations the GFlowNet spends per round, and proxy spend is a reported column in the results
table. This stage decides a number the results print, so it is measured rather than inherited.

### The grids were widened once, deliberately

The first exponent pass came back monotone to its own edge — 0.502, 0.473, 0.446 across
`beta` 1, 3, 10 — which cannot distinguish "10 is right" from "10 is the largest value we
offered". Widening upward answered it: regret turns hard above 10 and the optimum is interior.
The values below 1 close the same hole at the other end, where diversity was still rising at the
lowest exponent tried, and diversity is the axis the tie-break actually decides on.

Extending a grid after looking at it is only legitimate under conditions that happen to hold
here: the rule was fixed before any of these numbers existed, the landscape carries no claim,
and the rule is regret-first — an exponent buying diversity at a real regret cost is not
eligible, since only *statistical* ties go to diversity. Extending until the answer becomes
agreeable would not be legitimate, so the grids are fixed now and the full curve is reported
either way.

---

## What this design cannot see

The stages are sequential, so they cannot see interactions between the objective and its
hyperparameters. An objective that loses at the default exponent and would have won at another
one is invisible to this design.

That is not a hypothetical worry. The reward-exponent curve **reversed direction** between
trajectory balance and sub-trajectory balance, so these hyperparameters demonstrably do not
transfer across the objective. The same reasoning applies to stage C: the earlier step-count
evidence was measured on a different objective and is not carried over.

The design's mitigations are partial, and worth naming as partial:

* Stage A fixes the exponent at the default the objectives were compared at, rather than at an
  arbitrary value, so the comparison is at least at a shared and stated point.
* Every stage reports its whole table, not just its winner, so a reader can see how wide the
  ties were.

One further hole is inherited from the objective that wins: sub-trajectory balance's
length-weighting `lam` interpolates detailed balance as it approaches zero and trajectory balance
as it grows. It is a one-parameter family, and the selection phase evaluates it at a single
point.

---

## Running it

```bash
uv run python experiments/select_configuration.py            # all three stages
uv run python experiments/select_configuration.py --report   # read the store, run nothing
```

Each stage prints its full table — regret and diversity per arm, which arms tied, which was
chosen, and the sentence saying why — before moving to the next.

| Flag | What it does |
|---|---|
| `--report` | Read what is already stored and print the tables. Runs no campaigns |
| `--stage {a,b,c,both}` | Run one stage. Shards use `a`, then one process runs `b`, then `c` |
| `--only ARM` | Run just this arm. Repeatable; this is the sharding knob |
| `--print-winner` | Print stage A's chosen arm and exit, for a coordinator to shard stage B on |
| `--seeds N` | How many seeds an arm must hold before a stage is allowed to choose |
| `--results DIR` | Where results live. Defaults to `results/` |

The phase refuses to start if threading is not pinned (exit code 3), because an unpinned run
produces numbers a later run cannot reproduce. A stage that cannot yet choose — because some arm
is short of seeds — says so and stops rather than choosing on partial evidence.

### It is resumable

Every campaign is written the moment it finishes, so an interrupted run keeps everything up to
the interruption and a rerun computes only what is missing. Records also carry a fingerprint of
the code that could have produced them, so a stored campaign that the source has since moved
past is re-run rather than trusted.

### It shards, one process per arm

```bash
# Stage A: one process per objective, in parallel.
for arm in gfn-tb gfn-contrastive genetic-gfn gfn-db gfn-subtb gfn-fldb; do
  uv run python experiments/select_configuration.py --stage a --only "$arm" &
done
wait

# Whichever objective won stage A is what stage B scans.
winner=$(uv run python experiments/select_configuration.py --print-winner)
```

Sharding across *processes* rather than threads is what makes this safe. Campaigns are
independent, the store keeps one file per arm so writers never collide, and every campaign is
seeded from its own seed rather than from process order — so a sharded run and a serial one
produce identical records. Raising the thread count instead would not, since a multithreaded
reduction sums in completion order.

A shard that finishes first stops after its own arms: a winner drawn from a subset is a winner
of the subset, not of the stage.

---

## What comes out

The phase writes `results/selected.json`:

| Field | Meaning |
|---|---|
| `objective`, `beta`, `steps` | the chosen configuration, as fields rather than a name to be parsed |
| `arm` | the arm's name, as it appears in the results table |
| `objective_reason`, `arm_reason`, `steps_reason` | why each stage chose what it chose, in a form that can be pasted into a caption |
| `objective_tied` | the arms stage A could not separate. More than one name means the choice was settled on diversity rather than on regret |
| `seeds`, `task` | what the decision rests on |

The benchmark **reads** that file rather than re-deriving the choice. Re-deriving would silently
pick a different arm the moment a seed count or an arm list moved, and the table would then
report a configuration no selection ever made. The selected arm *replaces* the untuned GFlowNet
arms in the headline tiers rather than joining them — keeping both would put two configurations
of the same method in one table, and the better of the two would be exactly the thing the phase
was run to avoid reporting.

If the file is absent, the suite falls back to the untuned defaults and says so, rather than
quietly benchmarking them as though they had been chosen.

---

## API

- [`evogfn.benchmark.selection`](reference/benchmark.md) — the rule and the arm builders.
- [The benchmark suite](benchmark.md) — what each task decides, and what a protocol is.
- [What this does not show](limitations.md) — including what this staged design cannot see.
