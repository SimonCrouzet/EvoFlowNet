# What this does not show

This page exists because the rest of the documentation is easier to trust if the limits are
stated in one place rather than buried where each of them happens to arise.

Nothing here is hypothetical. Every entry is either a negative result from the project's own
benchmark suite, a control that was not run, or a defect in the machinery that produces the
numbers.

---

## 1. Almost every measured number in this project predates the code that would produce it now

The store holds 7,040 campaign records. **900 of them are records the current source could have
produced** — nine reward-exponent arms at 100 seeds each, all on the selection landscape.
Everything else is flagged stale against the fingerprint of the code it would have to be re-run
under.

That is not a filing problem. Three changes landed between those records and this page, and each
of them moves numbers rather than presentation:

* **Campaigns re-anchor.** The environment's parent moves to the best design measured so far at
  the end of every round, so a campaign's reach is `max_mutations × rounds` rather than
  `max_mutations`. With a fixed anchor every shape in the round sweep searched the identical
  Hamming ball, and every Ehrlich task's planted optimum sat 61 to 248 substitutions outside a
  radius of 4.
* **Regret is measured against what a task's search space was audited to contain**, not against
  the landscape's nominal optimum. On `large-space` 95% of the previously published regret was a
  floor no method could have cleared. On `feasibility` an arm sitting exactly on the reachable
  maximum was reported at a regret of 0.626.
* **Threading is pinned**, which is what makes a GFlowNet arm reproduce itself at all — see §7.

Three consequences, stated plainly:

* **The headline comparison has no numbers at all right now.** There is no stored GFlowNet
  campaign on `gb1-anchor`, `large-space`, `feasibility`, `protocol-alde` or
  `protocol-evolvepro`, and every classical arm on those tasks is stale.
* **`mlde` has never produced a stored campaign on any task.** See §8 for why.
* Figures quoted below are quoted only where the change that invalidated the rest cannot have
  moved them: structural counts of what a set contains, per-arm counts on a task whose anchor
  never moved, or a measurement re-run against the current source. Every other pre-audit regret
  comparison has been deleted from this page rather than hedged.

---

## 2. Some tasks can no longer separate the methods, and it is the opposite failure

An earlier version of this page said that no method solves any Ehrlich instance, and that every
arm on `large-space` sat between 0.974 and 0.992 of a maximum regret of 1.0. **That is retracted.**
The regret it described was measured against a planted optimum that had no legal construction
order at the budget in force, so most of the column was a constant nobody had computed.

What each task's search space was audited to contain:

| task | per round | re-anchors | attainable optimum |
|---|---|---|---|
| `gb1-anchor` | 4 | no | the landscape's own optimum — the ball is the whole published table |
| `large-space` | 62 | yes | **[0.2812, 1.0]** — the one bracket the audit could not close |
| `feasibility` | 4 | no | **0.3750, exact** — 26,580 constructible designs in a ball of 8 × 10¹⁰ |
| `protocol-alde` | 21 | yes | **1.0, exact** |
| `protocol-evolvepro` | 4 | yes | **1.0, exact** |
| the seven diagnostics | 4 | yes | **1.0, exact** |

The problem has changed shape rather than gone away. `feasibility`'s rejection-sampling arm
reaches 0.3750 — the reachable maximum — on 99 of 100 seeds. **A task an arm has exhausted cannot
rank the arms above it**, and a comparison drawn on it is vacuous regardless of how many seeds it
ran. Four of the five main tasks are now audited to contain their nominal optimum, which is what
makes them winnable and also what makes them saturable.

`large-space` is the opposite hazard. 0.2812 is witnessed by a design the audit's beam actually
built; 1.0 is what the reward's structure permits at 248 cumulative substitutions; nothing
measured says which is right. Any comparison there is read against an interval three quarters as
wide as the scale it sits on.

The sentence from the old version that still stands: **any claim that reads as "the GFlowNet
solves X" is wrong** — now for a second reason, since where a task is solved it is solved by a
rejection-sampling genetic algorithm too.

---

## 3. GB1 is the easiest geometry in the suite

GB1 is the empirical anchor: 149,361 measured variants out of 160,000, near-complete, so regret
is exact against a real measurement rather than against the best thing anyone happened to find.
That is what it is for, and it is worth having.

It is not a hard test:

* **No feasibility constraint.** All 160,000 strings are sequences you could order. The
  rejection-sampling control `genetic-feasible` returns bit-identical results to `genetic+proxy`
  there — same mean regret to four decimals, same mean diversity — because there is nothing to
  reject.
* **The mutation budget constrains nothing.** Four sites, four mutations: every sequence is
  reachable in one step, `Protocol.constrains_search` returns `False`, and the anchor has nowhere
  to move, which is why this is the one Ehrlich-free task that does not re-anchor.
* **Diversity has almost no room.** `L = 4` bounds mean pairwise Hamming distance at 4, and the
  observed spread across the classical arms is 2.5 to 3.5.

An earlier version of this project described GB1 as testing constrained search. That claim has
been retracted. GB1 says the numbers are not an artefact of synthetic landscapes. It says
nothing about constrained or feasibility-limited search.

Two further caveats internal to the dataset: 10,639 combinations were never assayed and are
imputed as zero by default (`is_measured` exists so an analysis can exclude them instead), and
the "optimum" is the best *measured* variant, so regret against it is exact only with respect
to what was assayed.

---

## 4. The feasibility result rests entirely on a synthetic proxy

This is the most load-bearing limitation on the page, because feasibility is where the
project's strongest measured results are.

Ehrlich's constructibility constraint is a **first-order Markov rule on adjacent tokens**:
certain residue pairs may not be adjacent. Real constructibility limits — codon usage,
synthesis failure modes, expression, folding, aggregation — are not first-order Markov in the
residue sequence.

What the proxy reproduces is the *shape* of the problem: a feasible set that shrinks
geometrically with sequence length, so that uniform proposals are almost all unbuildable. That
shape is the right one, and it is why the effect is large and clean. But:

* **GB1, the only real landscape in the suite, has no feasibility constraint at all.**
* Nothing in this repository tests any feasibility claim against a real constructibility rule.
* The transfer of these results to a real constraint is **not tested** and must be stated as a
  limitation wherever the feasibility numbers appear.

### Within the synthetic setting, be precise about which claim survives

`feasibility` keeps a fixed anchor, so the re-anchoring change did not touch it, and per-arm
*counts* are unaffected by rescoring regret against 0.375 instead of 1.0. Those survive. The
regret comparisons do not: there is no stored masked arm on this task, and the rejection arm has
since been shown to solve it.

| Claim | Status |
|---|---|
| Unmasked methods waste nearly all of a campaign on unbuildable designs | **Supported.** Buildable wells of 384: 3.0 ± 1.2 (random), 3.0 ± 1.5 (GA), 2.0 ± 1.0 (GA+proxy), 8.0 ± 2.3 (hill-climb), 2.6 ± 1.4 (annealing), 1.8 ± 0.9 (random+surrogate) |
| A masked GFlowNet is feasible by construction | **Supported, and definitional.** It is a property of the environment's mask, not an achievement of the sampler |
| Masking gets the same quality *while spending the budget* | **Supported on the throughput half.** Rejection sampling fills all 384 wells with buildable designs and halts after 84 ± 14 oracle calls of 384; masking has no such stall by construction. The *quality* half now has no measurement |
| Masking beats rejection sampling on regret | **Not measurable on this task.** Rejection sampling reaches the reachable maximum of 0.3750 on 99 of 100 seeds, so there is no headroom left to separate anything |
| Masking beats the strongest *unmasked* baseline on regret | **Unmeasured.** The pre-audit number was against a nominal optimum three quarters of which was unreachable, and there is no post-audit masked arm |
| Unmasked campaigns return no fittable signal | **Supported.** Rounds with a defined surrogate–oracle correlation: **0 of 400** for GA+proxy, 0 of 400 for annealing, 1 of 400 for random+surrogate, against 241 of 399 for rejection sampling — with the caveat that "undefined" can also arise from reward quantisation |

The honest one-sentence version: **masking converts more of a fixed budget into measurements than
rejection sampling does; whether it also searches better is currently unmeasured.**

### The set masking builds is not the feasible set, and one API still says it is

A design can satisfy the transition constraint, sit inside the mutation budget, and still have
no ordering of its mutations along which every intermediate is feasible. Masking builds the
*reachable* part of the feasible set, which is smaller. The gap is not marginal: on a length-8
Ehrlich toy at a budget of two, the Hamming ball holds 277 sequences, 26 of them feasible, and
18 reachable. On the instance `experiments/distributional_fidelity.py` defaults to — `L = 8` over
four tokens within three substitutions — the ball holds 1,789, of which 237 are feasible and 181
constructible, so **23.6% of the feasible set has no legal construction order**.

`MutationEnvironment.is_reachable` does not test this. It tests the mutation budget and the
adjacency constraint, which is the *feasibility* question under the reachability name. The only
method that answers the real question is `reachable_terminal_states`, which walks the graph, and
it refuses on Hamming-ball size rather than on the reachable set's own size. Any caller admitting
sequences from outside — a replay buffer, a genetic teacher, an assay — that reaches for the
obvious name gets the wrong test.

---

## 5. The budget argument is a survey finding, not our measurement

The survey is solid and is independent of any run of ours. Real ML-guided campaigns run at
20–800 assays (ALDE 396, LaMBO-2 374, EVOLVEpro 50–90, Hie et al. ≤40) while the machine-learning
convention runs at 1,000–10,000 (AdaLead / PEX / DyNA-PPO 1,000, LaMBO / MOGFN-AL 1,024,
δ-CS / SILO 1,280, GFN-AL 10,000, PMO 10,000). Fifty-five sources, verified from primaries.

The inference people want to draw from it — that benchmarking above the wet-lab regime
*reverses conclusions* — is the one thing we tested directly, and it did not reproduce: across
96, 384, 1,000 and 10,000 oracle calls the ordering never flipped.

**That null is now void.** The budget gradient ran with a fixed anchor, on a landscape whose
optimum sat outside the search radius, so every arm at every budget was stranded in the same ball
and the experiment could not have shown a flip. The four budget tasks now re-anchor, and none of
them has been re-run.

Use the budget survey as a reason to *measure at the wet-lab budget*. Do not use it as evidence
that the ML budget produces wrong rankings, and do not cite our null against it either — we no
longer have one.

---

## 6. Diversity is measured, and its usefulness is not

Two problems with reading a diversity advantage as a win, and both are prior to any number:

* **The metric is weak.** Mean pairwise Hamming distance is maximised trivially by a random
  baseline. On GB1, `random` is the most diverse method (3.48) *and* the worst on regret (4.30).
  Diversity must never be reported without regret beside it.
* **The value of diversity is not isolated.** No diversity-aware-selection ablation was run. The
  claim that a diverse batch is worth more to a lab than a concentrated one is an argument in
  this project, not a measurement.

The per-task diversity comparison that used to sit here — GFlowNet batches more diverse than a
rejection-sampling GA's on every task in the suite — has been removed rather than restated. There
is no stored GFlowNet campaign on any main task to draw it from.

Mode-level metrics would be better, and GB1 supports them exactly. They have not been run.

---

## 7. Determinism is enforced for torch, and only for torch

The GFlowNet arm used to be the defect on this page: at a fixed seed and configuration it
reproduced identically on 30–32 of 50 replicates while the classical baselines managed 50 of 50.
**That is fixed.** The cause was floating-point reduction order in a multithreaded matmul, one
thread pins it, `experiments/run_suite.py` exits with code 3 rather than running unpinned, and
every stored record carries whether it was produced that way.

The fix is narrower than the mechanism that carries it, and the gap is not guarded:

* `configure_determinism` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` with `os.environ.setdefault` — but it is called inside `main()`, long
  after the module imported numpy at the top of the file, and those variables are read by the
  BLAS at import. **The environment pinning is a no-op.** What actually binds is
  `torch.set_num_threads`, which governs torch and nothing else.
* `is_deterministic()` returns `torch.get_num_threads() == 1`. It therefore reports `True` while
  numpy reductions run across every core the machine has, and a record's `deterministic: true`
  field is a statement about torch alone.

In practice this has not bitten — the numpy arms are the ones that always reproduced — but
nothing in the suite would notice if it did. To pin the BLAS the variables have to be set before
numpy is imported, which means a launcher or a `sitecustomize`, not a function call in `main`.

Any GFlowNet effect is now readable down to the seed noise rather than to a 0.044 replicate
floor. Every effect *reported on this page* still predates the fix, which is §1.

---

## 8. Compute is not comparable between arms, and the column that would show it is wrong

Proxy spend is a **chosen budget, not an architectural constant**, and the arms in this suite
choose wildly different ones at an identical oracle budget. On a four-round, 384-assay campaign:

| arm | oracle calls | proxy evaluations |
|---|---|---|
| `random`, `genetic`, `hill-climb`, `random+surrogate` | 384 | 0 |
| `genetic+proxy`, `genetic-feasible`, `annealing`, `cmaes`, `mlde` | 384 | 38,400 — 50 inner generations of 256 candidates, for three rounds |
| the GFlowNet at the selected configuration | 384 | 57,600 — 300 gradient steps of batch 64, for three rounds |

Two things are wrong with the way that is currently reported.

**The stored column reads zero for the GFlowNet.** Re-running `gfn-subtb-beta-0.1` at seed 0
reproduces the stored best of 0.375 exactly and reports **57,600 proxy calls against the stored
record's 0** — and that record is *not* flagged stale, because the fingerprint covers what could
have changed the campaign rather than what could have changed the record written about it. Until
the affected arms are re-run, `proxy_calls` cannot be read out of the store, and a table built
from it would say the GFlowNet is the cheapest arm in the suite when it is the most expensive.

**`mlde` is wrapped in a proxy inner loop it should not have.** MLDE fits its own cross-validated
kernel ensemble; handing it an external surrogate as well makes it fit a model to another model's
predictions, because the wrapper feeds proxy values back through the same `observe` call the
assay results use. It also refits whenever its training set changes, and the wrapper changes it
fifty times a round — roughly 150 cross-validated refits of a twelve-member roster at five folds
per campaign, where the published method does one per round. This is the reason **`mlde` is the
only arm in the suite with no stored campaign on any task**, and the reason the most important
baseline here is currently missing from the comparison entirely.

Wall-clock is not a claim this project makes in either direction. It is a column that has to be
printed, because a paper that reports equal oracle budgets and hides a 20-fold difference in
everything else is making the same mistake as one that reports the difference and calls it a win.

---

## 9. The training objectives do separate — on one landscape, at one exponent

An earlier version of this page said nothing separates them, on the strength of five objectives
within ~5% of each other at 50 seeds. **That is superseded.** Those numbers came from a different
anchoring regime and are not comparable to what follows.

The [selection phase](selection.md) compared six training objectives at 100 seeds on the
diagnostic landscape, re-anchored, against an attainable optimum of 1.0:

| objective | mean regret | paired against SubTB |
|---|---|---|
| **sub-trajectory balance** | **0.4169** | — |
| forward-looking DB | 0.4556 | +0.039 [+0.007, +0.071] |
| contrastive balance | 0.4625 | +0.046 [+0.008, +0.084] |
| trajectory balance | 0.4706 | +0.054 [+0.016, +0.092] |
| detailed balance | 0.4744 | +0.058 [+0.019, +0.096] |
| Genetic-GFN | 0.4956 | +0.079 [+0.044, +0.113] |

Sub-trajectory balance is separated from all five, every interval excluding zero. The seed count
is the whole content of that sentence: at 30 seeds this was a five-way tie, and the power
estimate asked for about 97 seeds to resolve the closest pair.

What that does **not** license:

* **It is one diagnostic landscape.** The margins are 0.04–0.08 on a scale whose optimum is 1.0.
  Nothing here says sub-trajectory balance is better in general, only that it is the configuration
  to run this benchmark with.
* **It is an ordering at one reward exponent**, which is §13.
* **The arm the benchmark actually runs is not the lowest-regret arm.** The exponent scan came
  back a seven-way statistical tie, and the rule's diversity tie-break picked `beta = 0.1` at
  regret 0.4356 and diversity 10.74, against the numerically best exponent's 0.4075. That is the
  rule working as written, and it is still a 0.028 regret gap chosen on a secondary axis.

Two specific predictions failed outright, and both survive the re-measurement:

* **Contrastive balance was expected to fix a `log Z` instability.** It places third and is
  indistinguishable from trajectory balance across the suite — likely because trajectories here
  are short, so there is little for `log Z` to destabilise.
* **Forward-looking DB was expected to be favoured**, because every state in the mutation lattice
  is itself a scorable sequence. The structural argument holds; it places second, behind an
  objective the argument says nothing about.

The one thing the old section was right about and is worth keeping: a spread this narrow is not a
finding about GFlowNet objectives. It is a finding about this landscape.

---

## 10. Round structure: the measurement was void, and the replacement is confounded

Whether many small rounds beat few large ones at a fixed budget is genuinely open in the
literature, and we did not close it.

**The diagnostic that used to be quoted here cannot have measured anything.** `rounds-4x96`
against `rounds-8x48` ran with a fixed anchor, so every shape in the sweep searched the identical
Hamming ball and the curve was flat by construction. Whatever it showed was not about rounds.
Both tasks now re-anchor and neither has been re-run.

The main-tier replacement is `protocol-alde` (3 × 132) against `protocol-evolvepro` (8 × 48), and
it carries two confounds that have to be stated together:

* **The budgets are not matched.** 396 against 384 — a 3% advantage to the fewer-rounds arm.
* **The per-round radius is not matched either.** 21 substitutions a round against 4. That is
  deliberate: eight rounds buy from a radius of 4 the reach that three rounds need 21 for, and
  both are audited to pin 1.0 exactly, so the comparison is between two shapes that can each
  reach the answer. But it means the pair differs in two things rather than one, and an effect
  cannot be attributed to shape alone.

There is no defensible statement about round structure from this suite at present. The weaker
claim the old version made — that round effects are small next to acquisition or proxy access —
rested on the void diagnostic and has been removed rather than restated.

---

## 11. Novelty claims that were retracted

Kept here because a reader deserves to know what was checked and found already occupied.

| Retracted claim | What it collided with |
|---|---|
| Masking on a mutation-lattice GFlowNet is novel | MOGFN-AL (ICML 2023, App. D.6) already masks: "logits ... set to −1000" |
| The mutation-set state space is novel | GFlowNet Foundations §5.1 Def. 37 formalises the subset lattice; DAG-GFlowNet uses it |
| A correct closed-form `P_B` is a contribution | Malkin et al.: *any* valid `P_B` yields a unique correct `P_F`. `P_B` affects optimisation, not the target |
| First rounds-at-fixed-budget measurement at a three-digit budget | SSMuLA got there first, at 120–2,016 |

What survives as new is the **measured comparison of masking against rejection sampling at a
matched budget**, which is a much smaller claim than the ones above — and which, per §4, currently
has evidence on the throughput half and none on the quality half.

---

## 12. Controls that were not run

| Missing | Why it matters |
|---|---|
| A Bayesian-optimisation baseline (δ-CS, LaMBO-2, GameOpt) | The obvious reviewer question |
| A diversity-aware-selection ablation | Would decide whether diversity is worth anything |
| Realistic measurement noise (over-dispersed, top-flattening) | FLIGHTED reports r ≈ 0 between measured and true fitness for the top ~1,000 GB1 variants. Our noise model is wrong in the direction that flatters every method |
| TrpB as a second empirical landscape | Implemented and provenance-verified, but it has no task in the suite, so the empirical comparison rests on GB1 alone. Also an unresolved coverage discrepancy between sources — 99.45% vs 96.0% |
| Batch size × acquisition at fixed budget | The genuinely open cell in the literature. Acquisition is held at greedy everywhere here |
| `mlde` on any task | Implemented, strengthened to Wittmann et al.'s cross-validated ensemble, and never successfully run — see §8 |
| Exact distributional evaluation at scale | Now reported on an enumerable `L = 8` instance by `experiments/distributional_fidelity.py`, against the reachable support and against the Hamming ball. At scale it is still absent: the one attempt cost 1.54M oracle calls on GB1, roughly ten times the search space, and tested correctness rather than efficiency |

Genetic-GFN has been removed from this table. It was previously listed as implemented but not
run; it has since run at 100 seeds on the selection landscape, where it places last of six (§9).

---

## 13. The configuration was selected in stages, which cannot see interactions

The GFlowNet's configuration is chosen rather than inherited, for the reason given on the
[selection page](selection.md): every baseline here runs at hyperparameters its own authors
tuned, so a GFlowNet at defaults would not be being compared to them. The rule was fixed before
the numbers arrived and both scans sit on a landscape no headline task uses.

The design's cost is that the stages run in sequence — objective first, then the reward exponent
for the winner, then gradient steps — so **an objective that loses at the default exponent and
would have won at another one is invisible.** The full cross was not run; it costs several more
nights of compute.

This is not a hypothetical gap. The reward-exponent curve **reversed direction** between the two
objectives it has been scanned for. Across `beta` 1, 3, 10 trajectory balance gave 0.502, 0.473,
0.446 — falling — while sub-trajectory balance gives 0.4175, 0.4169, 0.4431 — rising. These
hyperparameters demonstrably do not transfer across the objective, and therefore the ordering in
stage A is an ordering *at one exponent* rather than an ordering of the objectives.

Two smaller versions of the same hole:

* Stage C's step-count scan is likewise run only for the configuration the earlier stages chose,
  and it decides a number the results table prints — gradient steps × batch size is the
  GFlowNet's proxy spend, which is §8.
* The winning objective's sub-trajectory length weighting is a one-parameter family — it
  interpolates detailed balance at one end and trajectory balance at the other — and it has been
  evaluated at a single value.

The selection phase is a **fairness precondition, not a finding**. Nothing on this page or in
the results table should cite it as evidence about which objective is better; it is the record
of how our own configuration was fixed, and of what fixing it that way could not see.

---

## How to cite anything on this site

If the claim is on this page as **supported**, cite the number and the seed count. If it is
listed as underpowered, negative, void or untested, say so in the same sentence. And check §1
first: at the time of writing, the only campaigns in the store that the current code would
reproduce are the selection phase's reward-exponent arms, so almost anything else needs a re-run
before it is quoted at all. The suite exists to make that distinction cheap; using it selectively
would waste the only thing it provides.
