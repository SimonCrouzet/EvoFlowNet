# What this does not show

This page exists because the rest of the documentation is easier to trust if the limits are
stated in one place rather than buried where each of them happens to arise.

Nothing here is hypothetical. Every entry is either a negative result from the project's own
benchmark suite or a control that was not run.

---

## 1. No method solves any Ehrlich instance

Ehrlich landscapes have a planted optimum of exactly `1.0`, verified at construction, so
regret is exact. On the flagship `large-space` task (Stanton et al.'s base configuration,
`L = 256`), **every method sits between 0.974 and 0.992 of a maximum regret of 1.0** — random
mutagenesis, hill climbing, genetic algorithms, simulated annealing, CMA-ES, MLDE, and every
GFlowNet variant.

The differences between them are real and statistically clean. On that task the GFlowNet beats
a proxy-optimising GA on 30 of 30 seeds. It is a 0.017 advantage inside a shared failure.

The comparisons in this project are between **failure modes**, not between a working method
and broken ones. Any sentence that reads as "the GFlowNet solves X" is wrong.

---

## 2. GB1 is the easiest geometry in the suite

GB1 is the empirical anchor: 149,361 measured variants, near-complete, so regret is exact
against a real measurement rather than against the best thing anyone happened to find. That is
what it is for, and it is worth having.

It is not a hard test:

* **No feasibility constraint.** All 160,000 strings are sequences you could order. The
  rejection-sampling control `genetic-feasible` is bit-identical to `genetic+proxy` there,
  because there is nothing to reject.
* **The mutation budget constrains nothing.** Four sites, four mutations: every sequence is
  reachable in one step. `Protocol.constrains_search` returns `False`.
* **Diversity has almost no room.** `L = 4` bounds mean pairwise Hamming distance at 4, and the
  observed spread across methods is 2.9 to 3.5.

An earlier version of this project described GB1 as testing constrained search. That claim has
been retracted. GB1 says the numbers are not an artefact of synthetic landscapes. It says
nothing about constrained or feasibility-limited search.

Two further caveats internal to the dataset: 10,639 combinations were never assayed and are
imputed as zero by default (`is_measured` exists so an analysis can exclude them instead), and
the "optimum" is the best *measured* variant, so regret against it is exact only with respect
to what was assayed.

---

## 3. The feasibility result rests entirely on a synthetic proxy

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

### And within the synthetic setting, be precise about which claim survives

| Claim | Status |
|---|---|
| Unmasked methods waste nearly all of a campaign on unbuildable designs | **Supported.** Buildable wells of 384: 3.0 (random), 2.8 (GA), 1.9 (GA+proxy), 8.0 (hill-climb) |
| A masked GFlowNet is feasible by construction | **Supported, and definitional.** It is a property of the environment's mask, not an achievement of the sampler |
| Masking beats the strongest *unmasked* baseline on regret | **Supported.** +0.243, 100/100 seeds |
| **Masking beats rejection sampling on regret** | **Not supported.** Advantage +0.000, W/T/L 0/100/0 — an exact tie on every seed |
| Masking gets the same quality *while spending the budget* | **Supported, and this is the headline.** Rejection halts at 82 ± 16 of 384 wells; masking fills 384 |
| Masking restores the surrogate's learning signal | **Supported.** Rounds with a defined surrogate–oracle correlation: 189/400 masked, 0/400 for GA+proxy — with the caveat that "undefined" can also arise from reward quantisation |

The honest one-sentence version: **masking converts more of a fixed budget into measurements
than rejection sampling does; it does not search better.**

---

## 4. The budget argument is a survey finding, not our measurement

The survey is solid. Real ML-guided campaigns run at 20–800 assays (ALDE 396, LaMBO-2 374,
EVOLVEpro 50–90, Hie et al. ≤40) while the machine-learning convention runs at 1,000–10,000
(AdaLead / PEX / DyNA-PPO 1,000, LaMBO / MOGFN-AL 1,024, δ-CS / SILO 1,280, GFN-AL 10,000, PMO
10,000). Fifty-five sources, verified from primaries.

The inference people want to draw from it — that benchmarking above the wet-lab regime
*reverses conclusions* — is the one thing we tested directly, and **it did not reproduce.**

Across 96, 384, 1,000 and 10,000 oracle calls the ordering GFN < rejection-GA < GA+proxy held
throughout. The gap to a proxy-optimising GA moved non-monotonically — +0.111, +0.191, +0.241,
+0.098 — and never flipped. Something does appear to change at 10,000: a blind GA goes from the
worst tier to nominally the best, overtaking the GFlowNet. But that difference is −0.014
[−0.043, +0.016], t = −0.94, W/T/L 8/32/10: not significant. The trace is suggestive (the
GFlowNet plateaus after round 4 while the GA is still climbing at round 10) and the statistic
is not.

Use the budget survey as a reason to *measure at the wet-lab budget*. Do not use it as
evidence that the ML budget produces wrong rankings — our own experiment says it does not, at
least here.

---

## 5. Diversity is measured, and its usefulness is not

GFlowNet batches are more diverse than a rejection-sampling GA's on **every** task in the
suite, same direction throughout: 5.38 / 3.65 on `feasibility`, 7.54 / 3.96 on ALDE, 7.79 /
4.42 on `large-space`, 3.29 / 2.92 on GB1.

Two problems with reading that as a win:

* **The metric is weak.** Mean pairwise Hamming distance is maximised trivially by a random
  baseline. On GB1, `random` is the most diverse method (3.48) *and* the worst on regret
  (4.30). Diversity must never be reported without regret beside it.
* **The value of diversity is not isolated.** No diversity-aware-selection ablation was run.
  The claim that a diverse batch is worth more to a lab than a concentrated one is an argument
  in this project, not a measurement.

Mode-level metrics would be better, and GB1 supports them exactly. They have not been run.

---

## 6. Reproducibility: the GFlowNet arm is not bit-identical

At a fixed seed and configuration, the classical baselines reproduce identically on 50 of 50
replicates. `gfn-tb` reproduces identically on **30–32 of 50**, with a per-seed standard
deviation of 0.044 and a maximum deviation of 0.144 — one full quantisation level of the
landscape.

This is a known open defect, not a rounding artefact. Any GFlowNet effect smaller than that
noise floor should not be read, which directly affects the objective comparison below.

---

## 7. Nothing separates the GFlowNet training objectives

Five objectives were compared at equal budget on the diagnostic landscape: detailed balance
0.654, SubTB 0.663, contrastive balance 0.669, forward-looking DB 0.674, trajectory balance
0.691 (lower is better). Total spread ~5%; adjacent pairs are inside their standard errors.

Worse, the replicate check undermines the one comparison that looked significant: trajectory
balance at the *identical* configuration returned 0.6913, 0.6737 and 0.6763 across three
replicates. The DB-vs-TB effect of 0.038 is roughly twice the replicate spread it has to clear.

Two specific predictions failed outright:

* **Contrastive balance was expected to fix a `log Z` instability.** It is indistinguishable
  from trajectory balance everywhere in the suite — likely because trajectories here are only
  1–5 steps long, so there is little for `log Z` to destabilise.
* **Forward-looking DB was expected to be favoured**, because every state in the mutation
  lattice is itself a scorable sequence. The structural argument holds; it places 4th of 5.

Both are reported as negative results.

---

## 8. Round structure: underpowered, and the better experiment disagrees

Whether many small rounds beat few large ones at a fixed budget is genuinely open in the
literature, and we did not close it.

At `L = 32`, 384 calls: the GFlowNet prefers 8 × 48 over 4 × 96 by +0.014 [−0.013, +0.040],
W/T/L 13/29/8 — underpowered. Across methods the direction is not consistent: hill climbing
prefers *fewer* rounds significantly (t = −3.07). And on the better-powered replicate at
`L = 64` with 100 seeds, the GFlowNet ties exactly while both GA variants prefer fewer rounds.

That replicate is also confounded: 8 × 48 = 384 against 3 × 132 = 396, a 3% budget advantage
to the fewer-rounds arm. It needs rerunning at matched budget.

The defensible statement is the weak one: round structure matters **less** than acquisition or
proxy access. Round effects are ≤0.028; the masking effect is 0.243.

---

## 9. Novelty claims that were retracted

Kept here because a reader deserves to know what was checked and found already occupied.

| Retracted claim | What it collided with |
|---|---|
| Masking on a mutation-lattice GFlowNet is novel | MOGFN-AL (ICML 2023, App. D.6) already masks: "logits ... set to −1000" |
| The mutation-set state space is novel | GFlowNet Foundations §5.1 Def. 37 formalises the subset lattice; DAG-GFlowNet uses it |
| A correct closed-form `P_B` is a contribution | Malkin et al.: *any* valid `P_B` yields a unique correct `P_F`. `P_B` affects optimisation, not the target |
| First rounds-at-fixed-budget measurement at a three-digit budget | SSMuLA got there first, at 120–2,016 |

What survives as new is the **measured comparison of masking against rejection sampling at a
matched budget**, which is a much smaller claim than the ones above.

---

## 10. Controls that were not run

| Missing | Why it matters |
|---|---|
| A Bayesian-optimisation baseline (δ-CS, LaMBO-2, GameOpt) | The obvious reviewer question |
| A diversity-aware-selection ablation | Would decide whether diversity is worth anything |
| Realistic measurement noise (over-dispersed, top-flattening) | FLIGHTED reports r ≈ 0 between measured and true fitness for the top ~1,000 GB1 variants. Our noise model is wrong in the direction that flatters every method |
| TrpB as a second empirical landscape | Also an unresolved coverage discrepancy between sources — 99.45% vs 96.0% |
| Batch size × acquisition at fixed budget | The genuinely open cell in the literature. Acquisition is held at greedy everywhere here |
| Genetic-GFN | Implemented, not run in this suite |
| Exact distributional evaluation at scale | The one attempt cost 1.54M oracle calls on GB1 — roughly ten times the search space. It tested correctness, and must never be quoted as efficiency |

---

## How to cite anything on this site

If the claim is on this page as **supported**, cite the number and the seed count. If it is
listed as underpowered, negative or untested, say so in the same sentence. The suite exists to
make that distinction cheap; using it selectively would waste the only thing it provides.
