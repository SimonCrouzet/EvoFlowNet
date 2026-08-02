---
date: 2026-07-29
topic: EvoGFN manuscript triage — masked lattice GFlowNets for directed evolution
verdict: SUPERSEDED 2026-07-30 → CONTINUE (see addendum; original verdict was PIVOT)
nugget: Action masking on a construction DAG does not sample the feasible set — it samples the parent-connected component of it — and the benchmark instances are configured so that the mutation radius, not the oracle budget, sets difficulty.
---

> **Superseded 2026-07-30.** `CLAIMS.md` became `HYPOTHESES.md`, the attainability
> audit landed (`cec5c7a`) and the reachable-support helper landed (`bb433a7`).
> The verdict below (PIVOT, framing is the blocker) no longer holds: the framing
> question is resolved and the risk has moved. **See the addendum at the end of
> this file.** The measurements in §"Measurements taken during this session"
> remain accurate as of 2026-07-29 and were independently confirmed by the audit.

# Evaluation: EvoGFN manuscript (DRAFT.md v1)

## Verdict: PIVOT

The current framing ("masked lattice GFlowNets for directed evolution at wet-lab
budgets") is not submittable, and the two reframings that the strategy agents
independently recommended are both refuted by measurements taken during this
session. What survives is a different and better paper: an audit of what masked
construction samplers actually reach, and of what the benchmark's own
configurations actually measure.

Timing evidence says urgency is low (the subfield is vacated, scooping risk LOW),
so the 6.5-week ICLR deadline is a self-imposed cost with no competitive benefit.

---

## Measurements taken during this session

All computed from `results/` and from the landscape/env code directly.

### 1. The evidence base is split across two code versions

`results/*/gfn-*.jsonl` exists in 1 of 12 tasks (`objectives`, and
`gfn-contrastive` there is 36/50). Every other GFlowNet arm exists only as
`*.jsonl.predeterminism`. The baselines were re-run after `19e060b`; the
GFlowNet arms were never reached, because they run last in method order and the
sweeps were interrupted.

The determinism fix **changed baseline results**: `rounds-8x48/genetic-feasible`
matches its pre-fix run on 12/50 seeds; `gb1-anchor/genetic+proxy` on 47/100.
So every paired-on-seed statistic in DRAFT.md currently spans two code versions.

The Jul-28 `ValueError: cannot fit a surrogate when no observation is finite`
crash is already fixed (`campaign.py:285`) and is *not* the current blocker.

### 2. Rerun cost is ~62 CPU-hours — not a bottleneck

From per-campaign timings in `logs/`. `budget-10000` alone is ~31 h;
`objectives` ~3.8 h. Sharded by task (race-free, documented), the full GFlowNet
rerun is under a day of wall clock. Dropping `budget-10000` and `objectives`
leaves ~27 CPU-hours — one evening, single-digit dollars.

### 3. The headline "exact tie on 100/100 seeds" is a structural ceiling

BFS of the masked-reachable set on the real `feasibility` instance
(L=64, c=2, k=4, d=0.15), wild type 0:

    |reachable set| = 26,580     max fitness = 0.3750     regret floor = 0.6250

0.625 is *exactly* the regret both `gfn-tb` and `genetic-feasible` report on
100/100 seeds. They are not tying as searchers — both are finding the optimum of
a 26,580-design reachable set with a 384-well budget. The task is **solved**, not
unsolved; regret against the attainable optimum is 0.000.

Consistent with this: across all methods and seeds the task has only **four**
attainable reward levels (0.125, 0.1875, 0.25, 0.375). `gfn-tb` scores the
maximum on 100/100 seeds; `genetic-feasible` on 199/200 records. Student-t paired
tests to three decimals are being run on a four-level ordinal outcome.

*Confidence:* exact for wild type 0. Other wild types exceeded the enumeration
cap; the inference that 0.375 is the ceiling generally rests on ~6,000 campaigns
never exceeding it. **Verify before relying on it.**

### 4. `large-space` is unsolvable by construction

Stanton's base config (L=256, c=4, k=8) needs **17–22 point mutations** to reach
reward 1.0. `MUTATIONS = 4` (`suite.py:58`). Maximum achievable reward is
**0.0938**; best attainable regret **0.9062**. DRAFT.md reports methods at
0.974–0.992 against a nominal optimum of 1.0 and attributes the failure to the
384-assay budget. The cause is the mutation cap. All method differences (0.018
spread) live inside a 0.094-wide window against a hard floor.

On L=32/64 tasks the attainable optimum **varies by wild type** (3–6 mutations
needed against a cap of 4), so per-seed ceilings differ (0.5625/0.75/1.0) while
regret is reported against a fixed 1.0. Paired tests survive; absolute numbers
and cross-task comparisons do not.

### 5. The "productive wells" collapse is false — the curves cross

Budget ladder (L=32, all four budgets pooled), mean regret by cumulative
productive wells:

| wells | gfn-tb | rejection-GA | blind GA | GA+proxy |
|---|---:|---:|---:|---:|
| 30–100 | **0.789** | 0.812 | 0.834 | 0.837 |
| 100–300 | **0.751** | 0.797 | 0.792 | 0.801 |
| 300–1000 | **0.651** | 0.705 | 0.719 | 0.670 |
| 1000–3000 | 0.682 | 0.612 | 0.587 | **0.542** |
| 3000–12000 | 0.625 | — | — | — |

Crossing at ~1,000 productive wells, not superposition. The GFlowNet is best
below and worst above; at 3,000–12,000 wells it is still worse (0.625) than a
blind GA at 1,000–3,000 (0.587). **This kills the productive-wells framing.**

### 6. THE LIVE RESULT — masking does not reach the feasible set

Exact enumeration (L=14, v=5, c=2, k=2, m=4). `F` = feasible sequences within the
mutation ball; `R` = those reachable through feasible single-mutation
intermediates, i.e. what `forward_mask` actually permits:

| density | \|F\| | \|F ∩ R\| | silently excluded |
|---|---:|---:|---:|
| 0.5 | 4,822 | 3,677 | **23.7%** |
| 0.3 | 373 | 139 | **62.7%** |
| 0.15 | 2 | 2 | 0% (only 2 feasible designs exist) |

**The excluded fraction grows as the constraint tightens** — exactly the regime
where masking is sold as the answer. This is unclaimed prior art territory:
SynFlowNet (ICLR 2025), RGFN (NeurIPS 2024), RxnFlow and MOGFN-AL all ship masked
construction without an enumerable support check. It also retroactively explains
§6.4's plateau (0.375 × 6 rounds then 0.379) as support exhaustion rather than
the "mode concentration" the draft asserts.

### 7. Against the rejection GA, the draft understates itself

Same-code-version paired comparisons: `gfn-tb` beats `genetic-feasible` on 9 of
10 tasks, ties on 1, never loses; significant on 6. The draft's "at parity" is
more pessimistic than its own data — but finding 3 means those effects sit
against ceilings, so this does not rescue the framing.

### 8. Housekeeping confirmed

Rename to **EvoGFN** committed (`743a0b2`) — closes retracted claim I4. MLDE is
committed and registered (`methods.py:307`), needs only running. `annealing` and
`cmaes` results exist on disk and appear nowhere in the draft; CMA-ES returns
infinite regret on every constrained task (feasible fraction 0.000–0.039).

---

## Dimension Scores (three framings stress-tested)

| Dimension | A: productive wells | B: masking-vs-rejection theory | C: budget position paper |
|---|---|---|---|
| Novelty | Weeks (ESS renamed) | Weeks (Freuder 1982 restated, and wrongly) | Months |
| Impact | Low–Medium | Low as framed | Medium, degrading |
| Timing | Well-timed | Too late (NLP side crowded) | Well-timed, closing |
| Feasibility | High risk | High risk | High risk |
| Competition | Moderate | Crowded (NLP) / open (GFN) | Moderate; Arnold lab dangerous |
| Nugget | Fuzzy — two welded together | **Wrong as stated** | Fuzzy — three stapled |
| Narrative | Compelling form, weak evidence | Weak | Self-undermining |
| **Verdict** | **REFINE → kill headline** | **REFINE → discard nugget** | **KILL vehicle, pursue content** |

Each was killed by a *different* measurement: A by the crossing (finding 5),
B by the ceiling (finding 3), C by the attainability confound (finding 4) plus
the venue error below.

---

## Key Concerns

1. **ICLR has no Position track.** Agent-verified against the ICLR 2027 CFP;
   `CallForPositionPapers` 404s. The Position track is **ICML's** (~January
   deadline). The "low-risk hedge" was a main-track no-experiments submission.
   *Worth confirming directly before acting on it.*
2. **The rejection GA's 200-attempt cap is a hyperparameter, not a property.**
   §7 concedes raising it trades wall clock for wells; §6.9 establishes wall
   clock does not bind. Those two sentences, both the author's, kill the headline
   in one line.
3. **All feasibility evidence rests on a non-reference Ehrlich generator**
   (limitation 12). Blocking for any benchmark-track claim.
4. **GB1, the only real landscape, has no feasibility constraint**, so it cannot
   support the central claim.
5. Timing: AbFlowNet and CAGFN both rejected at ICLR 2026; Awesome-GFlowNets has
   no 2025–2026 entries; reference implementations unmaintained since 2023.

---

## Watch List

- **Hyeonah Kim / Minsu Kim / Hernández-García / Bengio** (Mila, KAIST) — δ-CS,
  S3-GFN, Genetic-GFN. S3-GFN contrasts hard action-space restriction against
  soft regularisation and runs **no rejection control**. One port away.
- **Stanton / Frey / Cho** (Prescient Design) — own Ehrlich, holo-bench, LaMBO-2.
- **Wittmann / Yang / Arnold lab** (Caltech) — SSMuLA, ALDE, MLDE. Most likely
  to publish the budget-realism critique, in a biology venue.

Search terms: `GFlowNet AND ("action mask") AND ("rejection sampling")`;
`"Ehrlich function" OR "holo-bench"`; `"oracle budget" AND (protein OR sequence)`.

---

## Revisit Conditions

Re-promote the **method** framing if, after the rerun, the GFlowNet beats the
rejection GA on a task whose attainable ceiling is *not* saturated by both arms.
That requires instances configured so the reachable optimum is not trivially
found — i.e. fixing finding 4 first.

Re-promote the **benchmark** framing if the `pytorch-holo` adapter turns out to
be ~2 days and the feasibility effect reproduces on reference instances.

Kill the whole project only if finding 6 fails to replicate at scale — i.e. if
`|F \ R| / |F|` is near zero on the reference generator across densities. In
that case fall back to arXiv + workshop with the survey and the negative results.

---
---

# Addendum, 2026-07-30 — verdict revised to CONTINUE

`CLAIMS.md` → `HYPOTHESES.md`, plus commits `bb433a7` (reachable support) and
`cec5c7a` (attainability audit). Re-reviewed against those.

## What changed the verdict

**The audit is done, and it is stronger than what I specified.** I concluded the
mutation budget was too small. The correct result: mutations go one position at a
time, each position once, so every intermediate must independently satisfy the
transition matrix — and on all four Ehrlich tasks the planted optimum has **no
legal construction order at any fixed budget**. Availability is monotone in
placed positions, so a stuck greedy march is a *decision procedure*;
`planted_optimum_reachable` returns False as proof. Budget is necessary, not
sufficient. My 26,580 / 0.6250 enumeration was independently reproduced.

**The fix vindicates the wet-lab premise instead of wrecking it.** Four mutations
per round over eight rounds pins 1.0 exactly on `protocol-evolvepro` and all
seven diagnostics, where a fixed budget of 61 cannot reach it. The radius never
needed raising — the campaign needed to move. `MUTATIONS = 4` is gone, budgets
are protocol-derived, attainability is an interval with its method named, and a
test prevents recurrence.

## Where the 2026-07-29 evaluation was wrong

The proposed pivot — support gap as "a correctness bug in four published
methods" — was too strong. D1's own caveat defeats it: in SynFlowNet / RxnFlow /
RGFN, masking enforces synthesizability *via reaction steps*, so reachability
through valid intermediates **is the specification**. The claim only survives
where masking enforces an endpoint property, not a path property. Correctly
demoted to a footnote.

The surviving strong version is about **evaluation**, not design, and is
currently only in a commit message: the same policy scores L1 **0.061** against
the reachable set and **0.570** against the Hamming ball. No synthesizability
caveat applies. Promote it into `HYPOTHESES.md`.

## The new framing, and it is better than anything Phase 2–5 produced

**B2 — what transfers across an anchor move.** A policy learns
`P(action | state)` and transfers when the anchor moves; a GA holds specific
sequences and starts over; CMA-ES is parameterised relative to the parent, so the
same mean decodes differently after a move. First framing in this exercise that
is specifically about GFlowNets, is not scooped, and has a cheap decisive test
(the transfer probe: train in one region, re-anchor elsewhere, measure round one
before any new learning — a population provably cannot pass). B1 supplies the
measured mechanism. The note that "re-anchor on/off is the wrong test" — because
every method gains from a moving reachable set — is correct and pre-empts a
reviewer.

Load-bearing fairness condition: `_enforce_budget` is project scaffolding, not
part of a GA. If the GFlowNet wins because baselines were handicapped by an
imposed constraint, the result is worthless.

## The relocated risk

**Almost nothing is measured.** Done: G1, G2 (both about the project's own
process). Literature only: C1. Arithmetic: C3. Mechanism: B1. Roughly fifteen
hypotheses are untested or invalidated — **including B2, marked CENTRAL**.

Simultaneously the scope grew: CH65, multi-Ehrlich with a conflict dial,
NSGA-II, MOGFN-PC, TrpB, preference-count diagnostics. A second paper's
infrastructure, added while the central hypothesis has zero experiments.

The project moved from *wrong framing, lots of data* to *right framing, no data*.
Better science, worse for September.

## Corrected venue facts

- **ICML 2027 closes 7 September 2026** (conference 22–26 June 2027, Bolzano) —
  *earlier* than ICLR. My earlier "~January" was wrong; the 2027 edition moved.
- **ICLR 2027**: abstract 11 Sept, paper 16 Sept. No Position track.
- No confirmed Position track for ICML 2027 either — treat position papers as
  unavailable this cycle unless verified.
- **NeurIPS 2026 workshops**: suggested contribution deadline **29 August 2026**,
  notification by 29 September. Non-archival, so no novelty cost to a later
  main-track submission on B2.

## Revised plan

1. **Run the B2 transfer probe first.** Cheap, decisive, go/no-go on the thesis.
2. **Freeze multi-objective.** CH65 is good work and it is the scope risk.
3. **Workshop paper (29 Aug) on what is already finished** — G1 plus the 10×
   support-misreporting number. Self-contained, no rerun needed.
4. **Answer G1's own open question:** run `planted_optimum_reachable` against
   holo-bench's reference Ehrlich instances. If Stanton's own instances share the
   defect, the audit stops being self-critique and becomes a finding about the
   field. That is the difference between a footnote and a headline.
5. B2 becomes the main-track paper for a later cycle, once measured.

## Process note

The ledger kills claims well and has no mechanism for promoting findings into it
— `OUTLINE.md` lost C2 without a stated reason, and the 10× number is living in
a commit message. A rule that any measured number in a commit message owes an
entry in `HYPOTHESES.md` would close it.
