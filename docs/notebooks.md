# Notebooks

Four walkthroughs, in `notebooks/`, taking you from "what is a fitness landscape" to a full
design–build–test–learn campaign with its artifacts read back off disk.

They are stored as **`.py` "percent" scripts, not `.ipynb`**. Three reasons, in order of how
often they matter:

* **They diff.** A review of an `.ipynb` is a review of base64 and execution counts. A review
  of one of these is a review of the change.
* **They run.** `python notebooks/03-gflownet-from-scratch.py` executes the whole thing, so a
  notebook that has silently rotted fails a build instead of failing a reader.
* **They lint.** Ruff checks them alongside `src/`, under the same rules.

```bash
uv run python notebooks/01-landscapes-and-epistasis.py

# Prefer a real notebook? jupytext converts, and the Jupyter extension opens the
# .py directly as one.
uvx jupytext --to notebook notebooks/01-landscapes-and-epistasis.py
```

There are **no plots**. Everything prints as a table. That keeps the notebooks executable in
CI with no plotting dependency, and it keeps the numbers quotable.

---

## 01 — Landscapes, epistasis, and what "feasible" means

`notebooks/01-landscapes-and-epistasis.py` · seconds · **the GB1 section needs network**

Builds an Ehrlich landscape small enough to print in full, then demonstrates the three
properties that make these landscapes worth benchmarking on.

* **Epistasis.** Re-implements the motif satisfaction formula in four lines and verifies that
  fitness is exactly `s₀ × s₁` — a *product*, so a zero in any factor zeroes the score however
  perfect the others are. Then shows what the quantisation parameter `q` does by running the
  same landscape at `q = k` and `q = 1`: the second collapses the single-mutant scan to `{0,
  1}` with no gradient anywhere.
* **Feasibility.** Measures the fraction of uniformly random strings that are constructible,
  against the closed form. By `L = 16` you cannot find one in 20,000 draws.
* **Enumerability.** Shows the Hamming ball under a mutation budget, and why being able to
  write it down is what makes an exact distributional check possible at all.

Ends with GB1, and with an explicit statement of what GB1 does not test.

## 02 — Why the baselines collapse

`notebooks/02-why-baselines-collapse.py` · under a minute · numpy only, no network

Runs random mutagenesis, hill climbing and a genetic algorithm at 4 × 48 = 192 oracle calls,
driving them directly through `propose` / `observe` so what you see is the sampler and not the
harness.

Two failure modes, kept separate on purpose:

* **Mode collapse**, on a landscape with feasibility switched off. The finding is sharper than
  the usual telling: a hill climber collapses *structurally* (diversity pinned at ~2.0 from
  round zero) while a GA's diversity actually *rises* here, because there is no fitness signal
  to select on. Collapse is proportional to selection pressure — which means the batch narrows
  exactly when the campaign starts working.
* **Infeasibility**, on the suite's `feasibility` geometry. Feasible fractions of 0.02–0.4,
  and the observation that a `MutationEnvironment` handed a transition matrix makes **no
  difference** to these three, because a genetic algorithm does not take actions — it copies
  and mutates arrays. The mask only binds a sampler that walks the graph.

Then rejection sampling, and what it costs: feasibility bought at the price of near-total
collapse (2–3 distinct designs in a 48-design plate).

## 03 — A GFlowNet from scratch on the mutation lattice

`notebooks/03-gflownet-from-scratch.py` · about a minute · CPU torch, no network

The construction graph, then a trained policy, then the one check that matters.

* **The lattice.** Why "each position may be mutated at most once" is what makes the graph
  acyclic, and what follows from it: a variant with `k` mutations is reached by `k!`
  trajectories, and its backward policy is exactly `1/k` with no model and no learning —
  verified against `backward_mask` in the notebook.
* **Masking.** 7 legal first actions out of 33 under the feasibility constraint, against 25
  without it.
* **Trajectory balance**, trained for 400 steps — and an explicit note that this costs 25,600
  oracle calls, which is why a campaign never trains this way.
* **The distributional check.** The reachable set is 18 designs, so `p*(x) ∝ R(x)^β` can be
  computed exactly. The trained policy lands at L1 = 0.061 against a sampling-noise floor of
  0.047.

!!! tip "The trap this notebook walks into deliberately"
    `enumerate_terminal_states` returns the **Hamming ball**, which is not the reachable set: a
    design can be feasible and still unreachable if every ordering of its mutations passes
    through an infeasible intermediate. Measured against the Hamming ball the same policy
    scores L1 = 0.570 — nearly tenfold worse, and it would read as badly broken. Any exact
    distributional claim has to say which support it normalised over.

## 04 — A design–build–test–learn campaign

`notebooks/04-campaign.py` · a couple of minutes · CPU torch, no network

Two arms — a GFlowNet and a proxy-optimising genetic algorithm — through the same `Campaign`
loop at the same budget, then the per-round artifacts read back off disk.

The ledger shows three things:

* The GA's surrogate–oracle correlation is `nan` in every round. Not a missing number: with a
  feasible fraction of 0.08–0.25 there were too few finite measurements to correlate anything
  against. **Infeasibility does not only waste wells, it breaks the learning loop that
  justifies the campaign.**
* The GFlowNet's feasible fraction is 1.000 throughout, definitionally.
* Both arms spend exactly 72 oracle calls and 768 proposals. What differs is the *proxy* call
  count — and at these settings the genetic algorithm spends more of them, because
  `ProxyOptimising` runs a full inner search against the model each round.

Then `rounds.csv` and the per-round batch files, which carry the design, the surrogate's
prediction and the measurement in the same row — because the two disagreeing is the most
useful signal a round produces.

---

## Reading them honestly

Every notebook is sized down for runtime. Notebook 2 runs at half a real campaign's budget and
notebook 4 at under a fifth of one, on one seed. They demonstrate mechanisms; they do not
establish results. The [benchmark suite](benchmark.md) is what establishes results, and
[what this does not show](limitations.md) is what bounds them.

The notebooks say so themselves, in the places it matters.
