# %% [markdown]
# # 1 — Fitness landscapes, epistasis, and what "feasible" means
#
# A **fitness landscape** is the only thing in this library a wet lab would
# recognise as ground truth: give it sequences, it gives you numbers. Everything
# else — environments, samplers, campaigns — exists to decide *which* sequences to
# hand it, because handing it a sequence is what costs money.
#
# This notebook builds the two landscapes that ship with EvoGFN and shows the
# three properties that make them worth benchmarking on:
#
# 1. **Epistasis** — why the fitness of a variant is not the sum of its mutations.
# 2. **Feasibility** — why most strings are not sequences you could ever build.
# 3. **Enumerability** — why a landscape you can write down completely lets you
#    check whether a sampler is *right*, not merely whether it scored well.
#
# ## Runtime and network
#
# Everything up to the GB1 section is closed-form and runs in seconds on CPU with
# no downloads. **The GB1 section needs network access** the first time it runs (a
# ~3 MB download, cached afterwards). It is wrapped so that a machine without
# network prints a message and carries on rather than failing halfway down.
#
# There are no plots in any of these notebooks. That is deliberate: they are
# executed in CI as plain Python scripts, and a chart nobody looks at is a
# dependency nobody needs.

# %%
from __future__ import annotations

import numpy as np

from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape

RNG = np.random.default_rng(0)

# %% [markdown]
# ## An Ehrlich landscape
#
# Ehrlich functions (Stanton et al., ICML 2024 workshop) are procedurally
# generated closed-form landscapes. Their value here is not realism — it is that
# **the optimum is known and guaranteed reachable**, so regret is exact rather
# than "distance to the best thing we happened to find".
#
# The parameters below are deliberately tiny so the whole landscape can be
# printed. The benchmark suite runs `L = 32` (diagnostics), `L = 64` (mid-size)
# and `L = 256` (Stanton et al.'s own base configuration).

# %%
landscape = EhrlichLandscape(
    sequence_length=16,
    vocab_size=6,
    n_motifs=2,
    motif_length=4,
    max_spacing=2,
    transition_density=0.4,
    seed=0,
)

print(f"length          {landscape.sequence_length}")
print(f"alphabet        {''.join(landscape.alphabet.symbols)} ({landscape.alphabet.size} tokens)")
print(f"search space    {landscape.search_space_size:,} sequences")
print(f"motifs          c={landscape.n_motifs}, k={landscape.motif_length}")
print(f"quantisation    q={landscape.quantization}")
print(f"optimum         {float(landscape.optimum[0])}")

# %% [markdown]
# The optimum is `1.0` **by construction**, not by search. The constructor plants
# a feasible sequence, carves the motifs out of it, then verifies that the planted
# sequence really scores 1.0 — because a silent off-by-one in the motif
# construction would invalidate every regret number the landscape ever reports.

# %%
best = landscape.optimal_sequence
print(f"optimal sequence  {landscape.alphabet.decode(best)}")
print(f"scores            {float(landscape.evaluate(best[None, :])[0, 0])}")

for index, (motif, spacing) in enumerate(zip(landscape.motifs, landscape.spacings, strict=True)):
    tokens = "".join(landscape.alphabet.symbols[t] for t in motif)
    print(f"motif {index}: {tokens} at offsets {spacing.tolist()} from wherever it is placed")

# %% [markdown]
# Note that a motif is *spaced* and *placeable*: `CFCB` at offsets `[0, 2, 4, 6]`
# means those four tokens must appear at that relative spacing, anywhere in the
# sequence. There is no fixed site to mutate.

# %% [markdown]
# ## Epistasis: the score is a product, not a sum
#
# Each motif contributes a number in `[0, 1]` — how well its best placement
# matches, floored to one of `q` levels. The `c` motif scores are then
# **multiplied**. Re-implementing that formula is four lines, and worth doing once
# so the rest of the notebook is not taking the library's word for it.


# %%
def motif_satisfaction(sequences: np.ndarray, index: int) -> np.ndarray:
    """`h_q(x, m, s)` from Stanton et al., for one motif, over a batch."""
    motif = landscape.motifs[index]
    spacing = landscape.spacings[index]
    placements = np.arange(landscape.sequence_length - int(spacing[-1]))
    positions = placements[:, None] + spacing[None, :]
    matches = (sequences[:, positions] == motif[None, None, :]).sum(axis=2)
    step = landscape.motif_length // landscape.quantization
    return (matches.max(axis=1) // step) / landscape.quantization


def feasible_single_mutants(sequence: np.ndarray) -> np.ndarray:
    """Every one-substitution neighbour that is still constructible."""
    candidates = []
    for position in range(landscape.sequence_length):
        for token in range(landscape.alphabet.size):
            if token == sequence[position]:
                continue
            variant = sequence.copy()
            variant[position] = token
            candidates.append(variant)
    batch = np.stack(candidates)
    return batch[landscape.is_feasible(batch)]


neighbours = feasible_single_mutants(best)
s0 = motif_satisfaction(neighbours, 0)
s1 = motif_satisfaction(neighbours, 1)
actual = landscape.evaluate(neighbours)[:, 0]

print(
    f"{neighbours.shape[0]} of {landscape.sequence_length * (landscape.alphabet.size - 1)} "
    f"single mutants are feasible"
)
print(f"f(x) == s0 * s1 for all of them: {np.allclose(s0 * s1, actual)}")
print()
print(f"{'s0':>6}{'s1':>6}{'s0*s1':>8}{'f(x)':>8}{'count':>8}")
rows, counts = np.unique(np.stack([s0, s1], axis=1), axis=0, return_counts=True)
for (a, b), count in zip(rows, counts, strict=True):
    print(f"{a:>6.2f}{b:>6.2f}{a * b:>8.2f}{a * b:>8.2f}{count:>8}")

# %% [markdown]
# Fitness is `s0 x s1`. That product is the entire source of the epistasis, and it
# has a consequence a linear model of fitness cannot represent: **a zero in any
# factor zeroes the whole score, however perfect the others are.** A method that
# estimates per-position effects from single mutants and adds them up has no way
# to see that, and both the baselines and the surrogate in this library are
# exposed to it.
#
# `q` decides how visible the structure is. `q = k` (used above) pays out on every
# additional matched token, so there is a gradient to climb. `q = 1` pays nothing
# until a motif is matched **in full**. Same landscape, same seed, one parameter:

# %%
for quantization in (landscape.motif_length, 1):
    variant_landscape = EhrlichLandscape(
        sequence_length=16,
        vocab_size=6,
        n_motifs=2,
        motif_length=4,
        quantization=quantization,
        max_spacing=2,
        transition_density=0.4,
        seed=0,
    )
    peak = variant_landscape.optimal_sequence
    scan = []
    for position in range(variant_landscape.sequence_length):
        for token in range(variant_landscape.alphabet.size):
            if token != peak[position]:
                mutant = peak.copy()
                mutant[position] = token
                scan.append(mutant)
    batch = np.stack(scan)
    batch = batch[variant_landscape.is_feasible(batch)]
    values = variant_landscape.evaluate(batch)[:, 0]
    levels, counts = np.unique(values, return_counts=True)
    spread = "  ".join(f"{level:.2f}x{count}" for level, count in zip(levels, counts, strict=True))
    print(f"q={quantization}: single-mutant scores  {spread}")

# %% [markdown]
# At `q = 1` the landscape is a cliff: you are at 1.0 or you are at 0.0, and no
# single measurement tells you which direction to move. That is the regime real
# multi-residue binding motifs live in, and it is why "the method scored 0.0" is
# an ordinary outcome here rather than a bug.

# %% [markdown]
# ## Feasibility: most strings are not sequences
#
# The second half of an Ehrlich function is a constraint with nothing to do with
# fitness. A transition matrix `A` marks certain **adjacent token pairs as
# forbidden**; a sequence using any of them is outside the feasible set and scores
# `-inf` rather than a low number.
#
# The mechanism is a first-order Markov process, and it is a **synthetic proxy**.
# Real constructibility limits — codon usage, synthesis failure, expression,
# aggregation — are not first-order Markov in the residue sequence. What Ehrlich
# reproduces is the *shape*: a feasible set that shrinks geometrically with
# length. Nothing in this repository tests any feasibility claim against a real
# constructibility rule, and that limit is load-bearing for how far the results
# generalise.

# %%
transitions = landscape.transition_matrix
allowed = transitions > 0
print(f"transition matrix   {transitions.shape}")
print(f"permitted pairs     {allowed.sum()} of {allowed.size} ({allowed.mean():.1%})")

# %% [markdown]
# Now the part that decides a budget. Draw sequences uniformly at random and ask
# what fraction are constructible at all. Each of the `L - 1` adjacencies has to
# be permitted independently, so the fraction decays geometrically.

# %%
DRAWS = 20_000
print(f"transition_density=0.15, uniform random strings, {DRAWS:,} draws each")
print(f"{'L':>5}  {'closed form':>14}  {'measured':>10}")
for length in (4, 8, 16, 32, 64):
    scratch = EhrlichLandscape(
        sequence_length=length,
        vocab_size=6,
        n_motifs=1,
        motif_length=3,
        max_spacing=1,
        transition_density=0.15,
        seed=0,
    )
    # Uniform tokens make the adjacencies independent, so the feasible fraction
    # is exactly (share of permitted pairs) ** (L - 1).
    share = float((scratch.transition_matrix > 0).mean())
    draws = RNG.integers(0, scratch.alphabet.size, size=(DRAWS, length), dtype=np.int32)
    print(
        f"{length:>5}  {share ** (length - 1):>14.2e}  {int(scratch.is_feasible(draws).sum()):>10,}"
    )

# %% [markdown]
# By `L = 16` you can no longer find a buildable string by drawing 20,000 of them,
# and the closed form says why. A method that proposes designs and filters
# afterwards is, at these lengths, proposing nothing at all.
#
# EvoGFN's answer is to fold the constraint into the *construction graph*, so
# infeasible designs are never generated — notebook 3. Notebook 2 measures what
# happens when you do not.

# %% [markdown]
# ## Enumerability: why it matters more than it sounds
#
# `6^16` is 2.8 x 10^12 sequences, so this landscape cannot be enumerated. But a
# directed-evolution round does not search the whole space — it searches a
# **mutational neighbourhood** of a parent. Under a budget of `m` mutations the
# reachable set is the Hamming ball of radius `m`, and that is often small enough
# to write down completely.

# %%
parent = landscape.feasible_sequence(seed=0)
env = MutationEnvironment(
    parent,
    landscape.alphabet,
    max_mutations=2,
    transitions=landscape.transition_matrix,
)
ball = env.enumerate_terminal_states()
reachable = env.reachable_terminal_states()

print(f"whole space            {landscape.search_space_size:,}")
print(f"within 2 mutations     {ball.shape[0]:,}")
print(f"...of which feasible   {int(landscape.is_feasible(ball).sum()):,}")
print(f"...actually reachable  {reachable.shape[0]:,}")

# %% [markdown]
# Being able to write that set down enables one specific check, and it is the one
# that distinguishes a GFlowNet from an optimiser: comparing the sampler's
# **empirical distribution** against the exact target `p*(x) ∝ R(x)^β`. Every
# other metric here — best-found, top-K, diversity — can be satisfied by a good
# hill climber that never samples anything. This one cannot. Notebook 3 runs it.
#
# The three counts differ, and the second gap is the one that catches people out.
# `enumerate_terminal_states` returns the Hamming ball. Filtering it by
# feasibility still overcounts, because a *feasible destination* can be
# unreachable: if every ordering of its mutations passes through an infeasible
# intermediate, masking blocks the construction and the policy can never emit it.
# `reachable_terminal_states` walks the environment's own masks and returns what
# can actually be built.
#
# This matters beyond bookkeeping. Notebook 3 measures L1 distance to the target
# distribution, and that measurement needs the right support: the same trained
# policy scores 0.061 against the reachable set and 0.570 against the ball --
# tenfold, and it reads as a broken policy rather than a mis-specified support.

# %% [markdown]
# ## GB1: real measurements, and the easiest geometry in the suite
#
# GB1 (Wu et al., *eLife* 2016) measured 149,361 of the 160,000 variants at four
# epistatically coupled positions of protein G. It is here because it is real, and
# because near-completeness makes regret exact against a *measured* best.
#
# **This section downloads ~3 MB.** With no network it says so and skips.

# %%
try:
    from evogfn.landscapes.gb1 import GB1Landscape

    gb1 = GB1Landscape()
except Exception as error:
    gb1 = None
    print(f"GB1 unavailable ({type(error).__name__}); skipping this section.")
    print("Everything above ran without network and is unaffected.")

# %%
if gb1 is not None:
    wild = gb1.wild_type
    print(
        f"wild type       {gb1.alphabet.decode(wild)}  fitness {gb1.evaluate(wild[None, :])[0, 0]}"
    )
    print(f"best measured   {gb1.optimal_variant}  fitness {float(gb1.optimum[0]):.2f}")
    print(f"measured        {gb1.n_measured:,} of {gb1.search_space_size:,}")

    space = gb1.enumerate()
    values = gb1.evaluate(space)[:, 0]
    measured = gb1.is_measured(space)
    assayed = values[measured]
    print()
    print("distribution of measured fitness (wild type = 1.0):")
    for threshold in (0.0, 0.01, 0.1, 1.0):
        print(f"  <= {threshold:<5}  {float((assayed <= threshold).mean()):>6.1%}")
    print(f"  median   {float(np.median(assayed)):.4f}")

    optimum_tokens = gb1.alphabet.encode(gb1.optimal_variant)
    print()
    print(
        f"mutations from wild type to the best variant: {int((optimum_tokens != wild).sum())} of 4"
    )

# %% [markdown]
# The last line is the epistasis GB1 is known for: the best variant differs from
# wild type at **all four** positions, and no single, double or triple mutant
# comes close. A greedy walk from the wild type cannot get there.
#
# ### What GB1 does *not* test
#
# This has to be said plainly, because GB1's realism is easy to read as
# difficulty:
#
# * GB1 has **no feasibility constraint**. All 160,000 strings are orderable.
# * The suite's mutation budget is 4 and GB1 has 4 positions, so **every sequence
#   is reachable** and the mutation budget constrains nothing.
#   `Protocol.constrains_search` returns `False` here — that is the honest label,
#   not a bug.
#
# GB1 is therefore the **easiest geometry in the suite**: an empirical anchor
# saying the numbers are not an artefact of synthetic landscapes, and nothing
# more. Every claim about constrained or feasibility-limited search in this
# project rests on Ehrlich alone.

# %% [markdown]
# ## What to take away
#
# * Fitness is a product over motifs, so effects do not add and single-mutant
#   scans mislead. At `q = 1` there is no gradient at all.
# * Feasibility is a separate, geometrically decaying constraint — and the one
#   place where this benchmark's realism is a proxy rather than a measurement.
# * The reachable set under a mutation budget is small enough to enumerate, which
#   is the only way to check a sampler's *distribution* rather than its best hit.
#
# Next: `02-why-baselines-collapse.py`.
