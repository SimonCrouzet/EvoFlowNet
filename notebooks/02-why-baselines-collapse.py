# %% [markdown]
# # 2 — Why the baselines collapse
#
# Directed evolution *is* a genetic algorithm. So the baselines here are not
# strawmen to be cleared: a genetic algorithm and a hill climber are what protein
# engineers actually run, and any method proposing to replace them has to beat
# them on their own budget.
#
# This notebook runs them at a realistic budget and shows the two ways they fail.
# They are different failures, and conflating them is how a benchmark ends up
# measuring the wrong thing:
#
# 1. **Mode collapse.** The batch converges onto near-copies of one design. The
#    budget was spent on a single bet.
# 2. **Infeasibility.** On a landscape with a constructibility constraint, almost
#    every proposal is a design nobody could build. The budget was spent on
#    nothing at all.
#
# ## Runtime and honesty
#
# CPU, numpy only, well under a minute. No network, no torch, no surrogate — this
# notebook drives the samplers directly through `propose` / `observe` so that what
# you see is the sampler and not the harness. Notebook 4 adds the full campaign.
#
# The budget here is **4 rounds x 48 = 192 oracle calls**, half a real campaign,
# chosen so this runs fast. The suite's protocols are 384-480 calls, taken from
# ALDE (396), LaMBO-2 (374) and MLDE (480).
#
# **No method in this notebook solves anything.** Best-found stays far from the
# optimum of 1.0 for every method including the good ones. What is being compared
# is *how* each one fails, not which one wins.

# %%
from __future__ import annotations

import numpy as np

from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
from evoflownet.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.metrics.diversity import diversity

ROUNDS = 4
BATCH = 48
MAX_MUTATIONS = 4

# %% [markdown]
# ## A round loop, in eleven lines
#
# This is what a design-build-test-learn round is, stripped of everything
# optional. The sampler proposes, the oracle scores exactly the batch, the sampler
# is told what happened. `Campaign` in `evoflownet.loop` adds a surrogate, an
# acquisition rule and deduplication on top of this; nothing else changes.


# %%
def run_rounds(sampler, landscape, *, rounds=ROUNDS, batch=BATCH):
    """Drive a sampler for a few rounds and record what each batch looked like."""
    history = []
    best_so_far = -np.inf
    for index in range(rounds):
        proposals = sampler.propose(batch)
        values = landscape.evaluate(proposals)
        sampler.observe(proposals, values)

        flat = values[:, 0]
        finite = flat[np.isfinite(flat)]
        best_so_far = max(best_so_far, float(finite.max()) if finite.size else -np.inf)
        history.append(
            {
                "round": index,
                "feasible": float(np.isfinite(flat).mean()),
                "distinct": len(np.unique(proposals, axis=0)),
                "diversity": diversity(proposals),
                "best_so_far": best_so_far,
            }
        )
    return history


def report(name, history):
    """Print one sampler's rounds."""
    print(f"\n{name}")
    print(f"  {'round':>5} {'feasible':>9} {'distinct':>9} {'diversity':>10} {'best':>8}")
    for row in history:
        best = "-inf" if not np.isfinite(row["best_so_far"]) else f"{row['best_so_far']:.3f}"
        print(
            f"  {row['round']:>5} {row['feasible']:>9.3f} {row['distinct']:>9} "
            f"{row['diversity']:>10.2f} {best:>8}"
        )


# %% [markdown]
# ## Failure 1: mode collapse
#
# First a landscape with **no** feasibility constraint (`transition_density=1.0`
# permits every adjacency), so feasibility cannot confound what we are looking at.
# `distinct` counts unique designs in a 48-design batch; `diversity` is mean
# pairwise Hamming distance, which at a 4-mutation budget can be at most 8.

# %%
open_landscape = EhrlichLandscape(
    sequence_length=32,
    vocab_size=20,
    n_motifs=2,
    motif_length=4,
    max_spacing=2,
    transition_density=1.0,
    seed=3,
)
open_env = MutationEnvironment(
    open_landscape.feasible_sequence(seed=0),
    open_landscape.alphabet,
    max_mutations=MAX_MUTATIONS,
    transitions=open_landscape.transition_matrix,
)

for name, sampler in [
    ("random mutagenesis", RandomMutagenesis(open_env, seed=0)),
    ("hill climbing", HillClimbing(open_env, seed=0)),
    ("genetic algorithm", GeneticAlgorithm(open_env, seed=0)),
]:
    report(name, run_rounds(sampler, open_landscape))

# %% [markdown]
# `feasible` is 1.000 throughout, as intended — nothing here is about feasibility.
# Read the `diversity` column instead, and read it honestly, because it does not
# say quite what the headline version of this story says.
#
# * **Hill climbing** sits at ~2.0 from round zero and never leaves. It proposes
#   neighbours of a single incumbent, so its batch is a cloud around one design
#   *by construction*, before any selection pressure exists. This is mode collapse
#   in its purest form, and it is also the method that gets furthest here.
# * **Random mutagenesis** holds ~3.7. It is not learning, so it has nothing to
#   collapse onto — and it does not improve either.
# * **The genetic algorithm** starts collapsed (its population is seeded from
#   copies of the parent) and then *spreads out*, 1.93 → 4.08. It is not
#   concentrating because there is nothing to concentrate on: its best stays at
#   0.250 for all four rounds, so selection has almost no signal to act on.
#
# The lesson is sharper than "optimisers collapse": **collapse is proportional to
# selection pressure.** A hill climber collapses structurally. A GA collapses only
# once it is finding something — which means the batch narrows exactly when the
# campaign starts working, which is exactly when you least want it to. A plate of
# near-copies of one design is *one experiment*: if anything kills that design —
# expression, solubility, an off-target — the whole round is gone.
#
# The GFlowNet argument is precisely here: sample *proportionally to* reward
# rather than climbing toward its maximum, so the batch spreads across the
# high-fitness regions instead of piling onto one. Whether the diversity it buys
# is *useful* is a separate question, and one this project has **not** answered —
# no diversity-aware-selection ablation was run. On GB1, random mutagenesis is the
# most diverse method and the worst performing.

# %% [markdown]
# ## Failure 2: the budget spent on unbuildable designs
#
# Now the same samplers on the suite's `feasibility` geometry: a sparse transition
# matrix (`transition_density=0.15`), where most strings are not constructible.
# Everything else is identical.
#
# The `feasible` column is the fraction of the measured batch that could actually
# be built. Everything else was an ordered well that returns nothing.

# %%
hard_landscape = EhrlichLandscape(
    sequence_length=64,
    vocab_size=20,
    n_motifs=2,
    motif_length=4,
    max_spacing=2,
    transition_density=0.15,
    seed=1,
)
hard_env = MutationEnvironment(
    hard_landscape.feasible_sequence(seed=0),
    hard_landscape.alphabet,
    max_mutations=MAX_MUTATIONS,
    transitions=hard_landscape.transition_matrix,
)

for name, sampler in [
    ("random mutagenesis", RandomMutagenesis(hard_env, seed=0)),
    ("hill climbing", HillClimbing(hard_env, seed=0)),
    ("genetic algorithm", GeneticAlgorithm(hard_env, seed=0)),
]:
    report(name, run_rounds(sampler, hard_landscape))

# %% [markdown]
# Note that the environment *was* given the transition matrix. It makes no
# difference to these three, and that is the point worth being precise about: a
# `MutationEnvironment` enforces feasibility through **action masks**, and a
# genetic algorithm does not take actions. It copies and mutates arrays. The mask
# only binds a sampler that generates by walking the graph — which is what
# notebook 3 builds, and is the entire mechanism behind the claim.

# %% [markdown]
# The feasible fraction is a rounding error. In the benchmark suite, on this
# landscape at 384 calls over 100 seeds, the buildable wells were **3.0 of 384**
# for random mutagenesis, **2.8** for a blind genetic algorithm, **1.9** for one
# with surrogate access, and **8.0** for hill climbing. A 96-well plate returns
# roughly one usable measurement.
#
# Two consequences, and the second is the one people miss:
#
# * Nearly the whole budget bought nothing.
# * The **surrogate has nothing to learn from.** Infeasible designs score `-inf`,
#   so there is no finite signal to fit, and the model that was supposed to make
#   later rounds smarter never gets built. In the suite, the number of rounds with
#   a defined surrogate-oracle correlation on this task was 0 of 400 for the
#   proxy-optimising GA.
#
# ## Two ways to fix it, and they are not equivalent
#
# **Rejection sampling**: keep resampling offspring until they are legal. Correct,
# and available here as `feasible_only=True`. Watch what it costs.

# %%
rejection = GeneticAlgorithm(hard_env, seed=0, feasible_only=True, max_attempts=200)
try:
    report("genetic algorithm (rejection)", run_rounds(rejection, hard_landscape))
    filled = ROUNDS * BATCH
except RuntimeError as error:
    filled = 0
    print(f"\ngenetic algorithm (rejection): gave up\n  {error}")
print(f"\nproposals generated: {rejection.proposals_made:,} for {filled} wells filled")

# %% [markdown]
# Rejection sampling works: the feasible fraction is 1.000. It pays in proposals
# rather than oracle calls, and proposals are free, so this sounds like a solved
# problem. Two things say otherwise, and only one of them shows up at this scale.
#
# **What you can see above:** the `diversity` and `distinct` columns fall through
# the floor — a handful of unique designs in a 48-design plate. Rejection can only
# find feasible offspring in a very small neighbourhood, so it buys feasibility
# with the collapse from the first half of this notebook. In the benchmark suite
# the rejection GA's batch diversity is 3.65 against a masked GFlowNet's 5.38 on
# this task, and the same direction holds on every task in the suite.
#
# **What you cannot see at this scale:** it can *fail to fill the plate*. After
# `max_attempts` resampling rounds it raises rather than return infeasible
# designs. At the 4 x 48 used here it kept up (see the proposal count above); in
# the suite at 384 calls it halted at **81.9 ± 15.9 of 384 wells** (minimum 50,
# maximum 119).
#
# That stall is the actual result, and it needs stating precisely, because the
# tempting version of it is false. Rejection did **not** lose on quality: at
# matched budget its regret is statistically indistinguishable from a masked
# GFlowNet's — an exact tie on 100 of 100 seeds. It lost by *not spending the
# budget*. "Masking searches better" would be an overclaim; "masking converts more
# of the budget into measurements" is what the data support.
#
# **Masking**: make the constraint part of the construction graph, so an
# infeasible design is never generated in the first place. The environment already
# knows how — `is_reachable` says what its graph contains, and `forward_mask` is
# what stops a graph-walking sampler leaving it.

# %%
blind = GeneticAlgorithm(hard_env, seed=1)
proposals = blind.propose(500)
inside = int(hard_env.is_reachable(proposals).sum())
print(f"blind GA proposals inside the environment's graph: {inside} of 500")

state = hard_env.initial(1)
mask = hard_env.forward_mask(state)
print(f"legal first actions at the parent: {int(mask.sum())} of {hard_env.n_actions}")
print(f"...unconstrained this would be:    {hard_env.n_mutation_actions} substitutions + stop")

# %% [markdown]
# Twenty-seven legal actions out of 1,281. The mask is not a filter applied to a
# generated design; it is a statement about which edges of the construction graph
# exist at all. Notebook 3 builds a sampler that generates by walking those edges,
# so its feasible fraction is 1.000 by construction rather than by luck.
#
# One caveat, stated because it is easy to over-read: **"feasible by construction"
# is a property of the environment, not an achievement of the GFlowNet.** Any
# sampler that generates through a masked graph gets it. What is measured, and
# what is new, is the comparison against rejection sampling at a matched budget.

# %% [markdown]
# ## What to take away
#
# * Batch collapse tracks selection pressure. A hill climber collapses
#   structurally; a genetic algorithm collapses once it starts succeeding, which
#   is the worst possible timing for a design round.
# * Under a constructibility constraint, unmasked search does not merely perform
#   worse — it produces no usable measurements, and therefore no usable model.
# * Rejection sampling fixes correctness and not economics: it buys feasibility
#   with diversity, and at realistic budgets it stalls before spending them.
# * None of these methods, and none in the rest of this repository, solves an
#   Ehrlich instance. Best-found stays far below 1.0 everywhere. The comparison is
#   between failure modes.
#
# Next: `03-gflownet-from-scratch.py`.
