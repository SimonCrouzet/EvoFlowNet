# %% [markdown]
# # 4 — A design-build-test-learn campaign
#
# Notebooks 2 and 3 drove samplers directly. A real campaign has three more parts
# and a hard constraint:
#
# * a **surrogate** fitted to everything measured so far,
# * an **acquisition rule** turning prediction and uncertainty into one number,
# * a **batch selector** choosing which candidates actually go to the assay,
# * and a **budget** that only the assay is charged against.
#
# `evoflownet.loop.Campaign` runs that loop identically for every sampler, so a
# difference between runs is a difference between methods rather than between
# harnesses. This notebook runs two campaigns at the same budget, then reads the
# per-round artifacts back off disk.
#
# ## The one rule that decides whether the benchmark is meaningful
#
# Training a GFlowNet takes tens of thousands of reward evaluations — notebook 3
# spent 25,600 on a toy. Charging those against the oracle budget would exhaust a
# 384-call campaign before the first round finished. No published method does it:
# GFN-AL trains against a learned proxy and spends the real budget only on the
# selected batch.
#
# Getting this wrong does not raise an error. It produces a benchmark in which the
# GFlowNet looks catastrophically sample-inefficient for a reason that has nothing
# to do with GFlowNets. The seam is structural rather than remembered: the sampler
# is handed a `ProxyLandscape` over *the same surrogate instance* the campaign
# refits, and the proxy holds no oracle.
#
# ## Runtime
#
# CPU, a couple of minutes, no network. The settings are shrunk for speed — 3
# rounds of 24 (72 calls) against the suite's 4 x 96 = 384. Treat the numbers as
# illustrative of the mechanism, not as results.

# %%
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np

from evoflownet.acquisition.rules import TopK, UpperConfidenceBound
from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
from evoflownet.algorithms.gflownet.sampler import GFlowNetSampler
from evoflownet.algorithms.gflownet.training import TrainingConfig
from evoflownet.algorithms.inner_loop import ProxyOptimising
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.loop.campaign import Campaign
from evoflownet.models.policy import SequencePolicy
from evoflownet.rewards.base import TemperedReward
from evoflownet.surrogate.ensemble import DeepEnsemble
from evoflownet.surrogate.proxy import ProxyLandscape

ROUNDS = 3
BATCH = 24
POOL = 256
SEED = 0

# %% [markdown]
# ## The task
#
# A mid-size Ehrlich instance with a real feasibility constraint, and an
# environment that knows about it. Handing the transition matrix to the
# environment is not optional decoration: omit it and nothing raises, every
# proposal scores `-inf`, and the surrogate has no finite value to fit. That is
# how this was wrong in its first version.

# %%
landscape = EhrlichLandscape(
    sequence_length=32,
    vocab_size=20,
    n_motifs=2,
    motif_length=4,
    max_spacing=2,
    transition_density=0.5,
    seed=7,
)
parent = landscape.feasible_sequence(seed=0)
env = MutationEnvironment(
    parent,
    landscape.alphabet,
    max_mutations=4,
    transitions=landscape.transition_matrix,
)
print(f"landscape  L={landscape.sequence_length}, v={landscape.alphabet.size}, optimum 1.0")
print(f"budget     {ROUNDS} rounds x {BATCH} = {ROUNDS * BATCH} oracle calls")


# %% [markdown]
# ## Two arms, one loop
#
# Both arms get the same surrogate class, the same acquisition rule, the same
# selector and the same budget. The GFlowNet trains against a proxy over its
# surrogate; the genetic algorithm is wrapped in `ProxyOptimising` so it gets the
# *same* model access. Comparing a method that optimises the model against one
# that only meets it as a filter is not a comparison of methods.


# %%
def build_surrogate() -> DeepEnsemble:
    """A fresh ensemble, sized down for notebook runtime."""
    return DeepEnsemble(
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        n_members=3,
        epochs=60,
        seed=SEED,
    )


def build_campaign(sampler, surrogate, artifact_dir: Path) -> Campaign:
    """The loop, identical for every arm."""
    return Campaign(
        landscape=landscape,
        sampler=sampler,
        surrogate=surrogate,
        acquisition=UpperConfidenceBound(kappa=1.0),
        selector=TopK(),
        rounds=ROUNDS,
        batch_size=BATCH,
        pool_size=POOL,
        artifact_dir=artifact_dir,
    )


def gflownet_arm(artifact_dir: Path) -> Campaign:
    """A GFlowNet trained each round against the surrogate proxy."""
    surrogate = build_surrogate()
    policy = SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        hidden_dim=64,
        seed=SEED,
    )
    sampler = GFlowNetSampler(
        env,
        policy,
        proxy=ProxyLandscape(surrogate, alphabet=env.alphabet, sequence_length=env.sequence_length),
        reward=TemperedReward(beta=3.0),
        config=TrainingConfig(steps=80, batch_size=32, seed=SEED),
        seed=SEED,
    )
    return build_campaign(sampler, surrogate, artifact_dir)


def genetic_arm(artifact_dir: Path) -> Campaign:
    """A genetic algorithm given the same model access."""
    surrogate = build_surrogate()
    sampler = ProxyOptimising(
        GeneticAlgorithm(env, seed=SEED),
        proxy=ProxyLandscape(surrogate, alphabet=env.alphabet, sequence_length=env.sequence_length),
    )
    return build_campaign(sampler, surrogate, artifact_dir)


# %%
workspace = Path(tempfile.mkdtemp(prefix="evoflownet-notebook-"))
campaigns = {}
results = {}
for name, build in (("gflownet", gflownet_arm), ("genetic+proxy", genetic_arm)):
    campaigns[name] = build(workspace / name)
    results[name] = campaigns[name].run()
    print(f"{name} done")

# %% [markdown]
# ## The ledger
#
# `feasible` is the fraction of the measured batch that was constructible;
# `corr` is the Pearson correlation between what the surrogate predicted and what
# the oracle returned, which is `nan` in round 0 because there was no model yet.

# %%
for name, result in results.items():
    print(
        f"\n{name}  ({result.oracle_calls} oracle calls, {result.proposals:,} proposals generated)"
    )
    print(f"  {'round':>5}{'best':>8}{'mean':>8}{'feasible':>10}{'diversity':>11}{'corr':>8}")
    for record in result.rounds:
        print(
            f"  {record.index:>5}{record.best_so_far:>8.3f}{record.mean_in_round:>8.3f}"
            f"{record.feasible_fraction:>10.3f}{record.batch_diversity:>11.2f}"
            f"{record.surrogate_correlation:>8.2f}"
        )
    if (regret := result.simple_regret) is not None:
        print(f"  simple regret {regret:.4f}  (0.0 would be solved)")

# %% [markdown]
# Three things in that table, in order of how much they matter.
#
# **The `corr` column for `genetic+proxy` is `nan` in every round.** That is not a
# missing number, it is the finding: with a feasible fraction of 0.08-0.25, there
# were too few finite measurements in a round to correlate anything against. The
# surrogate that was supposed to make round two smarter than round one never got
# a signal to learn from. In the suite, on the `feasibility` task, the number of
# rounds with a defined surrogate-oracle correlation was **0 of 400** for the
# proxy-optimising GA against 189 of 400 for the GFlowNet. Infeasibility does not
# just waste the wells; it breaks the learning loop that justifies the campaign.
#
# **The `feasible` column is 1.000 for the GFlowNet in every round.** Definitional,
# as notebook 3 said: it generates through the masked graph, so it cannot do
# otherwise.
#
# **Diversity is 6.9 against 2.0-3.2.** Same direction as every task in the suite.
# Whether that diversity is *useful* is not established here — see the limitations
# page; on GB1 the most diverse method is also the worst.
#
# ### Read the regret the right way
#
# Simple regret is a long way from zero for both arms, and it stays that way at
# the suite's full 384-call budget with 100 seeds. **No method in this repository
# solves an Ehrlich instance.** On the large-space task every method sits between
# 0.974 and 0.992 of a maximum regret of 1.0 — the differences between them are
# real and statistically clean, and they are differences between degrees of stuck.
#
# One run at one seed also decides nothing. The benchmark suite runs 100 seeds and
# reports a paired comparison with a win/tie/loss count beside the mean, because
# on GB1 the GFlowNet's mean advantage of +0.96 comes with a 55/19/26 W/T/L — a
# single laboratory campaign wins about 55% of the time, not always.

# %% [markdown]
# ## Reading the per-round artifacts
#
# A benchmark wants aggregates. A campaign wants the opposite: the actual designs
# that went out in round two, what came back, and what the model believed at the
# time. Those are what someone asks for six months later when a variant turns out
# to matter, and a mean regret has discarded all of it.
#
# `artifact_dir` writes one CSV per round plus a `rounds.csv` manifest.

# %%
run_dir = workspace / "gflownet"
print(sorted(p.name for p in run_dir.iterdir()))

# %%
with (run_dir / "rounds.csv").open() as handle:
    for row in csv.DictReader(handle):
        print({key: row[key] for key in ("index", "proposed", "screened", "evaluated", "feasible")})

# %% [markdown]
# `proposed` is what the sampler generated; `screened` is what survived
# deduplication (a lab does not re-order a variant it has already assayed, and a
# collapsed sampler would otherwise spend its budget re-measuring one design);
# `evaluated` is what was charged to the budget.
#
# The gap between `proposed` and `evaluated` is where a rejection-sampling method
# pays its bill — free in oracle terms, and the reason `proposals_made` is tracked
# separately from oracle calls throughout this library.
#
# Now the batch itself. Each round file carries the design, the surrogate's
# prediction, and the measurement side by side — because the two disagreeing is
# the most useful signal a round produces, and reconstructing it later would mean
# refitting the model as it stood at the time.

# %%
with (run_dir / "round-002.csv").open() as handle:
    rows = list(csv.DictReader(handle))

rows.sort(key=lambda row: -float(row["measured"]))
print(f"{'design':>36}{'predicted':>11}{'measured':>10}")
for row in rows[:6]:
    tokens = np.array([int(t) for t in row["sequence"].split()], dtype=np.int32)
    print(f"{landscape.alphabet.decode(tokens):>36}{row['predicted']:>11}{row['measured']:>10}")

# %% [markdown]
# ## Where the budget did and did not go

# %%
print(f"{'arm':>14}{'oracle':>9}{'proposals':>12}{'proxy calls':>14}")
for name, result in results.items():
    proxy_calls = getattr(campaigns[name].sampler, "proxy_calls", 0)
    print(f"{name:>14}{result.oracle_calls:>9}{result.proposals:>12,}{proxy_calls:>14,}")

# %% [markdown]
# Both arms spent 72 oracle calls and generated 768 proposals — the budget and the
# proposal count are what is held equal. What differs is the proxy column: reward
# evaluations against the surrogate, free in budget terms and emphatically not
# free in wall clock.
#
# Note which way round it comes out. At these settings the *genetic algorithm*
# spends more proxy calls than the GFlowNet, because `ProxyOptimising` runs a
# full inner search loop against the model every round. That wrapper exists
# precisely so the comparison is between search methods rather than between one
# method allowed to optimise the model and one only allowed to be filtered by it.
# At the suite's defaults (300 gradient steps x batch 64 x 4 rounds) the GFlowNet
# is the more expensive of the two, on the order of 77,000 proxy calls.
#
# Reporting only oracle calls would make all of this look free; reporting only
# proposals would make an expensive assay look cheap. All three are tracked so the
# trade is visible rather than implied.

# %% [markdown]
# ## From the command line
#
# The same campaign is a CLI invocation, composed by Hydra:
#
# ```bash
# evoflownet campaign
# evoflownet campaign sampler=genetic acquisition=ucb selector=diverse
# evoflownet campaign campaign.rounds=8 campaign.batch_size=48
# evoflownet campaign --help      # every configurable option
# ```
#
# and the full benchmark suite, which is resumable and writes each campaign as it
# finishes:
#
# ```bash
# uv run python experiments/run_suite.py --tier main
# uv run python experiments/run_suite.py --report   # read, do not run
# ```

# %% [markdown]
# ## What to take away
#
# * A campaign is a budget plus a loop. The budget is spent only on measurement;
#   everything the sampler does to decide what to measure is charged elsewhere,
#   and a benchmark that conflates the two is measuring the wrong thing.
# * Per-round artifacts are the part a campaign actually leaves behind. Prediction
#   and measurement live in the same row on purpose.
# * The numbers here are one seed at a quarter budget. The suite exists because
#   that is not enough to conclude anything, and it reports win/tie/loss counts
#   alongside means because the mean alone was misleading in at least one case.

# %%
# Tidy up the temporary run directory. Point `artifact_dir` somewhere permanent
# to keep it -- that trail is the whole reason the artifacts exist.
shutil.rmtree(workspace)
