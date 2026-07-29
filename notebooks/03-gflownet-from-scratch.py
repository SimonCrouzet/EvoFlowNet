# %% [markdown]
# # 3 — A GFlowNet from scratch on the mutation lattice
#
# A GFlowNet does not optimise. It learns a policy that **samples** terminal
# states in proportion to their reward, `p(x) ∝ R(x)`. On a design round that is
# the difference between 48 bets and one.
#
# To do that it needs the space to be a **directed acyclic graph** whose terminal
# states are the objects you want. This notebook builds that graph for directed
# evolution, shows why it is acyclic, shows how feasibility becomes a property of
# the graph rather than a filter, trains a small policy with trajectory balance,
# and then — because the graph here is small enough to enumerate — checks whether
# the policy actually samples the right distribution.
#
# That last check is the point. Best-found, top-K and diversity are all satisfied
# by a good hill climber that never samples anything. Comparing the empirical
# distribution against the exact target is the one measurement that is not.
#
# ## Runtime
#
# CPU, torch, roughly a minute. No network. Everything is sized so it can be
# printed rather than plotted.

# %%
from __future__ import annotations

import numpy as np
import torch

from evoflownet.algorithms.gflownet.objectives import TrajectoryBalance
from evoflownet.algorithms.gflownet.sampling import sample_trajectories
from evoflownet.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.ehrlich import EhrlichLandscape
from evoflownet.metrics.distribution import (
    empirical_distribution,
    expected_l1_from_sampling_noise,
    l1_distance,
    target_distribution,
)
from evoflownet.models.policy import SequencePolicy
from evoflownet.rewards.base import TemperedReward

# %% [markdown]
# ## The mutation lattice
#
# A trajectory starts at the parent sequence, applies point mutations one at a
# time, and stops. The one rule that makes this work: **each position may be
# mutated at most once.**
#
# Without that rule, mutating a site and reverting it returns to a state already
# visited — a cycle — and the flow equations have no solution. With it, a state is
# exactly *the parent plus a set of applied mutations*, and the graph is the
# **subset lattice** over mutated positions, graded by how many have been applied.
#
# Two consequences follow, and both are used below:
#
# * A variant carrying `k` mutations is reached by exactly `k!` trajectories — one
#   per order the mutations could have been applied in.
# * Its parents in the graph are the `k` states reached by undoing any single one,
#   so the uniform backward policy is exactly `1/k` per parent, with no model and
#   no learning.

# %%
landscape = EhrlichLandscape(
    sequence_length=8,
    vocab_size=4,
    n_motifs=1,
    motif_length=3,
    max_spacing=2,
    transition_density=0.5,
    seed=0,
)
parent = landscape.feasible_sequence(seed=0)
env = MutationEnvironment(
    parent,
    landscape.alphabet,
    max_mutations=2,
    transitions=landscape.transition_matrix,
)

print(f"parent           {landscape.alphabet.decode(parent)}")
print(f"mutation budget  {env.max_mutations}")
print(f"actions          {env.n_mutation_actions} substitutions + 1 stop = {env.n_actions}")

# %% [markdown]
# ### Action encoding
#
# Action `a < length * alphabet_size` sets position `a // alphabet_size` to token
# `a % alphabet_size`; the last index is stop. One integer per action means the
# policy emits a single logit vector per state and the mask applies to it directly
# with no reshaping.

# %%
state = env.initial(1)
mask = env.forward_mask(state)
legal = np.flatnonzero(mask[0])
print(f"legal actions at the parent: {legal.size} of {env.n_actions}")
for action in legal[:6]:
    if action == env.stop_action:
        print("  stop")
    else:
        position, token = divmod(int(action), env.alphabet.size)
        print(f"  set position {position} to {env.alphabet.symbols[token]}")
print("  ...")

# %% [markdown]
# ### Where feasibility lives
#
# Without a transition matrix every substitution at an unmutated position is
# legal: `L x (v - 1)` of them. With one, a substitution is legal only if it also
# keeps every adjacency permitted. The constraint is enforced *on the edges*, so a
# sequence violating it is not reachable — there is no path to it.

# %%
unmasked = MutationEnvironment(parent, landscape.alphabet, max_mutations=2)
print(f"legal first actions, no constraint:  {int(unmasked.forward_mask(state).sum())}")
print(f"legal first actions, with constraint:{int(mask.sum()):>4}")

# %% [markdown]
# ### The backward policy is closed-form
#
# `backward_mask` marks the actions that could have produced a state. For an
# unstopped state with `k` mutations these are exactly the `k` substitutions that
# introduced them, so uniform-over-the-mask *is* the exact backward policy `1/k`.
#
# A stopped state is the case that is easy to get wrong: its only parent is the
# same state unstopped, because the stop action is the sole edge into it.


# %%
def first_legal(mask: np.ndarray) -> np.ndarray:
    """One legal non-stop action per row, or stop if nothing else is available."""
    chosen = []
    for row in mask:
        options = np.flatnonzero(row)
        substitutions = options[options != env.stop_action]
        chosen.append(int(substitutions[0]) if substitutions.size else env.stop_action)
    return np.asarray(chosen)


one = env.step(env.initial(3), np.array([legal[0], legal[1], legal[2]]))
two = env.step(one, first_legal(env.forward_mask(one)))

for label, walked in (("after 1 mutation", one), ("after 2 mutations", two)):
    print(f"{label}")
    print(f"  k                {env.n_mutations(walked)}")
    print(f"  backward parents {env.backward_mask(walked).sum(axis=1)}   (should equal k)")
    print(f"  log k!           {np.round(env.log_n_trajectories(walked), 4)}")

stopped = env.step(two, np.full(3, env.stop_action))
print(f"backward parents once stopped: {env.backward_mask(stopped).sum(axis=1)}   (always 1)")

# %% [markdown]
# It is worth being clear about what this does *not* buy. Malkin et al. show that
# for **any** valid backward policy there is a unique corresponding forward policy
# sampling proportional to reward. `P_B` affects optimisation, not the target.
# Choosing uniform here is a matter of cost and variance, not correctness — and
# MOGFN-AL, which uses this same graph, settles on uniform for the same reason.

# %% [markdown]
# ## Trajectory balance
#
# The training signal. For a complete trajectory `τ: s0 → ... → x`,
#
# ```
# L(τ) = ( log Z + Σ log P_F(s_{t+1} | s_t)  -  log R(x) - Σ log P_B(s_t | s_{t+1}) )²
# ```
#
# `Z` is a single learned scalar that must travel all the way to `log Σ R(x)`, so
# it gets a higher learning rate than the policy — otherwise it is the bottleneck.
# When the loss is zero for every trajectory, the policy samples `p(x) ∝ R(x)`.
#
# The reward is `R(x) = max(f(x), floor)^β`. The exponent `β` sharpens the target;
# `β = 3` is the value Jain et al. use and the default in the suite. The floor
# exists because infeasible and dead designs score zero or `-inf`, and `log 0` is
# not a number a loss can survive.

# %%
policy = SequencePolicy(
    n_tokens=env.alphabet.size,
    sequence_length=env.sequence_length,
    n_actions=env.n_actions,
    hidden_dim=64,
    seed=0,
)
reward = TemperedReward(beta=3.0)

result = train_trajectory_balance(
    env,
    policy,
    landscape,
    reward,
    TrainingConfig(steps=400, batch_size=64, log_every=100, seed=0),
    objective=TrajectoryBalance(),
)

print(f"loss   {result.losses[0]:.3f} -> {np.mean(result.losses[-20:]):.3f}")
print(f"log Z  {result.final_log_z:.3f}")
print(f"oracle calls used: {result.oracle_calls:,}")

# %% [markdown]
# **That oracle-call count is the reason a campaign never trains this way.** Four
# hundred steps at batch 64 is 25,600 evaluations — sixty-plus times a realistic
# campaign's entire budget. In `evoflownet.loop`, the GFlowNet trains against a
# *surrogate proxy* fitted to what has actually been measured, and the real oracle
# is spent only on the selected batch. Notebook 4 shows that. Here we hand it the
# true landscape because the point is to check the distribution, and for that we
# need the sampler aimed at a reward we know exactly.

# %% [markdown]
# ## Did it sample the right distribution?
#
# Here is a trap worth walking into deliberately, because it is easy to fall into
# by accident and it silently produces a wrong answer.
#
# `enumerate_terminal_states` returns the **Hamming ball** of radius
# `max_mutations` — everything within two substitutions of the parent. That is
# *not* what the masked environment can build. A design can be feasible and still
# unreachable, if every ordering of its mutations passes through an infeasible
# intermediate state. The docstring says as much: the Hamming ball is an upper
# bound.
#
# A target normalised over the wrong support is a target the policy is being
# marked against unfairly. So walk the graph instead:


# %%
def reachable_terminals(environment) -> np.ndarray:
    """Breadth-first over the masked construction graph from the parent."""
    from evoflownet.env.base import State  # noqa: PLC0415 - only needed here

    start = environment.parent
    seen = {start.tobytes(): start}
    frontier = [start]
    while frontier:
        batch = np.stack(frontier)
        mask = environment.forward_mask(
            State(sequences=batch, stopped=np.zeros(batch.shape[0], dtype=np.bool_))
        )
        nxt = []
        for row, sequence in enumerate(batch):
            for action in np.flatnonzero(mask[row]):
                if action == environment.stop_action:
                    continue
                position, token = divmod(int(action), environment.alphabet.size)
                child = sequence.copy()
                child[position] = token
                if child.tobytes() not in seen:
                    seen[child.tobytes()] = child
                    nxt.append(child)
        frontier = nxt
    return np.stack(list(seen.values()))


ball = env.enumerate_terminal_states()
space = reachable_terminals(env)
print(f"Hamming ball of radius 2      {ball.shape[0]:,}")
print(f"...feasible                   {int(landscape.is_feasible(ball).sum()):,}")
print(f"...actually reachable         {space.shape[0]:,}  <- the real support")

# %%
values = landscape.evaluate(space)
target = target_distribution(values, beta=reward.beta)

with torch.no_grad():
    trajectories = sample_trajectories(
        env, policy, 4000, epsilon=0.0, generator=torch.Generator().manual_seed(1)
    )
empirical = empirical_distribution(trajectories.terminal, space)

print(f"L1(empirical, target)         {l1_distance(empirical, target):.4f}")
print(f"L1 floor from sampling noise  {expected_l1_from_sampling_noise(target, 4000):.4f}")

# %% [markdown]
# The floor matters more than the distance. Drawing 4,000 samples from a
# distribution does not reproduce it exactly, so a perfect sampler still shows a
# non-zero L1. Without the floor beside it, an L1 of 0.3 is uninterpretable: it
# could be a badly fitted policy, or a perfect one measured with too few samples.
#
# Compare the top of the two distributions directly:

# %%
order = np.argsort(-target)[:8]
print(f"{'sequence':>10}{'f(x)':>8}{'target':>9}{'sampled':>9}")
for index in order:
    print(
        f"{landscape.alphabet.decode(space[index]):>10}"
        f"{values[index, 0]:>8.3f}{target[index]:>9.4f}{empirical[index]:>9.4f}"
    )

# %% [markdown]
# The policy lands close to the floor, which is the answer you want: on a space
# this size it is sampling proportionally to reward, not climbing to the top of
# it. The four `f(x) = 1.000` designs get ~15% of the mass each rather than 100%
# of it between them, and the 0.667 tier is still visited.
#
# Now the same measurement against the *Hamming ball* — the wrong support — to
# show how large the error from that one mistake is:

# %%
ball_values = landscape.evaluate(ball)
ball_target = target_distribution(ball_values, beta=reward.beta)
ball_empirical = empirical_distribution(trajectories.terminal, ball)
print(f"L1 against the Hamming ball   {l1_distance(ball_empirical, ball_target):.4f}")
print(f"L1 floor for that support     {expected_l1_from_sampling_noise(ball_target, 4000):.4f}")

# %% [markdown]
# Nearly a tenfold difference, and it would have read as a badly broken policy.
# Any exact distributional claim has to state which support it normalised over.

# %% [markdown]
# ## Feasibility, checked rather than asserted
#
# Every sequence the policy drew went through the masked graph, so all of them
# should be constructible. This is the property that notebook 2 spent a whole
# budget failing to get.

# %%
feasible = landscape.is_feasible(trajectories.terminal)
print(f"feasible fraction of 4,000 samples: {feasible.mean():.3f}")
print(f"distinct designs drawn:             {len(np.unique(trajectories.terminal, axis=0))}")

# %% [markdown]
# Say what this is, exactly. A feasible fraction of 1.000 is **definitional** — it
# is a property of the environment's mask, not an achievement of the GFlowNet.
# Any sampler generating through this graph gets it, and MOGFN-AL already masks
# this way (ICML 2023, App. D.6: "logits ... set to -1000"). Neither the masked
# mutation lattice nor closed-form uniform `P_B` is new here.
#
# What is measured in this project, and what notebook 2 set up, is the comparison
# against *rejection sampling* at a matched budget — where the two turn out to be
# tied on quality and to differ in how much of the budget gets spent.

# %% [markdown]
# ## Caveats worth carrying forward
#
# * This ran on `L = 8`, `v = 4`, 2 mutations — **18** reachable designs. That is
#   a toy, chosen so the exact check is possible at all. Nothing about a good L1
#   on 18 states transfers to `L = 256`. The one large-scale attempt at this in
#   the project cost 1.54 million oracle calls on GB1, roughly ten times the
#   search space: it tested correctness, and must never be quoted as efficiency.
# * The suite's GFlowNet arm is **not bit-reproducible**: at a fixed seed and
#   configuration it returns identical results on only 30-32 of 50 seeds, with a
#   per-seed standard deviation of 0.044. The classical baselines are identical
#   50/50. This is a known open defect, not a rounding artefact.
# * On the real benchmark landscapes, **no method solves any Ehrlich instance**.
#   Regret on the large-space task sits between 0.974 and 0.992 out of a maximum
#   of 1.0 for every method tried, GFlowNets included. "Marginally less stuck" is
#   the honest description.
#
# Next: `04-campaign.py`.
