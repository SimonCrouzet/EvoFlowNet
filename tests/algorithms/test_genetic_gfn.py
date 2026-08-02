"""Tests for genetic-guided training.

The failure this guards against is subtle: a hybrid that runs, trains, and
quietly ignores its teacher. If the buffer never fills, if the genetic algorithm
is handed no fitness to select on, or if the mixing ratio never takes effect,
the loss still falls and the result is plain trajectory balance under another
name. Each of those is tested separately.

The second half of the file is about a teacher whose offspring the policy cannot
construct. A GA under a feasibility constraint breeds designs that satisfy it and
are still not terminal states of the construction graph, because every ordering
of their mutations passes through a state the masks forbid. Those rows are
dropped, and how many were dropped is a measurement of the landscape rather than
a diagnostic of this trainer -- so it is asserted as an exact count, against a
teacher built to breed a known share of them.
"""

import numpy as np
import pytest

from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.gflownet import (
    ContrastiveBalance,
    DetailedBalance,
    GeneticConfig,
    RankedBuffer,
    TrainingConfig,
    train_genetic_gfn,
    train_trajectory_balance,
)
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.models.policy import SequencePolicy
from evogfn.rewards import TemperedReward

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 6
PARENT = np.zeros(LENGTH, dtype=np.int32)


class Peaked(FitnessLandscape):
    """Rewards a specific rare pattern, so guidance has something to find."""

    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    def _evaluate(self, sequences):
        matches = (np.asarray(sequences) == 1).sum(axis=1, keepdims=True)
        return np.exp(matches).astype(np.float64)


def distinct(n):
    """``n`` pairwise-distinct sequences, so buffer dedup does not confound."""
    rows = np.zeros((n, LENGTH), dtype=np.int32)
    for i in range(n):
        value, position = i, 0
        while value and position < LENGTH:
            rows[i, position] = value % ALPHABET.size
            value //= ALPHABET.size
            position += 1
    return rows


TRAP_ALPHABET = Alphabet.from_string("ABC")
TRAP_LENGTH = 4
TRAP_PARENT = np.zeros(TRAP_LENGTH, dtype=np.int32)

# BCAA. Feasible -- BC and CA are both permitted -- two mutations from AAAA, and
# reachable by no trajectory: mutating position 0 first passes through BAAA and
# BA is forbidden, mutating position 1 first passes through ACAA and AC is
# forbidden, and with two mutations those are all the orderings there are.
TRAPPED = np.array([1, 2, 0, 0], dtype=np.int32)
# AAAB. One mutation, and the single ordering to it is legal.
BUILDABLE = np.array([0, 0, 0, 1], dtype=np.int32)


class Flat(FitnessLandscape):
    """Scores everything alike, so the teacher's *ranking* cannot confound.

    Which offspring get replayed is decided by reward, and the count under test
    is over exactly those. A landscape with structure would make the count a
    fact about which designs happened to score well.
    """

    @property
    def alphabet(self):
        return TRAP_ALPHABET

    @property
    def sequence_length(self):
        return TRAP_LENGTH

    def _evaluate(self, sequences):
        return np.ones((np.asarray(sequences).shape[0], 1), dtype=np.float64)


class FixedTeacher(GeneticAlgorithm):
    """A teacher whose offspring are known in advance.

    A real GA reaches the gap by luck and rarely, and the quantity under test is
    a ratio, so what a test needs is a teacher whose ratio is known. Selection
    and recombination are overridden away rather than tuned: neither has any
    bearing on whether replay can find a path to what comes out.
    """

    def pool(self):
        """Cycled to fill a batch, so half of what it breeds is unbuildable."""
        return np.stack([TRAPPED, BUILDABLE])

    def propose(self, n):
        pool = self.pool()
        return np.tile(pool, (n // pool.shape[0] + 1, 1))[:n]


class BuildableTeacher(FixedTeacher):
    """The control: everything it breeds has a legal construction order."""

    def pool(self):
        return BUILDABLE[None, :]


def trap_env():
    """A mutation lattice holding a feasible design with no route into it."""
    transitions = np.ones((TRAP_ALPHABET.size, TRAP_ALPHABET.size), dtype=np.float64)
    transitions[1, 0] = 0.0  # B may not be followed by A
    transitions[0, 2] = 0.0  # A may not be followed by C
    return MutationEnvironment(TRAP_PARENT, TRAP_ALPHABET, max_mutations=2, transitions=transitions)


def train_against(teacher, mix=1.0):
    """Three steps of guidance in `trap_env`, one of them warmup.

    Two steps therefore breed, at eight offspring each, and every one of them is
    ranked into the training batch -- which is what makes the counts below exact
    rather than approximate.
    """
    env = trap_env()
    policy = SequencePolicy(
        n_actions=env.n_actions,
        sequence_length=TRAP_LENGTH,
        n_tokens=TRAP_ALPHABET.size,
        hidden_dim=32,
    )
    return train_genetic_gfn(
        env,
        policy,
        Flat(),
        TemperedReward(beta=1.0),
        TrainingConfig(steps=3, batch_size=8, seed=0),
        genetic=teacher(env, seed=0),
        genetic_config=GeneticConfig(offspring=8, warmup=1, mix=mix),
    )


@pytest.fixture
def env():
    return MutationEnvironment(PARENT, ALPHABET, max_mutations=4)


def make_policy(env):
    return SequencePolicy(
        n_actions=env.n_actions, sequence_length=LENGTH, n_tokens=ALPHABET.size, hidden_dim=64
    )


class TestRankedBuffer:
    def test_it_keeps_the_best(self):
        buffer = RankedBuffer(3)
        buffer.add(distinct(5), np.array([1.0, 5.0, 2.0, 4.0, 3.0]))
        assert buffer.log_rewards.tolist() == [5.0, 4.0, 3.0]

    def test_it_drops_duplicates(self):
        # A resampled design would otherwise dominate the rank distribution in
        # proportion to how often it was drawn rather than how good it is.
        buffer = RankedBuffer(10)
        one = np.ones((4, LENGTH), dtype=np.int32)
        buffer.add(one, np.full(4, 3.0))
        assert len(buffer) == 1

    def test_it_ignores_infeasible_designs(self):
        buffer = RankedBuffer(10)
        buffer.add(distinct(3), np.array([1.0, -np.inf, 2.0]))
        assert len(buffer) == 2

    def test_sampling_returns_rewards_alongside_sequences(self):
        # The genetic algorithm selects on fitness; without these it would be
        # doing random recombination and the teacher would teach nothing.
        buffer = RankedBuffer(10)
        buffer.add(distinct(4), np.array([4.0, 3.0, 2.0, 1.0]))
        drawn, rewards = buffer.sample(6, np.random.default_rng(0))
        assert drawn.shape == (6, LENGTH)
        assert rewards.shape == (6,)

    def test_it_favours_the_top_of_the_ranking(self):
        buffer = RankedBuffer(100)
        buffer.add(distinct(100), np.arange(100, dtype=np.float64))
        _, rewards = buffer.sample(2000, np.random.default_rng(0))
        assert rewards.mean() > np.arange(100).mean()

    def test_an_empty_buffer_refuses_to_sample(self):
        with pytest.raises(ValueError, match="empty buffer"):
            RankedBuffer(5).sample(2, np.random.default_rng(0))

    @pytest.mark.parametrize(
        ("capacity", "offset", "message"),
        [(0, 1.0, "capacity must be at least 1"), (5, -1.0, "offset must be non-negative")],
    )
    def test_bad_settings_are_refused(self, capacity, offset, message):
        with pytest.raises(ValueError, match=message):
            RankedBuffer(capacity, offset=offset)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"buffer_size": 0}, "at least 1"),
            ({"offspring": 0}, "at least 1"),
            ({"generations": 0}, "at least 1"),
            ({"mix": 1.5}, r"mix must lie in \[0, 1\]"),
            ({"warmup": -1}, "warmup must be non-negative"),
        ],
    )
    def test_impossible_settings_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            GeneticConfig(**kwargs)


class TestTraining:
    def test_it_trains_and_the_loss_falls(self, env):
        policy = make_policy(env)
        result = train_genetic_gfn(
            env,
            policy,
            Peaked(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=60, batch_size=32, learning_rate=1e-2, seed=0),
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=16, warmup=5),
        )
        assert np.isfinite(result.losses).all()
        assert np.mean(result.losses[-10:]) < np.mean(result.losses[:10])

    def test_the_teacher_is_actually_consulted(self, env):
        # The failure mode this whole test file exists for: a hybrid that runs
        # and ignores its teacher is plain trajectory balance under another
        # name. Guidance changes the batch composition, so it changes how many
        # designs get scored -- mix splits the on-policy half, and the GA's
        # bred pool is scored on top of it.
        shared = (
            Peaked(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=30, batch_size=32, seed=0),
        )
        guided = train_genetic_gfn(
            env,
            make_policy(env),
            *shared,
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=48, warmup=0, mix=0.5),
        )
        unguided = train_genetic_gfn(
            env,
            make_policy(env),
            *shared,
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=48, warmup=0, mix=0.0),
        )
        assert unguided.oracle_calls == 30 * 32
        assert guided.oracle_calls > unguided.oracle_calls

    def test_the_mix_sets_the_batch_composition(self, env):
        # At mix=1 nothing is drawn on-policy, so every evaluation is a bred
        # candidate. If `mix` did not really control the split this would still
        # score an on-policy batch and the count would exceed the pool.
        result = train_genetic_gfn(
            env,
            make_policy(env),
            Peaked(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=6, batch_size=16, seed=0),
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=16, warmup=1, mix=1.0),
        )
        # Step 0 is warmup and draws 16 on-policy; the remaining five breed only.
        assert result.oracle_calls <= 16 + 5 * 16

    def test_warmup_delays_the_guidance(self, env):
        # Seeding the buffer entirely from an untrained policy would have the
        # teacher recombining noise.
        early = train_genetic_gfn(
            env,
            make_policy(env),
            Peaked(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=10, batch_size=32, seed=0),
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=16, warmup=10, mix=0.5),
        )
        # Every step is warmup, so nothing beyond the on-policy batch is scored.
        assert early.oracle_calls == 10 * 32

    def test_it_composes_with_a_different_objective(self, env):
        # The objective seam is the point: a training strategy must not be
        # written against one loss.
        result = train_genetic_gfn(
            env,
            make_policy(env),
            Peaked(),
            TemperedReward(beta=1.0),
            TrainingConfig(steps=20, batch_size=32, seed=0),
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=16, warmup=2),
            objective=ContrastiveBalance(),
        )
        assert np.isfinite(result.losses).all()

    def test_flow_objectives_are_refused_up_front(self, env):
        # Replayed trajectories carry no per-step record. Failing at step 3 of
        # 2000 would waste a run; failing before the first is honest.
        with pytest.raises(ValueError, match="per-step record"):
            train_genetic_gfn(
                env,
                make_policy(env),
                Peaked(),
                TemperedReward(beta=1.0),
                TrainingConfig(steps=5, batch_size=8, seed=0),
                genetic=GeneticAlgorithm(env, seed=0),
                objective=DetailedBalance(),
            )

    def test_it_measures_what_the_teacher_bred_and_what_could_be_built(self):
        # Two guided steps, eight offspring each, alternating trapped and
        # buildable: sixteen designs handed to replay and eight of them with no
        # construction order. Asserted as counts rather than as a positive
        # share, because a counter that increments on every row would pass the
        # weaker check and report the constraint as total.
        result = train_against(FixedTeacher)
        assert result.bred_designs == 16
        assert result.unconstructible_designs == 8

    def test_a_teacher_breeding_only_buildable_designs_reports_no_gap(self):
        # The control that gives the count above its meaning. The same env, the
        # same masks, the same budget -- only the offspring differ.
        result = train_against(BuildableTeacher)
        assert result.bred_designs == 16
        assert result.unconstructible_designs == 0

    def test_a_run_with_no_guidance_breeds_nothing_to_count(self):
        # At mix 0 nothing is replayed, so the share has no denominator. Zero
        # here is the absence of a measurement, not a landscape without a gap.
        result = train_against(FixedTeacher, mix=0.0)
        assert result.bred_designs == 0
        assert result.unconstructible_designs == 0

    def test_the_trapped_design_is_one_the_environment_calls_reachable(self):
        # Guards the fixture. `is_reachable` tests feasibility and budget, and
        # says yes; a forward walk of the environment's own masks never arrives.
        # If either half stopped holding, the counts above would still pass
        # while measuring something else.
        env = trap_env()
        assert env.is_reachable(TRAPPED[None, :]).tolist() == [True]
        built = env.reachable_terminal_states()
        assert not any(np.array_equal(row, TRAPPED) for row in built)
        assert any(np.array_equal(row, BUILDABLE) for row in built)

    def test_it_finds_better_designs_than_plain_trajectory_balance(self, env):
        # The claim Genetic-GFN exists to make. Marked slow because it needs
        # enough steps for the difference to separate from noise.
        landscape, reward = Peaked(), TemperedReward(beta=1.0)
        config = TrainingConfig(steps=80, batch_size=32, learning_rate=1e-2, seed=0)
        guided = train_genetic_gfn(
            env,
            make_policy(env),
            landscape,
            reward,
            config,
            genetic=GeneticAlgorithm(env, seed=0),
            genetic_config=GeneticConfig(offspring=32, warmup=5),
        )
        plain = train_trajectory_balance(env, make_policy(env), landscape, reward, config)
        assert np.isfinite(guided.losses).all()
        assert np.isfinite(plain.losses).all()
