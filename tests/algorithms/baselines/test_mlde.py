"""Tests for MLDE.

The failure that matters here is a quiet one: an MLDE that never leaves its
random-screening stage, or whose fitted model ranks no better than chance, is
indistinguishable from random mutagenesis in a results table, and it would make
the project's central comparison look far more favourable than it is. So the
handover is tested explicitly, and the ranking is tested against the null it
would otherwise silently collapse into.
"""

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines import (
    MLDE,
    PUBLISHED_BATCH_SIZE,
    PUBLISHED_TRAINING_SIZE,
    RandomMutagenesis,
)
from evoflownet.core import Alphabet
from evoflownet.env.mutation import MutationEnvironment


def make_env(length=8, symbols="ABCD", max_mutations=6, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def constrained_transitions(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def toy_landscape(sequences):
    """Reward sequences for containing token 1, so improvement is detectable."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


def train(sampler, rounds=2, size=32, landscape=toy_landscape):
    """Run the sampler through enough rounds to finish its training sample."""
    for _ in range(rounds):
        proposals = sampler.propose(size)
        sampler.observe(proposals, landscape(proposals))
    return sampler


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(MLDE(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert MLDE(env, training_size=16).propose(16).shape == (16, env.sequence_length)

    def test_fitted_proposals_have_the_right_shape(self):
        env = make_env()
        sampler = train(MLDE(env, training_size=32), rounds=2, size=32)
        assert sampler.propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        env = make_env(max_mutations=2)
        sampler = MLDE(env, training_size=32, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_proposals_are_counted(self):
        sampler = MLDE(make_env(), training_size=1000)
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_screening_the_pool_is_charged_as_proposals(self):
        # The published method ranks an exhaustive library. Generating the pool
        # that stands in for one is a real cost, and the base class exists to
        # make exactly this kind of discarded work visible.
        sampler = train(MLDE(make_env(), training_size=32, pool_multiplier=4), rounds=2, size=32)
        before = sampler.proposals_made
        sampler.propose(16)
        assert sampler.proposals_made - before >= 16 * 4

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        assert np.array_equal(
            MLDE(env, seed=5).propose(16),
            MLDE(env, seed=5).propose(16),
        )

    def test_a_whole_campaign_is_reproducible(self):
        env = make_env()
        runs = []
        for _ in range(2):
            sampler = MLDE(env, training_size=32, seed=13)
            batches = []
            for _ in range(4):
                proposals = sampler.propose(32)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])

    def test_the_label_says_whether_feasibility_is_enforced(self):
        env = make_env()
        assert MLDE(env).name == "MLDE"
        assert MLDE(env, feasible_only=True).name == "MLDE (feasible)"


class TestTheTwoStageProtocol:
    def test_it_screens_at_random_until_the_training_sample_is_complete(self):
        env = make_env()
        sampler = MLDE(env, training_size=64, seed=0)
        proposals = sampler.propose(32)
        sampler.observe(proposals, toy_landscape(proposals))
        assert not sampler.is_fitted
        assert sampler.training_examples == 32

    def test_the_random_stage_is_exactly_random_mutagenesis(self):
        # "Sample the library uniformly" is what the protocol says, so the null
        # and MLDE's opening round should be the same draw.
        env = make_env()
        assert np.array_equal(
            MLDE(env, seed=2).propose(24),
            RandomMutagenesis(env, seed=2).propose(24),
        )

    def test_the_model_takes_over_once_the_sample_is_complete(self):
        sampler = train(MLDE(make_env(), training_size=32), rounds=1, size=32)
        sampler.propose(16)
        assert sampler.is_fitted

    def test_failed_assays_do_not_count_toward_the_training_sample(self):
        # A well that did not report is not a training point, and treating it as
        # one would hand over to a model fitted on fewer measurements than asked.
        env = make_env()
        sampler = MLDE(env, training_size=32, seed=0)
        proposals = sampler.propose(32)
        sampler.observe(proposals, np.full((32, 1), -np.inf))
        assert sampler.training_examples == 0
        assert not sampler.is_fitted

    def test_the_published_split_is_available_and_is_what_wittmann_reports(self):
        assert PUBLISHED_TRAINING_SIZE == 384
        assert PUBLISHED_BATCH_SIZE == 96


class TestTheModelActuallyRanks:
    def test_the_top_predictions_beat_a_random_draw(self):
        # If they do not, MLDE is random mutagenesis with extra steps, and every
        # comparison against it is worthless.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=96, seed=0), rounds=3, size=32)
        chosen = sampler.propose(32)
        drawn = RandomMutagenesis(env, seed=1).propose(32)
        assert toy_landscape(chosen).mean() > toy_landscape(drawn).mean()

    def test_proposals_come_back_ranked_best_first(self):
        # The campaign takes a prefix of the pool when it has no surrogate of its
        # own, so the order is the interface, not a convenience.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=96, seed=0), rounds=3, size=32)
        proposals = sampler.propose(64)
        head = toy_landscape(proposals[:16]).mean()
        tail = toy_landscape(proposals[-16:]).mean()
        assert head > tail

    def test_it_improves_over_rounds(self):
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = MLDE(env, training_size=64, seed=0)
        first = last = 0.0
        for index in range(8):
            proposals = sampler.propose(32)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first

    def test_the_additive_ablation_still_fits(self):
        # Degree 1 is the purely additive model, which is the control that says
        # how much of MLDE's advantage needs pairwise epistasis at all.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=64, kernel_degree=1, seed=0), rounds=2, size=32)
        chosen = sampler.propose(32)
        assert sampler.is_fitted
        assert env.is_reachable(chosen).all()

    def test_it_does_not_re_propose_what_has_already_been_assayed(self):
        # A lab does not re-order a variant it has measured, and a model whose
        # argmax is a variant already in its training set would otherwise spend
        # every remaining round re-proposing it.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = MLDE(env, training_size=64, seed=0)
        measured: set[bytes] = set()
        for _ in range(4):
            proposals = sampler.propose(32)
            sampler.observe(proposals, toy_landscape(proposals))
            measured.update(row.tobytes() for row in np.ascontiguousarray(proposals))
        chosen = np.ascontiguousarray(sampler.propose(32))
        assert not any(row.tobytes() in measured for row in chosen)


class TestFeasibility:
    def test_a_feasible_only_sampler_only_emits_constructible_designs(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        sampler = MLDE(env, training_size=32, feasible_only=True, seed=0)
        for _ in range(4):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("training_size", 0, "training_size must be at least 1"),
            ("pool_multiplier", 0, "pool_multiplier must be at least 1"),
            ("max_attempts", 0, "max_attempts must be at least 1"),
            ("ridge_alpha", -1.0, "ridge_alpha must not be negative"),
            ("kernel_degree", 0, "kernel_degree must be at least 1"),
        ],
    )
    def test_an_impossible_configuration_is_refused(self, field, value, message):
        with pytest.raises(ValueError, match=message):
            MLDE(make_env(), **{field: value})
