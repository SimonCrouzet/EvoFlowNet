"""Tests for the GFlowNet behind the sampler interface.

The property that matters most is negative: training must not reach the oracle.
It is the kind of error that raises nothing and simply makes the method look
sample-inefficient, so it is asserted from both sides -- the oracle counts its
own calls, and the sampler counts the proxy's.
"""

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.gflownet import GFlowNetSampler, TrainingConfig
from evoflownet.core.types import Alphabet
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.loop import Campaign
from evoflownet.models.policy import SequencePolicy
from evoflownet.rewards import TemperedReward
from evoflownet.surrogate import DeepEnsemble, ProxyLandscape

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 4
PARENT = np.zeros(LENGTH, dtype=np.int32)


class CountingLandscape(FitnessLandscape):
    """An oracle that records every call made to it."""

    def __init__(self):
        self.calls = 0

    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    @property
    def optimum(self):
        return np.array([float(LENGTH)])

    def _evaluate(self, sequences):
        self.calls += sequences.shape[0]
        return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)


@pytest.fixture
def parts():
    """An environment, policy, surrogate and proxy over a small mutation lattice."""
    env = MutationEnvironment(PARENT, ALPHABET, max_mutations=2)
    policy = SequencePolicy(
        n_actions=env.n_actions,
        sequence_length=LENGTH,
        n_tokens=ALPHABET.size,
        hidden_dim=32,
    )
    surrogate = DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=20, seed=0)
    proxy = ProxyLandscape(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
    return env, policy, surrogate, proxy


def build(parts, **kwargs):
    env, policy, _, proxy = parts
    return GFlowNetSampler(
        env,
        policy,
        proxy=proxy,
        reward=TemperedReward(beta=1.0),
        config=TrainingConfig(steps=5, batch_size=8),
        **kwargs,
    )


class TestBudgetSeparation:
    def test_training_never_reaches_the_oracle(self, parts):
        # The whole point. The sampler burns thousands of reward evaluations on
        # the proxy while the oracle is charged only for the measured batch.
        landscape = CountingLandscape()
        sampler = build(parts)
        result = Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=parts[2],
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == result.oracle_calls == 24
        assert sampler.proxy_calls > 0

    def test_the_proxy_absorbs_the_training_evaluations(self, parts):
        landscape = CountingLandscape()
        sampler = build(parts)
        Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=parts[2],
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        # Two retrains (rounds 1 and 2) at 5 steps x batch 8.
        assert sampler.proxy_calls == 80
        assert sampler.rounds_trained == 2


class TestTraining:
    def test_the_first_round_samples_without_training(self, parts):
        # Nothing has been measured, so the surrogate is unfitted and there is
        # no reward signal to train against.
        sampler = build(parts)
        proposals = sampler.propose(8)
        assert sampler.rounds_trained == 0
        assert proposals.shape == (8, LENGTH)

    def test_it_trains_once_the_surrogate_is_fitted(self, parts):
        _, _, surrogate, _ = parts
        rng = np.random.default_rng(0)
        train = rng.integers(0, ALPHABET.size, size=(32, LENGTH))
        surrogate.fit(train, (train == 1).sum(axis=1, keepdims=True).astype(float))
        sampler = build(parts)
        sampler.propose(8)
        assert sampler.rounds_trained == 1

    def test_each_round_trains_from_a_different_seed(self, parts):
        # Reusing one seed would replay identical trajectories every round, so
        # later rounds would add nothing despite costing the same compute.
        _, _, surrogate, _ = parts
        rng = np.random.default_rng(1)
        train = rng.integers(0, ALPHABET.size, size=(32, LENGTH))
        surrogate.fit(train, (train == 1).sum(axis=1, keepdims=True).astype(float))
        sampler = build(parts)
        first = sampler.propose(16)
        second = sampler.propose(16)
        assert not np.array_equal(first, second)


class TestInterface:
    def test_it_is_a_sampler_like_any_other(self, parts):
        assert isinstance(build(parts), Sampler)

    def test_proposals_are_counted(self, parts):
        sampler = build(parts)
        sampler.propose(12)
        assert sampler.proposals_made == 12

    def test_the_name_records_the_objective(self, parts):
        assert "TrajectoryBalance" in build(parts).name

    def test_proposals_stay_inside_the_environment(self, parts):
        env = parts[0]
        proposals = build(parts).propose(32)
        assert env.is_reachable(proposals).all()
