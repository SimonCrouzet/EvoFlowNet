"""Tests for giving a classical sampler proxy access.

This wrapper exists to remove a confound, so the tests are mostly about what it
must *not* do: reach the oracle, change a sampler that cannot learn, or act
before there is a model to act on.
"""

import numpy as np
import pytest

from evoflownet.algorithms import ProxyOptimising
from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines.genetic import GeneticAlgorithm
from evoflownet.algorithms.baselines.mutagenesis import RandomMutagenesis
from evoflownet.core.types import Alphabet
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.loop import Campaign
from evoflownet.surrogate import DeepEnsemble, ProxyLandscape

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 6
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

    def _evaluate(self, sequences):
        self.calls += sequences.shape[0]
        return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)


@pytest.fixture
def parts():
    env = MutationEnvironment(PARENT, ALPHABET, max_mutations=4)
    surrogate = DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=20, seed=0)
    proxy = ProxyLandscape(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
    return env, surrogate, proxy


def fit(surrogate, seed=0):
    rng = np.random.default_rng(seed)
    train = rng.integers(0, ALPHABET.size, size=(64, LENGTH))
    surrogate.fit(train, (train == 1).sum(axis=1, keepdims=True).astype(float))


class TestBudget:
    def test_the_inner_loop_never_reaches_the_oracle(self, parts):
        # The point of the wrapper is a fair comparison, which it would destroy
        # if the control it provides were charged for its own search.
        env, surrogate, proxy = parts
        landscape = CountingLandscape()
        sampler = ProxyOptimising(
            GeneticAlgorithm(env, seed=0), proxy=proxy, generations=5, population=32
        )
        result = Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=surrogate,
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == result.oracle_calls == 24
        assert sampler.proxy_calls > 0

    def test_the_proxy_spend_is_reported(self, parts):
        env, surrogate, proxy = parts
        fit(surrogate)
        sampler = ProxyOptimising(
            GeneticAlgorithm(env, seed=0), proxy=proxy, generations=5, population=32
        )
        sampler.propose(8)
        assert sampler.proxy_calls == 160

    @pytest.mark.parametrize("field", ["generations", "population"])
    def test_non_positive_sizes_are_refused(self, parts, field):
        env, _, proxy = parts
        with pytest.raises(ValueError, match="at least 1"):
            ProxyOptimising(RandomMutagenesis(env), proxy=proxy, **{field: 0})


class TestBehaviour:
    def test_it_does_nothing_before_the_surrogate_is_fitted(self, parts):
        # A campaign's first round has nothing to search against.
        env, _, proxy = parts
        sampler = ProxyOptimising(GeneticAlgorithm(env, seed=0), proxy=proxy, generations=5)
        sampler.propose(8)
        assert sampler.proxy_calls == 0

    def test_the_inner_search_moves_a_learning_sampler(self, parts):
        env, surrogate, proxy = parts
        fit(surrogate)
        plain = GeneticAlgorithm(env, seed=0).propose(16)
        wrapped = ProxyOptimising(
            GeneticAlgorithm(env, seed=0), proxy=proxy, generations=10, population=32
        ).propose(16)
        assert not np.array_equal(plain, wrapped)

    def test_wrapping_a_stateless_sampler_changes_nothing(self, parts):
        # Worth pinning, because it bounds what the control can show. Random
        # mutagenesis ignores observe entirely, so the inner loop only burns
        # proxy calls -- any difference in a benchmark row is noise, not an
        # effect of giving it the model.
        env, surrogate, proxy = parts
        fit(surrogate)
        sampler = ProxyOptimising(
            RandomMutagenesis(env, seed=3), proxy=proxy, generations=4, population=16
        )
        # Advance the bare sampler identically, then compare the next draw.
        bare = RandomMutagenesis(env, seed=3)
        for _ in range(4):
            bare.propose(16)
        assert np.array_equal(sampler.propose(8), bare.propose(8))

    def test_assay_results_reach_the_wrapped_sampler(self, parts):
        env, _, proxy = parts
        inner = GeneticAlgorithm(env, seed=0)
        sampler = ProxyOptimising(inner, proxy=proxy, generations=2, population=16)
        sequences = inner.propose(8)
        sampler.observe(sequences, np.ones((8, 1)))
        assert np.isfinite(inner.population).all()


class TestInterface:
    def test_it_is_a_sampler(self, parts):
        env, _, proxy = parts
        assert isinstance(ProxyOptimising(RandomMutagenesis(env), proxy=proxy), Sampler)

    def test_the_name_marks_the_proxy_access(self, parts):
        env, _, proxy = parts
        name = ProxyOptimising(GeneticAlgorithm(env), proxy=proxy).name
        assert "GeneticAlgorithm" in name
        assert "proxy" in name

    def test_the_wrapped_sampler_is_reachable(self, parts):
        env, _, proxy = parts
        inner = GeneticAlgorithm(env)
        assert ProxyOptimising(inner, proxy=proxy).inner is inner

    def test_proposals_are_counted(self, parts):
        env, _, proxy = parts
        sampler = ProxyOptimising(RandomMutagenesis(env), proxy=proxy)
        sampler.propose(12)
        assert sampler.proposals_made == 12
