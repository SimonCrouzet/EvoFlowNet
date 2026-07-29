"""Tests for the surrogate-backed proxy landscape.

The proxy exists so a sampler can train without touching the oracle. What is
tested here is that it faithfully reports the surrogate and that it refuses to
answer before there is a model to answer with -- an unfitted network's output
would look exactly like a reward signal and would train a policy toward its own
initialisation.
"""

import numpy as np
import pytest

from evogfn.core.types import Alphabet
from evogfn.surrogate import DeepEnsemble, ProxyLandscape

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 4


@pytest.fixture
def surrogate():
    return DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=20, seed=0)


def fit(surrogate, seed):
    rng = np.random.default_rng(seed)
    train = rng.integers(0, ALPHABET.size, size=(64, LENGTH))
    surrogate.fit(train, (train == 1).sum(axis=1, keepdims=True).astype(float))
    return train


def proxy_over(surrogate, **kwargs):
    return ProxyLandscape(surrogate, alphabet=ALPHABET, sequence_length=LENGTH, **kwargs)


class TestProxy:
    def test_it_tracks_refits_in_place(self, surrogate):
        # The campaign refits the surrogate it holds; a sampler built once at
        # the start must see that without being handed a new proxy.
        proxy = proxy_over(surrogate)
        assert not proxy.is_ready
        fit(surrogate, 2)
        assert proxy.is_ready

    def test_it_scores_with_the_surrogate_mean(self, surrogate):
        proxy = proxy_over(surrogate)
        train = fit(surrogate, 3)
        mean, _ = surrogate.predict(train)
        assert proxy.evaluate(train)[:, 0] == pytest.approx(mean)

    def test_optimism_adds_the_uncertainty(self, surrogate):
        train = fit(surrogate, 4)
        plain = proxy_over(surrogate)
        eager = proxy_over(surrogate, optimism=2.0)
        assert (eager.evaluate(train) >= plain.evaluate(train)).all()

    def test_zero_optimism_is_the_mean_exactly(self, surrogate):
        # The default must be the plain mean, so that a difference between arms
        # is the sampler rather than a hidden exploration bonus.
        train = fit(surrogate, 5)
        mean, _ = surrogate.predict(train)
        assert proxy_over(surrogate, optimism=0.0).evaluate(train)[:, 0] == pytest.approx(mean)

    def test_a_negative_optimism_is_refused(self, surrogate):
        with pytest.raises(ValueError, match="non-negative"):
            proxy_over(surrogate, optimism=-1.0)

    def test_scoring_before_fitting_is_refused(self, surrogate):
        with pytest.raises(RuntimeError, match="not been fitted"):
            proxy_over(surrogate).evaluate(np.zeros((4, LENGTH), dtype=np.int32))

    def test_it_reports_the_alphabet_it_was_given(self, surrogate):
        proxy = proxy_over(surrogate)
        assert proxy.alphabet == ALPHABET
        assert proxy.sequence_length == LENGTH
