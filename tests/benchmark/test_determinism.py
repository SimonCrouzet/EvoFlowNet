"""Tests that the benchmark can reproduce its own numbers.

A benchmark that cannot is worth nothing downstream, and this one could not.
Two independent causes, both silent: multithreaded floating-point reduction
order, and a policy drawn from global torch state so that "same seed" did not
mean "same starting network". Together they made the paired comparison the
statistics rest on quietly unpaired for every GFlowNet arm.

Both are asserted here, because either alone reintroduces the failure.
"""

import os

import pytest
import torch

from evoflownet.algorithms.gflownet.objectives import TrajectoryBalance
from evoflownet.benchmark.determinism import (
    THREAD_VARIABLES,
    configure_determinism,
    is_deterministic,
)
from evoflownet.benchmark.methods import gflownet
from evoflownet.benchmark.suite import objective_task
from evoflownet.models.policy import SequencePolicy


def policy(seed=None):
    return SequencePolicy(n_tokens=20, sequence_length=8, n_actions=32, hidden_dim=16, seed=seed)


class TestThreadPinning:
    def test_it_pins_intra_op_threads(self):
        configure_determinism()
        assert torch.get_num_threads() == 1
        assert is_deterministic()

    def test_it_is_idempotent(self):
        configure_determinism()
        configure_determinism()
        assert is_deterministic()

    def test_it_sets_the_library_variables(self, monkeypatch):
        for name in THREAD_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        configure_determinism()
        assert all(os.environ[name] == "1" for name in THREAD_VARIABLES)


class TestPolicySeeding:
    def test_the_same_seed_gives_the_same_network(self):
        # The bug that broke pairing: without this, two arms run under "seed 7"
        # started from different networks, so the paired statistic compared
        # initialisations as much as methods.
        first, second = policy(seed=7), policy(seed=7)
        assert torch.equal(first.forward_head.weight, second.forward_head.weight)
        assert torch.equal(first.embedding.weight, second.embedding.weight)

    def test_different_seeds_give_different_networks(self):
        assert not torch.equal(
            policy(seed=1).forward_head.weight, policy(seed=2).forward_head.weight
        )

    def test_every_parameter_is_seeded_not_just_the_first(self):
        # Seeding only some layers is the failure mode that looks fixed and is
        # not; the ensemble had exactly this bug in its output layer.
        first, second = policy(seed=3), policy(seed=3)
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_seeding_does_not_leak_into_global_state(self):
        # fork_rng is what makes construction reproducible without reseeding
        # everything built afterwards, which would be a different bug.
        torch.manual_seed(999)
        before = torch.randn(4)
        torch.manual_seed(999)
        policy(seed=7)
        after = torch.randn(4)
        assert torch.equal(before, after)

    def test_an_unseeded_policy_still_works(self):
        # Seeding is opt-in; the unseeded path must not break.
        assert policy(seed=None).forward_head.weight.shape[0] == 32


@pytest.mark.slow
def test_a_seeded_campaign_reproduces_exactly():
    """End to end: the property every paired statistic depends on."""
    configure_determinism()
    task, method = objective_task(), gflownet(TrajectoryBalance(), steps=20)
    values = [method(task, 3).run().best_value for _ in range(3)]
    assert len(set(values)) == 1, f"identical configuration diverged: {values}"
