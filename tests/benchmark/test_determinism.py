"""Tests that the benchmark can reproduce its own numbers.

A benchmark that cannot is worth nothing downstream, and this one could not.
Two independent causes, both silent: multithreaded floating-point reduction
order, and a policy drawn from global torch state so that "same seed" did not
mean "same starting network". Together they made the paired comparison the
statistics rest on quietly unpaired for every GFlowNet arm.

Both are asserted here, because either alone reintroduces the failure.

A third, found later: the pinning only ever reached torch's pool. numpy links a
separate BLAS whose thread count is fixed at import, so the environment
variables set here never touched it, while ``is_deterministic`` reported on
torch's counter and passed anyway. The tests below therefore assert against the
*live* pool sizes rather than against the call having been made -- a check that
cannot tell "pinned" from "never reached" is the bug, not the test for it.
"""

import os

import pytest
import torch

from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.benchmark.determinism import (
    THREAD_VARIABLES,
    configure_determinism,
    is_deterministic,
    thread_pools,
)
from evogfn.benchmark.methods import gflownet
from evogfn.benchmark.suite import objective_task
from evogfn.models.policy import SequencePolicy


def policy(seed=None):
    return SequencePolicy(n_tokens=20, sequence_length=8, n_actions=32, hidden_dim=16, seed=seed)


def pools(*counts):
    """Fake threadpoolctl output; only the two keys the module reads."""
    return [{"prefix": f"pool{i}", "num_threads": n} for i, n in enumerate(counts)]


class TestThreadPinning:
    def test_it_pins_intra_op_threads(self):
        configure_determinism()
        assert torch.get_num_threads() == 1
        assert is_deterministic()

    def test_it_pins_every_loaded_pool_not_just_torchs(self):
        # The regression: numpy's BLAS is a different library from torch's
        # OpenMP, kept its own count, and nothing here noticed.
        configure_determinism()
        assert thread_pools(), "no pool was found, so nothing was verified"
        assert all(count == 1 for _, count in thread_pools())

    def test_it_is_idempotent(self):
        configure_determinism()
        configure_determinism()
        assert is_deterministic()
        assert all(count == 1 for _, count in thread_pools())

    def test_it_sets_the_library_variables(self, monkeypatch):
        # These do nothing for the pools this process already loaded. They are
        # asserted because they still govern later imports and child processes.
        for name in THREAD_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        configure_determinism()
        assert all(os.environ[name] == "1" for name in THREAD_VARIABLES)


class TestDeterminismCheck:
    def test_an_unpinned_pool_makes_it_false(self, monkeypatch):
        configure_determinism()
        monkeypatch.setattr("evogfn.benchmark.determinism.threadpool_info", lambda: pools(1, 8))
        assert not is_deterministic()

    def test_an_unreadable_count_reads_as_unverified_not_fine(self, monkeypatch):
        # A pool found but not measurable is the case that must never pass:
        # every stored record carries this value as a claim about its run.
        configure_determinism()
        monkeypatch.setattr(
            "evogfn.benchmark.determinism.threadpool_info",
            lambda: [{"prefix": "opaque", "version": None}],
        )
        assert thread_pools() == (("opaque", None),)
        assert not is_deterministic()

    def test_finding_no_pool_at_all_reads_as_unverified(self, monkeypatch):
        configure_determinism()
        monkeypatch.setattr("evogfn.benchmark.determinism.threadpool_info", lambda: [])
        assert not is_deterministic()

    def test_unpinned_torch_makes_it_false_even_with_pinned_pools(self, monkeypatch):
        monkeypatch.setattr("evogfn.benchmark.determinism.threadpool_info", lambda: pools(1))
        monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
        assert not is_deterministic()

    def test_it_names_each_pool_so_a_failure_says_which(self, monkeypatch):
        monkeypatch.setattr("evogfn.benchmark.determinism.threadpool_info", lambda: pools(1, 8))
        assert thread_pools() == (("pool0", 1), ("pool1", 8))


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
