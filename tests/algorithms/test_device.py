"""Tests for running on an accelerator.

CI has no GPU, so the cross-device tests here skip there and only run on a
machine that has one. That gap is why the bug they cover survived: every test in
this suite ran on CPU, where a generator and a tensor are trivially on the same
device, and the mismatch only appears the first time someone passes
``device="cuda"``.
"""

import numpy as np
import pytest
import torch

from evoflownet.algorithms.gflownet import replay_trajectories, sample_trajectories
from evoflownet.algorithms.gflownet.sampling import _multinomial
from evoflownet.core import Alphabet
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.models import SequencePolicy

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def make_env():
    return MutationEnvironment(
        np.zeros(4, dtype=np.int32), Alphabet.from_string("ABC"), max_mutations=2
    )


def make_policy(env, device="cpu"):
    torch.manual_seed(0)
    return SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        hidden_dim=32,
        embedding_dim=8,
    ).to(device)


class TestGeneratorDeviceMismatch:
    """A CPU generator with a CUDA policy is the obvious way to write a
    reproducible GPU experiment, and ``torch.multinomial`` rejects it outright.
    """

    def test_a_same_device_draw_is_unaffected(self):
        probabilities = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        drawn = _multinomial(probabilities, torch.Generator().manual_seed(0))
        assert drawn.tolist() == [1, 0]

    def test_no_generator_is_allowed(self):
        # Mass entirely on index 1, so the draw is deterministic without needing
        # a generator at all.
        probabilities = torch.tensor([[0.0, 1.0]])
        assert _multinomial(probabilities, None).tolist() == [1]

    @needs_cuda
    def test_a_cpu_generator_drives_a_cuda_draw(self):
        probabilities = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cuda")
        drawn = _multinomial(probabilities, torch.Generator().manual_seed(0))
        assert drawn.device.type == "cuda"
        assert drawn.tolist() == [1, 0]

    @needs_cuda
    def test_the_draw_stays_reproducible_across_the_transfer(self):
        # The point of accepting the mismatch rather than erroring: the caller's
        # generator must still determine the outcome.
        probabilities = torch.rand((32, 8), device="cuda") + 1e-3
        first = _multinomial(probabilities, torch.Generator().manual_seed(7))
        second = _multinomial(probabilities, torch.Generator().manual_seed(7))
        assert torch.equal(first, second)


@needs_cuda
class TestSamplingOnCuda:
    def test_trajectories_can_be_sampled_on_cuda(self):
        env = make_env()
        policy = make_policy(env, "cuda")
        trajectories = sample_trajectories(
            env, policy, 8, epsilon=0.2, generator=torch.Generator().manual_seed(1), device="cuda"
        )
        assert trajectories.terminal.shape == (8, 4)
        assert trajectories.log_forward.device.type == "cuda"

    def test_trajectories_can_be_replayed_on_cuda(self):
        env = make_env()
        policy = make_policy(env, "cuda")
        sampled = sample_trajectories(
            env, policy, 8, generator=torch.Generator().manual_seed(1), device="cuda"
        )
        replayed = replay_trajectories(
            env, policy, sampled.terminal, generator=torch.Generator().manual_seed(2), device="cuda"
        )
        assert np.array_equal(replayed.terminal, sampled.terminal)

    def test_cuda_and_cpu_reach_the_same_terminals_from_one_seed(self):
        # Not guaranteed in general -- floating point differs between backends --
        # but on a graph this small the argmax structure should be identical, and
        # a divergence here would mean the device argument changes the sampled
        # distribution rather than only where it is computed.
        env = make_env()
        results = []
        for device in ("cpu", "cuda"):
            policy = make_policy(env, device)
            results.append(
                sample_trajectories(
                    env, policy, 32, generator=torch.Generator().manual_seed(3), device=device
                ).terminal
            )
        agreement = (results[0] == results[1]).all(axis=1).mean()
        assert agreement > 0.9, f"only {agreement:.0%} of terminals agreed across devices"
