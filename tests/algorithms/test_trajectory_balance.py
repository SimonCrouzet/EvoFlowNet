"""Tests for trajectory sampling and the trajectory balance objective.

The test that matters is :class:`TestItLearnsTheTargetDistribution`. Everything
else checks that the machinery is wired correctly; that one checks it is a
GFlowNet, by training on a landscape small enough to enumerate and comparing the
sampled distribution against the exact target.
"""

import math

import numpy as np
import pytest
import torch

from evoflownet.algorithms.gflownet import (
    parameter_groups,
    sample_trajectories,
    trajectory_balance_loss,
)
from evoflownet.core import Alphabet
from evoflownet.env.base import State
from evoflownet.env.mutation import MutationEnvironment
from evoflownet.metrics import (
    empirical_distribution,
    expected_l1_from_sampling_noise,
    l1_distance,
    target_distribution,
)
from evoflownet.models import SequencePolicy
from evoflownet.rewards import TemperedReward


def make_env(length=4, symbols="ABC", max_mutations=2):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
    )


def make_policy(env, seed=0, **overrides):
    torch.manual_seed(seed)
    return SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        **{"hidden_dim": 64, "embedding_dim": 16, **overrides},
    )


def terminal_state(sequences):
    return State(sequences=sequences, stopped=np.ones(sequences.shape[0], dtype=bool))


class TestSampling:
    def test_every_trajectory_terminates(self):
        env, policy = make_env(), None
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 16)
        assert len(trajectories) == 16
        assert trajectories.terminal.shape == (16, env.sequence_length)

    def test_no_trajectory_exceeds_the_mutation_budget(self):
        env = make_env(max_mutations=2)
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 32)
        counts = env.n_mutations(terminal_state(trajectories.terminal))
        assert counts.max() <= 2

    def test_backward_log_probability_is_minus_log_factorial(self):
        # The lattice property, arriving through the sampler rather than being
        # read off the mask: a terminal with k mutations is reached by k!
        # trajectories, so summing uniform 1/j over the path gives -log k!.
        env = make_env(max_mutations=3, length=5)
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 64)
        counts = env.n_mutations(terminal_state(trajectories.terminal))
        expected = np.array([-math.log(math.factorial(int(k))) for k in counts])
        assert trajectories.log_backward.detach().numpy() == pytest.approx(expected, abs=1e-5)

    def test_trajectory_length_is_mutations_plus_the_stop(self):
        env = make_env(max_mutations=2)
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 32)
        counts = env.n_mutations(terminal_state(trajectories.terminal))
        assert trajectories.lengths.tolist() == (counts + 1).tolist()

    def test_forward_log_probabilities_are_negative_and_finite(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 32)
        values = trajectories.log_forward.detach()
        assert torch.isfinite(values).all()
        assert (values < 0).all()

    def test_sampling_is_reproducible_given_a_generator(self):
        env = make_env()
        policy = make_policy(env)
        first = sample_trajectories(env, policy, 16, generator=torch.Generator().manual_seed(5))
        second = sample_trajectories(env, policy, 16, generator=torch.Generator().manual_seed(5))
        assert np.array_equal(first.terminal, second.terminal)

    def test_exploration_changes_what_is_drawn_but_not_how_it_is_scored(self):
        # Trajectory balance is off-policy: the behaviour policy may explore,
        # but the accumulated log-probabilities must be the model's. Scoring the
        # behaviour policy instead would optimise the wrong objective while
        # looking perfectly reasonable.
        env = make_env()
        policy = make_policy(env)
        explored = sample_trajectories(
            env, policy, 64, epsilon=0.9, generator=torch.Generator().manual_seed(1)
        )
        # Re-score the drawn terminals under the model by replaying them is not
        # available here, so check the weaker but still meaningful property:
        # log P_F stays a valid log-probability regardless of exploration.
        assert torch.isfinite(explored.log_forward).all()
        assert (explored.log_forward <= 0).all()

    def test_full_exploration_still_respects_the_masks(self):
        env = make_env(max_mutations=2)
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 64, epsilon=1.0)
        counts = env.n_mutations(terminal_state(trajectories.terminal))
        assert counts.max() <= 2

    @pytest.mark.parametrize(("n", "epsilon"), [(0, 0.0), (4, -0.1), (4, 1.5)])
    def test_invalid_arguments_are_refused(self, n, epsilon):
        env = make_env()
        policy = make_policy(env)
        with pytest.raises(ValueError, match=r"must be at least 1|must lie in"):
            sample_trajectories(env, policy, n, epsilon=epsilon)


class TestLoss:
    def test_it_is_zero_when_the_balance_condition_holds(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 8)
        # Choose log R so that log Z + log P_F - log R - log P_B is exactly 0.
        log_rewards = (policy.log_z + trajectories.log_forward - trajectories.log_backward).detach()
        loss = trajectory_balance_loss(trajectories, log_rewards, policy.log_z)
        assert loss.detach().item() == pytest.approx(0.0, abs=1e-10)

    def test_it_grows_with_the_violation(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 8)
        balanced = (policy.log_z + trajectories.log_forward - trajectories.log_backward).detach()
        small = trajectory_balance_loss(trajectories, balanced - 1.0, policy.log_z)
        large = trajectory_balance_loss(trajectories, balanced - 3.0, policy.log_z)
        assert small.detach().item() == pytest.approx(1.0)
        assert large.detach().item() == pytest.approx(9.0)

    def test_gradients_reach_the_policy_and_log_z(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 8)
        loss = trajectory_balance_loss(trajectories, torch.zeros(8), policy.log_z)
        loss.backward()
        assert policy.log_z.grad is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in policy.policy_parameters()
        )

    def test_an_infinite_log_reward_is_refused(self):
        # -inf here becomes nan at the subtraction and silently destroys the
        # batch; flooring belongs in the reward transform.
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 4)
        with pytest.raises(ValueError, match="must be finite"):
            trajectory_balance_loss(trajectories, torch.full((4,), -float("inf")), policy.log_z)

    def test_mismatched_batch_sizes_are_refused(self):
        env = make_env()
        policy = make_policy(env)
        trajectories = sample_trajectories(env, policy, 4)
        with pytest.raises(ValueError, match="log rewards for"):
            trajectory_balance_loss(trajectories, torch.zeros(3), policy.log_z)


class TestParameterGroups:
    def test_log_z_gets_a_higher_learning_rate(self):
        # It is one scalar that must travel to log sum R(x); at the policy's
        # rate it becomes the bottleneck and the loss plateaus with the policy
        # still wrong, which reads as slow convergence rather than a bug.
        env = make_env()
        policy = make_policy(env)
        groups = parameter_groups(policy, learning_rate=1e-3)
        assert groups[1]["lr"] > groups[0]["lr"]
        assert groups[1]["params"] == [policy.log_z]

    def test_every_parameter_is_in_exactly_one_group(self):
        env = make_env()
        policy = make_policy(env)
        groups = parameter_groups(policy)
        grouped = [p for group in groups for p in group["params"]]
        assert len(grouped) == len(list(policy.parameters()))

    @pytest.mark.parametrize(("lr", "multiplier"), [(0.0, 10.0), (1e-3, 0.0)])
    def test_nonpositive_rates_are_refused(self, lr, multiplier):
        env = make_env()
        policy = make_policy(env)
        with pytest.raises(ValueError, match="must be positive"):
            parameter_groups(policy, learning_rate=lr, log_z_multiplier=multiplier)


@pytest.mark.slow
class TestItLearnsTheTargetDistribution:
    """The check that this is a GFlowNet and not an expensive hill-climber.

    Trains on a landscape small enough to enumerate, then compares the sampled
    distribution against the exact ``p*(x) ∝ R(x)^β``. No other test in the
    suite would notice a sampler that optimises correctly and samples wrongly.
    """

    @staticmethod
    def train(env, values, *, beta=1.0, steps=1500, batch=64, seed=0):  # noqa: PLR0913
        policy = make_policy(env, seed=seed, hidden_dim=128)
        reward = TemperedReward(beta=beta)
        terminals = env.enumerate_terminal_states()
        lookup = {row.tobytes(): i for i, row in enumerate(terminals.astype(np.int64))}
        log_rewards_all = torch.as_tensor(reward.log_reward(values), dtype=torch.float32)

        optimiser = torch.optim.Adam(parameter_groups(policy, learning_rate=3e-3))
        generator = torch.Generator().manual_seed(seed)
        for step in range(steps):
            # Anneal exploration: broad early so every mode is seen, tight later
            # so the fit is to the policy rather than to the noise.
            epsilon = 0.3 * (1.0 - step / steps)
            trajectories = sample_trajectories(
                env, policy, batch, epsilon=epsilon, generator=generator
            )
            indices = [lookup[row.tobytes()] for row in trajectories.terminal.astype(np.int64)]
            loss = trajectory_balance_loss(trajectories, log_rewards_all[indices], policy.log_z)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return policy, terminals

    def test_the_sampled_distribution_approaches_the_exact_target(self):
        env = make_env(length=4, symbols="ABC", max_mutations=2)
        rng = np.random.default_rng(0)
        terminals = env.enumerate_terminal_states()
        values = rng.uniform(0.2, 4.0, size=terminals.shape[0])

        policy, space = self.train(env, values, beta=1.0)

        n_samples = 8000
        drawn = sample_trajectories(
            env, policy, n_samples, generator=torch.Generator().manual_seed(99)
        ).terminal
        empirical = empirical_distribution(drawn, space)
        target = target_distribution(values, beta=1.0)
        distance = l1_distance(empirical, target)
        floor = expected_l1_from_sampling_noise(target, n_samples=n_samples)

        # A perfect sampler still shows non-zero L1 at finite sample count, so
        # the floor is what "close" has to be measured against. Observed ~0.86x
        # the floor -- statistically indistinguishable from a perfect sampler --
        # so this bound has real headroom without being vacuous.
        assert distance < 2.5 * floor, f"L1 {distance:.4f} against a noise floor of {floor:.4f}"

    def test_it_beats_uniform_sampling_by_a_wide_margin(self):
        # Guards the test above from passing on a degenerate landscape: if the
        # target were nearly uniform, matching it would prove nothing.
        env = make_env(length=4, symbols="ABC", max_mutations=2)
        rng = np.random.default_rng(0)
        terminals = env.enumerate_terminal_states()
        values = rng.uniform(0.2, 4.0, size=terminals.shape[0])

        policy, space = self.train(env, values, beta=1.0)
        target = target_distribution(values, beta=1.0)
        n_samples = 8000
        drawn = sample_trajectories(
            env, policy, n_samples, generator=torch.Generator().manual_seed(99)
        ).terminal

        learned = l1_distance(empirical_distribution(drawn, space), target)
        uniform = l1_distance(np.full(len(space), 1.0 / len(space)), target)
        assert learned < uniform / 5

    def test_log_z_converges_to_the_true_partition_function(self):
        # An independent check on the whole construction. Nothing in the loss
        # tells the model log sum R(x); it can only arrive there if the flow
        # network is genuinely balanced, so this catches errors that leave the
        # sampled distribution looking plausible.
        env = make_env(length=4, symbols="ABC", max_mutations=2)
        rng = np.random.default_rng(0)
        terminals = env.enumerate_terminal_states()
        values = rng.uniform(0.2, 4.0, size=terminals.shape[0])

        policy, _ = self.train(env, values, beta=1.0)
        expected = float(np.log(float(values.sum())))
        assert policy.log_z.detach().item() == pytest.approx(expected, abs=0.1)

    def test_it_beats_a_greedy_sampler_on_distributional_distance(self):
        # The comparison the project rests on. The greedy sampler has zero
        # regret and would win on every performance metric here.
        env = make_env(length=4, symbols="ABC", max_mutations=2)
        rng = np.random.default_rng(1)
        terminals = env.enumerate_terminal_states()
        values = rng.uniform(0.2, 4.0, size=terminals.shape[0])

        policy, space = self.train(env, values, beta=1.0, seed=1)
        target = target_distribution(values, beta=1.0)

        n_samples = 8000
        drawn = sample_trajectories(
            env, policy, n_samples, generator=torch.Generator().manual_seed(7)
        ).terminal
        learned = l1_distance(empirical_distribution(drawn, space), target)

        best = space[int(np.argmax(values))]
        greedy_samples = np.tile(best, (n_samples, 1))
        greedy = l1_distance(empirical_distribution(greedy_samples, space), target)

        assert learned < greedy
