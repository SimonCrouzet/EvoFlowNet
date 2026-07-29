"""Tests for separable CMA-ES over the sequence relaxation.

Three things can silently break here and none of them shows up as an exception.
The argmax decoding can leave the mutation budget, in which case the sampler is
being scored on designs no other method is allowed to propose. The update can be
attributed to the wrong sample, because the harness scores a selected *subset* of
a round's proposals rather than all of it in order, which turns the covariance
update into noise. And the step size can run away to an infinity on a round where
everything scored -inf. Each has its own test.
"""

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines import CMAES
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment


def make_env(length=6, symbols="ABCD", max_mutations=3, transitions=None):
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


def unrelated_designs(env, proposals, count):
    """Reachable sequences that are definitely not among ``proposals``."""
    rng = np.random.default_rng(99)
    known = {row.tobytes() for row in np.ascontiguousarray(proposals)}
    length, budget = env.sequence_length, env.max_mutations
    picked: list[np.ndarray] = []
    while len(picked) < count:
        candidate = np.zeros(length, dtype=np.int32)
        positions = rng.choice(length, size=budget, replace=False)
        candidate[positions] = rng.integers(1, env.alphabet.size, size=budget)
        if candidate.tobytes() not in known:
            picked.append(candidate)
    return np.stack(picked)


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(CMAES(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert CMAES(env).propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        env = make_env(max_mutations=2)
        sampler = CMAES(env, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_proposals_are_counted(self):
        sampler = CMAES(make_env())
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        assert np.array_equal(CMAES(env, seed=4).propose(16), CMAES(env, seed=4).propose(16))

    def test_a_whole_campaign_is_reproducible(self):
        env = make_env(length=8, max_mutations=5)
        runs = []
        for _ in range(2):
            sampler = CMAES(env, seed=11)
            batches = []
            for _ in range(4):
                proposals = sampler.propose(24)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])

    def test_the_label_says_whether_feasibility_is_enforced(self):
        env = make_env()
        assert CMAES(env).name == "CMAES"
        assert CMAES(env, feasible_only=True).name == "CMAES (feasible)"


class TestTheRelaxation:
    def test_the_covariance_is_diagonal(self):
        # A full covariance over length * vocabulary would be 26 million entries
        # at L = 256, and its eigendecomposition is what makes it intractable.
        env = make_env(length=6, symbols="ABCD")
        sampler = CMAES(env)
        assert sampler._diagonal.shape == (6 * 4,)

    def test_the_mean_is_one_logit_per_position_and_token(self):
        env = make_env(length=6, symbols="ABCD")
        assert CMAES(env).mean_logits.shape == (6, 4)

    def test_decoding_projects_onto_the_mutation_budget(self):
        # The argmax of an unconstrained Gaussian differs from the parent at
        # nearly every position; without the projection the sampler would emit
        # designs outside the graph on its very first round.
        env = make_env(length=12, symbols="ABCD", max_mutations=2)
        proposals = CMAES(env, seed=0).propose(64)
        assert (proposals != env.parent[None, :]).sum(axis=1).max() <= 2

    def test_the_distribution_moves_when_the_ranking_is_informative(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        before = sampler.mean_logits.copy()
        proposals = sampler.propose(64)
        sampler.observe(proposals, toy_landscape(proposals))
        assert not np.allclose(before, sampler.mean_logits)

    def test_it_learns_to_prefer_the_rewarded_token(self):
        # The relaxation is only doing its job if the logit for the token the
        # landscape rewards ends up above the others.
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        sampler = CMAES(env, seed=0)
        for _ in range(20):
            proposals = sampler.propose(64)
            sampler.observe(proposals, toy_landscape(proposals))
        logits = sampler.mean_logits
        assert logits[:, 1].mean() > logits[:, [0, 2, 3]].mean()

    def test_it_improves_over_rounds(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        first = last = 0.0
        for index in range(20):
            proposals = sampler.propose(64)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first


class TestAttributingScoresToDraws:
    """The harness scores a selected subset, in its own order.

    CMA-ES updates from the Gaussian draw behind each ranked sample, so lining
    scores up by row index -- which is what a naive implementation does -- would
    attribute every score to some other candidate's draw. The distribution would
    still move, and it would still look like it was working.
    """

    def test_a_subset_in_a_different_order_still_updates(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        chosen = proposals[::-1][:16]
        before = sampler.mean_logits.copy()
        sampler.observe(chosen, toy_landscape(chosen))
        assert not np.allclose(before, sampler.mean_logits)

    def test_sequences_it_never_proposed_are_ignored(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        outsiders = unrelated_designs(env, proposals, count=8)
        before = sampler.mean_logits.copy()
        sampler.observe(outsiders, np.full((8, 1), 5.0))
        assert np.allclose(before, sampler.mean_logits)

    def test_a_single_usable_measurement_does_not_move_the_distribution(self):
        # One sample is a ranking of one: the rank-mu term is empty and the only
        # effect left is an inflated step size.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        before = sampler.mean_logits.copy()
        sampler.observe(proposals[:1], np.array([[3.0]]))
        assert np.allclose(before, sampler.mean_logits)


class TestNumericalRobustness:
    def test_a_round_of_failed_assays_leaves_the_search_usable(self):
        # -inf is what an infeasible design scores, and a whole round of them is
        # routine on a sparse feasible set. A NaN in the covariance here would
        # silently poison every later round.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        for _ in range(5):
            proposals = sampler.propose(32)
            sampler.observe(proposals, np.full((32, 1), -np.inf))
        assert np.isfinite(sampler.sigma)
        assert sampler.sigma > 0.0
        assert np.isfinite(sampler.mean_logits).all()

    def test_the_step_size_stays_finite_under_a_constant_ranking(self):
        # Every candidate scoring the same is the degenerate case for a rank
        # based method: the weights are applied to an arbitrary order.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        for _ in range(30):
            proposals = sampler.propose(32)
            sampler.observe(proposals, np.zeros((32, 1)))
        assert np.isfinite(sampler.sigma)
        assert np.isfinite(sampler._diagonal).all()
        assert (sampler._diagonal > 0.0).all()


class TestFeasibility:
    def test_a_rejecting_sampler_only_emits_constructible_designs(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        sampler = CMAES(env, feasible_only=True, seed=0)
        for _ in range(3):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_rejection_costs_proposals_rather_than_oracle_calls(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        plain = CMAES(env, seed=0)
        rejecting = CMAES(env, feasible_only=True, seed=0)
        plain.propose(32)
        rejecting.propose(32)
        assert rejecting.proposals_made > plain.proposals_made

    def test_an_impossible_constraint_raises_rather_than_returning_junk(self):
        forbidden = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
        env = make_env(
            length=8,
            symbols="ABCD",
            max_mutations=3,
            transitions=constrained_transitions(4, forbidden),
        )
        with pytest.raises(RuntimeError, match="feasible"):
            CMAES(env, feasible_only=True, max_attempts=3, seed=0).propose(32)


class TestValidation:
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_non_positive_step_size_is_refused(self, value):
        with pytest.raises(ValueError, match="initial_sigma must be positive"):
            CMAES(make_env(), initial_sigma=value)
