"""Tests for the GFlowNet behind the sampler interface.

The property that matters most is negative: training must not reach the oracle.
It is the kind of error that raises nothing and simply makes the method look
sample-inefficient, so it is asserted from both sides -- the oracle counts its
own calls, and the sampler counts the proxy's.

The second such property is what a move costs. Re-anchoring resolves the
sampler's own hook before the campaign's factory, and the factory's rebuild is
correct but restarts the accounting -- so the tests below assert that the
policy, the random stream and the counts all cross a move, and that proposals
after it land in the new anchor's ball rather than the old one's.
"""

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines import GeneticAlgorithm
from evogfn.algorithms.gflownet import GeneticConfig, GFlowNetSampler, TrainingConfig
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.loop import Campaign
from evogfn.loop.campaign import ReanchorableSampler
from evogfn.models.policy import SequencePolicy
from evogfn.rewards import TemperedReward
from evogfn.surrogate import DeepEnsemble, ProxyLandscape

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


def fit(surrogate, seed=0):
    """Give the proxy something to train against, as a measured round would."""
    rng = np.random.default_rng(seed)
    train = rng.integers(0, ALPHABET.size, size=(32, LENGTH))
    surrogate.fit(train, (train == 1).sum(axis=1, keepdims=True).astype(float))


def anchor():
    """A design the environment can re-anchor at, two mutations from the parent."""
    moved = PARENT.copy()
    moved[:2] = 1
    return moved


def snapshot(policy):
    """Every parameter of the policy, detached from it."""
    return {name: value.detach().clone().numpy() for name, value in policy.named_parameters()}


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


class TestReanchoring:
    """What the sampler carries when the campaign moves its anchor.

    The campaign's fallback is a factory rebuild, which keeps the policy -- the
    factory closes over it -- and drops the sampler's own accounting. That is a
    silent loss: ``proxy_calls`` is a printed column beside the oracle budget, so
    a count that restarts at each anchor reports the last anchor's rounds as the
    whole campaign's compute and understates exactly the method under test.
    """

    def test_the_sampler_satisfies_the_campaign_protocol(self, parts):
        # Without this the campaign silently takes the factory path instead, and
        # nothing raises, because rebuilding is a legitimate outcome.
        assert isinstance(build(parts), ReanchorableSampler)

    def test_the_trained_weights_survive_the_move(self, parts):
        # The action space is length x |alphabet| + 1 and the policy reads the
        # state sequence, so nothing the network holds is expressed relative to
        # the parent. A move that reset the weights would throw away every round
        # of training the campaign had paid for.
        env, policy, surrogate, _ = parts
        fit(surrogate)
        sampler = build(parts)
        sampler.propose(8)
        trained = snapshot(policy)
        moved = sampler.reanchored(env.reanchored(anchor()))
        carried = snapshot(moved._policy)
        assert set(carried) == set(trained)
        assert all(np.array_equal(trained[name], carried[name]) for name in trained)
        assert moved._policy is policy

    def test_the_proxy_spend_keeps_accumulating(self, parts):
        # The bug this hook exists for. A campaign total that restarts at each
        # anchor undercounts the arm's compute by however often it re-anchored,
        # in the column that reports what the method cost.
        env, _, surrogate, _ = parts
        fit(surrogate)
        sampler = build(parts)
        sampler.propose(8)
        spent = sampler.proxy_calls
        assert spent > 0

        moved = sampler.reanchored(env.reanchored(anchor()))
        assert moved.proxy_calls == spent
        moved.propose(8)
        assert moved.proxy_calls > spent
        assert moved.rounds_trained == 2

    def test_a_re_anchoring_campaign_reports_the_whole_campaigns_spend(self, parts):
        # End to end, and the number to compare is the one asserted by
        # TestBudgetSeparation with re-anchoring off: the same 80. Down the
        # factory path the campaign returns a rebuilt sampler whose counters
        # start at the last anchor, so this would read 40 -- or 0 -- and the
        # results table would print half of what the arm spent.
        env, _, surrogate, _ = parts
        sampler = build(parts)
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=sampler,
            surrogate=surrogate,
            rounds=3,
            batch_size=8,
            pool_size=64,
            environment=env,
            reanchor=True,
        )
        campaign.run()
        moved = campaign.sampler
        assert isinstance(moved, GFlowNetSampler)
        # The hook ran, so the campaign is reporting a sampler it was handed
        # rather than the one it was constructed with.
        assert moved is not sampler
        assert moved.proxy_calls == 80
        assert moved.rounds_trained == 2

    def test_proposals_land_inside_the_new_anchors_ball(self, parts):
        # The point of moving at all. A sampler still masking against the old
        # parent would return plausible designs the campaign's ledger attributes
        # to an anchor that cannot build them.
        env, _, _, _ = parts
        target = env.reanchored(anchor())
        moved = build(parts).reanchored(target)
        assert target.is_reachable(moved.propose(32)).all()

    def test_the_proposal_count_survives_the_move(self, parts):
        env, _, _, _ = parts
        sampler = build(parts)
        sampler.propose(12)
        assert sampler.reanchored(env.reanchored(anchor())).proposals_made == 12

    def test_the_next_round_trains_from_a_fresh_seed(self, parts):
        # rounds_trained is also the per-round training seed offset. Resetting it
        # would make the round after a move replay round zero's trajectories at
        # full compute and teach the policy nothing.
        env, _, surrogate, _ = parts
        fit(surrogate)
        sampler = build(parts)
        sampler.propose(8)
        moved = sampler.reanchored(env.reanchored(anchor()))
        assert moved.rounds_trained == sampler.rounds_trained == 1

    def test_the_sampling_stream_carries_rather_than_restarting(self, parts):
        # A restarted stream re-draws the designs the campaign has already
        # measured, which the deduplication then screens out -- so the round is
        # spent on proposals that cannot be selected.
        env, _, _, _ = parts
        sampler = build(parts)
        first = sampler.propose(16)
        moved = sampler.reanchored(env.reanchored(PARENT.copy()))
        assert not np.array_equal(first, moved.propose(16))

    def test_the_genetic_teacher_moves_with_the_sampler(self, parts):
        # A teacher left at the old anchor reverts surplus mutations to a parent
        # nobody is searching from, and the trainer then filters its offspring
        # out against the new environment -- so Genetic-GFN quietly degrades to
        # plain trajectory balance while still reporting itself as Genetic-GFN.
        env, _, _, _ = parts
        target = env.reanchored(anchor())
        sampler = build(parts, genetic=GeneticAlgorithm(env, population_size=8, seed=0))
        moved = sampler.reanchored(target)
        teacher = moved._genetic
        assert teacher is not None
        assert teacher is not sampler._genetic
        assert target.is_reachable(teacher.propose(16)).all()

    def test_it_refuses_an_environment_the_policy_is_not_sized_for(self, parts):
        # Reshaping instead would leave the policy emitting logits for actions
        # that do not exist, and it would keep proposing designs.
        sampler = build(parts)
        wider = MutationEnvironment(np.zeros(LENGTH + 2, dtype=np.int32), ALPHABET, max_mutations=2)
        with pytest.raises(ValueError, match="the anchor may move"):
            sampler.reanchored(wider)


class TestTheReachabilityGap:
    """How much of what the teacher bred the policy could be trained on.

    A design can satisfy the environment's constraint and sit inside the
    mutation budget and still have no ordering of its mutations along which
    every intermediate is feasible; replay finds no path to it and drops the
    row. That share is a property of the landscape rather than of this sampler,
    which is why it is accumulated here for the record to store instead of being
    logged and forgotten.

    Reported as two numbers rather than one share, because a share of no designs
    and a share of thousands with no gap are both ``0.0`` and only one of them is
    a measurement.
    """

    def guided(self, parts):
        env, _, surrogate, _ = parts
        fit(surrogate)
        return build(
            parts,
            genetic=GeneticAlgorithm(env, population_size=8, seed=0),
            # The default warmup outlasts this five-step retrain, so without
            # this the teacher would never be consulted and every count below
            # would be zero for a reason that has nothing to do with the gap.
            genetic_config=GeneticConfig(offspring=8, warmup=0),
        )

    def test_a_run_without_a_teacher_breeds_nothing_to_count(self, parts):
        fit(parts[2])
        sampler = build(parts)
        sampler.propose(8)
        assert sampler.bred_designs == 0
        assert sampler.unconstructible_fraction == 0.0

    def test_an_unconstrained_lattice_has_no_gap_to_report(self, parts):
        # Without a transition constraint every ordering of a mutation set is
        # legal, so nothing bred inside the budget can fail to be constructed. A
        # share above zero here would mean the counter is measuring some other
        # reason for a row to be dropped.
        sampler = self.guided(parts)
        sampler.propose(8)
        assert sampler.bred_designs > 0
        assert sampler.unconstructible_designs == 0
        assert sampler.unconstructible_fraction == 0.0

    def test_it_accumulates_over_rounds_rather_than_reporting_the_last(self, parts):
        # The campaign's share is what a record stores; a per-round count would
        # make it the last retrain's.
        sampler = self.guided(parts)
        sampler.propose(8)
        after_one_round = sampler.bred_designs
        sampler.propose(8)
        assert after_one_round > 0
        assert sampler.bred_designs > after_one_round

    def test_the_counts_carry_across_a_move(self, parts):
        # Same argument as `proxy_calls`, only sharper: the gap is a property of
        # the ball being searched, so a move is exactly when it changes, and
        # counts restarted at each anchor would report the last ball's gap as
        # the whole campaign's.
        env = parts[0]
        sampler = self.guided(parts)
        sampler.propose(8)
        moved = sampler.reanchored(env.reanchored(anchor()))
        assert moved.bred_designs == sampler.bred_designs > 0
        assert moved.unconstructible_designs == sampler.unconstructible_designs


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
