"""Tests for the campaign round engine.

The properties under test are budget properties. A campaign that quietly spends
more oracle calls than it was given produces numbers that look like results and
are not, and no assertion about fitness would catch it -- so the accounting is
tested harder than the optimisation.
"""

import itertools

import numpy as np
import pytest

from evogfn.acquisition import DiverseTopK, ExpectedImprovement, Greedy, TopK
from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.mutagenesis import RandomMutagenesis
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.loop import Campaign
from evogfn.surrogate import DeepEnsemble

ALPHABET = Alphabet.from_string("ABCD")
LENGTH = 6


class CountingLandscape(FitnessLandscape):
    """Counts every evaluation, so the budget can be checked from the outside."""

    def __init__(self, *, infeasible_token=None):
        self.calls = 0
        self._infeasible_token = infeasible_token

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
        value = (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)
        if self._infeasible_token is not None:
            blocked = (sequences == self._infeasible_token).any(axis=1)
            value[blocked] = -np.inf
        return value


class RandomSampler(Sampler):
    """Uniform proposals. Learns nothing, which makes it a clean control."""

    def __init__(self, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def propose(self, n):
        self._count(n)
        return self._rng.integers(0, ALPHABET.size, size=(n, LENGTH), dtype=np.int32)


class CollapsedSampler(Sampler):
    """Proposes one sequence over and over -- the mode-collapse failure mode."""

    def propose(self, n):
        self._count(n)
        return np.ones((n, LENGTH), dtype=np.int32)


def surrogate():
    return DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=30, seed=0)


class TestBudget:
    def test_the_oracle_is_called_exactly_the_budget(self):
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == 24
        assert result.oracle_calls == 24

    def test_the_ledger_agrees_with_the_landscape(self):
        # Two independent counts of the same thing. If the ledger could drift
        # from the oracle, every budget-indexed claim in the paper would be
        # unfalsifiable from inside the repo.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=5,
            pool_size=40,
        ).run()
        assert result.oracle_calls == landscape.calls == len(result.sequences)

    def test_training_the_sampler_is_not_charged(self):
        # The structural error this loop exists to prevent. The sampler generates
        # a 512-candidate pool per round; only the selected batch is measured.
        landscape = CountingLandscape()
        sampler = RandomSampler()
        result = Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=512,
        ).run()
        assert landscape.calls == 24
        assert sampler.proposals_made > 1000
        assert result.proposals > result.oracle_calls

    def test_the_budget_is_rounds_times_batch(self):
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=4,
            batch_size=96,
            pool_size=1024,
        )
        assert campaign.budget == 384

    def test_the_default_budget_matches_real_campaigns(self):
        # 4 x 96 = 384. ALDE screened 396 as six 96-well plates over three
        # rounds; LaMBO-2's wet lab measured 374. The 1,000-10,000 evaluations
        # common in iterative benchmarks is a different experiment.
        campaign = Campaign(landscape=CountingLandscape(), sampler=RandomSampler())
        assert campaign.budget == 384

    def test_a_pool_smaller_than_the_batch_is_refused(self):
        with pytest.raises(ValueError, match="nothing to select from"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(),
                batch_size=96,
                pool_size=32,
            )

    @pytest.mark.parametrize(
        "build",
        [
            lambda **kw: Campaign(rounds=0, pool_size=64, batch_size=8, **kw),
            lambda **kw: Campaign(batch_size=0, pool_size=64, **kw),
            lambda **kw: Campaign(pool_size=0, batch_size=8, **kw),
        ],
        ids=["rounds", "batch_size", "pool_size"],
    )
    def test_non_positive_sizes_are_refused(self, build):
        with pytest.raises(ValueError, match="at least 1"):
            build(landscape=CountingLandscape(), sampler=RandomSampler())


class TestRounds:
    def test_every_round_is_recorded(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=6,
            pool_size=48,
        ).run()
        assert [record.index for record in result.rounds] == [0, 1, 2]

    def test_best_so_far_never_decreases(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=64,
        ).run()
        trace = result.trace()
        assert trace == sorted(trace)

    def test_the_first_round_runs_without_a_surrogate(self):
        # There is nothing to fit it on. A campaign that tried would fail on the
        # empty-data check the surrogate raises.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 8

    def test_a_supplied_initial_design_is_measured_first(self):
        # Distinct rows: deduplication applies to round 0 too, so a design of
        # four identical sequences would correctly collapse to one measurement.
        design = np.stack([np.full(LENGTH, i, dtype=np.int32) for i in range(4)])
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=4,
            pool_size=32,
            initial_design=design,
        ).run()
        assert np.array_equal(result.sequences[:4], design)


class TestDeduplication:
    def test_a_collapsed_sampler_stops_rather_than_re_measuring(self):
        # A sampler proposing one sequence forever would otherwise spend the
        # entire budget measuring it repeatedly, and report a full ledger.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=CollapsedSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls < 32
        assert len({row.tobytes() for row in result.sequences}) == len(result.sequences)

    def test_nothing_is_measured_twice(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=64,
        ).run()
        unique = {row.tobytes() for row in np.ascontiguousarray(result.sequences)}
        assert len(unique) == len(result.sequences)

    def test_round_zero_deduplicates_too(self):
        # A collapsed sampler's opening plate is one experiment, not eight.
        landscape = CountingLandscape()
        Campaign(
            landscape=landscape,
            sampler=CollapsedSampler(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == 1

    def test_deduplication_can_be_turned_off(self):
        landscape = CountingLandscape()
        Campaign(
            landscape=landscape,
            sampler=CollapsedSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
            skip_measured=False,
        ).run()
        assert landscape.calls == 24


class TestFeasibility:
    def test_infeasible_designs_are_charged(self):
        # They cost the same to build. Not charging for them would make a
        # method that proposes unbuildable constructs look free, which is
        # exactly the comparison the masked sampler is meant to win.
        landscape = CountingLandscape(infeasible_token=3)
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24
        assert result.feasible_fraction < 1.0

    def test_a_fully_feasible_campaign_reports_one(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.feasible_fraction == pytest.approx(1.0)

    def test_infeasible_values_do_not_poison_the_best(self):
        landscape = CountingLandscape(infeasible_token=3)
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert np.isfinite(result.best_value)


class TestSurrogateEffect:
    def test_the_surrogate_improves_on_the_unassisted_sampler(self):
        # The point of the whole apparatus. Same sampler, same budget, same
        # seed -- the only difference is whether proposals are screened.
        blind = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=1),
            rounds=4,
            batch_size=8,
            pool_size=256,
        ).run()
        guided = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=1),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=256,
        ).run()
        assert guided.best_value >= blind.best_value

    def test_it_runs_without_a_surrogate_at_all(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24

    @pytest.mark.parametrize("rule", [Greedy(), ExpectedImprovement()])
    @pytest.mark.parametrize("selector", [TopK(), DiverseTopK(penalty=1.0)])
    def test_every_rule_and_selector_combination_runs(self, rule, selector):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            acquisition=rule,
            selector=selector,
            rounds=2,
            batch_size=6,
            pool_size=48,
        ).run()
        assert result.oracle_calls == 12


class TestResult:
    def test_regret_is_measured_against_the_true_optimum(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.simple_regret == pytest.approx(LENGTH - result.best_value)

    def test_regret_is_none_when_the_optimum_is_unknown(self):
        class Unknowable(CountingLandscape):
            @property
            def optimum(self):
                return None

        result = Campaign(
            landscape=Unknowable(),
            sampler=RandomSampler(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.simple_regret is None
        assert "simple_regret" not in result.summary()

    def test_the_summary_carries_the_budget(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=2,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.summary()["oracle_calls"] == 16

    def test_the_rejection_ratio_exposes_wasted_generation(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=4,
            pool_size=256,
        ).run()
        assert result.rounds[-1].rejection_ratio > 10


class BallSampler(Sampler):
    """Proposes everything its environment can build, in enumeration order.

    Deterministic on purpose. Re-anchoring is a statement about *which designs
    are reachable*, and a stochastic sampler would blur that into a question
    about how lucky the draw was.
    """

    def __init__(self, env):
        super().__init__()
        self._env = env

    def propose(self, n):
        states = self._env.reachable_terminal_states()
        self._count(states.shape[0])
        return states[:n]

    def reanchored(self, env):
        return BallSampler(env)


def mutation_env(max_mutations=2, parent=None, transitions=None):
    return MutationEnvironment(
        np.zeros(LENGTH, dtype=np.int32) if parent is None else parent,
        ALPHABET,
        max_mutations=max_mutations,
        transitions=transitions,
    )


def small_ehrlich():
    """A toy Ehrlich whose optimum is two mutations out, with a budget of one.

    Returns:
        The landscape and a feasible wild type. Chosen so the planted optimum is
        further than one round's mutation budget and well within the campaign's
        total, which is the regime the benchmark's real tasks are in -- 61 to 248
        mutations away against a per-round budget of four.
    """
    landscape = EhrlichLandscape(
        sequence_length=8,
        vocab_size=4,
        n_motifs=2,
        motif_length=2,
        quantization=2,
        max_spacing=2,
        transition_density=0.7,
        seed=0,
    )
    return landscape, landscape.feasible_sequence(seed=0)


class TestReanchoringIsOffByDefault:
    """Nothing already measured moves because this mechanism was added."""

    def test_a_campaign_without_an_environment_is_unchanged(self):
        # Bit-identical against a fixed seed. The numbers are hard-coded rather
        # than compared to a second run, so that a change to the loop cannot
        # move both sides together and still pass.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=7),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24
        assert result.best_value == 4.0
        assert result.trace() == [4.0, 4.0, 4.0]
        assert result.sequences[0].tolist() == [3, 2, 2, 3, 2, 3]
        assert int(result.sequences.sum()) == 232

    def test_supplying_an_environment_alone_changes_no_measurement(self):
        def run(**extra):
            return Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(seed=7),
                rounds=3,
                batch_size=8,
                pool_size=64,
                **extra,
            ).run()

        plain = run()
        watched = run(environment=mutation_env())
        assert np.array_equal(plain.sequences, watched.sequences)
        assert np.array_equal(plain.values, watched.values)
        assert plain.trace() == watched.trace()

    def test_a_fixed_anchor_never_leaves_the_wild_type(self):
        # The failure this mechanism exists to fix, stated as a measurement: with
        # the anchor held still, every round searches the same Hamming ball.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(mutation_env(max_mutations=2)),
            rounds=4,
            batch_size=16,
            pool_size=256,
            environment=mutation_env(max_mutations=2),
        ).run()
        assert result.anchor_trace() == [0, 0, 0, 0]
        assert result.best_value <= 2.0


class TestReanchoringMovesTheSearch:
    def test_cumulative_distance_outgrows_the_per_round_budget(self):
        # The property the whole mechanism exists for. Two mutations per round
        # is the budget; after four rounds the search is standing further out
        # than that, which one fixed environment can never do.
        env = mutation_env(max_mutations=2)
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=4,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        )
        result = campaign.run()
        assert result.anchor_trace() == [0, 1, 2, 3]
        assert max(result.anchor_trace()) > env.max_mutations
        moved = campaign.environment
        assert moved is not None
        assert moved.parent.tolist() == [1, 1, 1, 1, 0, 0]

    def test_it_reaches_fitness_a_fixed_anchor_cannot(self):
        def run(reanchor):
            env = mutation_env(max_mutations=2)
            return Campaign(
                landscape=CountingLandscape(),
                sampler=BallSampler(env),
                rounds=4,
                batch_size=16,
                pool_size=256,
                environment=env,
                reanchor=reanchor,
            ).run()

        assert run(reanchor=True).best_value > run(reanchor=False).best_value

    def test_the_ledger_names_the_design_each_round_started_from(self):
        env = mutation_env(max_mutations=2)
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=3,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        ).run()
        anchors = [record.anchor for record in result.rounds]
        assert anchors[0] == (0, 0, 0, 0, 0, 0)
        # Each anchor is a design that was actually measured, not a construction.
        measured = {tuple(int(t) for t in row) for row in result.sequences}
        assert all(anchor in measured for anchor in anchors[1:])

    def test_the_anchor_only_moves_on_an_improvement(self):
        # A round that learns nothing must not walk the search off a peak it
        # has already found.
        env = mutation_env(max_mutations=2)
        result = Campaign(
            landscape=CountingLandscape(infeasible_token=None),
            sampler=BallSampler(env),
            rounds=6,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        ).run()
        distances = result.anchor_trace()
        assert distances == sorted(distances)
        assert all(
            second - first <= env.max_mutations for first, second in itertools.pairwise(distances)
        )


class TestProposalsStayInsideTheMovedEnvironment:
    def test_every_measured_design_is_reachable_from_that_round_s_anchor(self):
        landscape, wild_type = small_ehrlich()
        env = MutationEnvironment(
            wild_type,
            landscape.alphabet,
            max_mutations=1,
            transitions=landscape.transition_matrix,
        )
        result = Campaign(
            landscape=landscape,
            sampler=RandomMutagenesis(env, feasible_only=True, seed=0),
            rounds=4,
            batch_size=12,
            pool_size=64,
            environment=env,
            reanchor=True,
        ).run()

        start = 0
        for record in result.rounds:
            batch = result.sequences[start : start + record.evaluated]
            start += record.evaluated
            anchored = MutationEnvironment(
                np.array(record.anchor, dtype=np.int32),
                landscape.alphabet,
                max_mutations=1,
                transitions=landscape.transition_matrix,
            )
            assert anchored.is_reachable(batch).all()
            assert landscape.is_feasible(batch).all()
        # Without this the assertions above would hold vacuously on a campaign
        # that never moved: every round would be checked against the wild type,
        # which is the environment the sampler was built on anyway. The sampler
        # follows the anchor through its own `reanchored` hook rather than
        # through a factory, which is the path a real campaign takes now that
        # every baseline implements one.
        assert max(result.anchor_trace()) > 0


class TestReanchoringOnEhrlich:
    """The test that says the mechanism matters rather than merely runs."""

    def test_the_optimum_is_out_of_reach_of_one_round_and_inside_the_campaign(self):
        # The guard on the instance itself. Without this the comparison below
        # could pass on a landscape where re-anchoring was never needed.
        landscape, wild_type = small_ehrlich()
        transitions = landscape.transition_matrix
        one_round = MutationEnvironment(
            wild_type, landscape.alphabet, max_mutations=1, transitions=transitions
        )
        whole_campaign = MutationEnvironment(
            wild_type, landscape.alphabet, max_mutations=4, transitions=transitions
        )
        within_one = landscape.evaluate(one_round.reachable_terminal_states()).max()
        within_four = landscape.evaluate(whole_campaign.reachable_terminal_states()).max()
        assert within_one < within_four == 1.0

    def test_re_anchoring_reaches_strictly_better_fitness(self):
        landscape, wild_type = small_ehrlich()

        def run(reanchor):
            env = MutationEnvironment(
                wild_type,
                landscape.alphabet,
                max_mutations=1,
                transitions=landscape.transition_matrix,
            )
            return Campaign(
                landscape=landscape,
                sampler=BallSampler(env),
                rounds=4,
                batch_size=32,
                pool_size=64,
                environment=env,
                reanchor=reanchor,
            ).run()

        fixed = run(reanchor=False)
        moving = run(reanchor=True)
        assert moving.best_value > fixed.best_value
        assert moving.best_value == 1.0
        assert moving.simple_regret == 0.0
        assert max(moving.anchor_trace()) > 1


class TestReanchoringIsRefusedWhenItCannotBeDone:
    def test_an_infeasible_anchor_is_refused(self):
        # The landscape scores this design finite; the environment cannot build
        # it. Anchoring there would void feasibility-by-construction for every
        # later round, silently.
        transitions = np.ones((ALPHABET.size, ALPHABET.size))
        transitions[1, 1] = 0.0
        env = mutation_env(max_mutations=2, transitions=transitions)
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=2,
            batch_size=4,
            pool_size=64,
            initial_design=np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int32),
            environment=env,
            reanchor=True,
        )
        with pytest.raises(ValueError, match="infeasible design"):
            campaign.run()

    def test_re_anchoring_without_an_environment_is_refused(self):
        with pytest.raises(ValueError, match="needs the environment"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=BallSampler(mutation_env()),
                reanchor=True,
            )

    def test_a_sampler_that_can_neither_be_told_nor_rebuilt_is_refused(self):
        # Refused at construction rather than after a round of oracle calls.
        with pytest.raises(ValueError, match="cannot follow a moved anchor"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(),
                environment=mutation_env(),
                reanchor=True,
            )

    def test_a_factory_is_enough_for_a_sampler_that_cannot_be_told(self):
        # RandomSampler deliberately has no `reanchored`, which is what makes
        # this the factory path. Every baseline in the package implements the
        # hook, so using one of those here would silently test the other branch.
        env = mutation_env(max_mutations=2)
        rebuilt = []

        def factory(moved):
            rebuilt.append(moved)
            return RandomSampler(seed=len(rebuilt))

        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=0),
            rounds=3,
            batch_size=8,
            pool_size=64,
            environment=env,
            reanchor=True,
            sampler_factory=factory,
        )
        result = campaign.run()
        assert max(result.anchor_trace()) > 0
        assert rebuilt, "the factory was never called, so nothing was rebuilt"
        assert campaign.sampler is not None
