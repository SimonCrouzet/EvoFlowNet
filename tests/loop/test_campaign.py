"""Tests for the campaign round engine.

The properties under test are budget properties. A campaign that quietly spends
more oracle calls than it was given produces numbers that look like results and
are not, and no assertion about fitness would catch it -- so the accounting is
tested harder than the optimisation.
"""

import numpy as np
import pytest

from evoflownet.acquisition import DiverseTopK, ExpectedImprovement, Greedy, TopK
from evoflownet.algorithms.base import Sampler
from evoflownet.core.types import Alphabet
from evoflownet.landscapes.base import FitnessLandscape
from evoflownet.loop import Campaign
from evoflownet.surrogate import DeepEnsemble

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
