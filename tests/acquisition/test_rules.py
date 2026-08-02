"""Tests for acquisition rules and batch selectors."""

import math

import numpy as np
import pytest

from evogfn.acquisition import (
    DiverseTopK,
    ExpectedImprovement,
    Greedy,
    ScalarizedAcquisition,
    Thompson,
    TopK,
    UpperConfidenceBound,
)
from evogfn.metrics import diversity
from evogfn.rewards.scalarization import Tchebycheff, WeightedSum


class TestAcquisitionRules:
    def test_greedy_returns_the_mean_untouched(self):
        mean = np.array([1.0, 3.0, 2.0])
        scored = Greedy().score(mean, np.array([9.0, 0.0, 5.0]), best_observed=0.0)
        assert scored == pytest.approx(mean)

    def test_greedy_ignores_uncertainty_entirely(self):
        mean = np.array([1.0, 2.0])
        first = Greedy().score(mean, np.zeros(2), best_observed=0.0)
        second = Greedy().score(mean, np.full(2, 100.0), best_observed=0.0)
        assert first == pytest.approx(second)

    def test_ucb_adds_a_weighted_spread(self):
        scored = UpperConfidenceBound(kappa=2.0).score(
            np.array([1.0, 1.0]), np.array([0.0, 0.5]), best_observed=0.0
        )
        assert scored == pytest.approx([1.0, 2.0])

    def test_the_published_kappa_is_the_default(self):
        # Jain et al. use 0.1 -- a notably small weight, close to greedy, and
        # worth knowing when reading results that depend on it.
        assert UpperConfidenceBound().kappa == pytest.approx(0.1)

    def test_ucb_at_the_published_kappa_barely_differs_from_greedy(self):
        # Measured behaviour, not an opinion: with typical ensemble spreads the
        # bonus is a rounding error next to differences in the mean.
        mean = np.array([1.0, 1.05, 0.9])
        std = np.array([0.05, 0.05, 0.4])
        greedy_order = np.argsort(-Greedy().score(mean, std, best_observed=0.0))
        ucb_order = np.argsort(-UpperConfidenceBound(kappa=0.1).score(mean, std, best_observed=0.0))
        assert greedy_order.tolist() == ucb_order.tolist()

    def test_a_negative_kappa_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            UpperConfidenceBound(kappa=-1.0)

    def test_expected_improvement_rewards_upside_over_the_incumbent(self):
        scored = ExpectedImprovement().score(
            np.array([0.0, 2.0]), np.array([1.0, 1.0]), best_observed=1.0
        )
        assert scored[1] > scored[0]

    def test_expected_improvement_prefers_spread_at_equal_mean(self):
        # With nothing to separate the means, the candidate that might surprise
        # you is worth more.
        scored = ExpectedImprovement().score(
            np.array([1.0, 1.0]), np.array([0.1, 2.0]), best_observed=1.0
        )
        assert scored[1] > scored[0]

    def test_expected_improvement_matches_the_closed_form(self):
        # Pinned against the analytic value rather than only against orderings.
        # The normal CDF is easy to refactor into an off-by-sqrt(2) version that
        # keeps every ranking test passing while the numbers are wrong.
        mean, std, incumbent = np.array([1.5]), np.array([2.0]), 1.0
        z = (mean[0] - incumbent) / std[0]
        expected = (mean[0] - incumbent) * 0.5 * (1 + math.erf(z / math.sqrt(2))) + std[
            0
        ] * math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        scored = ExpectedImprovement().score(mean, std, best_observed=incumbent)
        assert scored[0] == pytest.approx(expected)

    def test_expected_improvement_is_never_negative(self):
        scored = ExpectedImprovement().score(
            np.array([-5.0, -3.0]), np.array([0.1, 0.2]), best_observed=10.0
        )
        assert (scored >= 0).all()

    def test_thompson_draws_vary_between_calls(self):
        rule = Thompson(seed=0)
        mean, std = np.zeros(64), np.ones(64)
        assert not np.allclose(
            rule.score(mean, std, best_observed=0.0),
            rule.score(mean, std, best_observed=0.0),
        )

    def test_thompson_is_reproducible_from_a_seed(self):
        mean, std = np.zeros(32), np.ones(32)
        first = Thompson(seed=3).score(mean, std, best_observed=0.0)
        second = Thompson(seed=3).score(mean, std, best_observed=0.0)
        assert first == pytest.approx(second)

    def test_thompson_concentrates_where_the_model_is_confident(self):
        rule = Thompson(seed=0)
        draws = np.stack(
            [rule.score(np.zeros(2), np.array([0.01, 5.0]), best_observed=0.0) for _ in range(200)]
        )
        assert draws[:, 0].std() < draws[:, 1].std()


class TestScalarOnlyRulesRefuseObjectiveVectors:
    # The failure this exists to prevent is silent: an (n, 3) prediction handed
    # to UCB broadcasts into a well-formed ranking computed on nothing in
    # particular, and no field of the output says which objective won.
    @pytest.mark.parametrize(
        "rule", [Greedy(), UpperConfidenceBound(), ExpectedImprovement(), Thompson()]
    )
    def test_a_vector_valued_prediction_is_refused(self, rule):
        mean = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="ScalarizedAcquisition"):
            rule.score(mean, np.ones_like(mean), best_observed=0.0)

    @pytest.mark.parametrize(
        "rule", [Greedy(), UpperConfidenceBound(), ExpectedImprovement(), Thompson()]
    )
    def test_multi_objective_measurements_cannot_be_reduced(self, rule):
        with pytest.raises(ValueError, match="scalarised"):
            rule.reduce_objectives(np.zeros((4, 3)))

    def test_a_single_objective_column_is_still_accepted(self):
        assert Greedy().reduce_objectives(np.array([[1.0], [2.0]])).tolist() == [1.0, 2.0]

    def test_a_mismatched_spread_is_refused(self):
        with pytest.raises(ValueError, match="one spread per predicted value"):
            Greedy().score(np.zeros(3), np.zeros(4), best_observed=0.0)


class TestScalarizedAcquisition:
    def setup_method(self):
        self.rule = ScalarizedAcquisition(Greedy(), WeightedSum(), [0.5, 0.5])

    def test_it_declares_that_it_ranks_vectors(self):
        assert self.rule.supports_multi_objective
        assert not Greedy().supports_multi_objective

    def test_it_ranks_by_the_stated_trade_off(self):
        # Second design wins on the sum despite losing on the first objective.
        mean = np.array([[4.0, 0.0], [1.0, 5.0]])
        scored = self.rule.score(mean, np.zeros_like(mean), best_observed=0.0)
        assert scored == pytest.approx([2.0, 3.0])

    def test_a_lopsided_preference_moves_the_ranking(self):
        mean = np.array([[4.0, 0.0], [1.0, 5.0]])
        towards_first = ScalarizedAcquisition(Greedy(), WeightedSum(), [0.9, 0.1])
        scored = towards_first.score(mean, np.zeros_like(mean), best_observed=0.0)
        assert scored[0] > scored[1]

    def test_measurements_reduce_through_the_same_preference(self):
        # The property the whole design rests on: what the surrogate is fitted
        # to and what the rule ranks are the same function of the objectives.
        values = np.array([[4.0, 0.0], [1.0, 5.0]])
        assert self.rule.reduce_objectives(values) == pytest.approx([2.0, 3.0])

    def test_uncertainty_reaches_the_inner_rule(self):
        # A scalarisation of the means alone would leave an improvement rule
        # with no spread at all, and every candidate would look certain.
        rule = ScalarizedAcquisition(ExpectedImprovement(), WeightedSum(), [0.5, 0.5])
        mean = np.array([[1.0, 1.0], [1.0, 1.0]])
        std = np.array([[0.01, 0.01], [3.0, 3.0]])
        scored = rule.score(mean, std, best_observed=1.0)
        assert scored[1] > scored[0]

    def test_an_already_scalar_prediction_is_passed_straight_through(self):
        # A single-output surrogate has effectively been fitted to the
        # scalarised target already; applying the preference again would weight
        # a number that is not an objective vector.
        mean, std = np.array([1.0, 3.0]), np.array([0.0, 0.0])
        assert self.rule.score(mean, std, best_observed=0.0) == pytest.approx(mean)

    def test_it_works_with_a_non_linear_scalarisation(self):
        rule = ScalarizedAcquisition(Greedy(), Tchebycheff(), [0.5, 0.5])
        mean = np.array([[4.0, 4.0], [8.0, 0.0]])
        scored = rule.score(mean, np.zeros_like(mean), best_observed=0.0)
        # Tchebycheff scores the worst weighted objective, so the balanced
        # design wins outright where the weighted sum would tie them.
        assert scored[0] > scored[1]

    @pytest.mark.parametrize(
        ("preference", "message"),
        [
            ([0.5, 0.6], "sum to 1"),
            ([-0.5, 1.5], "non-negative"),
            ([[0.5, 0.5]], "shape"),
            ([np.inf, 0.0], "finite"),
        ],
    )
    def test_a_malformed_preference_is_refused(self, preference, message):
        with pytest.raises(ValueError, match=message):
            ScalarizedAcquisition(Greedy(), WeightedSum(), preference)

    def test_it_names_its_parts(self):
        assert "Greedy" in repr(self.rule)
        assert "WeightedSum" in repr(self.rule)


class TestBatchSelection:
    def setup_method(self):
        rng = np.random.default_rng(0)
        # A cluster of near-identical high scorers, plus scattered lower ones.
        cluster = np.tile(np.array([1, 1, 1, 1, 1, 1]), (20, 1))
        cluster[np.arange(20), rng.integers(0, 6, size=20)] = 2
        scattered = rng.integers(0, 4, size=(40, 6))
        self.candidates = np.concatenate([cluster, scattered])
        self.scores = np.concatenate([np.full(20, 10.0), rng.uniform(0, 5, size=40)])

    def test_top_k_takes_the_highest_scores(self):
        chosen = TopK().select(self.candidates, self.scores, 5)
        assert set(chosen.tolist()) <= set(range(20))

    def test_top_k_returns_a_near_duplicate_batch(self):
        # The behaviour the diverse selector exists to contrast with.
        chosen = TopK().select(self.candidates, self.scores, 10)
        assert diversity(self.candidates[chosen]) < 2.0

    def test_the_penalty_trades_score_for_spread(self):
        # The knob has to actually move both quantities, in opposite directions.
        # A penalty smaller than the score gap correctly changes nothing, which
        # is why a weak version of this test passes vacuously.
        rows = []
        for penalty in (0.0, 5.0, 20.0):
            chosen = DiverseTopK(penalty=penalty).select(self.candidates, self.scores, 10)
            rows.append((diversity(self.candidates[chosen]), self.scores[chosen].mean()))
        spreads = [r[0] for r in rows]
        means = [r[1] for r in rows]
        assert spreads[-1] > spreads[0], f"diversity did not rise: {spreads}"
        assert means[-1] < means[0], f"mean score did not fall: {means}"
        assert spreads == sorted(spreads), f"diversity not monotone in penalty: {spreads}"

    def test_zero_penalty_reproduces_top_k_exactly(self):
        # The comparison is one knob rather than two implementations, so a
        # difference in results cannot be an artefact of different code paths.
        plain = TopK().select(self.candidates, self.scores, 8)
        none = DiverseTopK(penalty=0.0).select(self.candidates, self.scores, 8)
        assert set(plain.tolist()) == set(none.tolist())

    def test_the_best_candidate_is_always_taken_first(self):
        # Diversity trades away mean quality, never the single best design.
        chosen = DiverseTopK(penalty=5.0).select(self.candidates, self.scores, 6)
        assert chosen[0] == int(np.argmax(self.scores))

    def test_no_candidate_is_selected_twice(self):
        chosen = DiverseTopK(penalty=1.0).select(self.candidates, self.scores, 25)
        assert len(set(chosen.tolist())) == len(chosen)

    @pytest.mark.parametrize("selector", [TopK(), DiverseTopK(penalty=1.0)])
    def test_asking_for_more_than_exists_returns_everything(self, selector):
        chosen = selector.select(self.candidates, self.scores, 1000)
        assert len(chosen) == len(self.candidates)

    @pytest.mark.parametrize("selector", [TopK(), DiverseTopK(penalty=1.0)])
    def test_asking_for_none_returns_none(self, selector):
        assert len(selector.select(self.candidates, self.scores, 0)) == 0

    def test_a_negative_penalty_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            DiverseTopK(penalty=-1.0)
