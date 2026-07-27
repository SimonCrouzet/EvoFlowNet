"""Tests for acquisition rules and batch selectors."""

import math

import numpy as np
import pytest

from evoflownet.acquisition import (
    DiverseTopK,
    ExpectedImprovement,
    Greedy,
    Thompson,
    TopK,
    UpperConfidenceBound,
)
from evoflownet.metrics import diversity


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
