"""Tests for a campaign run against a landscape that returns a vector.

Two things are being defended here. The first is that a multi-objective campaign
*runs at all* and reports quantities that mean something -- hypervolume of the
measured set, IGD+ against a reference front -- instead of a maximum taken
across objectives on different scales. The second is that adding all of that
moved nothing on the single-objective path, which every existing result is
indexed by.
"""

import numpy as np
import pytest

from evogfn.acquisition import ExpectedImprovement, Greedy, ScalarizedAcquisition
from evogfn.algorithms.base import Sampler
from evogfn.core.types import Alphabet
from evogfn.landscapes.base import FitnessLandscape
from evogfn.loop import Campaign
from evogfn.rewards.scalarization import Tchebycheff, WeightedSum
from evogfn.surrogate import DeepEnsemble

ALPHABET = Alphabet.from_string("ABCD")
LENGTH = 6

#: The trade-off every campaign below states. Equal weight on both antigens.
EVEN = (0.5, 0.5)


class TwoAntigenLandscape(FitnessLandscape):
    """A miniature of CH65: two objectives that pull in opposite directions.

    Objective 0 counts token ``1``, objective 1 counts token ``2``, and a
    position spent on one cannot be spent on the other -- so the front is a real
    trade-off rather than a single peak seen twice.
    """

    def __init__(self, *, floor=0.0):
        self.calls = 0
        self._floor = floor

    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    @property
    def n_objectives(self):
        return 2

    @property
    def optimum(self):
        return np.array([float(LENGTH), float(LENGTH)])

    def _evaluate(self, sequences):
        self.calls += sequences.shape[0]
        first = (sequences == 1).sum(axis=1)
        second = (sequences == 2).sum(axis=1)
        return np.stack([first, second], axis=1).astype(np.float64) + self._floor


class ThreeAntigenLandscape(TwoAntigenLandscape):
    """Three objectives, which is where the exact hypervolume gets expensive."""

    @property
    def n_objectives(self):
        return 3

    @property
    def optimum(self):
        return np.full(3, float(LENGTH))

    def _evaluate(self, sequences):
        self.calls += sequences.shape[0]
        counts = [(sequences == token).sum(axis=1) for token in (1, 2, 3)]
        return np.stack(counts, axis=1).astype(np.float64)


class StatesItsOwnReference(TwoAntigenLandscape):
    """A landscape that knows its own detection floor, as CH65 does."""

    @property
    def reference_point(self):
        return np.array([1.0, 1.0])


class StatesItsOwnFront(TwoAntigenLandscape):
    """A landscape that knows its own Pareto front."""

    @property
    def reference_front(self):
        return np.array([[6.0, 0.0], [3.0, 3.0], [0.0, 6.0]])


class RandomSampler(Sampler):
    """Uniform proposals. Learns nothing, which makes it a clean control."""

    def __init__(self, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def propose(self, n):
        self._count(n)
        return self._rng.integers(0, ALPHABET.size, size=(n, LENGTH), dtype=np.int32)


def surrogate():
    return DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=30, seed=0)


def scalarized(rule=None, preference=EVEN):
    return ScalarizedAcquisition(rule or Greedy(), WeightedSum(), preference)


def campaign(landscape=None, **kwargs):
    landscape = landscape if landscape is not None else TwoAntigenLandscape()
    even = np.full(landscape.n_objectives, 1.0 / landscape.n_objectives)
    kwargs.setdefault("acquisition", scalarized(preference=even))
    kwargs.setdefault("rounds", 3)
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("pool_size", 64)
    return Campaign(landscape=landscape, sampler=RandomSampler(seed=3), **kwargs)


class TestTheTradeOffMustBeStated:
    def test_a_preference_of_the_wrong_width_is_refused_at_construction(self):
        # Two weights against three antigens. Caught before the first plate is
        # measured rather than in round two, when the surrogate first asks the
        # rule to reduce what came back.
        landscape = ThreeAntigenLandscape()
        with pytest.raises(ValueError, match="objectives"):
            Campaign(
                landscape=landscape,
                sampler=RandomSampler(),
                acquisition=scalarized(preference=EVEN),
            )
        assert landscape.calls == 0

    def test_a_scalar_rule_is_refused_before_any_oracle_call(self):
        # The whole point of refusing in the constructor: the alternative is a
        # traceback after 384 measurements have been paid for.
        landscape = TwoAntigenLandscape()
        with pytest.raises(ValueError, match="ScalarizedAcquisition"):
            Campaign(landscape=landscape, sampler=RandomSampler(), acquisition=Greedy())
        assert landscape.calls == 0

    def test_the_default_acquisition_is_refused_too(self):
        # Defaulting to Greedy() would otherwise make a multi-objective campaign
        # silently rank on one antigen.
        with pytest.raises(ValueError, match="ranks one value"):
            Campaign(landscape=TwoAntigenLandscape(), sampler=RandomSampler())

    def test_a_stated_trade_off_runs(self):
        result = campaign(surrogate=surrogate()).run()
        assert result.oracle_calls == 24
        assert result.values.shape == (24, 2)

    def test_the_objective_vectors_survive_the_scalarisation(self):
        # The scalarisation directs the search; it must not decide what is
        # recorded, or the indicators would have nothing to be computed on.
        result = campaign(surrogate=surrogate()).run()
        assert result.n_objectives == 2
        assert result.is_multi_objective

    def test_best_so_far_is_the_scalarised_value(self):
        result = campaign(surrogate=surrogate()).run()
        combined = 0.5 * result.values[:, 0] + 0.5 * result.values[:, 1]
        assert result.trace()[-1] == pytest.approx(combined.max())

    @pytest.mark.parametrize(
        "acquisition",
        [
            ScalarizedAcquisition(Greedy(), WeightedSum(), EVEN),
            ScalarizedAcquisition(ExpectedImprovement(), WeightedSum(), EVEN),
            ScalarizedAcquisition(Greedy(), Tchebycheff(), EVEN),
        ],
    )
    def test_every_scalarisation_and_inner_rule_runs(self, acquisition):
        result = campaign(surrogate=surrogate(), acquisition=acquisition).run()
        assert result.oracle_calls == 24

    def test_three_objectives_run(self):
        result = campaign(ThreeAntigenLandscape(), surrogate=surrogate()).run()
        assert result.values.shape == (24, 3)


class TestWhatIsReported:
    def test_a_scalar_best_value_is_refused(self):
        result = campaign(surrogate=surrogate()).run()
        with pytest.raises(ValueError, match="no single best value"):
            _ = result.best_value

    def test_regret_is_none_rather_than_a_gap_to_an_ideal_point(self):
        # The landscape knows its per-objective maxima, and the gap to them is
        # not a regret: no design attains the ideal point, so the gap never
        # reaches zero and cannot score a run.
        result = campaign(surrogate=surrogate()).run()
        assert result.simple_regret is None
        assert result.optimum is None
        assert result.ideal_point == pytest.approx([6.0, 6.0])

    def test_the_summary_reports_indicators_instead_of_a_best_value(self):
        result = campaign(
            surrogate=surrogate(), reference_point=[0.0, 0.0], reference_front=[[6.0, 0.0]]
        ).run()
        summary = result.summary()
        assert "best_value" not in summary
        assert "simple_regret" not in summary
        assert summary["hypervolume"] > 0.0
        assert summary["igd_plus"] >= 0.0

    def test_hypervolume_is_none_without_a_reference_point(self):
        # Not zero. Zero is a result -- "nothing beat the reference" -- and
        # "nobody said where the reference is" is a different statement.
        result = campaign(surrogate=surrogate()).run()
        assert result.hypervolume is None
        assert "hypervolume" not in result.summary()

    def test_igd_plus_is_none_without_a_reference_front(self):
        result = campaign(surrogate=surrogate(), reference_point=[0.0, 0.0]).run()
        assert result.igd_plus is None

    def test_the_reference_point_used_is_kept_on_the_result(self):
        # A hypervolume that does not carry its reference point is not
        # comparable with anything, and nothing in the number says so.
        result = campaign(surrogate=surrogate(), reference_point=[1.0, 1.0]).run()
        assert result.reference_point == pytest.approx([1.0, 1.0])

    def test_a_landscape_can_state_its_own_reference_point(self):
        result = campaign(StatesItsOwnReference(), surrogate=surrogate()).run()
        assert result.reference_point == pytest.approx([1.0, 1.0])

    def test_an_explicit_reference_point_overrides_the_landscape(self):
        result = campaign(
            StatesItsOwnReference(), surrogate=surrogate(), reference_point=[0.0, 0.0]
        ).run()
        assert result.reference_point == pytest.approx([0.0, 0.0])

    def test_a_landscape_can_state_its_own_reference_front(self):
        result = campaign(StatesItsOwnFront(), surrogate=surrogate()).run()
        assert result.igd_plus is not None

    def test_a_reference_point_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match=r"shape \(2,\)"):
            campaign(reference_point=[0.0, 0.0, 0.0])

    def test_an_infinite_reference_point_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            campaign(reference_point=[-np.inf, 0.0])

    def test_the_pareto_front_holds_only_undominated_measurements(self):
        result = campaign(surrogate=surrogate()).run()
        front = result.pareto_front
        assert front.shape[1] == 2
        for point in front:
            beaten = (result.values >= point).all(axis=1) & (result.values > point).any(axis=1)
            assert not beaten.any()


class TestHypervolumeRises:
    def test_more_measurements_never_lose_volume(self):
        # Monotone by construction: measuring more can only add dominated
        # volume. A decrease would mean the indicator is being computed on
        # something other than the accumulated set.
        result = campaign(surrogate=surrogate(), rounds=4, reference_point=[0.0, 0.0]).run()
        trace = result.hypervolume_trace()
        assert len(trace) == 4
        assert trace == sorted(trace)

    def test_it_actually_moves_over_a_campaign(self):
        result = campaign(surrogate=surrogate(), rounds=4, reference_point=[0.0, 0.0]).run()
        trace = result.hypervolume_trace()
        assert trace[-1] > trace[0], f"hypervolume never moved: {trace}"

    def test_the_final_hypervolume_matches_the_last_round(self):
        result = campaign(surrogate=surrogate(), rounds=4, reference_point=[0.0, 0.0]).run()
        assert result.hypervolume == pytest.approx(result.hypervolume_trace()[-1])

    def test_a_lower_reference_point_gives_a_larger_volume(self):
        # Which is why two runs measured from different reference points are not
        # comparable, and why nothing here picks one on the caller's behalf.
        high = campaign(surrogate=surrogate(), reference_point=[2.0, 2.0]).run()
        low = campaign(surrogate=surrogate(), reference_point=[0.0, 0.0]).run()
        assert low.hypervolume > high.hypervolume

    def test_designs_below_the_reference_contribute_nothing(self):
        result = campaign(surrogate=surrogate(), reference_point=[float(LENGTH), float(LENGTH)])
        assert result.run().hypervolume == pytest.approx(0.0)

    def test_a_front_too_large_for_the_exact_method_reports_nan(self):
        # An exact indicator that refuses is better than an approximate one that
        # does not announce itself -- but refusing must not throw away a
        # campaign's measurements, so it is nan here and the values survive.
        result = campaign(
            ThreeAntigenLandscape(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=64,
            pool_size=512,
            reference_point=[-1.0, -1.0, -1.0],
        ).run()
        assert result.values.shape[0] == 256
        assert np.isnan(result.hypervolume_trace()[-1])


class TestTheSingleObjectiveResultDidNotMove:
    """Everything already reported must be bit-identical."""

    class OneObjective(FitnessLandscape):
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
            return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)

    def run_one(self, rule):
        return Campaign(
            landscape=self.OneObjective(),
            sampler=RandomSampler(seed=7),
            surrogate=surrogate(),
            acquisition=rule,
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()

    def test_the_pinned_numbers_are_unchanged(self):
        # Hard-coded rather than compared against a second run, so that a change
        # to the loop cannot move both sides together and still pass.
        result = self.run_one(Greedy())
        assert result.oracle_calls == 24
        assert result.best_value == 4.0
        assert result.sequences[0].tolist() == [3, 2, 2, 3, 2, 3]
        assert result.simple_regret == pytest.approx(2.0)

    def test_the_summary_keys_are_unchanged(self):
        assert list(self.run_one(Greedy()).summary()) == [
            "oracle_calls",
            "proposals",
            "best_value",
            "feasible_fraction",
            "rounds",
            "simple_regret",
        ]

    def test_no_indicator_appears_on_a_single_objective_run(self):
        result = self.run_one(ExpectedImprovement())
        assert result.hypervolume is None
        assert result.igd_plus is None
        assert result.ideal_point is None
        assert all(np.isnan(v) for v in result.hypervolume_trace())

    def test_the_incumbent_still_comes_from_the_measurements(self):
        # Expected improvement is the rule that reads best_observed, so it is
        # the one that would move if the reduction had changed shape.
        result = self.run_one(ExpectedImprovement())
        assert result.best_value == pytest.approx(result.values.max())
