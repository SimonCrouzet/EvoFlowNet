"""Tests for the rule that picks the configuration the benchmark reports.

The rule is only worth having if it is mechanical. A selection procedure that
can be nudged is not a procedure, so what these pin is that the rule reaches its
answer from the numbers alone: that a genuine winner is not overturned by a
tie-break it never triggered, that a tie *is* broken by diversity rather than by
arrival order or alphabetical accident, and that the cases where selecting at
all would be dishonest raise instead of returning something confident-looking.

The failure to fear is not a crash. It is a `Selection` naming an arm that the
evidence does not support, since nothing downstream re-checks the choice.
"""

import numpy as np
import pytest

from evogfn.benchmark.selection import select


class Record:
    """The two fields the rule reads, which is all a stored record needs here."""

    def __init__(self, regret, diversity=1.0):
        self.regret = regret
        self.diversity = diversity


def arm(regrets, diversity=1.0):
    return {i: Record(r, diversity) for i, r in enumerate(regrets)}


class TestAClearWinner:
    def test_a_separated_arm_wins_on_regret(self):
        # Wide, non-overlapping separation, so no tie-break should be consulted.
        chosen = select({"good": arm([0.1] * 20), "bad": arm([0.9] * 20)})
        assert chosen.chosen == "good"
        assert chosen.tied == ("good",)

    def test_its_reason_says_it_was_separated(self):
        chosen = select({"good": arm([0.1] * 20), "bad": arm([0.9] * 20)})
        assert "separated from every other arm" in chosen.reason

    def test_diversity_cannot_overturn_a_separated_arm(self):
        # The rule is regret-first. An arm that is genuinely worse on fitness
        # does not win by being more diverse, or the headline column would be
        # selected against the thing the benchmark is measuring.
        chosen = select(
            {
                "good": arm([0.1] * 20, diversity=1.0),
                "diverse-but-worse": arm([0.9] * 20, diversity=99.0),
            }
        )
        assert chosen.chosen == "good"


class TestATie:
    @pytest.fixture
    def tied(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.2, 30)
        return {
            "narrow": {i: Record(0.5 + n, 2.0) for i, n in enumerate(noise)},
            "wide": {i: Record(0.5 + n, 8.0) for i, n in enumerate(noise)},
        }

    def test_diversity_breaks_it(self, tied):
        assert select(tied).chosen == "wide"

    def test_both_arms_are_named_as_tied(self, tied):
        assert select(tied).tied == ("narrow", "wide")

    def test_the_reason_says_the_choice_rested_on_diversity(self, tied):
        # A caption drawn from this must not imply the arm won on fitness.
        assert "tied on regret" in select(tied).reason

    def test_an_arm_behind_by_less_than_the_noise_is_still_eligible(self):
        # Losing by less than the evidence can resolve is not losing. Were this
        # wrong, the rule would silently become "lowest mean", and the tie-break
        # would never fire on real data.
        rng = np.random.default_rng(1)
        # Independent noise per arm, so the *paired difference* carries variance.
        # Sharing one noise draw would make the difference a constant, which a
        # paired test resolves at any sample size -- a tie that cannot occur.
        records = {
            "leader": {i: Record(0.50 + n, 1.0) for i, n in enumerate(rng.normal(0, 0.3, 30))},
            "behind": {i: Record(0.51 + n, 9.0) for i, n in enumerate(rng.normal(0, 0.3, 30))},
        }
        assert select(records).chosen == "behind"


class TestRefusals:
    def test_nothing_to_choose_between(self):
        with pytest.raises(ValueError, match="no arms"):
            select({})

    def test_arms_sharing_no_seed(self):
        # Unpaired arms would still produce a mean, and a mean here would look
        # exactly like a result.
        with pytest.raises(ValueError, match="share no seed"):
            select({"a": {0: Record(0.1)}, "b": {1: Record(0.9)}})

    def test_every_arm_failing_everywhere(self):
        with pytest.raises(ValueError, match="failed on every"):
            select({"a": arm([np.inf] * 5), "b": arm([np.inf] * 5)})


class TestPartialFailure:
    def test_an_arm_is_scored_on_the_seeds_it_survived(self):
        records = {
            "flaky": {0: Record(0.1), 1: Record(np.inf), 2: Record(0.1)},
            "steady": {0: Record(0.5), 1: Record(0.5), 2: Record(0.5)},
        }
        assert select(records).regret["flaky"] == pytest.approx(0.1)

    def test_an_arm_that_failed_somewhere_does_not_win_on_a_tie_break(self):
        # Its mean is computed over survivors, which flatters it; letting that
        # number into a paired comparison would compare different seed sets.
        records = {
            "flaky": {i: Record(np.inf if i % 2 else 0.01, 99.0) for i in range(20)},
            "steady": {i: Record(0.5, 1.0) for i in range(20)},
        }
        assert select(records).tied == ("flaky",)


def test_only_shared_seeds_are_used():
    records = {
        "a": {0: Record(0.1), 1: Record(0.1), 2: Record(99.0)},
        "b": {0: Record(0.5), 1: Record(0.5)},
    }
    # Seed 2 belongs to one arm only, so it must not reach either mean.
    assert select(records).regret["a"] == pytest.approx(0.1)
