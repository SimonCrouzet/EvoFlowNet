"""Tests for the CH65 three-antigen antibody landscape.

Values are checked against the published dataset rather than against the
implementation. The counts asserted here -- 62,926 complete variants, the three
per-antigen retained counts, the per-objective censored fractions -- are the
figures in Phillips et al. (eLife 2023), so a loader that read the wrong columns
or intersected them wrongly fails rather than quietly scoring the wrong antigen.

The censoring tests exist because that is the failure mode which makes the
sibling CR9114 library unusable: an objective pinned to its detection floor for
almost every variant is constant, not informative, and nothing in the values
themselves says so.

Marked ``requires_data``: these download ~15MB on first run. Deselect offline
with ``-m "not requires_data"``.
"""

import numpy as np
import pytest

from evogfn.landscapes.ch65 import (
    CH65_ANTIGENS,
    CH65_DETECTION_FLOOR,
    CH65_GERMLINE,
    CH65_MATURE,
    CH65_MUTATIONS,
    CH65_N_LIGHT_CHAIN,
    CH65_N_MEASURED,
    CH65_N_SITES,
    CH65Landscape,
)
from evogfn.rewards.base import TemperedReward
from evogfn.rewards.scalarization import ScalarizedReward, WeightedSum

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def landscape():
    """Loaded once; construction streams a 65,535-row CSV."""
    return CH65Landscape()


@pytest.fixture(scope="module")
def every(landscape):
    """The whole 65,536-sequence space."""
    return landscape.enumerate()


class TestSpaceShape:
    def test_the_space_is_two_to_the_sixteenth(self, landscape):
        assert landscape.sequence_length == CH65_N_SITES == 16
        assert landscape.search_space_size == 65_536

    def test_every_site_is_binary(self, landscape):
        # The whole point of this landscape: it is a Boolean subset lattice, not
        # a 20-letter space with a restriction imposed on top of it.
        assert landscape.alphabet.size == 2
        assert landscape.alphabet.symbols == ("0", "1")

    def test_a_decoded_variant_is_the_source_files_genotype_key(self, landscape, every):
        # Token 0 is germline, so decoding reproduces the deposit's `geno`
        # string and a value can be checked against the CSV without translation.
        assert landscape.alphabet.decode(every[0]) == CH65_GERMLINE
        assert landscape.alphabet.decode(every[-1]) == CH65_MATURE
        assert landscape.alphabet.decode(every[37]) == format(37, "016b")

    def test_the_sixteen_mutations_are_named_and_split_by_chain(self):
        assert len(CH65_MUTATIONS) == 16
        assert CH65_MUTATIONS[:CH65_N_LIGHT_CHAIN] == (
            "N26D",
            "S29R",
            "Y35N",
            "Y48C",
            "D49Y",
            "V98I",
        )
        assert CH65_MUTATIONS[CH65_N_LIGHT_CHAIN] == "G31D"
        assert CH65_MUTATIONS[-1] == "R87K"


class TestThreeObjectives:
    def test_evaluation_returns_one_column_per_antigen(self, landscape, every):
        values = landscape.evaluate(every)
        assert values.shape == (65_536, 3)
        assert landscape.n_objectives == 3

    def test_the_objectives_are_the_three_published_antigens(self, landscape):
        assert CH65_ANTIGENS == ("MA90", "MA90_G189E", "SI06")
        assert landscape.objective_names == (
            "affinity_MA90",
            "affinity_MA90_G189E",
            "affinity_SI06",
        )

    def test_a_batch_scores_the_same_as_its_rows(self, landscape):
        variants = [CH65_GERMLINE, CH65_MATURE, "1010101010101010", "0000000011111111"]
        tokens = landscape.alphabet.encode_many(variants)
        batch = landscape.evaluate(tokens)
        one_by_one = np.stack([landscape.evaluate(row[None, :])[0] for row in tokens])
        assert np.allclose(batch, one_by_one)

    def test_scoring_is_a_permutation_free_lookup(self, landscape):
        # Distinct mutation sets must index distinct table entries: a flattening
        # bug that collapsed positions would make these agree.
        a = landscape.alphabet.encode("1100000000000000")[None, :]
        b = landscape.alphabet.encode("0000000000000011")[None, :]
        assert not np.allclose(landscape.evaluate(a), landscape.evaluate(b))

    def test_affinities_span_the_assays_four_decade_range(self, landscape, every):
        measured = landscape.evaluate(every)[landscape.is_measured(every)]
        assert measured.min() == pytest.approx(CH65_DETECTION_FLOOR)
        assert measured.max() == pytest.approx(10.52604683)


class TestQualityControl:
    def test_the_measured_count_matches_the_publication(self, landscape):
        # 62,926 of 65,536 (96%) is the figure in Phillips et al. Fig 1B, and it
        # is the intersection of the three per-antigen counts the loader checks
        # separately on load.
        assert landscape.n_measured == CH65_N_MEASURED == 62_926

    def test_is_measured_separates_retained_from_dropped(self, landscape, every):
        measured = landscape.is_measured(every)
        assert measured.sum() == 62_926
        assert (~measured).sum() == 65_536 - 62_926 == 2_610

    def test_the_germline_and_the_mature_antibody_were_both_measured(self, landscape):
        tokens = landscape.alphabet.encode_many([CH65_GERMLINE, CH65_MATURE])
        assert landscape.is_measured(tokens).all()

    def test_dropped_variants_are_imputed_at_the_detection_floor_by_default(self, landscape, every):
        dropped = ~landscape.is_measured(every)
        assert dropped.any()
        assert (landscape.evaluate(every)[dropped] == CH65_DETECTION_FLOOR).all()

    def test_the_imputed_value_is_configurable(self, every):
        # Failing QC is not the same as binding weakly, so an analysis must be
        # able to make the absence visible rather than silently a floor value.
        landscape = CH65Landscape(unmeasured_value=float("nan"))
        values = landscape.evaluate(every)
        dropped = ~landscape.is_measured(every)
        assert np.isnan(values[dropped]).all()
        assert not np.isnan(values[~dropped]).any()


class TestCensoring:
    def test_censoring_is_reported_per_objective(self, landscape):
        # Pooled, this would be ~21% and look tolerable. Per objective it says
        # that MA90 is clean, that MA90-G189E is mildly affected, and that any
        # ordering SI06 implies within its censored half is an artefact.
        ma90, g189e, si06 = landscape.censored_fraction
        assert ma90 == 0.0
        assert g189e == pytest.approx(0.1502, abs=5e-5)
        assert si06 == pytest.approx(0.4743, abs=5e-5)

    def test_is_censored_answers_per_variant_and_per_objective(self, landscape, every):
        censored = landscape.is_censored(every)
        assert censored.shape == (65_536, 3)
        assert censored.sum(axis=0).tolist() == [0, 9_451, 29_847]

    def test_censored_measurements_sit_exactly_on_the_floor(self, landscape, every):
        values = landscape.evaluate(every)
        censored = landscape.is_censored(every)
        assert (values[censored] == CH65_DETECTION_FLOOR).all()
        measured = landscape.is_measured(every)
        assert (values[measured][~censored[measured]] > CH65_DETECTION_FLOOR).all()

    def test_unmeasured_variants_are_not_reported_as_censored(self, landscape, every):
        # Absent and below-the-floor are different failures, and the default
        # imputation puts unmeasured variants at the floor value -- so this is
        # exactly where the two would get conflated.
        dropped = ~landscape.is_measured(every)
        assert not landscape.is_censored(every)[dropped].any()

    def test_the_germline_has_no_detectable_breadth(self, landscape):
        # It binds the strain CH65 was raised against and nothing else, which is
        # the fact the whole library exists to explain.
        germline = landscape.alphabet.encode(CH65_GERMLINE)[None, :]
        assert landscape.is_censored(germline).tolist() == [[False, True, True]]
        assert landscape.evaluate(germline)[0, 0] == pytest.approx(8.55177344)


class TestOptimum:
    def test_the_optimum_is_a_three_component_ideal_point(self, landscape):
        optimum = landscape.optimum
        assert optimum.shape == (3,)
        assert optimum == pytest.approx([10.52604683, 10.330712, 9.76193295])

    def test_each_component_is_the_best_measured_value_on_that_antigen(self, landscape, every):
        measured = landscape.evaluate(every)[landscape.is_measured(every)]
        assert measured.max(axis=0) == pytest.approx(landscape.optimum)

    def test_no_single_variant_attains_the_ideal_point(self, landscape, every):
        # This is why the ideal point is not a target and the gap to it is not a
        # regret: the three maxima belong to three different sequences.
        assert len(set(landscape.optimal_variants)) == 3
        values = landscape.evaluate(every)
        assert not (values == landscape.optimum).all(axis=1).any()

    def test_the_mature_antibody_is_good_but_dominated(self, landscape, every):
        mature = landscape.evaluate(landscape.mature[None, :])[0]
        assert mature == pytest.approx([10.10262723, 9.7489624, 9.34545642])
        values = landscape.evaluate(every)[landscape.is_measured(every)]
        dominates = ((values >= mature).all(axis=1) & (values > mature).any(axis=1)).sum()
        assert dominates > 0

    def test_the_wild_type_is_the_germline_ancestor(self, landscape):
        # Not the mature antibody: a campaign starts where evolution started, or
        # every action available to it is a reversion.
        assert landscape.alphabet.decode(landscape.wild_type) == CH65_GERMLINE
        assert (landscape.wild_type == 0).all()

    def test_maturation_improves_every_objective(self, landscape):
        germline = landscape.evaluate(landscape.wild_type[None, :])[0]
        mature = landscape.evaluate(landscape.mature[None, :])[0]
        assert (mature > germline).all()


class TestMultiObjectiveWiring:
    def test_a_scalarized_reward_consumes_the_objective_matrix_directly(self, landscape, every):
        # The end-to-end claim of `evogfn train landscape=ch65 reward=scalarized`:
        # a three-column landscape reaches a scalar log reward with no landscape,
        # sampler or loss change anywhere.
        reward = ScalarizedReward(
            WeightedSum(),
            [1 / 3, 1 / 3, 1 / 3],
            reward=TemperedReward(beta=3.0),
        )
        log_rewards = reward.log_reward(landscape.evaluate(every))
        assert log_rewards.shape == (65_536,)
        assert np.isfinite(log_rewards).all()

    def test_the_preference_changes_which_variant_wins(self, landscape, every):
        # If it did not, the multi-objective machinery would be decoration.
        values = landscape.evaluate(every)
        base = ScalarizedReward(WeightedSum(), [1 / 3, 1 / 3, 1 / 3])
        ma90_only = base.with_preference([1.0, 0.0, 0.0])
        si06_only = base.with_preference([0.0, 0.0, 1.0])
        best_ma90 = landscape.alphabet.decode(every[int(ma90_only.log_reward(values).argmax())])
        best_si06 = landscape.alphabet.decode(every[int(si06_only.log_reward(values).argmax())])
        assert best_ma90 == landscape.optimal_variants[0]
        assert best_si06 == landscape.optimal_variants[2]
        assert best_ma90 != best_si06
