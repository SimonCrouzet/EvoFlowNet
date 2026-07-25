"""Tests for the GB1 four-site landscape.

Values here are checked against the published dataset rather than against the
implementation, so a loader that silently mis-indexed the table would fail.

Marked ``requires_data``: these download ~2.6MB on first run. Deselect offline
with ``-m "not requires_data"``.
"""

import numpy as np
import pytest

from evoflownet.landscapes.gb1 import (
    GB1_N_MEASURED,
    GB1_WILD_TYPE,
    GB1Landscape,
)

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def landscape():
    """Loaded once; the table is ~1.3MB and construction reads a zipped CSV."""
    return GB1Landscape()


class TestKnownValues:
    def test_wild_type_fitness_is_one_by_definition(self, landscape):
        wild_type = landscape.alphabet.encode(GB1_WILD_TYPE)[None, :]
        assert landscape.evaluate(wild_type)[0, 0] == pytest.approx(1.0)

    def test_the_best_variant_is_fwaa(self, landscape):
        assert landscape.optimal_variant == "FWAA"
        assert landscape.optimum[0] == pytest.approx(8.76196565571)

    def test_the_optimum_requires_all_four_mutations(self, landscape):
        # This is the epistasis GB1 is known for: no single, double or triple
        # mutant approaches the quadruple optimum.
        best = landscape.alphabet.encode(landscape.optimal_variant)
        wild_type = landscape.alphabet.encode(GB1_WILD_TYPE)
        assert int((best != wild_type).sum()) == 4

    def test_the_assay_size_matches_the_publication(self, landscape):
        assert landscape.n_measured == GB1_N_MEASURED == 149_361

    def test_the_space_is_twenty_to_the_fourth(self, landscape):
        assert landscape.sequence_length == 4
        assert landscape.alphabet.size == 20
        assert landscape.search_space_size == 160_000


class TestLookup:
    def test_scoring_is_a_permutation_free_lookup(self, landscape):
        # Distinct positions must index distinct table entries: a flattening bug
        # that collapsed positions would make these agree.
        a = landscape.alphabet.encode("AVGG")[None, :]
        b = landscape.alphabet.encode("GGVA")[None, :]
        assert landscape.evaluate(a)[0, 0] != landscape.evaluate(b)[0, 0]

    def test_every_sequence_in_the_space_can_be_scored(self, landscape):
        values = landscape.evaluate(landscape.enumerate())
        assert values.shape == (160_000, 1)
        assert np.isfinite(values).all()

    def test_the_enumerated_maximum_is_the_reported_optimum(self, landscape):
        values = landscape.evaluate(landscape.enumerate())[:, 0]
        assert values.max() == pytest.approx(landscape.optimum[0])

    def test_a_batch_scores_the_same_as_its_rows(self, landscape):
        variants = ["VDGV", "FWAA", "AAAA", "WWWW"]
        batch = landscape.evaluate(landscape.alphabet.encode_many(variants))[:, 0]
        one_by_one = [
            landscape.evaluate(landscape.alphabet.encode(v)[None, :])[0, 0] for v in variants
        ]
        assert np.allclose(batch, one_by_one)


class TestUnmeasuredVariants:
    def test_the_measured_fraction_matches_the_assay(self, landscape):
        measured = landscape.is_measured(landscape.enumerate())
        assert measured.sum() == GB1_N_MEASURED
        assert 160_000 - measured.sum() == 10_639

    def test_the_wild_type_was_measured(self, landscape):
        wild_type = landscape.alphabet.encode(GB1_WILD_TYPE)[None, :]
        assert landscape.is_measured(wild_type)[0]

    def test_unmeasured_variants_are_imputed_as_dead_by_default(self, landscape):
        every = landscape.enumerate()
        unmeasured = ~landscape.is_measured(every)
        assert unmeasured.any()
        assert (landscape.evaluate(every)[unmeasured, 0] == 0.0).all()

    def test_the_imputed_value_is_configurable(self):
        # Absent from the library is not the same as unfit, so an analysis must
        # be able to make their absence visible rather than silently zero.
        landscape = GB1Landscape(unmeasured_value=float("nan"))
        every = landscape.enumerate()
        unmeasured = ~landscape.is_measured(every)
        values = landscape.evaluate(every)[:, 0]
        assert np.isnan(values[unmeasured]).all()
        assert not np.isnan(values[~unmeasured]).any()

    def test_imputation_does_not_disturb_measured_values(self):
        default = GB1Landscape()
        alternative = GB1Landscape(unmeasured_value=-99.0)
        every = default.enumerate()
        measured = default.is_measured(every)
        assert np.allclose(
            default.evaluate(every)[measured, 0], alternative.evaluate(every)[measured, 0]
        )


class TestLandscapeShape:
    def test_about_a_fifth_of_measured_variants_are_dead(self, landscape):
        every = landscape.enumerate()
        measured = landscape.is_measured(every)
        dead = (landscape.evaluate(every)[measured, 0] == 0.0).mean()
        assert 0.15 < dead < 0.25

    def test_most_variants_are_worse_than_wild_type(self, landscape):
        # Wild-type sits at 1.0 while the mean is far below it, which is what
        # makes this a search problem rather than a sampling-anything problem.
        every = landscape.enumerate()
        measured = landscape.is_measured(every)
        values = landscape.evaluate(every)[measured, 0]
        assert (values < 1.0).mean() > 0.9

    def test_the_optimum_is_far_above_wild_type(self, landscape):
        assert landscape.optimum[0] > 8.0
