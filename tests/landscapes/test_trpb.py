"""Tests for the TrpB four-site landscape.

Values here are checked against the published dataset rather than against the
implementation, so a loader that mis-sliced the four-site sub-landscape out of
FLIP2's ten-sub-landscape file would fail rather than quietly score the wrong
protein.

Marked ``requires_data``: these download ~3.5MB on first run. Deselect offline
with ``-m "not requires_data"``.
"""

import numpy as np
import pytest

from evogfn.landscapes.trpb import (
    TRPB_N_MEASURED,
    TRPB_POSITIONS,
    TRPB_WILD_TYPE,
    TrpBLandscape,
)

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def landscape():
    """Loaded once; construction streams a 228,298-row gzipped CSV."""
    return TrpBLandscape()


class TestKnownValues:
    def test_wild_type_fitness_is_one_by_definition(self, landscape):
        # FLIP2 normalises wild-type to 1.0, matching the GB1 convention.
        wild_type = landscape.alphabet.encode(TRPB_WILD_TYPE)[None, :]
        assert landscape.evaluate(wild_type)[0, 0] == pytest.approx(1.0)

    def test_the_best_variant_is_aikg(self, landscape):
        assert landscape.optimal_variant == "AIKG"
        assert landscape.optimum[0] == pytest.approx(2.4505363811874465)

    def test_the_runner_up_is_clkg(self, landscape):
        runner_up = landscape.alphabet.encode("CLKG")[None, :]
        assert landscape.evaluate(runner_up)[0, 0] == pytest.approx(2.283982063667191)

    def test_the_optimum_requires_all_four_mutations(self, landscape):
        # The same epistasis GB1 shows: the best variant is not reachable by
        # improving wild-type one position at a time.
        best = landscape.alphabet.encode(landscape.optimal_variant)
        wild_type = landscape.alphabet.encode(TRPB_WILD_TYPE)
        assert int((best != wild_type).sum()) == 4

    def test_the_assay_size_matches_the_publication(self, landscape):
        # 159,129 is the PNAS figure, and disagrees with the 153,620 reported by
        # holo-bench; see notes/review/07-trpb-provenance.md.
        assert landscape.n_measured == TRPB_N_MEASURED == 159_129

    def test_the_space_is_twenty_to_the_fourth(self, landscape):
        assert landscape.sequence_length == 4
        assert landscape.alphabet.size == 20
        assert landscape.search_space_size == 160_000

    def test_the_assayed_positions_are_the_active_site_four(self):
        assert TRPB_POSITIONS == (183, 184, 227, 228)


class TestLookup:
    def test_scoring_is_a_permutation_free_lookup(self, landscape):
        # Distinct positions must index distinct table entries: a flattening bug
        # that collapsed positions would make these agree.
        a = landscape.alphabet.encode("AIKG")[None, :]
        b = landscape.alphabet.encode("GKIA")[None, :]
        assert landscape.evaluate(a)[0, 0] != landscape.evaluate(b)[0, 0]

    def test_every_sequence_in_the_space_can_be_scored(self, landscape):
        values = landscape.evaluate(landscape.enumerate())
        assert values.shape == (160_000, 1)
        assert np.isfinite(values).all()

    def test_the_enumerated_maximum_is_the_reported_optimum(self, landscape):
        values = landscape.evaluate(landscape.enumerate())[:, 0]
        assert values.max() == pytest.approx(landscape.optimum[0])

    def test_a_batch_scores_the_same_as_its_rows(self, landscape):
        variants = ["VFVS", "AIKG", "AAAA", "WWWW"]
        batch = landscape.evaluate(landscape.alphabet.encode_many(variants))[:, 0]
        one_by_one = [
            landscape.evaluate(landscape.alphabet.encode(v)[None, :])[0, 0] for v in variants
        ]
        assert np.allclose(batch, one_by_one)


class TestUnmeasuredVariants:
    def test_the_measured_fraction_matches_the_assay(self, landscape):
        measured = landscape.is_measured(landscape.enumerate())
        assert measured.sum() == TRPB_N_MEASURED
        assert 160_000 - measured.sum() == 871

    def test_the_wild_type_was_measured(self, landscape):
        wild_type = landscape.alphabet.encode(TRPB_WILD_TYPE)[None, :]
        assert landscape.is_measured(wild_type)[0]

    def test_unmeasured_variants_are_imputed_as_dead_by_default(self, landscape):
        every = landscape.enumerate()
        unmeasured = ~landscape.is_measured(every)
        assert unmeasured.any()
        assert (landscape.evaluate(every)[unmeasured, 0] == 0.0).all()

    def test_the_imputed_value_is_configurable(self):
        # Absent from the library is not the same as unfit, so an analysis must
        # be able to make their absence visible rather than silently dead.
        landscape = TrpBLandscape(unmeasured_value=float("nan"))
        every = landscape.enumerate()
        unmeasured = ~landscape.is_measured(every)
        values = landscape.evaluate(every)[:, 0]
        assert np.isnan(values[unmeasured]).all()
        assert not np.isnan(values[~unmeasured]).any()


class TestLandscapeShape:
    def test_nearly_everything_is_worse_than_wild_type(self, landscape):
        # 99.3%, against 90% for GB1. This is what makes TrpB the harder of the
        # two: almost the whole space is flat and dead.
        every = landscape.enumerate()
        measured = landscape.is_measured(every)
        values = landscape.evaluate(every)[measured, 0]
        assert (values < 1.0).mean() > 0.99

    def test_the_median_variant_is_effectively_dead(self, landscape):
        every = landscape.enumerate()
        measured = landscape.is_measured(every)
        values = landscape.evaluate(every)[measured, 0]
        assert np.median(values) < 0.1

    def test_the_optimum_is_far_above_wild_type(self, landscape):
        assert landscape.optimum[0] > 2.0


class TestDistinctFromGB1:
    def test_the_alphabet_and_shape_match_gb1_but_the_values_do_not(self, landscape):
        # The point of adding TrpB is a second landscape of the same shape whose
        # numbers are unrelated, so a method tuned to GB1's reward geometry has
        # nowhere to hide.
        gb1 = pytest.importorskip("evogfn.landscapes.gb1").GB1Landscape()
        assert landscape.sequence_length == gb1.sequence_length
        assert landscape.alphabet == gb1.alphabet
        assert landscape.optimal_variant != gb1.optimal_variant
        assert landscape.objective_names != gb1.objective_names
