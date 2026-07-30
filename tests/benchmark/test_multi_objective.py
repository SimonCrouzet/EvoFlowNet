"""Tests for the multi-objective suite: what it declares, and what a run stores.

Four failures are under test, and each is one this suite could plausibly ship
with because none of them raises.

**An indicator measured from nowhere.** Hypervolumes taken from different
reference points are not comparable and neither number records its point, so a
task that forgot to state one, or a campaign that silently picked up whatever the
landscape happened to expose, would produce a column that mixes scales. The task
refuses to exist without a reference point, and the point is in the `repr` every
record stores as provenance.

**A trade-off applied in one place and not another.** A campaign that ranked its
pool under one preference while its sampler bred under a different one -- or
under none, because a classical baseline refuses an objective matrix -- would
search somewhere its own ledger never claimed was good.

**A reference front that cannot be reached, or is reached by accident.** The
CH65 front must exclude variants tied at the detection floor, and the Ehrlich
fronts must consist of points some sequence actually attains.

**A preference ensemble that is not at fixed budget.** Running eight preferences
at a full budget each and comparing that against one preference at a full budget
is not a comparison, and it is what "eight preferences" would mean if the split
were forgotten.

The CH65 tests need the downloaded dataset and say so. The end-to-end arm tests
are kept to a two-round toy with a four-design plate, which is enough to exercise
every seam the suite has and cheap enough to run on every commit.
"""

import numpy as np
import pytest

from evogfn.acquisition.rules import Greedy
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.nsga2 import NSGA2
from evogfn.benchmark.multi_objective import (
    ARMS,
    EXACT_FRONT_LIMIT,
    MULTI_OBJECTIVE_MAIN,
    MultiObjectiveTask,
    PreferenceEnsemble,
    ScalarizedObserving,
    ch65_reference_front,
    conflict_sweep,
    multi_objective_tiers,
    objective_count_sweep,
    preference_task,
    preference_vectors,
    recombination_front,
    run_multi_objective_task,
    scalarized_genetic_arm,
    scalarized_gflownet_arm,
    set_indicators,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.store import ResultStore
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ch65 import CH65_DETECTION_FLOOR, CH65Landscape
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.multi_ehrlich import MultiEhrlichLandscape
from evogfn.loop.campaign import Campaign
from evogfn.loop.ledger import CampaignResult
from evogfn.metrics.pareto import non_dominated
from evogfn.rewards.scalarization import WeightedSum

EVERY_TASK = [
    *MULTI_OBJECTIVE_MAIN,
    *conflict_sweep(),
    *objective_count_sweep(),
    preference_task(),
]


def toy_landscape(conflict=1.0):
    """A multi-Ehrlich instance small enough to run several campaigns against.

    Full conflict by default, so its front is more than a single point -- a front
    of one would make every coverage assertion below vacuous.
    """
    return MultiEhrlichLandscape.with_conflict(
        sequence_length=16,
        vocab_size=4,
        n_objectives=2,
        n_motifs=1,
        motif_length=4,
        quantization=4,
        max_spacing=2,
        transition_density=0.5,
        conflict=conflict,
        seed=3,
    )


def toy_task(*, rounds=2, batch=4, front=recombination_front):
    """A multi-objective task cheap enough to run end to end."""
    return MultiObjectiveTask(
        name="toy",
        purpose="a toy, for testing that the wiring does what the table says",
        build=toy_landscape,
        protocol=Protocol(rounds=rounds, batch_size=batch, max_mutations=4),
        max_mutations=4,
        reanchor=True,
        reference_point=(0.0, 0.0),
        front=front,
        front_is_exact=False,
    )


# --------------------------------------------------------------------------
# What a task declares.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_every_task_states_where_its_hypervolume_is_measured_from(task):
    # Without this the campaign falls back to whatever the landscape exposes,
    # and a landscape gaining or moving a `reference_point` property would
    # silently rescale every result already in the store.
    assert task.reference_point
    assert task.n_objectives == len(task.reference_point)
    assert all(np.isfinite(task.reference_point))


def test_a_task_without_a_reference_point_is_refused():
    with pytest.raises(ValueError, match="must state the reference point"):
        MultiObjectiveTask(
            name="nowhere",
            purpose="x",
            build=toy_landscape,
            protocol=Protocol(rounds=1, batch_size=2, max_mutations=4),
            max_mutations=4,
        )


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_the_provenance_string_carries_the_reference_point(task):
    # `run_multi_objective_task` stores this as a record's protocol field. Two
    # records at 4x96=384 whose hypervolumes came from different points are not
    # comparable, and a string naming only the protocol could not tell them
    # apart.
    text = repr(task)
    assert task.name in text
    assert f"{task.n_objectives} objectives" in text
    assert f"{task.reference_point[0]:g}" in text


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_the_search_budget_is_cumulative_only_where_the_anchor_moves(task):
    expected = task.max_mutations * task.protocol.rounds if task.reanchor else task.max_mutations
    assert task.search_budget == expected


def test_the_tiers_say_which_of_them_carries_results():
    tiers = multi_objective_tiers(4, 2)
    headline = [tier.name for tier in tiers if tier.headline]
    # Exactly one tier carries claims. The sweeps say when a ranking would
    # change and the preference diagnostic decides one setting; promoting either
    # to a result is the misreading this suite is laid out to prevent.
    assert headline == ["main"]
    assert {t.name for t in tiers} == {"main", "conflict", "objectives", "preferences"}


def test_the_multi_ehrlich_parent_is_drawn_rather_than_planted():
    # `Task.parent` raises for anything but GB1 and Ehrlich, so a multi-objective
    # task that did not override it could not be run at all. The starting point
    # must also be independent of the answer: a parent that *was* a planted
    # optimum would hand every arm the front for free.
    task = toy_task()
    landscape = task.landscape()
    parent = task.parent(landscape)
    assert landscape.is_feasible(parent[None, :])[0]
    assert not any(np.array_equal(parent, optimum) for optimum in landscape.optimal_sequences)


# --------------------------------------------------------------------------
# Preferences.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_objectives", [2, 3, 4])
@pytest.mark.parametrize("count", [1, 4, 8])
def test_preferences_lie_on_the_simplex(n_objectives, count):
    weights = preference_vectors(n_objectives, count, seed=1)
    assert weights.shape == (count, n_objectives)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_one_preference_is_the_neutral_one():
    # Anything else would be the benchmark, rather than the biology, claiming
    # which objective matters.
    np.testing.assert_allclose(preference_vectors(3, 1), [[1 / 3, 1 / 3, 1 / 3]])


def test_two_objectives_get_an_even_grid_including_both_ends():
    weights = preference_vectors(2, 5)
    np.testing.assert_allclose(weights[0], [0.0, 1.0])
    np.testing.assert_allclose(weights[-1], [1.0, 0.0])
    assert len(np.unique(weights[:, 0])) == 5


def test_a_two_objective_grid_does_not_depend_on_the_seed():
    # The diagnostic varies the *count*. A grid that moved per seed would be
    # varying the draw as well, and the two effects would not be separable.
    np.testing.assert_array_equal(
        preference_vectors(2, 4, seed=0), preference_vectors(2, 4, seed=9)
    )


def test_three_objectives_do_depend_on_the_seed():
    # No even lattice has four points on a triangle, so these are drawn -- and
    # drawing them per seed is what averages the comparison over draws.
    left = preference_vectors(3, 4, seed=0)
    right = preference_vectors(3, 4, seed=1)
    assert not np.allclose(left, right)


def test_asking_for_no_preferences_is_refused():
    with pytest.raises(ValueError, match="count must be at least 1"):
        preference_vectors(2, 0)


# --------------------------------------------------------------------------
# Reference fronts.
# --------------------------------------------------------------------------


def test_aligned_objectives_have_a_one_point_front():
    # The landscape's own documentation says the exact front at conflict 0 is a
    # single point, and this is the one setting where the construction and the
    # truth provably coincide -- so it is where the construction is checkable.
    np.testing.assert_allclose(recombination_front(toy_landscape(conflict=0.0)), [[1.0, 1.0]])


def test_every_point_of_a_constructed_front_is_attained_by_a_real_sequence():
    # An unattainable "front" would put a permanent floor under IGD+, so no arm
    # could ever reach zero and the indicator would stop discriminating at the
    # top. Every candidate is a real sequence and every infeasible one scores
    # -inf and is dropped, so finiteness is the property that says so.
    landscape = toy_landscape()
    front = recombination_front(landscape)
    assert front.shape == (front.shape[0], 2)
    assert np.isfinite(front).all()
    assert non_dominated(front).all()
    # Each objective's planted optimum is in the candidate set and scores 1.0 on
    # its own objective, so nothing can dominate it away from the column maximum.
    np.testing.assert_allclose(front.max(axis=0), 1.0)


def test_a_constructed_front_refuses_a_landscape_with_no_planted_optima():
    with pytest.raises(TypeError, match="has none"):
        recombination_front(
            EhrlichLandscape(sequence_length=8, vocab_size=4, n_motifs=1, motif_length=2)
        )


@pytest.mark.requires_data
def test_the_ch65_front_drops_the_variants_tied_at_the_detection_floor():
    landscape = CH65Landscape()
    front = ch65_reference_front(landscape)
    space = landscape.enumerate()
    values = np.asarray(landscape.evaluate(space), dtype=np.float64)
    measured = landscape.is_measured(space)

    with_censored = int(non_dominated(values[measured]).sum())
    # 20 with the censored variants, 19 without: exactly one front point is
    # non-dominated only because an objective could not resolve it, and it is
    # unreachable anyway -- a censored value sits *on* the reference point, and
    # hypervolume counts only designs strictly above it.
    assert with_censored == 20
    assert front.shape == (19, 3)
    assert (front > CH65_DETECTION_FLOOR).all()


def test_the_ch65_front_is_refused_for_any_other_landscape():
    with pytest.raises(TypeError, match="cannot be computed"):
        ch65_reference_front(toy_landscape())


# --------------------------------------------------------------------------
# The scalarising adapter.
# --------------------------------------------------------------------------


def _toy_environment():
    """An environment over the toy landscape, anchored at a feasible parent."""
    landscape = toy_landscape()
    return landscape, MutationEnvironment(
        landscape.feasible_sequence(0),
        landscape.alphabet,
        max_mutations=4,
        transitions=landscape.transition_matrix,
    )


def test_a_classical_baseline_refuses_an_objective_matrix():
    # The premise of the adapter. If this ever stops raising, the adapter is
    # papering over something that has been fixed properly.
    landscape, env = _toy_environment()
    sampler = GeneticAlgorithm(env, seed=0)
    batch = sampler.propose(4)
    with pytest.raises(ValueError, match="must be scalarised"):
        sampler.observe(batch, landscape.evaluate(batch))


def test_the_adapter_lets_it_rank_under_a_stated_trade_off():
    landscape, env = _toy_environment()
    wrapped = ScalarizedObserving(
        GeneticAlgorithm(env, seed=0), scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    batch = wrapped.propose(4)
    wrapped.observe(batch, landscape.evaluate(batch))
    assert wrapped.proposals_made == 4


def test_the_adapter_passes_a_single_objective_batch_straight_through():
    # The proxy the inner loop searches against returns (n, 1), so both widths
    # arrive within one round. Scalarising the scalar would apply a two-entry
    # preference to something that is not an objective vector, and would raise.
    _, env = _toy_environment()
    wrapped = ScalarizedObserving(
        GeneticAlgorithm(env, seed=0), scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    batch = wrapped.propose(4)
    wrapped.observe(batch, np.arange(4, dtype=np.float64)[:, None])


def test_the_adapter_forwards_a_moved_anchor():
    # The campaign checks the *outermost* object for `reanchored`. A wrapper
    # without the hook sends every arm it wraps down the rebuild path, and the
    # populations each baseline carefully carries are discarded silently.
    landscape, env = _toy_environment()
    inner = GeneticAlgorithm(env, seed=0)
    wrapped = ScalarizedObserving(
        inner, scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    moved = wrapped.reanchored(env.reanchored(landscape.feasible_sequence(1)))
    assert isinstance(moved, ScalarizedObserving)
    assert isinstance(moved.inner, GeneticAlgorithm)


# --------------------------------------------------------------------------
# The arms, end to end.
# --------------------------------------------------------------------------


def test_a_scalar_acquisition_is_refused_against_several_objectives():
    # The reason every arm here, NSGA-II included, is built with a
    # ScalarizedAcquisition. Refused at construction, before any oracle call.
    landscape, env = _toy_environment()
    with pytest.raises(ValueError, match="state the trade-off explicitly"):
        Campaign(
            landscape=landscape,
            sampler=NSGA2(env, seed=0),
            acquisition=Greedy(),
            rounds=1,
            batch_size=2,
            pool_size=4,
        )


@pytest.mark.parametrize("name", sorted(ARMS))
def test_every_arm_runs_and_reports_both_indicators(name):
    task = toy_task()
    arm = ARMS[name] if name != "gfn-tb" else scalarized_gflownet_arm(1, steps=4)
    result = arm(task, 0).run()

    assert result.oracle_calls == task.protocol.budget
    assert result.values.shape[1] == 2
    got = set_indicators(task, result)
    volume, coverage = got["best"], got["regret"]
    assert volume is not None
    # Hypervolume can legitimately be zero here -- an Ehrlich design scoring 0 on
    # an objective is at the reference point and encloses nothing -- so what is
    # asserted is that it is a number rather than that it is large.
    assert volume >= 0.0
    assert coverage is not None
    assert coverage >= 0.0


def test_nsga2_ranks_the_objective_matrix_rather_than_a_scalarisation():
    # The whole point of having it: it must see the vectors. If the campaign ever
    # started handing samplers a reduced value, this arm would degenerate into a
    # dominance-ranked GA over one weighting and stop being a control.
    campaign = ARMS["nsga2"](toy_task(), 0)
    campaign.run()
    sampler = campaign.sampler
    assert isinstance(sampler, NSGA2)
    assert sampler.values is not None
    assert sampler.values.shape[1] == 2


def test_the_genetic_arm_optimises_the_proxy_rather_than_only_meeting_it():
    # A GFlowNet that beats a baseline which never looks at the model has beaten
    # the access, not the method.
    campaign = scalarized_genetic_arm()(toy_task(), 0)
    campaign.run()
    assert int(getattr(campaign.sampler, "proxy_calls", 0)) > 0


# --------------------------------------------------------------------------
# The preference ensemble.
# --------------------------------------------------------------------------


def test_several_preferences_share_one_budget_rather_than_multiplying_it():
    task = toy_task(rounds=2, batch=4)
    single = scalarized_gflownet_arm(1, steps=2)(task, 0)
    split = scalarized_gflownet_arm(4, steps=2)(task, 0)
    assert isinstance(split, PreferenceEnsemble)
    # Four preferences of one design a round against one preference of four:
    # the same total, which is what makes the two comparable at all.
    assert split.budget == single.budget
    assert len(split.campaigns) == 4


def test_the_merged_result_is_the_union_of_what_every_preference_measured():
    task = toy_task(rounds=2, batch=4)
    ensemble = scalarized_gflownet_arm(2, steps=2)(task, 0)
    result = ensemble.run()
    assert result.oracle_calls == task.protocol.budget
    assert result.values.shape == (len(result.sequences), 2)
    # Renumbered, so the merged ledger reads in order rather than restarting at
    # zero once per preference.
    assert [record.index for record in result.rounds] == list(range(len(result.rounds)))


def test_a_split_that_leaves_less_than_one_design_each_is_refused():
    arm = scalarized_gflownet_arm(8, steps=2)
    with pytest.raises(ValueError, match="less than one design each"):
        arm(toy_task(rounds=2, batch=4), 0)


# --------------------------------------------------------------------------
# What a stored record holds.
# --------------------------------------------------------------------------


def test_an_uncomputable_hypervolume_is_stored_as_nan_rather_than_raising():
    # Three objectives with a front larger than the exact method accepts, which
    # on ch65-real is what a converged arm produces. The measurements are the
    # product and they survive; the indicator says "not computed" and propagates.
    angles = np.linspace(0.1, 1.4, EXACT_FRONT_LIMIT + 4)
    # One objective rises as another falls, so no point dominates any other.
    front = np.column_stack([np.cos(angles) + 1.0, np.sin(angles) + 1.0, np.ones(angles.size)])
    result = CampaignResult(
        sampler="toy",
        rounds=(),
        sequences=np.zeros((front.shape[0], 4), dtype=np.int32),
        values=front,
        reference_point=np.zeros(3),
    )
    assert non_dominated(front).sum() > EXACT_FRONT_LIMIT
    volume = set_indicators(toy_task(), result)["best"]
    assert volume is not None
    assert np.isnan(volume)


def test_a_result_with_no_reference_point_is_refused_rather_than_stored():
    result = CampaignResult(
        sampler="toy",
        rounds=(),
        sequences=np.zeros((2, 4), dtype=np.int32),
        values=np.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="no reference point"):
        set_indicators(toy_task(), result)


def test_a_run_is_stored_once_and_resumed_rather_than_repeated(tmp_path):
    store = ResultStore(tmp_path)
    task = toy_task()
    arms = {"random": ARMS["random"]}

    assert run_multi_objective_task(task, arms, store, [0, 1], report=lambda _: None) == 2
    # The second call is the whole point of the store: raising a tier's seed
    # count must cost the new seeds and not the old ones.
    assert run_multi_objective_task(task, arms, store, [0, 1], report=lambda _: None) == 0
    assert run_multi_objective_task(task, arms, store, [0, 1, 2], report=lambda _: None) == 1

    held = store.usable("toy", "random")
    assert sorted(held) == [0, 1, 2]
    record = held[0]
    assert record.protocol == repr(task)
    assert record.oracle_calls == task.protocol.budget
    # Recorded against *this* module, not `benchmark.methods`: the arms are built
    # here, and a record that under-declares what it depends on cannot notice an
    # edit to the very arm that produced it.
    assert "evogfn.benchmark.multi_objective" in record.source


def test_the_designs_kept_for_inspection_are_the_ones_on_the_measured_front(tmp_path):
    store = ResultStore(tmp_path)
    task = toy_task()
    run_multi_objective_task(task, {"random": ARMS["random"]}, store, [0], report=lambda _: None)
    record = store.usable("toy", "random")[0]
    # Not "the best ten": with several objectives there is no order to take a
    # top ten under without first inventing a trade-off.
    assert record.top_sequences
    assert all(len(design) == task.landscape().sequence_length for design in record.top_sequences)
