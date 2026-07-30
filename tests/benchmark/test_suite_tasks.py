"""Tests for how the suite's tasks are configured, and what a run stores.

Three failures are under test here, and all three had already happened.

**A campaign that cannot move.** Every task anchored its search at the wild type
for its whole life, so four rounds of a four-mutation budget reached four
mutations rather than sixteen and the planted optimum was outside the search
space by construction. `test_a_reanchored_campaign_moves_its_anchor` is the
end-to-end check that the mechanism is actually wired through the methodologies,
and `test_the_optimum_is_reachable_under_the_configured_budget` re-derives, per
task, the audit numbers the suite now declares rather than trusting them.

**A regret against a target nothing could reach.** `run_task` stored
``landscape.optimum - best``, which on ``large-space`` was 95% floor. The tests
here pin the stored number to the *attainable* optimum instead, and pin the
absence of one where no audit exists -- an unaudited task storing no regret is
the behaviour, not an oversight.

**A declaration that drifts from the landscape.** A declared bound above the
landscape's own optimum claims a design scoring above the maximum, and a
declared bound below what an arm reaches makes every regret on the task
negative. Both are refused rather than reported.

The heavier re-derivations are marked ``slow``. ``large-space`` is not among
them at all: its beam search at L=256 runs for minutes, and the place to
re-check it is ``experiments/audit_optima.py``, which exists for exactly that.
"""

import itertools

import numpy as np
import pytest

from evogfn.benchmark.attainable import attainable_optimum, reanchored_attainable
from evogfn.benchmark.methods import BASELINES
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    CAPPED,
    MAIN,
    budget_gradient,
    objective_task,
    rounds_curve,
    run_task,
)
from evogfn.benchmark.tasks import Attainable, Task
from evogfn.landscapes.ehrlich import EhrlichLandscape

#: A landscape small enough to run a real campaign against inside a test, and
#: chosen so that random mutagenesis improves on the parent more than once in
#: four rounds. That last part is not incidental: the campaign moves its anchor
#: only on a *strict* improvement, and an Ehrlich reward is a product of
#: quantised terms and so flat across most of a neighbourhood, which means an
#: arm can run a whole campaign without the anchor ever moving. A toy where that
#: happened would make this file pass while testing nothing.
TOY = {
    "sequence_length": 16,
    "vocab_size": 4,
    "n_motifs": 1,
    "motif_length": 4,
    "quantization": 4,
    "max_spacing": 2,
    "transition_density": 0.5,
    "seed": 2,
}

#: Tasks whose re-anchoring audit is affordable in a test. ``large-space`` is
#: excluded by size rather than by choice; ``gb1-anchor`` has no planted optimum
#: to march to and no anchor to move.
AUDITABLE = ("feasibility", "protocol-alde", "protocol-evolvepro")


def toy_landscape() -> EhrlichLandscape:
    """Build the shared toy instance."""
    return EhrlichLandscape(**TOY)  # type: ignore[arg-type]


def toy_task(*, reanchor: bool, attainable: Attainable | None, rounds: int = 4) -> Task:
    """A task cheap enough to run end to end.

    Args:
        reanchor: Whether the anchor follows the best design measured so far.
        attainable: What to declare the search space contains.
        rounds: Design-build-test-learn cycles.

    Returns:
        The task.
    """
    return Task(
        name="toy",
        purpose="a toy, for testing that the wiring does what the table says",
        build=toy_landscape,
        protocol=Protocol(rounds=rounds, batch_size=16, max_mutations=4),
        max_mutations=4,
        reanchor=reanchor,
        attainable=attainable,
    )


def nominal(task: Task) -> float:
    """The landscape's own optimum, which regret used to be measured against."""
    optimum = task.landscape().optimum
    assert optimum is not None
    return float(np.max(optimum))


# --------------------------------------------------------------------------
# The declaration: consistent with the landscape, and complete.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [*MAIN, *budget_gradient(), *rounds_curve(), objective_task()],
    ids=lambda t: t.name,
)
def test_every_task_declares_what_it_can_reach(task):
    # The gap this closes: a task with no declaration stores regret against the
    # landscape's optimum, which is the failure the whole mechanism replaces.
    # Absence is allowed by the type and must not be allowed by the suite.
    assert task.attainable is not None, (
        f"{task.name} declares no attainable optimum, so every regret stored on it would be "
        f"against a target nobody has checked is reachable"
    )
    audited = task.attainable_optimum(nominal(task))
    assert audited is not None
    assert audited.lower <= audited.upper <= audited.nominal


@pytest.mark.parametrize(
    "task",
    [*MAIN, *budget_gradient(), *rounds_curve(), objective_task()],
    ids=lambda t: t.name,
)
def test_the_search_budget_is_cumulative_only_where_the_anchor_moves(task):
    rounds = task.protocol.rounds
    expected = task.max_mutations * rounds if task.reanchor else task.max_mutations
    assert task.search_budget == expected
    # The whole point of the fix, stated as an assertion: a re-anchored campaign
    # reaches further than one round, and a fixed one never does.
    if task.reanchor and rounds > 1:
        assert task.search_budget > task.max_mutations


def test_the_capped_phrase_is_the_one_the_audit_greps_for():
    # ``experiments/audit_optima.py`` decides "capped on purpose" from
    # "DEFECT" by looking for this substring in a task's purpose. Rewording
    # `CAPPED` without it would turn every Ehrlich task in the suite into a
    # reported defect, and the audit exits non-zero on one.
    assert "mutation budget is deliberately capped" in CAPPED


def test_a_declaration_above_the_landscapes_optimum_is_refused():
    # A bound claiming a design scores above the maximum is a broken audit, and
    # a broken bound reads exactly like a fact once it is in a table.
    task = toy_task(reanchor=False, attainable=Attainable.exactly(2.0, "wishful"))
    with pytest.raises(ValueError, match="above the landscape's own optimum"):
        task.attainable_optimum(1.0)


def test_a_half_declared_interval_is_refused():
    with pytest.raises(ValueError, match="both ends of an interval or as neither"):
        Attainable(lower=0.5, upper=None, source="x")


def test_an_inverted_interval_is_refused():
    with pytest.raises(ValueError, match="disagrees with itself"):
        Attainable.between(0.9, 0.5, "x")


def test_a_bound_without_provenance_is_refused():
    with pytest.raises(ValueError, match="how it was measured"):
        Attainable.exactly(0.5, "  ")


def test_deferring_to_the_landscape_means_no_regret_floor():
    resolved = Attainable.whole_optimum("nothing is out of reach").resolve(
        task="t", budget=4, nominal=8.76
    )
    assert resolved.is_exact
    assert resolved.regret_floor == (0.0, 0.0)


# --------------------------------------------------------------------------
# The audit, re-derived rather than trusted.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", AUDITABLE)
def test_the_optimum_is_reachable_under_the_configured_budget(name):
    """What the task says it can reach, measured at the shape it actually runs.

    Not "is the planted optimum inside the radius": the radius is per round now
    and is deliberately smaller than that distance on every Ehrlich task here.
    The question is whether the *campaign* -- its rounds, its radius and its
    anchor rule together -- can construct a design attaining the declared value.
    """
    task = next(t for t in MAIN if t.name == name)
    declared = task.attainable_optimum(nominal(task))
    assert declared is not None

    if task.reanchor:
        measured = reanchored_attainable(
            task, per_round=task.max_mutations, rounds=task.protocol.rounds
        )
    else:
        measured = attainable_optimum(task)

    # The searched bound is witnessed by a design the environment built, so it
    # is a claim about reachability rather than about the search's luck.
    assert measured.lower >= declared.lower - 1e-9, (
        f"{name} declares it can reach {declared.lower} and the audit reached only "
        f"{measured.lower}; the declaration is optimistic"
    )
    assert measured.upper <= declared.upper + 1e-9, (
        f"{name} declares an upper bound of {declared.upper} and the audit certifies "
        f"{measured.upper}; the declaration is looser than what was proved"
    )


@pytest.mark.slow
def test_the_diagnostics_reach_their_optimum_at_the_shared_radius():
    # Seven diagnostics, one landscape, one radius. Measured at the fewest rounds
    # any of them runs, since rounds only add reach -- so this one run covers
    # the lot without paying for it seven times.
    task = objective_task()
    fewest = min(t.protocol.rounds for t in (*budget_gradient(), *rounds_curve(), objective_task()))
    measured = reanchored_attainable(task, per_round=task.max_mutations, rounds=fewest)
    declared = task.attainable_optimum(nominal(task))
    assert declared is not None
    assert measured.lower >= declared.lower - 1e-9


# --------------------------------------------------------------------------
# The mechanism, end to end through a real methodology.
# --------------------------------------------------------------------------


def test_a_reanchored_campaign_moves_its_anchor():
    # The measurement the whole of job one exists to produce. `anchor_distance`
    # flat at zero across every round is what a campaign that re-searched one
    # Hamming ball for its whole life looked like, and it looked like nothing at
    # all in the report.
    campaign = BASELINES["random"](toy_task(reanchor=True, attainable=None), 0)
    trace = campaign.run().anchor_trace()

    assert len(trace) == 4
    assert trace[0] == 0, "the first round searches from the wild type by definition"
    assert all(later >= earlier for earlier, later in itertools.pairwise(trace)), (
        f"anchor distance went backwards: {trace}; the anchor only moves on an improvement"
    )
    assert max(trace) > 0, f"the anchor never moved: {trace}"


def test_a_fixed_anchor_campaign_never_moves():
    campaign = BASELINES["random"](toy_task(reanchor=False, attainable=None), 0)
    result = campaign.run()
    assert result.anchor_trace() == [0, 0, 0, 0]
    # And the environment is the object it was handed, not a re-anchored copy.
    assert campaign.environment is not None
    assert np.array_equal(campaign.environment.parent, toy_landscape().feasible_sequence(0))


def test_a_reanchored_campaign_outreaches_a_fixed_one():
    # Same sampler, same seed, same radius: the only difference is whether the
    # anchor was allowed to follow the ledger.
    moved = BASELINES["random"](toy_task(reanchor=True, attainable=None), 0).run()
    fixed = BASELINES["random"](toy_task(reanchor=False, attainable=None), 0).run()
    wild_type = toy_landscape().feasible_sequence(0)

    def furthest(result):
        return int((np.asarray(result.sequences) != wild_type[None, :]).sum(axis=1).max())

    assert furthest(moved) > 4, "a re-anchored campaign should leave the first Hamming ball"
    assert furthest(fixed) <= 4, "a fixed anchor cannot measure anything outside its own ball"


@pytest.mark.parametrize("name", ["random", "hill-climb", "genetic", "cmaes", "mlde"])
def test_every_baseline_can_follow_a_moved_anchor(name):
    # The campaign refuses at construction when a sampler can neither be informed
    # of a move nor rebuilt for it, so this is the check that every methodology
    # supplies a factory -- and it is a construction-time check because that is
    # when the refusal happens, before a quarter of the budget has been spent.
    campaign = BASELINES[name](toy_task(reanchor=True, attainable=None, rounds=2), 0)
    assert campaign.environment is not None


# --------------------------------------------------------------------------
# What a run stores: regret against the attainable optimum, or nothing.
# --------------------------------------------------------------------------


def stored(tmp_path, task):
    """Run one arm on one seed and read back the record it wrote."""
    store = ResultStore(tmp_path)
    run_task(task, {"random": BASELINES["random"]}, store, [0], report=lambda _: None)
    return store.load(task.name, "random")[0]


def test_regret_is_stored_against_the_attainable_optimum(tmp_path):
    # 0.5 rather than 1.0, so the two candidate targets are far apart and the
    # test cannot pass by coincidence.
    declared = Attainable.exactly(0.5, "declared for the test")
    record = stored(tmp_path, toy_task(reanchor=True, attainable=declared))

    assert record.regret == pytest.approx(0.5 - record.best)
    assert record.regret != pytest.approx(
        nominal(toy_task(reanchor=True, attainable=declared)) - record.best
    )


def test_an_arm_at_the_attainable_optimum_has_no_regret_left(tmp_path):
    # A task an arm has exhausted reports zero or below, which is what makes
    # "solved" detectable in the report rather than something to be inferred
    # from a small number.
    floor = Attainable.exactly(0.0, "everything is at least this good")
    record = stored(tmp_path, toy_task(reanchor=True, attainable=floor))
    assert record.regret is not None
    assert record.regret <= 0.0


def test_an_unaudited_task_stores_no_regret(tmp_path):
    # Absence rather than a plausible wrong number: a regret against an
    # unaudited optimum is indistinguishable from a real one once stored.
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))
    assert record.regret is None
    assert np.isfinite(record.best)


def test_the_stored_provenance_names_the_radius_and_the_anchor_rule(tmp_path):
    # Rounds and batch size alone cannot tell a 4-mutation fixed-anchor run from
    # a 4-per-round re-anchored one, and those are different experiments.
    moved = stored(tmp_path / "moved", toy_task(reanchor=True, attainable=None))
    fixed = stored(tmp_path / "fixed", toy_task(reanchor=False, attainable=None))
    assert "re-anchored" in moved.protocol
    assert "fixed anchor" in fixed.protocol
    assert "4/round" in moved.protocol
    assert moved.protocol != fixed.protocol
