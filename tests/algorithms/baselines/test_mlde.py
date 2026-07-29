"""Tests for MLDE.

The failure that matters here is a quiet one: an MLDE that never leaves its
random-screening stage, or whose fitted model ranks no better than chance, is
indistinguishable from random mutagenesis in a results table, and it would make
the project's central comparison look far more favourable than it is. So the
handover is tested explicitly, and the ranking is tested against the null it
would otherwise silently collapse into.

The second failure, added with the ensemble, is subtler and worse: shipping a
*weakened* version of somebody else's published method and then reporting that we
beat it. Two things guard against that here. The ensemble is measured against the
single kernel ridge it replaced, on a landscape with the pairwise epistasis MLDE
exists to capture, so a regression in the baseline's strength shows up as a test
failure rather than as a favourable number. And the budget arithmetic -- that the
published protocol needs 480 assays where this repository's campaigns have 384 --
is asserted rather than described, so the handicap cannot quietly disappear.
"""

import numpy as np
import pytest

from evoflownet.algorithms.base import Sampler
from evoflownet.algorithms.baselines import (
    MLDE,
    PUBLISHED_BATCH_SIZE,
    PUBLISHED_BUDGET,
    PUBLISHED_CV_FOLDS,
    PUBLISHED_MODELS_AVERAGED,
    PUBLISHED_TRAINING_SIZE,
    RandomMutagenesis,
)
from evoflownet.core import Alphabet
from evoflownet.env.mutation import MutationEnvironment

#: The repository's four-plate campaign budget, which MLDE-as-published exceeds.
FOUR_PLATE_BUDGET = 384


def make_env(length=8, symbols="ABCD", max_mutations=6, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def constrained_transitions(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def toy_landscape(sequences):
    """Reward sequences for containing token 1, so improvement is detectable."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


def epistatic_landscape(length, vocab, seed):
    """An additive field plus random pairwise couplings.

    This is a Potts model, and it is the landscape MLDE is *for*: a purely
    additive fitness needs no interaction terms and would let a linear model tie
    the ensemble, which would test nothing about whether epistasis is captured.
    """
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(length, vocab))
    pairs = [(i, (i * 3 + 1) % length) for i in range(length) if i != (i * 3 + 1) % length]
    couplings = rng.normal(size=(len(pairs), vocab, vocab)) * 2.0

    def score(sequences):
        X = np.asarray(sequences)
        total = field[np.arange(X.shape[1]), X].sum(axis=1)
        for index, (i, j) in enumerate(pairs):
            total += couplings[index, X[:, i], X[:, j]]
        return total[:, None]

    return score


def single_kernel_ridge(train_X, train_y, query_X, length):
    """The model this file used to fit: one kernel ridge, degree 2, alpha 1.

    Reproduced here rather than imported because it no longer exists in the
    source. It is the thing the ensemble has to beat to have been worth adding,
    so its two hyperparameters are pinned to the old defaults rather than
    exposed -- a tunable reference is not the model that was replaced.
    """

    def gram(left, right):
        return ((left[:, None, :] == right[None, :, :]).sum(axis=2) / length) ** 2

    offset = train_y.mean()
    K = gram(train_X, train_X)
    K[np.diag_indices_from(K)] += 1.0
    dual = np.linalg.solve(K, train_y - offset)
    return gram(query_X, train_X) @ dual + offset


def spearman(a, b):
    """Rank correlation, which is all MLDE's predictions are ever used for."""
    ranked_a = np.argsort(np.argsort(a)).astype(float)
    ranked_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def held_out_comparison(seed, length=12, vocab=4, max_mutations=10, training=96):
    """Rank correlation of the ensemble and of the single ridge, on one seed."""
    env = make_env(length=length, symbols="ABCDEFGHIJ"[:vocab], max_mutations=max_mutations)
    score = epistatic_landscape(length, vocab, seed)
    drawer = RandomMutagenesis(env, seed=100 + seed)
    train_X = np.asarray(drawer.propose(training))
    held_X = np.asarray(drawer.propose(256))
    train_y = score(train_X)
    held_y = score(held_X)[:, 0]

    sampler = MLDE(env, training_size=training, seed=seed)
    sampler.observe(train_X, train_y)
    ensemble = sampler.predict(held_X)
    single = single_kernel_ridge(train_X, train_y[:, 0], held_X, length)
    return spearman(ensemble, held_y), spearman(single, held_y)


def train(sampler, rounds=2, size=32, landscape=toy_landscape):
    """Run the sampler through enough rounds to finish its training sample."""
    for _ in range(rounds):
        proposals = sampler.propose(size)
        sampler.observe(proposals, landscape(proposals))
    return sampler


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(MLDE(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert MLDE(env, training_size=16).propose(16).shape == (16, env.sequence_length)

    def test_fitted_proposals_have_the_right_shape(self):
        env = make_env()
        sampler = train(MLDE(env, training_size=32), rounds=2, size=32)
        assert sampler.propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        env = make_env(max_mutations=2)
        sampler = MLDE(env, training_size=32, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_proposals_are_counted(self):
        sampler = MLDE(make_env(), training_size=1000)
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_screening_the_pool_is_charged_as_proposals(self):
        # The published method ranks an exhaustive library. Generating the pool
        # that stands in for one is a real cost, and the base class exists to
        # make exactly this kind of discarded work visible.
        sampler = train(MLDE(make_env(), training_size=32, pool_multiplier=4), rounds=2, size=32)
        before = sampler.proposals_made
        sampler.propose(16)
        assert sampler.proposals_made - before >= 16 * 4

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        assert np.array_equal(
            MLDE(env, seed=5).propose(16),
            MLDE(env, seed=5).propose(16),
        )

    def test_a_whole_campaign_is_reproducible(self):
        env = make_env()
        runs = []
        for _ in range(2):
            sampler = MLDE(env, training_size=32, seed=13)
            batches = []
            for _ in range(4):
                proposals = sampler.propose(32)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])

    def test_the_label_says_whether_feasibility_is_enforced(self):
        env = make_env()
        assert MLDE(env).name == "MLDE"
        assert MLDE(env, feasible_only=True).name == "MLDE (feasible)"


class TestTheTwoStageProtocol:
    def test_it_screens_at_random_until_the_training_sample_is_complete(self):
        env = make_env()
        sampler = MLDE(env, training_size=64, seed=0)
        proposals = sampler.propose(32)
        sampler.observe(proposals, toy_landscape(proposals))
        assert not sampler.is_fitted
        assert sampler.training_examples == 32

    def test_the_random_stage_is_exactly_random_mutagenesis(self):
        # "Sample the library uniformly" is what the protocol says, so the null
        # and MLDE's opening round should be the same draw.
        env = make_env()
        assert np.array_equal(
            MLDE(env, seed=2).propose(24),
            RandomMutagenesis(env, seed=2).propose(24),
        )

    def test_the_model_takes_over_once_the_sample_is_complete(self):
        sampler = train(MLDE(make_env(), training_size=32), rounds=1, size=32)
        sampler.propose(16)
        assert sampler.is_fitted

    def test_failed_assays_do_not_count_toward_the_training_sample(self):
        # A well that did not report is not a training point, and treating it as
        # one would hand over to a model fitted on fewer measurements than asked.
        env = make_env()
        sampler = MLDE(env, training_size=32, seed=0)
        proposals = sampler.propose(32)
        sampler.observe(proposals, np.full((32, 1), -np.inf))
        assert sampler.training_examples == 0
        assert not sampler.is_fitted

    def test_the_published_split_is_available_and_is_what_wittmann_reports(self):
        assert PUBLISHED_TRAINING_SIZE == 384
        assert PUBLISHED_BATCH_SIZE == 96


class TestTheModelActuallyRanks:
    def test_the_top_predictions_beat_a_random_draw(self):
        # If they do not, MLDE is random mutagenesis with extra steps, and every
        # comparison against it is worthless.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=96, seed=0), rounds=3, size=32)
        chosen = sampler.propose(32)
        drawn = RandomMutagenesis(env, seed=1).propose(32)
        assert toy_landscape(chosen).mean() > toy_landscape(drawn).mean()

    def test_proposals_come_back_ranked_best_first(self):
        # The campaign takes a prefix of the pool when it has no surrogate of its
        # own, so the order is the interface, not a convenience.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=96, seed=0), rounds=3, size=32)
        proposals = sampler.propose(64)
        head = toy_landscape(proposals[:16]).mean()
        tail = toy_landscape(proposals[-16:]).mean()
        assert head > tail

    def test_it_improves_over_rounds(self):
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = MLDE(env, training_size=64, seed=0)
        first = last = 0.0
        for index in range(8):
            proposals = sampler.propose(32)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first

    def test_dropping_the_pairwise_kernel_still_fits(self):
        # Degree 1 removes the pairwise-epistasis members, which is the control
        # that says how much of MLDE's advantage needs interactions at all. It is
        # not a *purely* additive ablation -- the local-kernel and k-NN members
        # are nonlinear and stay -- and the docstring says so.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = train(MLDE(env, training_size=64, kernel_degree=1, seed=0), rounds=2, size=32)
        chosen = sampler.propose(32)
        assert sampler.is_fitted
        assert not any(name.startswith("poly2") for name in sampler.members)
        assert env.is_reachable(chosen).all()

    def test_it_does_not_re_propose_what_has_already_been_assayed(self):
        # A lab does not re-order a variant it has measured, and a model whose
        # argmax is a variant already in its training set would otherwise spend
        # every remaining round re-proposing it.
        env = make_env(length=10, symbols="ABCD", max_mutations=8)
        sampler = MLDE(env, training_size=64, seed=0)
        measured: set[bytes] = set()
        for _ in range(4):
            proposals = sampler.propose(32)
            sampler.observe(proposals, toy_landscape(proposals))
            measured.update(row.tobytes() for row in np.ascontiguousarray(proposals))
        chosen = np.ascontiguousarray(sampler.propose(32))
        assert not any(row.tobytes() in measured for row in chosen)


class TestFeasibility:
    def test_a_feasible_only_sampler_only_emits_constructible_designs(self):
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3)])
        env = make_env(length=8, symbols="ABCD", max_mutations=3, transitions=transitions)
        sampler = MLDE(env, training_size=32, feasible_only=True, seed=0)
        for _ in range(4):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))


class TestTheEnsemble:
    def test_the_roster_holds_more_than_one_model_class(self):
        # The point of the rewrite. A roster of one is a single model wearing an
        # ensemble's name, which is what this file used to ship.
        members = MLDE(make_env()).members
        assert len(members) > 1
        assert any(name.startswith("poly") for name in members)
        assert any(name.startswith("local") for name in members)
        assert any(name.startswith("knn") for name in members)

    def test_it_averages_only_the_cross_validated_best_few(self):
        # Wittmann et al. rank by cross-validation error and average the top
        # three. Averaging the whole roster would drag the good members toward
        # the bad ones, which is the failure this selection step exists to avoid.
        sampler = train(MLDE(make_env(), training_size=64), rounds=2, size=32)
        sampler.propose(16)
        assert len(sampler.selected_members) == PUBLISHED_MODELS_AVERAGED
        assert set(sampler.selected_members) <= set(sampler.members)
        assert len(set(sampler.selected_members)) == PUBLISHED_MODELS_AVERAGED

    def test_nothing_is_selected_before_the_model_takes_over(self):
        assert MLDE(make_env(), training_size=64).selected_members == ()

    def test_n_averaged_controls_how_many_are_averaged(self):
        sampler = train(MLDE(make_env(), training_size=64, n_averaged=1), rounds=2, size=32)
        sampler.propose(16)
        assert len(sampler.selected_members) == 1

    def test_n_averaged_is_clamped_to_the_roster(self):
        # Asking for more members than exist is a configuration error the caller
        # cannot act on, so it is met by averaging everything rather than raising.
        sampler = train(MLDE(make_env(), training_size=64, n_averaged=999), rounds=2, size=32)
        sampler.propose(16)
        assert len(sampler.selected_members) == len(sampler.members)

    def test_the_selection_is_reproducible(self):
        # Selection is part of the fitted model, so a campaign is only
        # reproducible if the same seed picks the same members.
        runs = []
        for _ in range(2):
            sampler = train(MLDE(make_env(), training_size=64, seed=11), rounds=2, size=32)
            sampler.propose(16)
            runs.append(sampler.selected_members)
        assert runs[0] == runs[1]

    def test_predicting_before_anything_is_measured_is_refused(self):
        env = make_env()
        with pytest.raises(RuntimeError, match="at least 2 finite measurements"):
            MLDE(env).predict(np.zeros((4, env.sequence_length), dtype=np.int32))

    def test_it_predicts_better_than_the_single_kernel_ridge_it_replaced(self):
        # The reason for the rewrite, and the number that has to hold: on a
        # landscape with pairwise epistasis the ensemble must rank held-out
        # variants better than the lone degree-2 kernel ridge this file used to
        # fit. If it does not, the baseline was weakened, not strengthened, and
        # every comparison drawn against it is worth less than it looks.
        results = [held_out_comparison(seed) for seed in range(8)]
        ensemble = np.array([rho for rho, _ in results])
        single = np.array([rho for _, rho in results])
        assert ensemble.mean() > single.mean()
        assert (ensemble > single).sum() >= 6

    def test_it_still_wins_at_the_published_training_size(self):
        # Where MLDE is actually meant to run. Selection has four times the data
        # here, so this is the comparison least confounded by selection noise.
        results = [held_out_comparison(seed, training=PUBLISHED_TRAINING_SIZE) for seed in range(4)]
        ensemble = np.array([rho for rho, _ in results])
        single = np.array([rho for _, rho in results])
        assert ensemble.mean() > single.mean()

    def test_a_fitted_ensemble_only_proposes_reachable_designs(self):
        # The ensemble ranks a pool; a bug in the ranking must not be able to
        # promote something the environment cannot build.
        env = make_env(length=10, symbols="ABCD", max_mutations=3)
        sampler = MLDE(env, training_size=32, seed=4)
        for _ in range(5):
            proposals = sampler.propose(32)
            sampler.observe(proposals, toy_landscape(proposals))
        assert sampler.is_fitted
        assert env.is_reachable(sampler.propose(32)).all()

    def test_folds_shrink_rather_than_break_on_a_tiny_training_set(self):
        # `training_size` can legitimately be smaller than `cv_folds`; a fold
        # with nothing in it would raise mid-campaign rather than at construction.
        env = make_env()
        sampler = MLDE(env, training_size=3, cv_folds=PUBLISHED_CV_FOLDS, seed=0)
        proposals = sampler.propose(3)
        sampler.observe(proposals, toy_landscape(proposals))
        assert env.is_reachable(sampler.propose(4)).all()
        assert sampler.is_fitted


class TestTheBudgetTension:
    def test_the_published_protocol_does_not_fit_the_four_plate_budget(self):
        # The whole reason the default is a compression. If this ever passes
        # quietly, someone has changed a constant that a results table depends on.
        assert PUBLISHED_BUDGET == PUBLISHED_TRAINING_SIZE + PUBLISHED_BATCH_SIZE
        assert PUBLISHED_BUDGET > FOUR_PLATE_BUDGET
        assert PUBLISHED_TRAINING_SIZE >= FOUR_PLATE_BUDGET

    def test_the_default_is_flagged_as_below_the_published_training_size(self):
        sampler = MLDE(make_env())
        assert sampler.runs_below_published_training_size
        assert sampler.required_budget <= FOUR_PLATE_BUDGET

    def test_the_budget_note_names_the_handicap(self):
        note = MLDE(make_env()).budget_note
        assert str(PUBLISHED_TRAINING_SIZE) in note
        assert str(PUBLISHED_BUDGET) in note
        assert "handicapped" in note

    def test_as_published_runs_wittmanns_own_split(self):
        sampler = MLDE.as_published(make_env(), seed=3)
        assert not sampler.runs_below_published_training_size
        assert sampler.required_budget == PUBLISHED_BUDGET
        assert "handicapped" not in sampler.budget_note

    def test_as_published_screens_at_random_for_the_whole_four_plate_budget(self):
        # The concrete form of the tension: run the published method inside this
        # repository's budget and it never gets to propose a designed variant.
        env = make_env()
        sampler = MLDE.as_published(env, seed=0)
        for _ in range(FOUR_PLATE_BUDGET // PUBLISHED_BATCH_SIZE):
            proposals = sampler.propose(PUBLISHED_BATCH_SIZE)
            sampler.observe(proposals, toy_landscape(proposals))
        assert sampler.training_examples == FOUR_PLATE_BUDGET
        assert not sampler.is_fitted

    def test_as_published_does_fit_once_it_is_given_its_own_budget(self):
        env = make_env()
        sampler = MLDE.as_published(env, seed=0)
        for _ in range(PUBLISHED_BUDGET // PUBLISHED_BATCH_SIZE):
            proposals = sampler.propose(PUBLISHED_BATCH_SIZE)
            sampler.observe(proposals, toy_landscape(proposals))
        assert sampler.is_fitted
        assert env.is_reachable(sampler.propose(PUBLISHED_BATCH_SIZE)).all()

    def test_the_default_training_size_is_unchanged(self):
        # Guards the instruction that strengthening the model must not silently
        # move the operating point of an existing benchmark arm.
        assert MLDE(make_env()).required_budget == 96 + PUBLISHED_BATCH_SIZE


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("training_size", 0, "training_size must be at least 1"),
            ("pool_multiplier", 0, "pool_multiplier must be at least 1"),
            ("max_attempts", 0, "max_attempts must be at least 1"),
            ("n_averaged", 0, "n_averaged must be at least 1"),
            ("ridge_alpha", -1.0, "ridge_alpha must not be negative"),
            ("kernel_degree", 0, "kernel_degree must be at least 1"),
            ("cv_folds", 1, "cv_folds must be at least 2"),
        ],
    )
    def test_an_impossible_configuration_is_refused(self, field, value, message):
        with pytest.raises(ValueError, match=message):
            MLDE(make_env(), **{field: value})
