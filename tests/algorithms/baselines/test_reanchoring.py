"""Tests for what each baseline carries when the campaign moves its anchor.

The campaign's fallback is a rebuild, which is correct and forgetful. Under a
claim that a learned policy transfers across anchors where classical state does
not, a forgetful rebuild is not a neutral default -- it manufactures the gap the
claim is about. So each baseline implements ``reanchored`` and these tests are
about two things: that what it says it carries really arrives, and that what
arrives is legal in the environment it arrives in.

The second half matters as much as the first. Carrying state across a move is
only an improvement if the state still means what it meant; an individual that
now sits outside the mutation budget, or a fitness attached to a sequence that
was reverted underneath it, is worse than having carried nothing, because it is
wrong rather than merely absent.
"""

import numpy as np
import pytest

from evogfn.algorithms.baselines import (
    CMAES,
    MLDE,
    GeneticAlgorithm,
    HillClimbing,
    RandomMutagenesis,
    SimulatedAnnealing,
)
from evogfn.algorithms.baselines.nsga2 import NSGA2
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.loop.campaign import ReanchorableSampler

ALPHABET = Alphabet.from_string("ABCD")
LENGTH = 8


def make_env(max_mutations=4, transitions=None):
    return MutationEnvironment(
        np.zeros(LENGTH, dtype=np.int32),
        ALPHABET,
        max_mutations=max_mutations,
        transitions=transitions,
    )


def constrained_transitions(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def moved_anchor(env, distance=3):
    """A design the environment can be re-anchored at, ``distance`` mutations out."""
    anchor = env.parent
    anchor[:distance] = 1
    return anchor


def toy_landscape(sequences):
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


def pareto_landscape(sequences):
    """Two objectives that trade off, for NSGA-II."""
    array = np.asarray(sequences)
    return np.stack([(array == 1).sum(axis=1), (array == 2).sum(axis=1)], axis=1).astype(np.float64)


#: Every baseline, built and then exercised for a round so that it has state
#: worth carrying before anything is asked to carry it.
def exercised(make, env, landscape=toy_landscape, rounds=2, n=32):
    sampler = make(env)
    for _ in range(rounds):
        proposals = sampler.propose(n)
        sampler.observe(proposals, landscape(proposals))
    return sampler


SINGLE_OBJECTIVE = [
    pytest.param(lambda env: RandomMutagenesis(env, seed=0), id="random"),
    pytest.param(lambda env: HillClimbing(env, seed=0), id="hill-climb"),
    pytest.param(lambda env: SimulatedAnnealing(env, seed=0), id="annealing"),
    pytest.param(lambda env: GeneticAlgorithm(env, population_size=16, seed=0), id="genetic"),
    pytest.param(lambda env: CMAES(env, seed=0), id="cmaes"),
    pytest.param(lambda env: MLDE(env, training_size=8, seed=0), id="mlde"),
]

EVERY = [*SINGLE_OBJECTIVE, pytest.param(lambda env: NSGA2(env, population_size=16), id="nsga2")]

#: Those whose move consumes nothing from the random stream. The two population
#: samplers are excluded because re-projection draws to choose which surplus
#: substitutions to revert, which advances the shared generator.
NON_DRAWING = [case for case in SINGLE_OBJECTIVE if case.id != "genetic"]

#: The same arms configured to respect the adjacency rule. CMA-ES needs no flag
#: -- its decoder projects onto the constructible set by default, which is what
#: stopped it scoring -inf on every seed -- while the rest reject or redraw.
CONSTRAINT_RESPECTING = [
    pytest.param(lambda env: RandomMutagenesis(env, feasible_only=True, seed=0), id="random"),
    pytest.param(lambda env: HillClimbing(env, feasible_only=True, seed=0), id="hill-climb"),
    pytest.param(lambda env: SimulatedAnnealing(env, feasible_only=True, seed=0), id="annealing"),
    pytest.param(
        lambda env: GeneticAlgorithm(env, population_size=16, feasible_only=True, seed=0),
        id="genetic",
    ),
    pytest.param(lambda env: CMAES(env, seed=0), id="cmaes"),
    pytest.param(lambda env: MLDE(env, training_size=8, feasible_only=True, seed=0), id="mlde"),
]


class TestTheContract:
    @pytest.mark.parametrize("make", EVERY)
    def test_every_baseline_satisfies_the_campaign_protocol(self, make):
        # The campaign decides between carrying and rebuilding on this check,
        # and it is structural: a missing method is a silent downgrade, never an
        # error.
        assert isinstance(make(make_env()), ReanchorableSampler)

    @pytest.mark.parametrize("make", SINGLE_OBJECTIVE)
    def test_proposals_land_inside_the_new_environment(self, make):
        env = make_env()
        moved_env = env.reanchored(moved_anchor(env))
        moved = exercised(make, env).reanchored(moved_env)
        assert moved_env.is_reachable(moved.propose(32)).all()

    @pytest.mark.parametrize("make", CONSTRAINT_RESPECTING)
    def test_proposals_stay_feasible_under_a_transition_constraint(self, make):
        # Only the arms that enforce feasibility at all. The plain ones are the
        # controls the constraint experiment compares against and are *expected*
        # to emit unbuildable designs; what they must not do is emit designs
        # outside the new budget, which the test above covers for all of them.
        transitions = constrained_transitions(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        env = make_env(max_mutations=3, transitions=transitions)
        # An anchor the constraint admits: 0 may follow 0 and 2 may follow 0,
        # and re-anchoring refuses an infeasible design anyway.
        anchor = np.zeros(LENGTH, dtype=np.int32)
        anchor[1] = 2
        moved_env = env.reanchored(anchor)
        moved = exercised(make, env).reanchored(moved_env)
        assert moved_env.is_reachable(moved.propose(32)).all()

    @pytest.mark.parametrize("make", NON_DRAWING)
    def test_the_original_sampler_is_left_alone(self, make):
        # A campaign holds the old sampler until it swaps in the new one, and a
        # move that edited state in place would corrupt the round in progress.
        # Checked against an untouched twin rather than against the mover's own
        # later behaviour, which would pass even if the move had rewritten it.
        env = make_env()
        moved_env = env.reanchored(moved_anchor(env))
        untouched = exercised(make, env)
        original = exercised(make, env)
        original.reanchored(moved_env)
        assert np.array_equal(original.propose(8), untouched.propose(8))

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(
                lambda env: GeneticAlgorithm(env, population_size=16, seed=0), id="genetic"
            ),
            pytest.param(lambda env: NSGA2(env, population_size=16), id="nsga2"),
        ],
    )
    def test_a_population_move_edits_nothing_it_leaves_behind(self, make):
        # The two population samplers are the exception to the test above, and
        # only in one respect: re-projection draws from the shared random stream
        # to pick which surplus substitutions to revert, so the sampler being
        # moved advances by however many individuals had to be edited. What it
        # must not do is edit the population or the scores themselves, which is
        # what would corrupt a round still in progress.
        env = make_env()
        landscape = pareto_landscape if isinstance(make(env), NSGA2) else toy_landscape
        original = exercised(make, env, landscape=landscape)
        population = original.population.copy()
        original.reanchored(env.reanchored(moved_anchor(env)))
        assert np.array_equal(original.population, population)

    @pytest.mark.parametrize("make", SINGLE_OBJECTIVE)
    def test_the_proposal_count_keeps_accumulating(self, make):
        # It is a campaign total. Restarting it at each anchor would understate
        # exactly the cost `proposals_made` exists to report.
        env = make_env()
        original = exercised(make, env)
        spent = original.proposals_made
        assert spent > 0
        assert original.reanchored(env.reanchored(moved_anchor(env))).proposals_made == spent


class TestReproducibility:
    @pytest.mark.parametrize("make", SINGLE_OBJECTIVE)
    def test_a_campaign_across_a_move_is_reproducible_from_the_seed(self, make):
        env = make_env()
        moved_env = env.reanchored(moved_anchor(env))
        runs = []
        for _ in range(2):
            sampler = exercised(make, env)
            sampler = sampler.reanchored(moved_env)
            batches = []
            for _ in range(2):
                proposals = sampler.propose(16)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])

    @pytest.mark.parametrize("make", SINGLE_OBJECTIVE)
    def test_rebuilding_at_the_same_seed_repeats_the_same_proposals(self, make):
        """The defect the rebuild path has and this hook exists to avoid.

        Every constructor here does ``np.random.default_rng(seed)``, so a
        sampler rebuilt at the seed it was first built with restarts its stream
        from the beginning and re-draws, number for number, what it drew before.
        A campaign that rebuilds on every anchor move therefore re-proposes its
        own earlier designs, and the damage is invisible: deduplication drops
        them, so the ledger shows a shrinking *screened* count rather than an
        error, and the round quietly buys fewer measurements than it paid for.

        Pinned as a test because it is a property of the constructor rather than
        a mistake in one file, so nothing else would catch it coming back.
        """
        env = make_env()
        assert np.array_equal(make(env).propose(16), make(env).propose(16))

    @pytest.mark.parametrize("make", SINGLE_OBJECTIVE)
    def test_re_anchoring_carries_the_stream_instead_of_restarting_it(self, make):
        # The fix: the live generator is carried, not the seed. Re-anchored at
        # the *same* anchor on purpose, so the only thing that could make the
        # next batch differ is the stream having advanced -- if it matched, the
        # sampler would be repeating itself exactly as the rebuild path does.
        env = make_env()
        original = exercised(make, env)
        before = original.propose(16)
        after = original.reanchored(env.reanchored(env.parent)).propose(16)
        assert not np.array_equal(before, after)


class TestWhatEachOneCarries:
    """The claims in each ``reanchored`` docstring, checked one at a time."""

    def test_random_mutagenesis_has_nothing_to_carry(self):
        env = make_env()
        moved = RandomMutagenesis(env, seed=0).reanchored(env.reanchored(moved_anchor(env)))
        assert isinstance(moved, RandomMutagenesis)

    def test_hill_climbing_keeps_its_incumbent_and_its_patience(self):
        env = make_env()
        sampler = HillClimbing(env, patience=3, seed=0)
        proposals = sampler.propose(16)
        sampler.observe(proposals, toy_landscape(proposals))
        moved = sampler.reanchored(env.reanchored(env.parent))
        assert moved.best_value == sampler.best_value
        assert np.array_equal(moved._current, sampler._current)
        assert moved._stale == sampler._stale

    def test_hill_climbing_restarts_when_its_incumbent_left_the_ball(self):
        # The campaign picks its anchor through the acquisition rule, not
        # through the climber's bookkeeping, so the two can disagree.
        env = make_env(max_mutations=2)
        sampler = HillClimbing(env, seed=0)
        sampler._current = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        moved_env = env.reanchored(np.array([0, 0, 2, 2, 0, 0, 0, 0], dtype=np.int32))
        moved = sampler.reanchored(moved_env)
        assert np.array_equal(moved._current, moved_env.parent)

    def test_annealing_keeps_its_temperature(self):
        # The finding this test exists for: a cooling schedule is a scalar in
        # objective units and refers to nothing about which ball is searched.
        # A rebuild resets it to `initial_temperature`, so a chain that had
        # cooled to 0.125 by round four would be hot again -- and "the annealer
        # wanders in late rounds" would be a fact about the harness.
        env = make_env()
        sampler = SimulatedAnnealing(env, initial_temperature=1.0, cooling_rate=0.5, seed=0)
        for _ in range(3):
            proposals = sampler.propose(16)
            sampler.observe(proposals, toy_landscape(proposals))
        assert sampler.temperature == pytest.approx(0.125)
        moved = sampler.reanchored(env.reanchored(env.parent))
        assert moved.temperature == pytest.approx(0.125)
        assert SimulatedAnnealing(env, seed=0).temperature == pytest.approx(1.0)

    def test_annealing_keeps_the_chain_where_it_stood(self):
        env = make_env()
        sampler = SimulatedAnnealing(env, seed=0)
        proposals = sampler.propose(16)
        sampler.observe(proposals, toy_landscape(proposals))
        moved = sampler.reanchored(env.reanchored(env.parent))
        assert np.array_equal(moved._current, sampler._current)
        assert moved.current_value == sampler.current_value
        assert moved.best_value == sampler.best_value

    def test_annealing_forgets_its_accepted_value_only_when_it_restarts(self):
        # The value on record is the value of where the chain stands. If the
        # chain has to restart at the anchor, keeping it would have Metropolis
        # compare candidates against a design the chain is no longer on.
        env = make_env(max_mutations=2)
        sampler = SimulatedAnnealing(env, seed=0)
        sampler._current = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        sampler._current_value = 5.0
        moved = sampler.reanchored(
            env.reanchored(np.array([0, 0, 2, 2, 0, 0, 0, 0], dtype=np.int32))
        )
        assert moved.current_value == -np.inf
        assert moved.best_value == sampler.best_value

    def test_the_genetic_population_survives_a_loose_budget_whole(self):
        # Every constrained task in this repository runs a budget of 62 of 64 or
        # 248 of 256 against an anchor that moves a handful of substitutions, so
        # this is the case the benchmark actually meets: nothing is re-projected
        # and the population and its fitnesses transfer intact.
        env = make_env(max_mutations=LENGTH)
        sampler = exercised(lambda e: GeneticAlgorithm(e, population_size=16, seed=0), env)
        carried = sampler.carried_fitness
        assert carried > 0
        moved = sampler.reanchored(env.reanchored(moved_anchor(env)))
        assert np.array_equal(moved.population, sampler.population)
        assert moved.carried_fitness == carried

    def test_a_reprojected_individual_loses_the_measurement_it_no_longer_owns(self):
        # Reverting a substitution makes a different sequence, and the fitness
        # on record was measured on the old one. Keeping it would let selection
        # promote a design on an assay never run on it.
        env = make_env(max_mutations=LENGTH)
        sampler = exercised(lambda e: GeneticAlgorithm(e, population_size=16, seed=0), env)
        tight = MutationEnvironment(moved_anchor(env, distance=6), ALPHABET, max_mutations=1)
        moved = sampler.reanchored(tight)
        assert tight.is_reachable(moved.population).all()
        edited = ~(moved.population == sampler.population).all(axis=1)
        assert edited.any()
        assert not np.isfinite(moved._fitness[edited]).any()

    def test_mlde_carries_its_entire_dataset_and_its_fitted_model(self):
        # The headline of the second job. A training set is pairs of a sequence
        # and a measured value and the kernel is an agreement count, so nothing
        # in the method refers to an anchor: predictions must come out identical,
        # not merely similar.
        env = make_env(max_mutations=LENGTH)
        sampler = MLDE(env, training_size=16, seed=0)
        for _ in range(3):
            proposals = sampler.propose(24)
            sampler.observe(proposals, toy_landscape(proposals))
        assert sampler.is_fitted or sampler.training_examples >= 16

        moved = sampler.reanchored(env.reanchored(moved_anchor(env)))
        assert moved.training_examples == sampler.training_examples

        probe = np.tile(env.parent, (12, 1))
        probe[np.arange(12), np.arange(12) % LENGTH] = 1
        assert np.allclose(moved.predict(probe), sampler.predict(probe))
        assert moved.selected_members == sampler.selected_members

    def test_a_rebuilt_mlde_is_the_baseline_this_avoids(self):
        # Stated as a test because it is the size of the effect: a rebuild puts
        # MLDE back in the random-screening stage, so it re-spends its whole
        # training budget every time the anchor moves.
        env = make_env(max_mutations=LENGTH)
        sampler = MLDE(env, training_size=16, seed=0)
        for _ in range(3):
            proposals = sampler.propose(24)
            sampler.observe(proposals, toy_landscape(proposals))
        rebuilt = MLDE(env.reanchored(moved_anchor(env)), training_size=16, seed=0)
        assert rebuilt.training_examples == 0
        assert sampler.reanchored(env.reanchored(moved_anchor(env))).training_examples > 0

    def test_mlde_refuses_a_change_of_sequence_length(self):
        env = make_env()
        other = MutationEnvironment(np.zeros(LENGTH + 1, dtype=np.int32), ALPHABET)
        with pytest.raises(ValueError, match="sequence length may not"):
            MLDE(env, seed=0).reanchored(other)

    def test_cmaes_carries_the_distribution_it_learned(self):
        # More survives than the intuitive reading suggests: the relaxation is
        # indexed by (position, token) and neither index is anchor-relative, so
        # "position 7 prefers token 3" is still true after the move. Only the
        # decoder is anchored.
        env = make_env(max_mutations=LENGTH)
        sampler = exercised(lambda e: CMAES(e, seed=0), env, rounds=4, n=32)
        moved = sampler.reanchored(env.reanchored(moved_anchor(env)))
        assert np.array_equal(moved.mean_logits, sampler.mean_logits)
        assert moved.sigma == sampler.sigma
        assert np.array_equal(moved._diagonal, sampler._diagonal)
        assert np.array_equal(moved._path_sigma, sampler._path_sigma)

    def test_cmaes_drops_the_pairing_between_draws_and_designs(self):
        # Those sequences came out of the old decoder. Under the new one the
        # same draw decodes elsewhere, so a score arriving after the move would
        # update the rank-mu term against a draw that did not produce it.
        env = make_env(max_mutations=LENGTH)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(32)
        moved = sampler.reanchored(env.reanchored(env.parent))
        before = moved.mean_logits.copy()
        moved.observe(proposals, toy_landscape(proposals))
        assert np.array_equal(moved.mean_logits, before)

    def test_cmaes_refuses_a_relaxation_of_a_different_size(self):
        env = make_env()
        other = MutationEnvironment(np.zeros(LENGTH + 2, dtype=np.int32), ALPHABET)
        with pytest.raises(ValueError, match="sequence length and alphabet may not"):
            CMAES(env, seed=0).reanchored(other)

    def test_nsga2_carries_its_population_and_recomputes_its_fronts(self):
        env = make_env(max_mutations=LENGTH)
        sampler = exercised(lambda e: NSGA2(e, population_size=16), env, landscape=pareto_landscape)
        moved = sampler.reanchored(env.reanchored(moved_anchor(env)))
        assert np.array_equal(moved.population, sampler.population)
        assert np.array_equal(moved.values, sampler.values)
        assert np.array_equal(moved.ranks, sampler.ranks)

    def test_nsga2_drops_the_objectives_of_a_reprojected_individual(self):
        # Half the population inside the new budget and half outside, built by
        # hand so the split is exact rather than whatever breeding happened to
        # produce.
        tight = MutationEnvironment(np.zeros(LENGTH, dtype=np.int32), ALPHABET, max_mutations=2)
        sampler = NSGA2(make_env(), population_size=4)
        sampler._population = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],  # one mutation: inside
                [1, 1, 0, 0, 0, 0, 0, 0],  # two: inside
                [1, 1, 1, 0, 0, 0, 0, 0],  # three: re-projected
                [1, 1, 1, 1, 0, 0, 0, 0],  # four: re-projected
            ],
            dtype=np.int32,
        )
        sampler._values = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])

        moved = sampler.reanchored(tight)
        assert tight.is_reachable(moved.population).all()
        edited = ~(moved.population == sampler.population).all(axis=1)
        assert list(edited) == [False, False, True, True]
        carried, before = moved.values, sampler.values
        assert carried is not None
        assert before is not None
        assert np.array_equal(carried[:2], before[:2])
        assert not np.isfinite(carried[edited]).any()

    def test_nsga2_returns_to_its_founding_state_when_nothing_survives(self):
        # Under a budget tight enough that every individual is edited, no
        # objective vector still belongs to the design it is attached to, so the
        # honest report is that nothing transferred -- which is the founding
        # state. It is also the only safe one: ranks over an all -inf table are
        # ties and the crowding distance across such a front is `inf - inf`,
        # which would randomise the tournament while looking like it ranked.
        env = make_env(max_mutations=LENGTH)
        sampler = exercised(lambda e: NSGA2(e, population_size=16), env, landscape=pareto_landscape)
        tight = MutationEnvironment(moved_anchor(env, distance=6), ALPHABET, max_mutations=1)
        moved = sampler.reanchored(tight)
        assert tight.is_reachable(moved.population).all()
        assert moved.values is None

    def test_nsga2_proposals_land_inside_the_new_environment(self):
        env = make_env()
        moved_env = env.reanchored(moved_anchor(env))
        sampler = exercised(lambda e: NSGA2(e, population_size=16), env, landscape=pareto_landscape)
        assert moved_env.is_reachable(sampler.reanchored(moved_env).propose(32)).all()


class TestTheRebuildStaysReachable:
    """Whether carrying beats a fresh start is measured, not assumed.

    So the fresh start stays available *through* the hook rather than only by
    hiding the hook and letting the campaign's factory fallback take over. The
    difference matters: the fallback rebuilds at the original seed, which makes
    the sampler re-propose designs the campaign has already measured.
    """

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(
                lambda env: GeneticAlgorithm(
                    env, population_size=16, carry_population=False, seed=0
                ),
                id="genetic",
            ),
            pytest.param(
                lambda env: NSGA2(env, population_size=16, carry_population=False),
                id="nsga2",
            ),
        ],
    )
    def test_declining_to_carry_founds_a_fresh_population(self, make):
        env = make_env(max_mutations=LENGTH)
        landscape = toy_landscape if "Genetic" in make(env).name else pareto_landscape
        sampler = exercised(make, env, landscape=landscape)
        anchor = moved_anchor(env)
        moved = sampler.reanchored(env.reanchored(anchor))
        assert np.array_equal(moved.population, np.tile(anchor, (16, 1)))

    def test_declining_to_carry_still_carries_the_random_stream(self):
        # The whole reason this is a parameter rather than an absence: the
        # campaign's factory fallback would rebuild at the original seed and
        # re-propose what it has already measured.
        env = make_env()
        sampler = GeneticAlgorithm(env, population_size=16, carry_population=False, seed=0)
        before = sampler.propose(16)
        after = sampler.reanchored(env.reanchored(env.parent)).propose(16)
        assert not np.array_equal(before, after)
