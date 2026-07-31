r"""The multi-objective suite: what a second objective costs, and when it changes the answer.

Every task in [evogfn.benchmark.suite][] returns one number per design. This
module holds the ones that do not, and it is a separate module rather than five
more rows in `suite.MAIN` because almost nothing carries over. A multi-objective
campaign refuses a scalar acquisition rule at construction, scores itself with
set indicators instead of a regret, needs a reference point *and* a reference
front before either indicator means anything, and has a baseline
([NSGA2][evogfn.algorithms.baselines.nsga2.NSGA2]) that no single-objective task
can run. Interleaving the two would put a `nan` in the regret column of half the
suite and an unusable "best value" in the other half.

## Three tiers, and they are not read the same way

**Main tests** carry results. Two of them, chosen so that between them they say
something about a real assay and something about a controlled one:

* `ch65-real` -- three antibody affinities actually measured by Tite-Seq, on the
  16-site somatic-mutation lattice of CH65. The trade-off here is not a
  construction; the paper it comes from is *about* breadth costing potency.
* `mo-ehrlich-hard` -- two Ehrlich objectives at maximum conflict, at L=10, v=4,
  c=2, k=5, density 0.5, seed 2. The alphabet is what that task is for: four
  letters is DNA/RNA, and it makes $4^{10}$ = 1,048,576 sequences enumerable, so
  the reference front IGD+ measures coverage of is the **true** one. The earlier
  version of this task mirrored `protocol-alde`'s parameters exactly (L=64,
  v=20, c=2, k=4) to make "what does adding an objective cost?" a paired
  question, and that pairing is gone: a 20-letter alphabet is enumerable only to
  L=5, so the price of the pairing was a two-point substitute front. See
  [MO_VOCAB_SIZE][evogfn.benchmark.multi_objective.MO_VOCAB_SIZE] and the
  constants below it for every alignment that moved and what it bought.

**Explanatory sweeps** answer "under what conditions does the ranking change".
They vary conflict at fixed objective count, and objective count at fixed
conflict, on the same instance family at a different seed. They are *not* inputs
to the main table -- a method that wins at low conflict and loses at 1.0 has not
won anything, and the sweep is how that gets said rather than a source of extra
headline rows. The conflict rungs sit on a measured transition rather than on
round numbers; see
[CONFLICT_SWEEP][evogfn.benchmark.multi_objective.CONFLICT_SWEEP].

**The one diagnostic that decides something** is `mo-preferences`. A scalarised
method searches wherever its preference $\omega$ points, so comparing one
preference against a population method is comparing a point against a set, and
the usual response is to assert that this is either fine or fatal. Here it is
measured: the same GFlowNet runs at 1, 4 and 8 preference vectors **at fixed
total budget**, so 8 preferences buys 48 assays each rather than 8 x 384. What
comes out sets how many preferences the main-table GFlowNet arm gets, and
nothing else.

## What is measured, and what refuses to be

Hypervolume above a reference point and IGD+ against a reference front, which is
the pair [CampaignResult][evogfn.loop.ledger.CampaignResult] already reports.
Both need something stated from outside the run, and both are stated here rather
than defaulted:

* **The reference point** is a claim about the assay -- the worst value worth
  counting on each objective. CH65's is the Tite-Seq detection floor `(6, 6, 6)`,
  where the titration stops resolving. The Ehrlich tasks' is the origin, because
  an Ehrlich objective at 0 has satisfied no motif and has genuinely contributed
  nothing. Each task carries its own explicitly instead of relying on the
  landscape to expose one, so a landscape growing or losing a `reference_point`
  property cannot silently move a published number.
* **The reference front** is what IGD+ measures coverage of, and every task here
  now supplies the exact one: CH65's by sweeping its measured library, the
  Ehrlich tasks' by enumerating their $4^{10}$ space. `front_is_exact` is
  therefore `True` throughout, which it was not -- the Ehrlich tasks used to
  carry a constructed subset, and `mo-ehrlich-hard`'s ran to two points.

Hypervolume is **computable here only if the optional `moo` extra is
installed**, and where it is not that is reported rather than patched.
[evogfn.metrics.pareto][] is exact in every dimension it accepts; its built-in
method stops at 16 front points in three or more objectives, and pymoo takes
over past that when it is available. The limit is reached in practice, and by
the wrong runs: on CH65 a random 384-design draw has a front of 2--9 points and
computes either way, while the 384 designs a method actually converges on have a
front of up to 19. `mo-objectives-4`'s exact front alone is 23 points. So a
core-only install loses the hypervolume of the arms that did best -- which is why
`ch65-real` also carries an exact reference front and is read on IGD+. A `nan`
in that column is "not computed", never "no volume".

## Left-censoring on CH65, and what it does to a front

47.4% of CH65's `affinity_SI06` values sit at the detection floor. Those variants
are *tied at a bound*, not measured, and two of them reported at 6.0 may differ
by any amount below it. A front computed over the measured library therefore
risks carrying points that are non-dominated only because a censored objective
could not resolve them.

Measured, the damage is one point: the library's Pareto front is **20 points
including censored variants and 19 excluding them**, and the difference is a
single variant. The reason it is not worse is arithmetical --
[hypervolume][evogfn.metrics.pareto.hypervolume] only counts designs *strictly
above* the reference point, and a censored value sits exactly on it, so the
33,073 variants uncensored on every objective are precisely the ones that
contribute any volume at all. `CH65_FRONT_UNCENSORED` is the front this module
scores IGD+ against, for that reason.

## Arms

Four, and deliberately not five.
[MOGFN-PC][evogfn.rewards.scalarization] -- one preference-conditioned model
covering the whole front -- is out of scope here; `mo-preferences` measures what
running several single-preference models costs instead, which is the comparison
that has to exist before a conditioned model can be said to beat anything.

* `random` -- mutagenesis, no surrogate. The floor: a hypervolume with nothing
  below it is a number rather than a result.
* `nsga2` -- NSGA-II, ranking by dominance. The incumbent, and the arm that picks
  no trade-off at all, which is exactly what scalarisation gives up.
* `genetic+proxy` -- a genetic algorithm optimising the scalarised proxy. The
  strongest classical control: same model access, same trade-off, no learning.
* `gfn-tb` -- a trajectory-balance GFlowNet. The method under test.

Every one of them is handed a
[ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition], NSGA-II
included, because the campaign refuses a scalar rule against a vector-valued
landscape at construction. For NSGA-II that rule ranks nothing -- the arm runs
without a surrogate, so no pool is ever scored -- but the campaign still uses
[reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives] for
the ledger's ``best_so_far`` and for the re-anchoring step. Since every arm gets
the *same* uniform preference for that, the anchor rule is a property of the
protocol rather than of an arm.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from evogfn.acquisition.rules import Greedy, ScalarizedAcquisition, TopK
from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.mutagenesis import RandomMutagenesis
from evogfn.algorithms.baselines.nsga2 import NSGA2
from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.benchmark.determinism import is_deterministic
from evogfn.benchmark.protocol import PLATE, Protocol
from evogfn.benchmark.suite import DIAGNOSTIC_MUTATIONS, Purpose, Tier
from evogfn.benchmark.tasks import Task
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ch65 import CH65_DETECTION_FLOOR, CH65_N_SITES, CH65Landscape
from evogfn.landscapes.multi_ehrlich import MultiEhrlichLandscape
from evogfn.loop.campaign import Campaign
from evogfn.loop.ledger import CampaignResult, RoundRecord
from evogfn.metrics.diversity import diversity
from evogfn.metrics.pareto import non_dominated
from evogfn.models.policy import SequencePolicy
from evogfn.rewards.base import TemperedReward
from evogfn.rewards.scalarization import WeightedSum
from evogfn.surrogate.ensemble import DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from evogfn.benchmark.store import ResultStore
    from evogfn.core.types import Fitness, Tokens
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.rewards.scalarization import Scalarization

#: An arm turns a task and a seed into something runnable. Two shapes qualify: an
#: ordinary [Campaign][evogfn.loop.campaign.Campaign], and a
#: [PreferenceEnsemble][evogfn.benchmark.multi_objective.PreferenceEnsemble],
#: which is several campaigns sharing one budget. A union rather than a Protocol
#: because those are the only two and naming them is more informative than
#: naming the two methods they have in common.
MultiObjectiveMethodology = Callable[["Task", int], "Campaign | PreferenceEnsemble"]

#: CH65's 16 binary sites, and the per-round mutation budget on `ch65-real`. Equal
#: to the sequence length, so the ball of radius 16 around the germline *is* the
#: whole 65,536-variant lattice and there is nothing an anchor could move
#: towards -- the same geometry as ``gb1-anchor`` in the single-objective suite.
CH65_MUTATIONS = CH65_N_SITES

#: Alphabet size on every multi-Ehrlich task. Four, and it was measurement that
#: forced it rather than taste. These tasks are read on IGD+ against a reference
#: front, and a reference front is only worth the word if it is the *exact* one --
#: which means enumerating the space. At the 20-letter alphabet the
#: single-objective suite uses, $20^L$ clears
#: [MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE] at $L = 6$,
#: so **no** Ehrlich task here was enumerable and every one of them fell back to
#: a constructed front. `mo-ehrlich-hard`'s had two points: a coverage indicator
#: with two things to cover.
#:
#: Four letters is DNA/RNA rather than a toy -- aptamer and ribozyme selection are
#: directed evolution over exactly this alphabet -- and $4^{10} = 1{,}048{,}576$
#: is a fifth of the enumeration guard, so the front becomes a fact instead of a
#: construction.
MO_VOCAB_SIZE = 4

#: Sequence length on every multi-Ehrlich task, hard and diagnostic alike.
#:
#: Ten is the largest length at which the space is enumerable *and* two Ehrlich
#: motifs still fit: $4^{11} = 4.2$M is also inside the guard but costs six times
#: the enumeration for measurably **smaller** fronts (4 points at maximum
#: conflict against 5, seed 2), and $4^{12}$ is past it.
#:
#: The single-objective suite's two lengths -- 64 for the protocol tasks, 32 for
#: the diagnostics -- cannot survive that, so `mo-ehrlich-hard` and the sweeps now
#: run at one length and differ by instance seed and conflict alone. That is a
#: real loss and is recorded here rather than in a changelog: "hard" no longer
#: means "longer", it means "the seed the protocol tasks use, at maximum
#: conflict".
MO_LENGTH = 10

#: Tokens per motif, $k$. Five rather than the single-objective suite's four,
#: and this is the alignment the exact front cost most. An Ehrlich value is
#: quantised to $q = k$ levels per motif, so $k$ is what bounds how many distinct
#: objective vectors a front can even contain: at $k = 4$ the exact front at
#: maximum conflict measured **one to two points**, which is no better than the
#: constructed front it replaces. At $k = 5$ it measures 3 to 5 at two
#: objectives, 9 to 12 at three and 23 to 25 at four.
MO_MOTIF_LENGTH = 5

#: Largest gap between consecutive motif positions. One -- contiguous motifs --
#: rather than the single-objective suite's three, and it is arithmetic rather
#: than a choice: a motif spans $(k - 1) s + 1$ positions and must fit inside a
#: block of $L / c = 5$, so $k = 5$ leaves no room for spacing at all. Keeping
#: $s = 3$ would have meant $k = 2$, and a two-level objective.
MO_MAX_SPACING = 1

#: Per-round radius on every multi-Ehrlich task. `protocol-alde`'s audited 21 is
#: meaningless at $L = 10$ -- a radius wider than the sequence is no radius -- so
#: the diagnostics' 4 is what carries over, and with re-anchoring four rounds of
#: it reach further than the sequence is long.
#:
#: What this costs is worth stating plainly: 4 substitutions of the shared chain
#: reach **493** constructible designs, out of 8,616 feasible sequences in the
#: whole space. A four-plate protocol spends 384 assays against that, so these
#: tasks now sit in a far more saturated regime than their single-objective
#: counterparts, where the reachable set is astronomical. They measure how well a
#: method covers a front, not whether it can find one at all.
MO_EHRLICH_MUTATIONS = DIAGNOSTIC_MUTATIONS

#: Ehrlich instance seed shared with `protocol-alde` and `protocol-evolvepro`.
MO_HARD_SEED = 2

#: Ehrlich instance seed shared with the single-objective diagnostics.
MO_DIAGNOSTIC_SEED = 7

#: Conflict values swept in the explanatory tier, chosen from a scan rather than
#: from the round numbers. The dial is not a smooth knob on front size: it
#: redraws each objective's planted optimum and its motif placement, so the
#: instance changes wholesale at every rung and the front size is noisy in
#: conflict. What the scan is for is locating the *transition*, and the exact
#: front over the whole $4^{10}$ space at these settings gives it:
#:
#: | conflict | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
#: | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
#: | seed 7 (swept) | 1 | 2 | 1 | 1 | 1 | 1 | 3 | 2 | 3 | 2 | 3 |
#: | seed 2 (hard)  | 1 | 1 | 1 | 1 | 2 | 3 | 2 | 3 | 3 | 3 | 5 |
#:
#: The old rungs `(0.0, 0.25, 0.5, 0.75, 1.0)` were round numbers rather than
#: measurements, and at the old settings 0.0 and 0.25 gave the *same* one-point
#: front -- two of five rungs measuring no trade-off at all. On this instance
#: they land no better: `1, 1, 1, 3, 3`, three rungs below the transition and a
#: gap of 0.25 across it. These five give `1, 1, 3, 3, 3`. They keep 0.0 as the
#: control, where a single-point front is provable from the construction rather
#: than merely observed, and put 0.4 and 0.6 either side of where the front
#: starts to spread -- on the swept seed it moves between exactly those two, and
#: on seed 2 it has already moved by 0.4. The transition is then visible in the
#: sweep's own front sizes instead of being asserted here.
CONFLICT_SWEEP: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.0)

#: Objective counts swept in the explanatory tier. Two is the same instance as
#: `mo-conflict-1.00`, deliberately: running both is a free reproducibility check
#: on the sweep machinery, since the two task names must produce identical
#: numbers or one of the two builders is wrong.
OBJECTIVE_COUNTS: tuple[int, ...] = (2, 3, 4)

#: Preference counts in the diagnostic, at **fixed total budget**. Eight
#: preferences means eight campaigns of 48 assays, not eight campaigns of 384.
PREFERENCE_COUNTS: tuple[int, ...] = (1, 4, 8)

#: Reference point for the Ehrlich tasks. An Ehrlich value is a product of
#: quantised motif satisfactions, so zero means "matched nothing" -- a design
#: there has contributed nothing on that objective and should enclose no volume.
#: A point below zero would hand every arm the same constant box and shrink the
#: differences a comparison reads into it.
EHRLICH_REFERENCE = 0.0

#: Where hypervolume starts on CH65: the Tite-Seq detection floor on all three
#: antigens. Stated here rather than read off the landscape so that a change to
#: [CH65Landscape][evogfn.landscapes.ch65.CH65Landscape] cannot move a published
#: number without moving this line too.
CH65_REFERENCE_POINT: tuple[float, ...] = (
    CH65_DETECTION_FLOOR,
    CH65_DETECTION_FLOOR,
    CH65_DETECTION_FLOOR,
)

#: Front points [hypervolume][evogfn.metrics.pareto.hypervolume] accepts in three
#: or more objectives **without** the optional `moo` extra, before its built-in
#: inclusion--exclusion becomes intractable. Restated here because it decides
#: whether this suite has a hypervolume column at all: on `ch65-real` a converged
#: arm produces a 19-point front and a random one 2--9, and on `mo-objectives-4`
#: the exact front alone runs to 25 points -- so a core-only install loses the
#: number for precisely the arms that did best.
#:
#: With the extra installed the limit does not apply and nothing here records
#: `nan`. Without it, read those tasks on IGD+.
#: [pymoo_available][evogfn.metrics.pareto.pymoo_available] says which install
#: is running.
EXACT_FRONT_LIMIT = 16

#: Where a multi-objective campaign's code actually starts, for staleness. The
#: arms are built here rather than in [evogfn.benchmark.methods][], so declaring
#: that module instead -- as [evogfn.benchmark.suite][] does -- would leave every
#: record in this suite unable to notice an edit to the arm that produced it.
RESULT_DEPENDENCIES = (
    "evogfn.benchmark.multi_objective",
    "evogfn.loop.campaign",
)

#: Gradient steps per round for the GFlowNet arm, matching
#: [evogfn.benchmark.methods][] so a multi-objective GFlowNet is the same amount
#: of training as a single-objective one.
DEFAULT_TRAINING_STEPS = 300

#: Reward exponent, as in Jain et al. and the MOGFN papers.
DEFAULT_BETA = 3.0

#: Candidates generated per round before selection, for the arms that select.
DEFAULT_POOL = 2048

#: Pool multiplier for the arms that do **not** select. NSGA-II runs without a
#: surrogate, so its round is "breed, measure the first plateful, survive"; a
#: pool of exactly one plate would be shortened by deduplication and the arm
#: would quietly underspend its budget. Four plates leaves room for that without
#: turning the generation into a pool the arm never uses.
UNSELECTED_POOL_PLATES = 4

# A campaign needs at least two measurements before a pairwise diversity exists.
_MIN_FOR_DIVERSITY = 2

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2

# Where the simplex is an interval and an even grid is exact rather than sampled.
_TWO_OBJECTIVES = 2

# How many designs a record keeps for inspection.
_TOP_DESIGNS = 10

# Reference fronts, by the task name and the builder that produced them. Building
# CH65's means re-reading a 14 MB CSV and sweeping 62,926 variants, and a campaign
# asks for it at construction -- so without this a fifty-seed arm would pay for
# the same array fifty times. The builder is part of the key rather than the name
# alone: two tasks of one name and different fronts is a thing a test does, and a
# cache that could not tell them apart would hand one of them the other's front.
_FrontKey = tuple[str, "Callable[[FitnessLandscape], npt.NDArray[np.float64]] | None"]
_FRONT_CACHE: dict[_FrontKey, npt.NDArray[np.float64] | None] = {}


# --------------------------------------------------------------------------
# Reference fronts: what IGD+ measures coverage of, and how good each one is.
# --------------------------------------------------------------------------


def ch65_reference_front(landscape: FitnessLandscape) -> npt.NDArray[np.float64]:
    """CH65's exact Pareto front, over the variants that were actually resolved.

    Exact in the strong sense: the 16-site lattice is 65,536 variants, 62,926 of
    them cleared QC on all three antigens, and
    [non_dominated][evogfn.metrics.pareto.non_dominated] sweeps that in well
    under a second. There is no sampling and no method-dependent reference set,
    which is what makes an IGD+ on this task a number rather than a ranking that
    moves when an arm is added.

    Censored variants are excluded, and the exclusion is the whole reason this
    function exists rather than a one-line call. 47.4% of `affinity_SI06` sits at
    the Tite-Seq detection floor, where the value reported is a *bound* and two
    variants tied there may differ by any amount below it. Including them adds
    one point to the front (20 rather than 19) that is non-dominated only because
    an objective could not resolve it -- and that point can never be attained by
    any arm anyway, since a design must beat the reference point *strictly* on
    every objective to enclose any volume, and a censored value sits exactly on
    it.

    Args:
        landscape: The CH65 landscape.

    Returns:
        A `(19, 3)` array of the non-dominated affinity vectors among variants
        measured, and uncensored, on all three antigens.

    Raises:
        TypeError: If handed a landscape that is not CH65. The censoring rule is
            specific to this assay and there is nothing sensible to fall back to.
    """
    if not isinstance(landscape, CH65Landscape):
        raise TypeError(
            f"the CH65 reference front is defined by that assay's detection floor and "
            f"cannot be computed for a {type(landscape).__name__}"
        )
    space = landscape.enumerate()
    resolved = landscape.is_measured(space) & ~landscape.is_censored(space).any(axis=1)
    values = np.asarray(landscape.evaluate(space[resolved]), dtype=np.float64)
    front = values[non_dominated(values)]
    return np.asarray(np.unique(front, axis=0), dtype=np.float64)


def enumerated_front(landscape: FitnessLandscape) -> npt.NDArray[np.float64]:
    r"""The true Pareto front of a multi-Ehrlich instance, by enumerating the space.

    What every Ehrlich task in this suite is now scored against, and the reason
    [MO_VOCAB_SIZE][evogfn.benchmark.multi_objective.MO_VOCAB_SIZE] is 4. A
    coverage indicator is only as strong as what it measures coverage *of*:
    against a constructed subset, IGD+ = 0 means "covered everything one
    particular construction found", while against the enumerated front it means
    covered the front. Only the second reading is a claim.

    $4^{10} = 1{,}048{,}576$ sequences, scored in about 1.5 seconds and cached by
    [MultiObjectiveTask.reference_front][evogfn.benchmark.multi_objective.MultiObjectiveTask.reference_front],
    so a fifty-seed arm pays for it once. Measured front sizes at maximum
    conflict: 3--5 points at two objectives, 9--12 at three, 23--25 at four --
    against the **two** points
    [recombination_front][evogfn.benchmark.multi_objective.recombination_front]
    produced for `mo-ehrlich-hard` at the old settings.

    Note that the front over the whole space is generally *not* attainable under
    a mutation budget: at 4 substitutions from the campaign's parent only 493
    designs are reachable, and their front is a subset. That is the honest target
    -- an arm that cannot reach a front point should be charged for it -- but it
    does mean IGD+ = 0 is not expected here, where against the constructed front
    it was.

    Args:
        landscape: The multi-Ehrlich landscape.

    Returns:
        An `(m, n_objectives)` array of the distinct non-dominated objective
        vectors over the whole space.

    Raises:
        TypeError: If handed a landscape that is not a multi-Ehrlich instance.
        ValueError: If the space is larger than
            [MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE],
            which is the guard that makes "exact" affordable rather than a
            promise the machine cannot keep.
    """
    if not isinstance(landscape, MultiEhrlichLandscape):
        raise TypeError(
            f"a {type(landscape).__name__} is not enumerated for its front here; only a "
            f"multi-Ehrlich instance is small enough, and CH65 has its own routine "
            f"because its front turns on which variants the assay resolved"
        )
    return landscape.exact_pareto_front()


def recombination_front(landscape: FitnessLandscape) -> npt.NDArray[np.float64]:
    r"""An attainable front for a multi-Ehrlich instance, built rather than enumerated.

    ## What this is for now

    Nothing in this suite: every task here moved to
    [enumerated_front][evogfn.benchmark.multi_objective.enumerated_front] when
    the alphabet dropped to 4 and the space became enumerable. It is kept because
    the problem it solves has not gone away -- a multi-Ehrlich instance at the
    20-letter alphabet, or at any length past
    [MO_LENGTH][evogfn.benchmark.multi_objective.MO_LENGTH], has no enumerable
    space and still needs something for IGD+ to measure against.

    ## Why the exact front was not available at those settings

    [exact_pareto_front][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape.exact_pareto_front]
    enumerates the whole space, and the whole space at $v = 20$, $L = 32$ is
    $20^{32}$ -- twenty-five orders of magnitude past
    [MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE]. It was not
    a matter of patience: a 20-letter alphabet is enumerable only up to $L = 5$,
    so *no* instance in this suite was enumerable and something had to take the
    exact front's place visibly.

    ## What this is instead

    The exact front **over a declared candidate set**: each objective's planted
    optimum, and every *contiguous-segment transplant* between every ordered pair
    of them -- take objective $a$'s optimum and overwrite positions $[i, j)$ with
    objective $b$'s. Each objective attains 1.0 at its own planted optimum by
    construction, so the extremes of the true front are in the set; the
    transplants walk between them and pick up whatever intermediate trade-offs a
    hybrid can express. Infeasible hybrids -- a transplant can land on a
    forbidden adjacency at either junction -- score $-\infty$ and are dropped by
    the landscape's own front routine rather than by a filter here.

    Segments rather than the single crossovers a genetic algorithm uses, and the
    difference is measured rather than assumed: at four objectives on the L=32
    instance, prefix crossovers alone find 13 non-dominated points and segments
    find 22, several of which *dominate* what the prefixes found. The cost is a
    pool of $O(k^2 L^2)$ sequences -- 4,292 at L=64 with two objectives, 6,736 at
    L=32 with four -- which an Ehrlich function scores in milliseconds.

    Two properties follow, and both matter for reading a number computed against
    it:

    * Every point is **attained by a sequence that exists**, so IGD+ = 0 is
      reachable and the indicator does not carry a permanent floor. This is the
      failure a "front" made of unattainable corner points would have.
    * It is a **subset of the true front, not the front**. An arm scoring IGD+ = 0
      has covered everything this construction found, which is a weaker
      statement than having covered the front. The right way to read it is as a
      *coverage-of-a-known-attainable-set*, and the right response to an arm
      saturating it is a larger candidate set, not a claim.

    How thick the answer is varies sharply with the instance, and thin is not
    wrong but is coarse -- and it was the measured thinness that retired this
    from the suite. At the old settings `mo-ehrlich-hard` (v=20, L=64) yielded
    **two** points, `(0.375, 1.0)` and `(1.0, 0.5)`, so IGD+ there was
    essentially "did you find both ends of the trade-off"; the L=32 sweeps
    yielded 1 to 22.

    At `conflict = 0` every planted optimum is the same sequence, so the set
    collapses to one point and the front is `[[1, ..., 1]]` -- which is what the
    landscape's own documentation says the exact front is in that regime, and is
    the one setting where this construction and the truth provably coincide.

    Args:
        landscape: The multi-Ehrlich landscape.

    Returns:
        An `(m, n_objectives)` array of distinct non-dominated objective vectors.

    Raises:
        TypeError: If handed a landscape with no planted optima to recombine.
    """
    if not isinstance(landscape, MultiEhrlichLandscape):
        raise TypeError(
            f"the recombination front is built from planted optima and a "
            f"{type(landscape).__name__} has none"
        )
    optima = np.asarray(landscape.optimal_sequences)
    candidates = [optima]
    for left, right in itertools.permutations(range(optima.shape[0]), 2):
        candidates.append(_transplants(optima[left], optima[right]))
    return landscape.exact_pareto_front(np.concatenate(candidates))


def _transplants(host: Tokens, donor: Tokens) -> Tokens:
    """Every contiguous segment of one sequence grafted into another.

    Args:
        host: The sequence positions outside the segment are taken from.
        donor: The sequence the segment is taken from.

    Returns:
        An `(m, length)` array with one row per segment `[i, j)`, including the
        empty segment (the host unchanged) and the whole sequence (the donor).
    """
    length = int(host.shape[0])
    grafts = [
        np.concatenate([host[:start], donor[start:stop], host[stop:]])
        for start in range(length + 1)
        for stop in range(start, length + 1)
    ]
    return np.stack(grafts)


# --------------------------------------------------------------------------
# The task: a landscape, a protocol, and the two references its metrics need.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiObjectiveTask(Task):
    """A task that states where its indicators are measured from.

    [Task][evogfn.benchmark.tasks.Task] carries an ``attainable`` optimum, which
    is a single-objective idea: with several objectives the landscape's optimum
    is an ideal point no design attains and the gap to it is not a regret. What
    replaces it is a pair of references, and both are carried here rather than
    read off the landscape at run time.

    **The reference point is carried, not fetched.** A campaign will happily take
    a landscape's own through
    [StatesReferencePoint][evogfn.loop.campaign.StatesReferencePoint], and that
    is the right default for a one-off run. It is the wrong one for a stored
    benchmark: hypervolumes measured from different reference points are not
    comparable, neither number records which point it came from, and a landscape
    that grows, loses or moves a `reference_point` property would silently
    re-scale every result already in the store. Stating it on the task puts it in
    the task's `repr`, which is what `run_multi_objective_task` writes into every
    record as provenance.

    Attributes:
        reference_point: The `(n_objectives,)` worst value worth counting on each
            objective. Empty is refused rather than defaulted.
        front: Builds the `(m, n_objectives)` reference front IGD+ is scored
            against, from the landscape. `None` leaves IGD+ unreported, which is
            honest and makes the task nearly unreadable -- hypervolume alone
            rewards one design enclosing a large box and says nothing about
            whether the rest of the front was approached.
        front_is_exact: Whether `front` returns the true Pareto front or a
            constructed subset of it. Recorded because an IGD+ of zero means
            different things in the two cases, and a table that does not say
            which is inviting the stronger reading.

    Raises:
        ValueError: If no reference point is stated. There is no defensible
            default: see
            [hypervolume][evogfn.loop.ledger.CampaignResult.hypervolume].
    """

    reference_point: tuple[float, ...] = ()
    front: Callable[[FitnessLandscape], npt.NDArray[np.float64]] | None = None
    front_is_exact: bool = False

    def __post_init__(self) -> None:
        """Refuse a task whose hypervolume would be measured from nowhere."""
        if not self.reference_point:
            raise ValueError(
                f"{self.name}: a multi-objective task must state the reference point its "
                f"hypervolume is measured from; two hypervolumes taken from different "
                f"points are not comparable and neither number carries the point"
            )

    @property
    def n_objectives(self) -> int:
        """How many objectives this task's landscape returns."""
        return len(self.reference_point)

    def parent(self, landscape: FitnessLandscape) -> Tokens:
        """The wild type a campaign starts from.

        Extends [Task.parent][evogfn.benchmark.tasks.Task.parent], which knows
        only the two single-objective landscapes and raises on anything else.
        Both multi-objective landscapes need an answer and neither answer is
        obvious:

        * CH65's is the **germline** ancestor, not the mature antibody. It is the
          sequence evolution started from, and anchoring at the mature antibody
          would make every available action a reversion.
        * A multi-Ehrlich instance draws a feasible sequence from the shared
          Markov chain, independent of every planted optimum, so the starting
          point leaks nothing about the answer on any objective.

        Args:
            landscape: The landscape being searched.

        Returns:
            A starting sequence of the landscape's length.

        Raises:
            TypeError: If the landscape is none of the four this suite knows.
        """
        if isinstance(landscape, CH65Landscape):
            return landscape.wild_type
        if isinstance(landscape, MultiEhrlichLandscape):
            return landscape.feasible_sequence(self.parent_seed)
        return Task.parent(self, landscape)

    def reference_front(self) -> npt.NDArray[np.float64] | None:
        """Build the front IGD+ is scored against, or ``None`` if none is stated.

        Memoised on the task's name. Every campaign asks for this at
        construction, and CH65's answer costs a 14 MB CSV parse and a sweep of
        62,926 variants -- which at a hundred seeds times four arms is the
        dominant cost of the whole task and produces the identical array each
        time. A copy is returned, so a caller holding one cannot edit the front
        every later campaign will be scored against.

        Returns:
            An `(m, n_objectives)` array, or ``None`` when the task states no
            front and IGD+ therefore goes unreported.
        """
        key: _FrontKey = (self.name, self.front)
        if key not in _FRONT_CACHE:
            _FRONT_CACHE[key] = None if self.front is None else self.front(self.landscape())
        cached = _FRONT_CACHE[key]
        return None if cached is None else cached.copy()

    def __repr__(self) -> str:
        """Name the task, its budget, its search radius, its anchor and its references.

        All of it, because all of it decides what a stored number means. The
        objective count and the reference point are the additions over
        [Task][evogfn.benchmark.tasks.Task]: two records at ``4x96=384`` whose
        hypervolumes were measured from different points are not comparable, and
        a provenance string naming only the protocol could not tell them apart.
        """
        anchor = "re-anchored" if self.reanchor else "fixed anchor"
        point = ",".join(f"{value:g}" for value in self.reference_point)
        exact = "exact" if self.front_is_exact else "constructed"
        front = "no front" if self.front is None else f"{exact} front"
        return (
            f"{self.name} ({self.protocol!r}, {self.max_mutations}/round, {anchor}, "
            f"{self.n_objectives} objectives, ref=({point}), {front})"
        )


def _task(  # noqa: PLR0913 - a task is defined by what it declares
    name: str,
    purpose: str,
    build: Callable[[], FitnessLandscape],
    protocol: Protocol,
    *,
    reanchor: bool,
    reference_point: tuple[float, ...],
    front: Callable[[FitnessLandscape], npt.NDArray[np.float64]] | None,
    front_is_exact: bool,
) -> MultiObjectiveTask:
    """A multi-objective task whose radius, anchor rule and references are all stated.

    The search radius is taken from the protocol rather than accepted separately,
    for the reason `evogfn.benchmark.suite._task` gives: the two are read by
    different code and a task searching at one radius while reporting another
    would be undetectable from a stored record.

    Args:
        name: Short identifier.
        purpose: What this task decides that the others cannot.
        build: Makes the landscape.
        protocol: Rounds, batch size and the per-round mutation budget.
        reanchor: Whether the anchor follows the best design measured so far, on
            the acquisition rule's scalarisation.
        reference_point: Where hypervolume is measured from.
        front: Builds the reference front, or ``None``.
        front_is_exact: Whether that front is the true one.

    Returns:
        The task.

    Raises:
        ValueError: If the protocol states no mutation budget, or if no reference
            point is given.
    """
    if protocol.max_mutations is None:
        raise ValueError(f"{name}: protocol must state a mutation budget")
    return MultiObjectiveTask(
        name=name,
        purpose=purpose,
        build=build,
        protocol=protocol,
        max_mutations=protocol.max_mutations,
        reanchor=reanchor,
        reference_point=reference_point,
        front=front,
        front_is_exact=front_is_exact,
    )


def _multi_ehrlich(
    *,
    n_objectives: int,
    conflict: float,
    seed: int,
) -> Callable[[], FitnessLandscape]:
    """A factory for a multi-Ehrlich instance at the suite's fixed motif parameters.

    Only the objective count, the conflict and the seed vary, so any two tasks
    built here differ in the axis being swept and in nothing else. What does *not*
    vary is stated as constants above, each with the alignment it keeps or breaks
    against the single-objective suite: `c = 2` motifs and a density of 0.5 carry
    over unchanged, while
    [MO_VOCAB_SIZE][evogfn.benchmark.multi_objective.MO_VOCAB_SIZE],
    [MO_LENGTH][evogfn.benchmark.multi_objective.MO_LENGTH],
    [MO_MOTIF_LENGTH][evogfn.benchmark.multi_objective.MO_MOTIF_LENGTH] and
    [MO_MAX_SPACING][evogfn.benchmark.multi_objective.MO_MAX_SPACING] were all
    moved to buy an enumerable space -- so "what does adding an objective cost?"
    is no longer a paired question against `protocol-alde` and must be read as a
    comparison within this suite.

    Args:
        n_objectives: How many Ehrlich functions to compose.
        conflict: How far the objectives' planted optima are allowed to disagree.
        seed: Seeds the transition matrix, the base sequence and the divergence.

    Returns:
        A factory building the landscape.
    """

    def build() -> FitnessLandscape:
        return MultiEhrlichLandscape.with_conflict(
            sequence_length=MO_LENGTH,
            vocab_size=MO_VOCAB_SIZE,
            n_objectives=n_objectives,
            n_motifs=2,
            motif_length=MO_MOTIF_LENGTH,
            max_spacing=MO_MAX_SPACING,
            transition_density=0.5,
            conflict=conflict,
            seed=seed,
        )

    return build


#: The two main tests. Between them: one real assay whose trade-off nobody
#: constructed, and one controlled instance whose only difference from a
#: single-objective task already in the suite is the objective count. Named at
#: length rather than ``MAIN`` so that it cannot be confused with
#: [evogfn.benchmark.suite][]'s, which is scored on an entirely different column.
MULTI_OBJECTIVE_MAIN: tuple[MultiObjectiveTask, ...] = (
    _task(
        "ch65-real",
        "Does any of this hold on measured affinities? Three antigens assayed by "
        "Tite-Seq over the 16 somatic mutations separating CH65 from its "
        "germline ancestor -- a trade-off the biology imposes rather than one a "
        "benchmark constructed. The mutation budget reaches every variant in the "
        "library, so this is the easiest geometry here and the hardest data.",
        CH65Landscape,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=CH65_MUTATIONS, label="four plates"),
        # Sixteen mutations over sixteen binary sites is the entire lattice, so
        # the first round already sees every design a later one could be
        # anchored at and there is nothing for an anchor to move towards.
        reanchor=False,
        reference_point=CH65_REFERENCE_POINT,
        front=ch65_reference_front,
        front_is_exact=True,
    ),
    _task(
        "mo-ehrlich-hard",
        "How well is a whole front covered when the objectives fight? Two "
        "Ehrlich objectives at maximum conflict on a DNA-sized alphabet -- "
        "L=10, v=4, c=2, k=5, density 0.5, seed 2 -- where the 1,048,576-sequence "
        "space is enumerable and the reference front is therefore the true one "
        "rather than a construction. The per-round radius is 4 and the campaign "
        "re-anchors, so its reach is cumulative.",
        _multi_ehrlich(
            n_objectives=2,
            conflict=1.0,
            seed=MO_HARD_SEED,
        ),
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=MO_EHRLICH_MUTATIONS, label="four plates"
        ),
        reanchor=True,
        reference_point=(EHRLICH_REFERENCE, EHRLICH_REFERENCE),
        front=enumerated_front,
        # Enumerated, not constructed: 4^10 is a fifth of the enumeration guard,
        # so IGD+ here measures coverage of the front rather than of a candidate
        # pool. It is 5 points at this seed, against the 2 the construction gave.
        front_is_exact=True,
    ),
)


def conflict_sweep() -> tuple[MultiObjectiveTask, ...]:
    """Two objectives at five degrees of disagreement, on the cheap instance.

    The explanatory sweep, and the one that decides whether a ranking is a
    finding or a coincidence: `conflict = 0` collapses the exact front to a
    single point, so a multi-objective method has nothing to spread over and
    should not beat a scalar one, while `conflict = 1` is where the trade-off is
    real. A method that wins at one end and loses at the other has not won.

    The rungs are placed on a measured transition rather than on round numbers --
    see [CONFLICT_SWEEP][evogfn.benchmark.multi_objective.CONFLICT_SWEEP] for the
    scan and for what the old spacing got wrong. Their exact fronts run 1, 1, 3,
    3, 3 points, so the sweep now contains the point at which a trade-off starts
    to exist instead of straddling it invisibly.

    Returns:
        One task per conflict value.
    """
    return tuple(
        _task(
            f"mo-conflict-{conflict:.2f}",
            f"Does the ranking survive the degree of objective conflict? Two "
            f"Ehrlich objectives at conflict {conflict:.2f} on the L=10, v=4 "
            f"diagnostic instance, scored against its enumerated front. "
            f"Explanatory: this decides how to read the main table, not what "
            f"goes in it.",
            _multi_ehrlich(
                n_objectives=2,
                conflict=conflict,
                seed=MO_DIAGNOSTIC_SEED,
            ),
            Protocol(rounds=4, batch_size=PLATE, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            reference_point=(EHRLICH_REFERENCE, EHRLICH_REFERENCE),
            front=enumerated_front,
            front_is_exact=True,
        )
        for conflict in CONFLICT_SWEEP
    )


def objective_count_sweep() -> tuple[MultiObjectiveTask, ...]:
    """Two, three and four objectives at maximum conflict.

    The other explanatory axis. Cost is expected to rise with the objective
    count for reasons that have nothing to do with any method -- the exact front
    grows from 3 points at two objectives to 9 at three and 23 at four, and a
    single preference covers proportionally less of it -- so this sweep exists to
    separate that from anything a method is doing.

    It is also where the hypervolume column depends on the install. A 23-point
    reference front means a converged arm's *measured* front can outgrow
    `EXACT_FRONT_LIMIT`, and without the optional `moo` extra there is no exact
    method for that, so `mo-objectives-4` is the task that goes `nan` first. With
    the extra it does not.

    The two-objective entry is the same instance as ``mo-conflict-1.00``. Running
    both is deliberate: the two builders must produce identical numbers, and a
    divergence is a bug in one of them rather than a result.

    Returns:
        One task per objective count.
    """
    return tuple(
        _task(
            f"mo-objectives-{count}",
            f"What does the objective count itself cost? {count} Ehrlich "
            f"objectives at maximum conflict on the L=10, v=4 diagnostic "
            f"instance, scored against its enumerated front. Explanatory: it "
            f"says how to read the main table, not what is in it.",
            _multi_ehrlich(
                n_objectives=count,
                conflict=1.0,
                seed=MO_DIAGNOSTIC_SEED,
            ),
            Protocol(rounds=4, batch_size=PLATE, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            reference_point=(EHRLICH_REFERENCE,) * count,
            front=enumerated_front,
            front_is_exact=True,
        )
        for count in OBJECTIVE_COUNTS
    )


def preference_task() -> MultiObjectiveTask:
    """The task the preference-count diagnostic is measured on.

    Deliberately the *same* landscape and protocol as `mo-ehrlich-hard`, under a
    different name so the results land in their own file. A diagnostic that
    decides a main-table setting should be run on the thing it decides for: a
    preference count that pays off on the diagnostic instance and not on the main
    one would be a recommendation drawn from the wrong place.

    Returns:
        The task.
    """
    return _task(
        "mo-preferences",
        "How many preference vectors should the scalarised arm get? The same "
        "landscape and protocol as mo-ehrlich-hard, with the budget split "
        "between preferences instead of spent on one. This is the only "
        "diagnostic here that decides something.",
        _multi_ehrlich(
            n_objectives=2,
            conflict=1.0,
            seed=MO_HARD_SEED,
        ),
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=MO_EHRLICH_MUTATIONS, label="four plates"
        ),
        reanchor=True,
        reference_point=(EHRLICH_REFERENCE, EHRLICH_REFERENCE),
        front=enumerated_front,
        front_is_exact=True,
    )


# --------------------------------------------------------------------------
# Preferences: where a scalarised method is pointed.
# --------------------------------------------------------------------------


def preference_vectors(n_objectives: int, count: int, *, seed: int = 0) -> npt.NDArray[np.float64]:
    r"""Points on the simplex for a scalarised method to search towards.

    Two schemes, because the simplex is a different object in two dimensions than
    in three or more:

    * **Two objectives.** The simplex is an interval, so an even grid including
      both endpoints *is* the answer -- there is nothing to sample and nothing a
      draw could do better. It is also independent of `seed`, which is correct:
      the diagnostic is measuring the count, and a count whose realisation
      changed per seed would be measuring the draw as well.
    * **Three or more.** No even lattice has an arbitrary number of points --
      Das--Dennis gives $\binom{H + k - 1}{k - 1}$, which is 3, 6, 10, ... for
      three objectives and never 4 or 8 -- so truncating one would be an
      arbitrary choice dressed as a grid. A Dirichlet(1) draw is the scheme Jain
      et al. (2023) use for exactly this, and seeding it from the campaign's seed
      means the comparison across counts is averaged over draws rather than
      resting on one.

    Args:
        n_objectives: How many objectives the preference must cover.
        count: How many preference vectors to return.
        seed: Seeds the draw, for three or more objectives.

    Returns:
        A `(count, n_objectives)` array, each row non-negative and summing to one.

    Raises:
        ValueError: If `count` or `n_objectives` is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be at least 1, got {n_objectives}")
    if count == 1:
        # The neutral trade-off. Anything else would be a claim about which
        # objective matters, made by the benchmark rather than by the biology.
        return np.full((1, n_objectives), 1.0 / n_objectives, dtype=np.float64)
    if n_objectives == _TWO_OBJECTIVES:
        first = np.linspace(0.0, 1.0, count, dtype=np.float64)
        return np.column_stack([first, 1.0 - first])
    drawn = np.random.default_rng(seed).dirichlet(np.ones(n_objectives), size=count)
    return np.asarray(drawn, dtype=np.float64)


def _acquisition(preference: npt.NDArray[np.float64]) -> ScalarizedAcquisition:
    """The rule every arm here is built with, pointed at one preference.

    [WeightedSum][evogfn.rewards.scalarization.WeightedSum] rather than
    [Tchebycheff][evogfn.rewards.scalarization.Tchebycheff], with the limitation
    stated rather than hidden: a linear scalarisation cannot reach a concave part
    of the Pareto front at any preference (Miettinen, 1999, Thm 3.1.4). It is
    chosen because it has no parameter beyond the preference itself, so the
    preference-count diagnostic varies one thing; a Tchebycheff arm additionally
    carries a reference point, and a sweep over counts would then be confounded
    with where that point sits.

    Args:
        preference: The `(n_objectives,)` trade-off to rank under.

    Returns:
        The acquisition rule.
    """
    return ScalarizedAcquisition(Greedy(), WeightedSum(), preference)


# --------------------------------------------------------------------------
# Arms.
# --------------------------------------------------------------------------


class ScalarizedObserving(Sampler):
    """Hands a scalar-only sampler the one number its ranking needs.

    Every classical baseline in this package refuses a multi-objective array at
    `observe` -- see `evogfn.algorithms.baselines._values.single_objective` --
    because ranking a population needs one order and flattening an `(n, k)` array
    would pair each design with somebody else's score. That refusal is right, and
    it means a genetic algorithm cannot be run against CH65 at all without
    something stating the trade-off first.

    This is that something, and it is a wrapper rather than a change to the
    baseline for two reasons. The refusal stays where it is, so a future caller
    that forgets to scalarise still gets an error instead of a plausible number.
    And the scalarisation is visible in the arm's construction next to the
    acquisition rule's, which is what makes it checkable that the two are the
    *same* trade-off -- a sampler breeding towards one weighting while the
    campaign ranked and re-anchored on another would search somewhere its own
    ledger never claimed was good.

    Args:
        sampler: The scalar-only sampler to wrap.
        scalarization: How the objectives are combined.
        preference: The `(n_objectives,)` trade-off, which must be the one the
            campaign's acquisition rule uses.
    """

    def __init__(
        self,
        sampler: Sampler,
        *,
        scalarization: Scalarization,
        preference: npt.NDArray[np.float64],
    ) -> None:
        """Wrap the sampler without running it."""
        super().__init__()
        self._sampler = sampler
        self._scalarization = scalarization
        self._preference = np.asarray(preference, dtype=np.float64)

    @property
    def name(self) -> str:
        """Short label, marking that the ranking is under a stated trade-off."""
        return f"{self._sampler.name} (scalarised)"

    @property
    def inner(self) -> Sampler:
        """The wrapped sampler, for inspection."""
        return self._sampler

    @property
    def proxy_calls(self) -> int:
        """Reward evaluations the wrapped sampler spent on the proxy.

        Forwarded rather than owned. ``run_multi_objective_task`` reads this off
        whatever the campaign's last sampler is, and a wrapper reporting its own
        zero would erase the compute the arm actually spent.
        """
        return int(getattr(self._sampler, "proxy_calls", 0))

    def reanchored(self, env: MutationEnvironment) -> ScalarizedObserving:
        """Move the wrapped sampler and re-wrap it under the same trade-off.

        The wrapper owns nothing anchored -- a preference is a direction in
        objective space, not in sequence space -- so this forwards. It has to
        exist: the campaign's check for
        [ReanchorableSampler][evogfn.loop.campaign.ReanchorableSampler] is a
        check on the *outermost* object, so a wrapper without this hook would
        send every arm it wraps down the rebuild path and discard the population
        each one carefully carries.

        Args:
            env: The re-anchored environment.

        Returns:
            A wrapper around the re-anchored inner sampler.

        Raises:
            TypeError: If the wrapped sampler cannot re-anchor. Returning `self`
                instead would leave the inner sampler breeding around the
                previous round's parent while the ledger recorded the new one,
                which is wrong in a way that still produces plausible designs.
        """
        from evogfn.loop.campaign import ReanchorableSampler  # noqa: PLC0415 - avoids a cycle

        if not isinstance(self._sampler, ReanchorableSampler):
            raise TypeError(
                f"{self._sampler.name} does not implement reanchored(env), so the "
                f"scalarising wrapper around it cannot move either; pass a "
                f"sampler_factory to the campaign to rebuild it instead"
            )
        moved = ScalarizedObserving(
            self._sampler.reanchored(env),
            scalarization=self._scalarization,
            preference=self._preference,
        )
        moved._proposals_made = self._proposals_made
        return moved

    def propose(self, n: int) -> Tokens:
        """Ask the wrapped sampler for candidates.

        Args:
            n: How many candidates to return.

        Returns:
            An `(n, sequence_length)` array.
        """
        proposals = self._sampler.propose(n)
        self._count(n)
        return np.asarray(proposals)

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Scalarise the measurements, then pass them on.

        A single-objective batch is passed through untouched. That is not a
        special case to be tidied away: the proxy the inner loop searches against
        returns `(n, 1)`, so this method is called with both widths in a single
        round, and scalarising a scalar would apply the preference to a number
        that is not an objective vector.

        Args:
            sequences: The candidates that were measured.
            values: An `(n, n_objectives)` array of their objective values.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == _MATRIX_NDIM and array.shape[1] > 1:
            array = self._scalarization.scalarize(array, self._preference)[:, None]
        self._sampler.observe(sequences, array)

    def __repr__(self) -> str:
        """Name the wrapped sampler and the trade-off it was given."""
        return (
            f"ScalarizedObserving({self._sampler!r}, "
            f"{np.array2string(self._preference, precision=3)})"
        )


class PreferenceEnsemble:
    """Several single-preference campaigns sharing one oracle budget.

    What makes this an experiment rather than a convenience is the *sharing*. The
    obvious way to run a scalarised method at several preferences is one full
    campaign each, and it is not a comparison: eight preferences at 384 assays
    apiece is 3,072 oracle calls against a population method's 384, so of course
    it covers more front. Here the total is held at the protocol's budget and
    split, so eight preferences buys 48 assays each. Whether that trade pays is
    the question, and it can only be asked at equal spend.

    The merged result is the union of what every sub-campaign measured, which is
    the right set to score: hypervolume and IGD+ are indicators of a *set* of
    designs, and the set a practitioner would have in hand at the end is all of
    them.

    One field of the merged result does not survive the merge cleanly, and is
    documented rather than repaired.
    [trace][evogfn.loop.ledger.CampaignResult.trace] is a best-so-far under the
    acquisition rule's scalarisation, and the sub-campaigns scalarise differently
    -- so the concatenated trace is a sequence of within-preference curves laid
    end to end, not one curve. Read it per block of `rounds` or not at all; the
    indicators are what this arm is scored on.

    Args:
        campaigns: The sub-campaigns, one per preference, in preference order.

    Raises:
        ValueError: If no campaigns are given.
    """

    def __init__(self, campaigns: Sequence[Campaign]) -> None:
        """Hold the sub-campaigns without running them."""
        if not campaigns:
            raise ValueError("a preference ensemble needs at least one campaign")
        self._campaigns = tuple(campaigns)

    @property
    def campaigns(self) -> tuple[Campaign, ...]:
        """The sub-campaigns, for inspection."""
        return self._campaigns

    @property
    def budget(self) -> int:
        """Total oracle calls across every preference."""
        return sum(campaign.budget for campaign in self._campaigns)

    @property
    def sampler(self) -> Sampler:
        """The last preference's sampler.

        There is no single sampler here, and the caller reading this wants one
        thing from it -- ``proxy_calls``, for the compute a run spent. Reporting
        one preference's share understates that by the ensemble size, which is
        why `run_multi_objective_task` sums across the ensemble instead of
        reading this. It is exposed for inspection and for interface parity with
        [Campaign][evogfn.loop.campaign.Campaign].
        """
        return self._campaigns[-1].sampler

    @property
    def proxy_calls(self) -> int:
        """Proxy evaluations across every preference, which is what the arm spent."""
        return sum(int(getattr(c.sampler, "proxy_calls", 0)) for c in self._campaigns)

    def run(self) -> CampaignResult:
        """Run every preference and merge what they measured.

        Returns:
            One result over the union of the measurements, carrying the first
            sub-campaign's references -- they are identical across the ensemble
            by construction, since they come from the task.
        """
        results = [campaign.run() for campaign in self._campaigns]
        rounds: list[RoundRecord] = []
        for result in results:
            rounds.extend(_reindexed(result, len(rounds)))
        first = results[0]
        return CampaignResult(
            sampler=f"{first.sampler} x{len(results)} preferences",
            rounds=tuple(rounds),
            sequences=np.concatenate([r.sequences for r in results]),
            values=np.concatenate([r.values for r in results]),
            optimum=first.optimum,
            ideal_point=first.ideal_point,
            reference_point=first.reference_point,
            reference_front=first.reference_front,
        )

    def __repr__(self) -> str:
        """Name the ensemble by its size and shared budget."""
        return f"PreferenceEnsemble({len(self._campaigns)} preferences, budget={self.budget})"


def _reindexed(result: CampaignResult, offset: int) -> list[RoundRecord]:
    """Renumber one sub-campaign's rounds so the merged ledger reads in order.

    Args:
        result: One preference's completed campaign.
        offset: How many rounds the merged ledger already holds.

    Returns:
        The same records, renumbered.
    """
    return [replace(record, index=record.index + offset) for record in result.rounds]


def _parts(task: Task, seed: int) -> tuple[FitnessLandscape, MutationEnvironment, DeepEnsemble]:
    """Everything a campaign needs that is not the method itself.

    Built identically for every arm on a given task and seed, which is what makes
    the comparison paired rather than merely simultaneous. Deliberately a copy of
    `evogfn.benchmark.methods._parts` rather than an import of it: that is a
    private helper in a module this one does not otherwise depend on, and reaching
    into it would couple the multi-objective suite's provenance to edits in the
    single-objective one.

    Args:
        task: Fixes the landscape, the parent and the search radius.
        seed: Fixes the surrogate's initialisation.

    Returns:
        The landscape, the environment anchored at the wild type, and the
        surrogate.
    """
    landscape = task.landscape()
    env = MutationEnvironment(
        task.parent(landscape),
        landscape.alphabet,
        max_mutations=task.max_mutations,
        # Without this, feasibility-by-construction is silently off and every
        # proposal on a constrained landscape scores minus infinity.
        transitions=_feasibility_of(landscape),
    )
    surrogate = DeepEnsemble(
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        epochs=150,
        seed=seed,
    )
    return landscape, env, surrogate


def _anchor_seed(seed: int, generation: int) -> int:
    """A distinct, reproducible seed for each anchor a campaign moves to.

    The same mechanism, and for the same measured reason, as
    `evogfn.benchmark.methods._anchor_seed`. A sampler rebuilt from a factory
    starts from its constructor and therefore from its seed, so a sampler
    re-seeded identically at every anchor proposes the identical pool at every
    anchor; the campaign then deduplicates almost all of it and spends rounds two
    onward on the tail of a batch it had already generated.

    Reproducibility is not given up to fix it: the stream is a pure function of
    the campaign's seed and of how many times its anchor has moved, both fixed by
    the run.

    Args:
        seed: The campaign's seed.
        generation: How many times the anchor has moved, from zero.

    Returns:
        The seed to build the sampler for that anchor with. Generation zero
        returns `seed` unchanged, so a task that never re-anchors is bit-for-bit
        what it would be without the mechanism.
    """
    if generation == 0:
        return seed
    return int(np.random.SeedSequence([seed, generation]).generate_state(1)[0])


def _feasibility_of(landscape: FitnessLandscape) -> npt.NDArray[np.float64] | None:
    """The landscape's transition matrix, when it has one.

    Args:
        landscape: The oracle.

    Returns:
        The matrix whose zeros mark forbidden adjacent token pairs, or ``None``
        for a landscape with no constructibility constraint -- CH65, whose
        16-site lattice is fully connected.
    """
    matrix = getattr(landscape, "transition_matrix", None)
    return None if matrix is None else np.asarray(matrix, dtype=np.float64)


def _campaign(  # noqa: PLR0913 - a campaign is defined by its protocol
    task: MultiObjectiveTask,
    landscape: FitnessLandscape,
    env: MutationEnvironment,
    build: Callable[[MutationEnvironment], Sampler],
    surrogate: DeepEnsemble | None,
    *,
    preference: npt.NDArray[np.float64],
    rounds: int,
    batch_size: int,
    pool_size: int,
) -> Campaign:
    """Assemble one campaign under a stated trade-off.

    Args:
        task: Fixes the search radius, the anchor rule and both references.
        landscape: The oracle.
        env: The environment the sampler was built against.
        build: Rebuilds the sampler for a moved anchor. Called once here for the
            opening round, so the sampler the campaign starts with and the ones
            it rebuilds come from one place and cannot drift apart.
        surrogate: Model fitted to the scalarised measurements, or ``None`` for
            an arm that ranks by dominance and needs no scalar model.
        preference: The trade-off the acquisition rule ranks, the surrogate is
            fitted to, and the anchor follows.
        rounds: Design-build-test-learn cycles for this campaign.
        batch_size: Variants measured per round.
        pool_size: Candidates generated per round before selection.

    Returns:
        The campaign, which refuses at construction if the acquisition rule
        cannot rank the landscape's objectives or if re-anchoring is asked for
        without the means to do it.
    """
    return Campaign(
        landscape=landscape,
        sampler=build(env),
        surrogate=surrogate,
        acquisition=_acquisition(preference),
        selector=TopK(),
        rounds=rounds,
        batch_size=batch_size,
        pool_size=pool_size,
        # Passed explicitly, never taken from the landscape: see
        # [MultiObjectiveTask][evogfn.benchmark.multi_objective.MultiObjectiveTask].
        reference_point=np.asarray(task.reference_point, dtype=np.float64),
        reference_front=task.reference_front(),
        environment=env,
        reanchor=task.reanchor,
        sampler_factory=build,
    )


def random_arm() -> MultiObjectiveMethodology:
    """Mutagenesis with no model and no memory: the floor.

    A hypervolume with nothing below it is a number rather than a result, and
    this is what puts something below it. It is also the arm that says how much
    of a front is available for free on a given task -- on a low-conflict
    instance that can be most of it.

    Returns:
        An arm.
    """

    def arm(task: Task, seed: int) -> Campaign:
        landscape, env, _ = _parts(task, seed)
        preference = preference_vectors(landscape.n_objectives, 1)[0]
        generation = itertools.count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build mutagenesis against whichever anchor the campaign is at."""
            return RandomMutagenesis(anchored, seed=_anchor_seed(seed, next(generation)))

        return _campaign(
            _as_multi_objective(task),
            landscape,
            env,
            make,
            None,
            preference=preference,
            rounds=task.protocol.rounds,
            batch_size=task.protocol.batch_size,
            pool_size=max(DEFAULT_POOL, task.protocol.batch_size * UNSELECTED_POOL_PLATES),
        )

    return arm


def nsga2_arm() -> MultiObjectiveMethodology:
    r"""NSGA-II, ranking by dominance and picking no trade-off at all.

    Run **without a surrogate**, and that is the method rather than an ablation.
    NSGA-II's round is breed-measure-survive: the offspring it generates are the
    designs it wants measured, and its elitist $(\mu + \lambda)$ step is what
    turns the measurements into the next generation. Inserting a surrogate would
    make the campaign re-rank that batch by a scalarised prediction, which is
    precisely the trade-off NSGA-II exists not to make -- the arm would then be a
    scalarised GA wearing a dominance-ranked population.

    It still receives a
    [ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition],
    because the campaign refuses a scalar rule against a vector-valued landscape
    at construction. With no surrogate that rule never scores a pool; it supplies
    the ledger's ``best_so_far`` and the anchor, both of which every arm here
    gets identically.

    Returns:
        An arm.
    """

    def arm(task: Task, seed: int) -> Campaign:
        landscape, env, _ = _parts(task, seed)
        preference = preference_vectors(landscape.n_objectives, 1)[0]
        generation = itertools.count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build NSGA-II against whichever anchor the campaign is at."""
            return NSGA2(anchored, seed=_anchor_seed(seed, next(generation)))

        return _campaign(
            _as_multi_objective(task),
            landscape,
            env,
            make,
            None,
            preference=preference,
            rounds=task.protocol.rounds,
            batch_size=task.protocol.batch_size,
            pool_size=task.protocol.batch_size * UNSELECTED_POOL_PLATES,
        )

    return arm


def scalarized_genetic_arm() -> MultiObjectiveMethodology:
    """A genetic algorithm optimising the scalarised proxy: the strongest control.

    The multi-objective counterpart of ``genetic+proxy``, and here for the same
    reason. A GFlowNet that only beats a baseline which never looks at the model
    has beaten the access, not the method:
    [ProxyOptimising][evogfn.algorithms.inner_loop.ProxyOptimising] gives the GA
    the same unlimited free evaluations of the surrogate that training gives the
    GFlowNet, so both spend real oracle calls only on the measured batch.

    The scalarisation the GA breeds under is the same object the acquisition rule
    ranks with, which is what stops the sampler climbing one trade-off while the
    ledger reports another.

    Returns:
        An arm.
    """

    def arm(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed)
        preference = preference_vectors(landscape.n_objectives, 1)[0]
        # One proxy for the whole campaign, closed over rather than rebuilt: it
        # wraps the surrogate instance the campaign refits in place.
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
        generation = itertools.count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the GA, its inner loop and its scalarising adapter."""
            stream = _anchor_seed(seed, next(generation))
            return ScalarizedObserving(
                ProxyOptimising(GeneticAlgorithm(anchored, seed=stream), proxy=proxy),
                scalarization=WeightedSum(),
                preference=preference,
            )

        return _campaign(
            _as_multi_objective(task),
            landscape,
            env,
            make,
            ensemble,
            preference=preference,
            rounds=task.protocol.rounds,
            batch_size=task.protocol.batch_size,
            pool_size=max(DEFAULT_POOL, task.protocol.batch_size * UNSELECTED_POOL_PLATES),
        )

    return arm


def scalarized_gflownet_arm(
    preferences: int = 1,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
) -> MultiObjectiveMethodology:
    """A trajectory-balance GFlowNet trained against the scalarised proxy.

    The method under test. The preference enters **once**, through the
    acquisition rule: the campaign fits its surrogate to
    [reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives] of
    the measurements, so the proxy the policy trains against already predicts the
    scalarised value and a
    [ScalarizedReward][evogfn.rewards.scalarization.ScalarizedReward] on top of
    it would have nothing left to combine. That single point of application is
    what makes the trained policy, the ranked pool, the ledger's best-so-far and
    the moved anchor the same trade-off rather than four that happen to agree.

    With `preferences > 1` this returns a
    [PreferenceEnsemble][evogfn.benchmark.multi_objective.PreferenceEnsemble]:
    one policy per preference, each with `1 / preferences` of the protocol's
    budget. The rounds are kept and the plate is cut, rather than the reverse,
    so every preference still sees the same number of design-build-test-learn
    cycles and the comparison across counts is not confounded with a comparison
    across campaign shapes.

    Args:
        preferences: How many preference vectors to split the budget between.
        steps: Gradient steps per round, per preference. **Not** divided by the
            preference count: training is free in oracle terms and this arm is
            being compared at equal *budget*, not at equal wall clock. The
            compute it costs is recorded as ``proxy_calls`` so the trade is
            visible rather than implied.
        beta: Reward exponent.

    Returns:
        An arm.

    Raises:
        ValueError: If `preferences` is not positive.
    """
    if preferences < 1:
        raise ValueError(f"preferences must be at least 1, got {preferences}")

    def arm(task: Task, seed: int) -> Campaign | PreferenceEnsemble:
        landscape, _, _ = _parts(task, seed)
        weights = preference_vectors(landscape.n_objectives, preferences, seed=seed)
        rounds = task.protocol.rounds
        # Integer division, and the remainder is deliberately not redistributed:
        # a split that gave one preference an extra design would make the arms
        # differ in budget as well as in count. What it costs is at most
        # `preferences - 1` unspent assays out of the protocol's total, and the
        # ledger records the real spend rather than the nominal one.
        batch = task.protocol.batch_size // preferences
        if batch < 1:
            raise ValueError(
                f"{task.name}: splitting a batch of {task.protocol.batch_size} between "
                f"{preferences} preferences leaves less than one design each"
            )
        campaigns = [
            _one_gflownet(
                task,
                seed,
                index=index,
                preference=weights[index],
                steps=steps,
                beta=beta,
                rounds=rounds,
                batch=batch,
            )
            for index in range(preferences)
        ]
        return campaigns[0] if preferences == 1 else PreferenceEnsemble(campaigns)

    return arm


def _one_gflownet(  # noqa: PLR0913 - one campaign per preference, and it needs all of it
    task: Task,
    seed: int,
    *,
    index: int,
    preference: npt.NDArray[np.float64],
    steps: int,
    beta: float,
    rounds: int,
    batch: int,
) -> Campaign:
    """One preference's campaign, with its own policy and its own surrogate.

    Nothing is shared between preferences. A shared surrogate would be fitted to
    whichever trade-off wrote to it last and would carry information between arms
    of the ensemble that a practitioner running them in parallel would not have;
    a shared policy would be retrained against a moving target every time the
    preference changed. Both would make the ensemble cheaper and neither would be
    the experiment.

    Args:
        task: Fixes the landscape, the radius and the references.
        seed: The campaign's seed.
        index: Which preference this is, mixed into the seed so the sub-campaigns
            do not all propose the identical opening pool.
        preference: This sub-campaign's trade-off.
        steps: Gradient steps per round.
        beta: Reward exponent.
        rounds: Design-build-test-learn cycles.
        batch: Variants measured per round.

    Returns:
        The campaign.
    """
    stream = _anchor_seed(seed, index)
    landscape, env, ensemble = _parts(task, stream)
    # Built once and closed over, so a rebuild for a moved anchor keeps the
    # trained weights: the action space and the policy's input are properties of
    # the space rather than of the anchor, and only the masks move.
    policy = SequencePolicy(
        n_actions=env.n_actions,
        sequence_length=env.sequence_length,
        n_tokens=env.alphabet.size,
        hidden_dim=128,
        seed=stream,
    )
    proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
    generation = itertools.count()

    def make(anchored: MutationEnvironment) -> Sampler:
        """Build the sampler against whichever anchor the campaign is at."""
        anchor_stream = _anchor_seed(stream, next(generation))
        return GFlowNetSampler(
            anchored,
            policy,
            proxy=proxy,
            # A plain tempered reward, and deliberately not a
            # `ScalarizedReward`. The surrogate this proxy wraps is single-output
            # and the campaign fits it to
            # `acquisition.reduce_objectives(measurements)`, so the preference
            # has already been applied by the time the proxy predicts anything.
            # Applying it a second time here would not be redundant, it would
            # raise: a `(n, 1)` prediction is not an objective vector and a
            # two-entry preference does not fit it.
            reward=TemperedReward(beta=beta),
            config=TrainingConfig(steps=steps, batch_size=64, seed=anchor_stream),
            objective=TrajectoryBalance(),
            seed=anchor_stream,
        )

    return _campaign(
        _as_multi_objective(task),
        landscape,
        env,
        make,
        ensemble,
        preference=preference,
        rounds=rounds,
        batch_size=batch,
        pool_size=max(DEFAULT_POOL, batch * UNSELECTED_POOL_PLATES),
    )


def _as_multi_objective(task: Task) -> MultiObjectiveTask:
    """Narrow a task to one that states its references, or refuse.

    Args:
        task: The task an arm was handed.

    Returns:
        The same task, narrowed.

    Raises:
        TypeError: If the task states no reference point, in which case its
            hypervolume would be measured from wherever the landscape happened to
            suggest and its IGD+ from nothing at all.
    """
    if not isinstance(task, MultiObjectiveTask):
        raise TypeError(
            f"{task.name} is a {type(task).__name__} and states no reference point or "
            f"reference front; the multi-objective arms need both, so build it with "
            f"MultiObjectiveTask"
        )
    return task


#: The four arms of the main and explanatory tiers, in a stable order so a report
#: reads the same way each run.
ARMS: dict[str, MultiObjectiveMethodology] = {
    "random": random_arm(),
    "nsga2": nsga2_arm(),
    "genetic+proxy": scalarized_genetic_arm(),
    "gfn-tb": scalarized_gflownet_arm(),
}


def preference_arms() -> dict[str, MultiObjectiveMethodology]:
    """The same GFlowNet at each preference count, at fixed total budget.

    Returns:
        One arm per entry of `PREFERENCE_COUNTS`, named by the count.
    """
    return {f"gfn-tb-pref{count}": scalarized_gflownet_arm(count) for count in PREFERENCE_COUNTS}


# --------------------------------------------------------------------------
# Running, and what a stored record holds.
# --------------------------------------------------------------------------


def multi_objective_tiers(main_seeds: int, explanatory_seeds: int) -> list[Tier]:
    """The multi-objective suite, ordered cheap and decisive first.

    The three tiers differ in **role**, not only in seed count, and a report
    should say so. Only `main` carries results. `conflict` and `objectives`
    answer "under what conditions does the ranking change" and are not inputs to
    the main table. `preferences` is the only one that decides anything, namely
    how many preference vectors the main-table GFlowNet arm gets.

    Args:
        main_seeds: Seeds for the headline tier.
        explanatory_seeds: Seeds for the sweeps and the diagnostic.

    Returns:
        Tiers in the order they should run, so an interrupted night still yields
        something readable.
    """
    return [
        Tier(
            "preferences",
            (preference_task(),),
            tuple(range(explanatory_seeds)),
            purpose=Purpose.DIAGNOSTIC,
        ),
        Tier("main", MULTI_OBJECTIVE_MAIN, tuple(range(main_seeds)), purpose=Purpose.BENCHMARK),
        Tier(
            "conflict",
            conflict_sweep(),
            tuple(range(explanatory_seeds)),
            purpose=Purpose.DIAGNOSTIC,
        ),
        Tier(
            "objectives",
            objective_count_sweep(),
            tuple(range(explanatory_seeds)),
            purpose=Purpose.DIAGNOSTIC,
        ),
    ]


def arms_for_tier(tier: Tier) -> dict[str, MultiObjectiveMethodology]:
    """Which arms a tier runs.

    Args:
        tier: The tier being run.

    Returns:
        The preference sweep for the diagnostic tier, the four comparison arms
        for everything else.
    """
    return preference_arms() if tier.name == "preferences" else dict(ARMS)


def set_indicators(task: MultiObjectiveTask, result: CampaignResult) -> dict[str, float | None]:
    """The two numbers a stored record is indexed by, for a multi-objective run.

    [RunRecord][evogfn.benchmark.store.RunRecord] has one larger-is-better field
    and one smaller-is-better field, and every table built off the store reads
    them positionally. The multi-objective pair with those orientations is
    hypervolume, which rises as the set improves, and IGD+, which falls to zero
    when the front is covered -- so ``best`` carries the volume and ``regret``
    the coverage.

    Hypervolume comes back as ``nan`` where no exact method in
    [evogfn.metrics.pareto][] can run: past `EXACT_FRONT_LIMIT` front points in
    three or more objectives, **without** the optional `moo` extra. With the
    extra there is no such case and this column is always populated; install it
    before a run rather than reading the gaps afterwards, since a `nan` cannot be
    filled in later without re-running the campaign that produced it.

    Where it does happen it is reported and not patched: an approximation written
    into the same column as an exact value would be indistinguishable from one,
    and the measurements survive on the result for anyone who wants to score them
    separately. On `ch65-real` and `mo-objectives-4` this bites precisely the arms
    that did best, which is why those tasks are read on IGD+.

    Args:
        task: The task being run, named in any error.
        result: The completed campaign.

    Returns:
        ``best`` and ``regret``, ready to pass to
        [ResultStore.stamp][evogfn.benchmark.store.ResultStore.stamp].

    Raises:
        ValueError: If the campaign supplied no reference point, so there is no
            indicator to store the run under at all.
    """
    if result.reference_point is None:
        raise ValueError(
            f"{task.name} measured {result.n_objectives} objectives but its campaign "
            f"carried no reference point, so its result has no hypervolume to be stored "
            f"under; the task states one and something dropped it"
        )
    try:
        volume = result.hypervolume
    except NotImplementedError:
        volume = float("nan")
    return {"best": float("nan") if volume is None else volume, "regret": result.igd_plus}


def run_multi_objective_task(
    task: MultiObjectiveTask,
    arms: Mapping[str, MultiObjectiveMethodology],
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever is missing for one task, storing each result as it lands.

    Deliberately not `evogfn.benchmark.suite.run_task`. That function declares
    its records' dependencies as [evogfn.benchmark.methods][] and
    [evogfn.loop.campaign][], and the arms here are built in *this* module --
    which neither of those imports. A record stamped through it would therefore
    fail to notice an edit to the very arm that produced it, which is the
    stale-result failure the fingerprint exists to prevent.

    Args:
        task: What to run.
        arms: Arms by name.
        store: Where results go, and what says which are already held.
        seeds: Seeds wanted.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.
    """
    landscape = task.landscape()
    ran = 0

    for name, arm in arms.items():
        outstanding = store.missing(task.name, name, seeds)
        if not outstanding:
            report(f"  {task.name}/{name}: {len(seeds)} seeds cached")
            continue
        started = time.perf_counter()
        for seed in outstanding:
            campaign = arm(task, seed)
            result = campaign.run()
            feasible = (
                float(landscape.is_feasible(result.sequences).mean())
                if len(result.sequences)
                else 0.0
            )
            store.append(
                store.stamp(
                    depends_on=RESULT_DEPENDENCIES,
                    task=task.name,
                    method=name,
                    seed=seed,
                    # The task's repr, which carries the reference point: two
                    # records at 4x96=384 whose hypervolumes were measured from
                    # different points are not comparable, and nothing in either
                    # number says so.
                    protocol=repr(task),
                    **set_indicators(task, result),
                    diversity=(
                        diversity(result.sequences)
                        if len(result.sequences) >= _MIN_FOR_DIVERSITY
                        else 0.0
                    ),
                    feasible_fraction=feasible,
                    oracle_calls=result.oracle_calls,
                    proposals=result.proposals,
                    proxy_calls=_proxy_calls(campaign),
                    deterministic=is_deterministic(),
                    top_sequences=_front_designs(result),
                    trace=result.trace(),
                    rounds=[
                        {
                            "index": float(record.index),
                            "proposed": float(record.proposed),
                            "screened": float(record.screened),
                            "evaluated": float(record.evaluated),
                            "feasible": float(record.feasible),
                            "best_in_round": record.best_in_round,
                            "best_so_far": record.best_so_far,
                            "mean_in_round": record.mean_in_round,
                            "batch_diversity": record.batch_diversity,
                            "surrogate_correlation": record.surrogate_correlation,
                            "hypervolume": record.hypervolume,
                            "anchor_distance": float(record.anchor_distance),
                        }
                        for record in result.rounds
                    ],
                )
            )
            ran += 1
        elapsed = time.perf_counter() - started
        report(
            f"  {task.name}/{name}: ran {len(outstanding)} "
            f"({elapsed / max(len(outstanding), 1):.1f}s each), "
            f"{len(seeds) - len(outstanding)} cached"
        )
    return ran


def run_multi_objective_tier(
    tier: Tier,
    arms: Mapping[str, MultiObjectiveMethodology],
    store: ResultStore,
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run every task in a tier.

    Args:
        tier: What to run.
        arms: Arms by name.
        store: Where results go.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.

    Raises:
        TypeError: If the tier carries a task that states no references.
    """
    report(f"{tier!r}")
    return sum(
        run_multi_objective_task(_as_multi_objective(task), arms, store, tier.seeds, report=report)
        for task in tier.tasks
    )


def _proxy_calls(campaign: Campaign | PreferenceEnsemble) -> int:
    """Proxy evaluations the arm spent, whether it was one campaign or several.

    Summed across an ensemble rather than read off one of its members: an arm at
    eight preferences trains eight policies for one oracle budget, and reporting
    an eighth of that would make the compute trade the preference diagnostic
    exists to expose invisible.

    Args:
        campaign: What the arm returned.

    Returns:
        Proxy evaluations, or zero for a sampler that keeps no such count.
    """
    if isinstance(campaign, PreferenceEnsemble):
        return campaign.proxy_calls
    return int(getattr(campaign.sampler, "proxy_calls", 0))


def _front_designs(result: CampaignResult, k: int = _TOP_DESIGNS) -> list[list[int]]:
    """The non-dominated designs a campaign found, for inspection.

    Not "the best ten": with several objectives there is no order to take a top
    ten under that does not first invent a trade-off, and the designs worth
    looking at when a number surprises someone are the ones on the measured
    front. Where the front is larger than `k` it is truncated in measurement
    order, which is arbitrary but at least does not smuggle in a weighting.

    Args:
        result: A completed campaign.
        k: How many to keep.

    Returns:
        Token lists, in measurement order.

    Raises:
        ValueError: If any measurement is ``nan``, where dominance is undefined.
    """
    if not len(result.sequences):
        return []
    values = np.asarray(result.values, dtype=np.float64)
    keep = np.flatnonzero(non_dominated(values))[:k]
    return [[int(token) for token in result.sequences[index]] for index in keep]
