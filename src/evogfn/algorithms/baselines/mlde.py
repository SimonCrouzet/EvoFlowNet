"""MLDE: the supervised baseline protein engineers actually run.

Machine-learning-assisted directed evolution (Wittmann, Yue & Arnold, *Cell
Systems* 2021) is three steps and no more: screen a random sample of the library,
fit a regressor to it, order the variants it predicts best. There is no
acquisition function, no uncertainty, no policy and no second thought. On the
GB1 four-site landscape that protocol found a variant in the top 0.1% far more
often than a directed-evolution walk did, which is why it is the reference point
the field cites.

It is here because it is the cheapest way for this project's central claim to be
wrong. If one supervised fit on a random plate reaches the same designs a
GFlowNet reaches, then everything the GFlowNet adds -- the flow objective, the
masked policy, the multi-round loop -- has bought nothing, and the honest report
is that a regression was enough. Beating a genetic algorithm while losing to
ridge regression on a random sample would be a hollow result.

A baseline is only worth beating if it is the real thing, so what is and is not
the published method is spelled out below rather than left for a reader to infer.

Single-shot by design, multi-round by this harness
--------------------------------------------------

MLDE is a *two-stage* method: one training sample, one prediction, done. This
repository's campaign loop instead calls ``propose`` and ``observe`` once per
round, so this implementation refits on the accumulated measurements every round
and proposes against the refreshed model. That is a deviation and must be
reported as one -- iterating the fit makes it closer to CLADE (Qiu & Wei, 2021)
or to ftMLDE than to MLDE as published, and it can only help the baseline, which
is the right direction for a deviation in a baseline to run.

The budget it cannot have
-------------------------

State this plainly, because the results table depends on it. The published
protocol is **384 training variants plus a top-96 design plate, 480 assays**
(:data:`PUBLISHED_BUDGET`). This repository's four-plate campaign budget is 384
assays *in total*. MLDE's training set alone therefore exceeds the entire budget
it is benchmarked under, and there is no configuration in which the published
method fits inside 384 calls.

So the default here trains on one 96-well plate, a quarter of Wittmann et al.'s
sample. That is a **compression, not a design choice**, and it is a handicap:
the method's own paper shows outcome improving with training set size, so a
96-variant MLDE is a weaker MLDE. Any table reporting this arm must carry the
note; :attr:`MLDE.budget_note` returns it in a form a report can print, and
:attr:`MLDE.runs_below_published_training_size` is the boolean to branch on.

Two ways out are provided and neither is silent. :meth:`MLDE.as_published`
builds the method at its own split for a reference run at a 480-call budget, and
``training_size=PUBLISHED_TRAINING_SIZE`` does the same by hand. Neither changes
what an existing caller gets.

The ensemble: theirs, and ours
------------------------------

Wittmann et al. do not fit one model. Their released implementation
(``github.com/fhalab/MLDE``) trains **22 architectures** -- 5 Keras networks
(three MLP depths, two CNNs), 4 XGBoost variants, and 13 scikit-learn regressors
including KernelRidge, KNeighborsRegressor, BayesianRidge, ElasticNet,
RandomForest and GradientBoosting -- each under **5-fold cross-validation**,
ranks the architectures by cross-validation error, and averages the predictions
of **all cross-validation instances of the top 3**. The previous version of this
file fitted a single kernel ridge, which is one of their 22 members, and calling
that MLDE understated the method.

What is reproduced here, and is the paper's:

* One-hot encoding of the variant, which is their ``onehot`` encoding and the
  one their unsupervised-encoding ablations are measured against.
* 5-fold cross-validation of every member (:data:`PUBLISHED_CV_FOLDS`).
* Ranking members by cross-validated mean squared error and averaging only the
  top few (:data:`PUBLISHED_MODELS_AVERAGED`), rather than averaging blindly.
* Averaging across the *fold instances* of the selected members, so the final
  predictor is 15 models each trained on 4/5 of the data -- exactly their
  ``n_averaged x n_cv`` construction, not a refit on everything.

What is ours, and why:

* **The roster of 12 members, not 22.** XGBoost and Keras are two dependencies
  this library does not have and will not take for one baseline, and sklearn is
  not a dependency either (numpy and torch are). The members below are written
  against numpy so that the *shape* of the ensemble -- different inductive
  biases, selected by held-out error -- survives, which is the part their
  ablations show matters. It is a smaller and less varied ensemble than theirs.
* **Which 12.** Polynomial-kernel ridge at degrees 1 and 2 and three penalties
  each (6); *local*-kernel ridge at two bandwidths and two penalties (4);
  k-nearest-neighbours at k=5 and k=15 (2). Degree 1 is a linear model, so the
  penalty sweep stands in for their many linear-ish regressors (BayesianRidge,
  ElasticNet, ARDRegression, LinearSVR, SGDRegressor); degree 2 is their
  KernelRidge and is the pairwise-epistasis model MLDE exists to capture; k-NN
  is their KNeighborsRegressor, at sklearn's default k=5 and one wider setting.
* **The local kernel.** ``exp(-gamma * hamming)`` with ``gamma`` set by the
  median pairwise distance in the training set. Kernel ridge is theirs; this
  particular kernel and the median-heuristic bandwidth are ours. It is here
  because the polynomial kernel is global -- every training point influences
  every prediction -- and a local kernel is a genuinely different bias rather
  than another point on the same regularisation path. It is the member that
  earns most of the improvement over the single model this file used to fit.
* **No neural member, and no trees.** Their 5 Keras networks and 7 tree models
  have no stand-in here, which is the largest gap. We built and measured a
  random-ReLU-feature network as an affordable substitute for the Keras MLPs and
  then removed it: at finite width such a model is a Monte-Carlo approximation
  to the arc-cosine kernel, which for one-hot inputs is again a function of the
  same agreement count the polynomial kernel already uses, so it contributed
  sampling noise rather than a new bias -- and being near-constant, it was
  *selected* by cross-validated MSE precisely when the real models were
  struggling, which cost accuracy. A trained MLP would be a different model; one
  trained per fold per round across thousands of campaigns is not affordable
  here. Tree ensembles are absent for the same cost reason.
* **The candidate set.** Their library is four combinatorial sites -- 160,000
  variants, exhaustively scorable. The environments here reach ``10^13`` designs
  and upwards, so a random pool stands in for exhaustive enumeration and the
  ensemble ranks that.

The kernel deserves one line of justification since it does the most work: the
inner product of two one-hot sequences is the number of positions at which they
agree, so a degree-2 polynomial kernel is exactly the space of position-pair
interactions, reached in the dual without ever forming the ``(L·V)²`` features,
and the Hamming distance the local kernel needs is the same count again. That
identity is standard; using it here instead of sklearn's KernelRidge is ours.

Where the compression shows
---------------------------

The two adaptations above interact, and the measurement is worth recording. At
Wittmann et al.'s own training size the cross-validated selection is stable --
across a pairwise-epistasis landscape at ``L = 256`` it chose the same three
members on every seed tried, and the ensemble beat the single kernel ridge on
every one. At the compressed 96-variant default the same selection becomes
noisy: with only ~77 points per fold the members' cross-validated errors are
within noise of each other, and the ensemble occasionally selects a member with
no ranking power at all. It still wins on average, but the *variance* of running
MLDE below its published training size is itself a cost, and it is a cost the
budget note above is describing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._values import single_objective
from evogfn.algorithms.baselines.mutagenesis import RandomMutagenesis

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment

#: Training variants in Wittmann et al.'s published protocol.
PUBLISHED_TRAINING_SIZE = 384

#: Variants in the plate their fitted model then designs.
PUBLISHED_BATCH_SIZE = 96

#: Assays the published protocol spends in total. This repository's four-plate
#: budget is 384, so the published method does not fit inside it -- see the
#: module docstring, and do not quietly drop this from a results table.
PUBLISHED_BUDGET = PUBLISHED_TRAINING_SIZE + PUBLISHED_BATCH_SIZE

#: Cross-validation folds per member, matching their ``n_cv`` default of 5.
PUBLISHED_CV_FOLDS = 5

#: Members whose predictions are averaged, matching their ``n_averaged`` of 3.
PUBLISHED_MODELS_AVERAGED = 3

#: One 96-well plate, and this package's default round. Used as the training
#: sample size because the published 384 is more than the whole default campaign
#: budget. A compression of the method, not a choice.
DEFAULT_TRAINING_SIZE = 96

#: Fewest measurements worth fitting. A ridge on one point predicts that point's
#: value everywhere, which is a constant ranking and no ranking at all.
_MIN_TRAINING = 2

#: Rows of the candidate pool compared against the training set at a time. The
#: kernel is an agreement count over positions, so the intermediate is
#: ``(chunk, n_train, length)`` booleans; chunking bounds that at tens of MB
#: instead of letting it scale with the pool.
_KERNEL_CHUNK = 256

#: Multipliers applied to ``ridge_alpha`` to give the kernel members genuinely
#: different biases. Two orders of magnitude either side is enough to separate an
#: almost-interpolating fit from an almost-constant one; cross-validation then
#: picks, which is the whole point of having the spread.
_ALPHA_SCALES = (0.1, 1.0, 10.0)

#: Neighbourhood sizes for the k-NN members. 5 is sklearn's default, which is
#: what their KNeighborsRegressor member runs at; 15 is the smoothed counterpart.
#: k=1 was tried and dropped -- at ``L = 256`` with a mutation budget in the
#: tens, every pair of variants agrees at over 80% of positions, so "the nearest
#: neighbour" is close to arbitrary and the member is pure variance.
_NEIGHBOURHOODS = (5, 15)

#: Bandwidths for the local kernel, as multiples of the reciprocal median
#: pairwise Hamming distance in the training set. The median heuristic is what
#: makes one setting work across sequence lengths and mutation budgets, which
#: differ by an order of magnitude between the landscapes here.
_LOCAL_SCALES = (0.5, 2.0)

#: Penalties for the local-kernel members, relative to ``ridge_alpha``. Two
#: rather than three, so that no single model class contributes half the roster
#: and wins selection by having the most tickets.
_LOCAL_ALPHA_SCALES = (0.1, 1.0)

#: Added to every ridge diagonal on top of the requested penalty. A caller may
#: legitimately ask for ``ridge_alpha=0``, and duplicate variants in the training
#: set then make the system singular; this keeps the solve from raising on data
#: that is merely repetitive.
_RIDGE_FLOOR = 1e-8

#: Floor on the median pairwise distance used to set the local bandwidth. Zero
#: happens when every measured variant is the same sequence, and dividing by it
#: would turn one degenerate training set into a NaN-valued ranking.
_DISTANCE_FLOOR = 1e-6


@dataclass(frozen=True, eq=False)
class _Learner:
    """One member of the ensemble: a model class with its hyperparameters.

    Held as data rather than as a class per member because every member is a
    function of one shared intermediate -- the matrix of position agreements --
    and objects would each want their own copy of it.

    ``kernel_key`` names the transform of that matrix the member reads, so that
    members sharing a kernel share the work of building it. The k-NN members
    rank on raw agreement, which is the degree-1 kernel, so they name it too.
    """

    name: str
    kind: Literal["kernel", "local", "neighbours"]
    kernel_key: str
    alpha: float = 0.0
    degree: int = 1
    scale: float = 1.0
    neighbours: int = 1


@dataclass(frozen=True, eq=False)
class _Fit:
    """One member trained on one cross-validation fold.

    Fifteen of these -- three selected members by five folds -- are what
    ``predict`` averages, following Wittmann et al.'s construction.

    ``coefficients`` are dual weights for the kernel members and the fold's own
    measured values for the k-NN members, which is the whole difference between
    a solved model and a lazy one.
    """

    learner: _Learner
    rows: npt.NDArray[np.intp]
    coefficients: npt.NDArray[np.float64]
    intercept: float


def _roster(ridge_alpha: float, max_degree: int) -> tuple[_Learner, ...]:
    """Build the ensemble's model classes.

    Args:
        ridge_alpha: Centre of the penalty sweep for every regularised member.
        max_degree: Highest polynomial kernel degree to include.

    Returns:
        The members, in a fixed order so that ties in cross-validated error
        break the same way on every run.
    """
    learners: list[_Learner] = []
    for degree in range(1, max_degree + 1):
        for scale in _ALPHA_SCALES:
            alpha = ridge_alpha * scale
            learners.append(
                _Learner(
                    f"poly{degree}-a{alpha:g}",
                    "kernel",
                    f"poly{degree}",
                    alpha=alpha,
                    degree=degree,
                )
            )
    for bandwidth in _LOCAL_SCALES:
        for scale in _LOCAL_ALPHA_SCALES:
            alpha = ridge_alpha * scale
            learners.append(
                _Learner(
                    f"local{bandwidth:g}-a{alpha:g}",
                    "local",
                    f"local{bandwidth:g}",
                    alpha=alpha,
                    scale=bandwidth,
                )
            )
    for k in _NEIGHBOURHOODS:
        # Ranks on raw agreement, which is the degree-1 kernel already built.
        learners.append(_Learner(f"knn-{k}", "neighbours", "poly1", neighbours=k))
    return tuple(learners)


class MLDE(Sampler):
    """Fits an ensemble to a random sample, then proposes its top predictions.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        training_size: Measurements gathered at random before the model takes
            over. Defaults to one plate rather than Wittmann et al.'s 384,
            because 384 exceeds this repository's whole campaign budget; see the
            module docstring, and pass :data:`PUBLISHED_TRAINING_SIZE` or use
            :meth:`as_published` to run the method at its own split.
        pool_multiplier: Candidates generated per candidate returned. Our
            choice: the published method ranks an exhaustive library, and this
            is how much of an unenumerable one the ensemble gets to rank.
        ridge_alpha: Centre of the penalty sweep. Every regularised member is
            fitted at this value scaled by :data:`_ALPHA_SCALES`, and
            cross-validation chooses between them, so this sets where the sweep
            sits rather than fixing one penalty. Meaningful as 1.0 only because
            every kernel here is normalised to ``[0, 1]``.
        kernel_degree: Highest polynomial kernel degree in the roster; members
            are included at every degree up to it. 2 is the pairwise epistasis
            model. Note that 1 does *not* give a purely additive ensemble -- the
            local-kernel and k-NN members are nonlinear and remain in the roster
            -- it removes the pairwise kernel members only.
        cv_folds: Cross-validation folds used to rank and to fit members.
            Defaults to Wittmann et al.'s 5. Reduced automatically when there
            are fewer measurements than folds.
        n_averaged: Members whose fold instances are averaged. Defaults to
            Wittmann et al.'s 3. Clamped to the roster size.
        feasible_only: Draw only constructible candidates.
        max_attempts: Draws before giving up on filling the pool.
        seed: Seeds the random training sample, the candidate pool and the
            random projection behind the feature members.

    Raises:
        ValueError: If a size is not positive, the ridge penalty is negative,
            the kernel degree is below 1, or ``cv_folds`` is below 2.
    """

    def __init__(  # noqa: PLR0913 - the protocol is a training split plus an ensemble
        self,
        env: MutationEnvironment,
        *,
        training_size: int = DEFAULT_TRAINING_SIZE,
        pool_multiplier: int = 4,
        ridge_alpha: float = 1.0,
        kernel_degree: int = 2,
        cv_folds: int = PUBLISHED_CV_FOLDS,
        n_averaged: int = PUBLISHED_MODELS_AVERAGED,
        feasible_only: bool = False,
        max_attempts: int = 10,
        seed: int = 0,
    ) -> None:
        """Start in the random-screening stage, with nothing fitted."""
        super().__init__()
        for label, value in [
            ("training_size", training_size),
            ("pool_multiplier", pool_multiplier),
            ("max_attempts", max_attempts),
            ("n_averaged", n_averaged),
        ]:
            if value < 1:
                raise ValueError(f"{label} must be at least 1, got {value}")
        if ridge_alpha < 0.0:
            raise ValueError(f"ridge_alpha must not be negative, got {ridge_alpha}")
        if kernel_degree < 1:
            raise ValueError(f"kernel_degree must be at least 1, got {kernel_degree}")
        if cv_folds < 2:  # noqa: PLR2004 - one fold holds nothing out
            raise ValueError(f"cv_folds must be at least 2, got {cv_folds}")

        self._env = env
        self._training_size = training_size
        self._pool_multiplier = pool_multiplier
        self._max_attempts = max_attempts
        self._feasible_only = feasible_only
        self._cv_folds = cv_folds
        self._roster = _roster(ridge_alpha, kernel_degree)
        self._n_averaged = min(n_averaged, len(self._roster))
        # The random stage *is* random mutagenesis -- "sample the library
        # uniformly" is what the protocol says -- so it is the same object
        # rather than a second copy of the same sampling code.
        self._explorer = RandomMutagenesis(env, feasible_only=feasible_only, seed=seed)

        self._sequences: list[Tokens] = []
        self._values: list[float] = []
        self._measured: set[bytes] = set()
        self._fitted: Tokens | None = None
        self._fits: list[_Fit] = []
        self._selected: tuple[str, ...] = ()
        self._offset = 0.0
        # Bandwidth of the local kernel, fixed at fit time from the training set
        # alone. Recomputing it from the candidate pool would give the pool a
        # different kernel from the one the model was fitted under.
        self._median_distance = 1.0
        self._stale = True

    @classmethod
    def as_published(
        cls,
        env: MutationEnvironment,
        *,
        feasible_only: bool = False,
        seed: int = 0,
    ) -> MLDE:
        """Build MLDE at Wittmann et al.'s own split, for a reference run.

        This configuration cannot complete inside this repository's 384-call
        campaign budget: it screens :data:`PUBLISHED_TRAINING_SIZE` variants
        before proposing anything, and needs :data:`PUBLISHED_BUDGET` calls to
        reach a designed plate. It exists so that "MLDE as published" can be run
        as a separate reference arm at its own budget and quoted honestly beside
        the compressed arm, rather than the compressed arm being described as
        the published method.

        Args:
            env: Supplies the parent, alphabet, mutation budget and feasibility.
            feasible_only: Draw only constructible candidates.
            seed: Seeds the random training sample and the candidate pool.

        Returns:
            An :class:`MLDE` whose training sample is the published one.
        """
        return cls(
            env,
            training_size=PUBLISHED_TRAINING_SIZE,
            feasible_only=feasible_only,
            seed=seed,
        )

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "MLDE" + (" (feasible)" if self._feasible_only else "")

    @property
    def is_fitted(self) -> bool:
        """Whether the ensemble has taken over from random screening."""
        return self._fitted is not None

    @property
    def training_examples(self) -> int:
        """Measurements gathered so far."""
        return len(self._values)

    @property
    def members(self) -> tuple[str, ...]:
        """Names of every model class in the roster, in selection-tie order."""
        return tuple(learner.name for learner in self._roster)

    @property
    def selected_members(self) -> tuple[str, ...]:
        """Model classes the last cross-validation chose, best first.

        Empty before the first fit. Worth logging: if the same member wins on
        every landscape, the rest of the roster is decoration and the ensemble
        claim is weaker than it looks.
        """
        return self._selected

    @property
    def runs_below_published_training_size(self) -> bool:
        """Whether this configuration trains on less than Wittmann et al. did.

        True for the default, and true for every configuration that fits inside
        this repository's 384-call budget. A results table that reports this arm
        without the caveat is reporting a handicapped method under the published
        method's name.
        """
        return self._training_size < PUBLISHED_TRAINING_SIZE

    @property
    def required_budget(self) -> int:
        """Assays needed to complete one published-shaped protocol.

        Returns:
            The training sample plus one :data:`PUBLISHED_BATCH_SIZE` design
            plate. Compare against the campaign budget: if it does not fit, the
            method is not being run as published.
        """
        return self._training_size + PUBLISHED_BATCH_SIZE

    @property
    def budget_note(self) -> str:
        """One sentence a results table can print next to this arm.

        Returns:
            A statement of how this configuration's training sample compares
            with the published one, so the compression is carried alongside the
            number rather than left in a docstring nobody reads.
        """
        if not self.runs_below_published_training_size:
            return (
                f"trained on {self._training_size} variants, at or above Wittmann et al.'s "
                f"{PUBLISHED_TRAINING_SIZE}; needs {self.required_budget} assays in total"
            )
        fraction = 100.0 * self._training_size / PUBLISHED_TRAINING_SIZE
        return (
            f"trained on {self._training_size} variants, {fraction:.0f}% of Wittmann et al.'s "
            f"{PUBLISHED_TRAINING_SIZE}; the published protocol needs {PUBLISHED_BUDGET} assays "
            f"and does not fit this budget, so this arm is a handicapped MLDE"
        )

    def propose(self, n: int) -> Tokens:
        """Screen at random, or return the ensemble's ``n`` best predictions.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array, ranked best-first once the model
            is fitted so that a caller taking a prefix takes the top designs.

        Raises:
            RuntimeError: If ``feasible_only`` and no feasible candidate can be
                drawn; raised by the underlying random draw rather than here.
        """
        if not self._ready():
            return self._draw(n)

        self._refit()
        pool = self._pool(n)
        predictions = self._predict(pool)
        order = np.argsort(-predictions, kind="stable")[:n]
        return pool[order]

    def predict(self, sequences: Tokens) -> npt.NDArray[np.float64]:
        """Score sequences with the ensemble, refitting first if it is stale.

        Exposed because the ensemble's *predictions* are the thing worth
        checking against the single model it replaced, and a comparison that has
        to reach through :meth:`propose` measures the candidate pool as much as
        the regressor.

        Args:
            sequences: An ``(n, sequence_length)`` array to score.

        Returns:
            An ``(n,)`` array of predicted objective values, on the same scale as
            the observed ones.

        Raises:
            RuntimeError: If nothing has been observed yet, since an ensemble
                fitted on no measurements would return a constant and a caller
                ranking by it would be ranking noise without knowing.
        """
        if len(self._values) < _MIN_TRAINING:
            raise RuntimeError(
                f"MLDE needs at least {_MIN_TRAINING} finite measurements to fit, "
                f"got {len(self._values)}"
            )
        self._refit()
        return self._predict(np.ascontiguousarray(np.asarray(sequences)))

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Add measurements to the training set and mark the model out of date.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which the
                single-output surrogate has no target to regress on.
        """
        flat = single_objective(values)
        rows = np.ascontiguousarray(np.asarray(sequences))
        for row, value in zip(rows, flat, strict=False):
            self._measured.add(row.tobytes())
            if not np.isfinite(value):
                # A failed or infeasible assay carries no fitness to regress on.
                # Keeping it in `_measured` still stops it being proposed again.
                continue
            self._sequences.append(row.copy())
            self._values.append(float(value))
        self._stale = True

    def _ready(self) -> bool:
        """Whether enough has been measured to hand over to the model."""
        return len(self._values) >= max(self._training_size, _MIN_TRAINING)

    def _draw(self, n: int) -> Tokens:
        """Random library members, charging their cost to this sampler.

        The delta is taken from the explorer's own counter so that rejection
        sampling under ``feasible_only`` is charged here too, rather than
        vanishing into a nested object nobody reports on.
        """
        before = self._explorer.proposals_made
        drawn = self._explorer.propose(n)
        self._count(self._explorer.proposals_made - before)
        return drawn

    def _pool(self, n: int) -> Tokens:
        """Candidates for the model to rank, excluding anything already assayed.

        Returns:
            At least ``n`` distinct unmeasured sequences where the environment
            can supply them, and a plain random draw where it cannot -- a short
            plate would be a worse answer than a repeated one.
        """
        wanted = n * self._pool_multiplier
        collected: list[Tokens] = []
        seen: set[bytes] = set()
        found = 0
        for _ in range(self._max_attempts):
            batch = np.ascontiguousarray(self._draw(wanted))
            keep = []
            for position, row in enumerate(batch):
                key = row.tobytes()
                if key in seen or key in self._measured:
                    continue
                seen.add(key)
                keep.append(position)
            if keep:
                collected.append(batch[keep])
                found += len(keep)
            if found >= n:
                break
        if found < n:
            # The reachable set is nearly exhausted, or nearly all of it has
            # been assayed. Ranking is meaningless at that point; fall back to
            # the null so the round still fills.
            return self._draw(n)
        return np.concatenate(collected)

    def _refit(self) -> None:
        """Cross-validate the roster and keep the top members, if data changed.

        Follows Wittmann et al.: every member is fitted on each of ``cv_folds``
        training folds, members are ranked by mean squared error on the held-out
        folds, and the fold instances of the best ``n_averaged`` become the
        predictor. Every member is a transform of one shared agreement matrix,
        which is what keeps twelve models from costing twelve times one.
        """
        if not self._stale:
            return
        X = np.stack(self._sequences)
        y = np.asarray(self._values, dtype=np.float64)
        # Centring removes the intercept, which must not be penalised: a ridge
        # that shrinks the mean pulls every prediction toward zero, which on a
        # landscape with a large offset is most of the signal.
        self._offset = float(y.mean())
        target = y - self._offset

        agreement = self._agreement(X, X)
        self._median_distance = self._bandwidth(agreement)
        kernels = self._kernels(agreement)

        folds = max(2, min(self._cv_folds, target.size))
        # Interleaved rather than contiguous folds. The training set arrives in
        # acquisition order, so contiguous blocks would hold out an entire late
        # round -- which is systematically better than the early ones once the
        # model starts steering -- and rank members on a distribution shift
        # instead of on fit quality.
        assignment = np.arange(target.size) % folds

        errors = np.empty(len(self._roster), dtype=np.float64)
        instances: list[list[_Fit]] = []
        for index, learner in enumerate(self._roster):
            kernel = kernels[learner.kernel_key]
            fitted: list[_Fit] = []
            squared = 0.0
            for fold in range(folds):
                held = np.flatnonzero(assignment == fold)
                rows = np.flatnonzero(assignment != fold)
                fit = self._fit_member(learner, rows, kernel, target)
                fitted.append(fit)
                predicted = self._predict_member(fit, kernel[held])
                squared += float(((predicted - target[held]) ** 2).sum())
            errors[index] = squared / target.size
            instances.append(fitted)

        keep = np.argsort(errors, kind="stable")[: self._n_averaged]
        self._fits = [fit for index in keep for fit in instances[index]]
        self._selected = tuple(self._roster[index].name for index in keep)
        self._fitted = X
        self._stale = False

    def _fit_member(
        self,
        learner: _Learner,
        rows: npt.NDArray[np.intp],
        kernel: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> _Fit:
        """Train one member on one fold.

        Args:
            learner: The model class and its hyperparameters.
            rows: Indices of the training points this fold may see.
            kernel: Full ``(n, n)`` matrix of this member's kernel.
            target: Full ``(n,)`` centred objective values.

        Returns:
            The fitted instance, carrying whatever it needs to predict.
        """
        fold_target = target[rows]

        if learner.kind == "neighbours":
            # Lazy by construction: the "coefficients" are the fold's own values,
            # and prediction averages the k of them nearest the query.
            return _Fit(learner, rows, fold_target.copy(), 0.0)

        # Each fold instance carries its own mean. Using the global one would
        # leak the held-out fold's mean into the fit, which is small but is
        # exactly the kind of leak that makes cross-validated ranking optimistic.
        intercept = float(fold_target.mean())
        K = kernel[np.ix_(rows, rows)].copy()
        K[np.diag_indices_from(K)] += learner.alpha + _RIDGE_FLOOR
        dual: npt.NDArray[np.float64] = np.linalg.solve(K, fold_target - intercept).astype(
            np.float64, copy=False
        )
        return _Fit(learner, rows, dual, intercept)

    def _predict_member(
        self,
        fit: _Fit,
        kernel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Predict centred objective values from one fitted instance.

        Args:
            fit: A member trained on one fold.
            kernel: ``(m, n)`` kernel of the queries against *all* training
                points; the fold's own columns are selected here.

        Returns:
            An ``(m,)`` array, on the centred scale that :meth:`_predict`
            restores the offset to.
        """
        block = kernel[:, fit.rows]

        if fit.learner.kind == "neighbours":
            k = min(fit.learner.neighbours, fit.rows.size)
            nearest = np.argpartition(-block, k - 1, axis=1)[:, :k]
            neighbours: npt.NDArray[np.float64] = fit.coefficients[nearest].mean(axis=1)
            return neighbours

        weighted: npt.NDArray[np.float64] = block @ fit.coefficients
        return weighted + fit.intercept

    def _predict(self, sequences: Tokens) -> npt.NDArray[np.float64]:
        """Averaged prediction of every selected member's fold instances.

        Returns:
            An ``(n,)`` array.

        Raises:
            RuntimeError: If called before the ensemble has been fitted.
        """
        if self._fitted is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("MLDE predicted before it was fitted")
        kernels = self._kernels(self._agreement(sequences, self._fitted))
        total = np.zeros(sequences.shape[0], dtype=np.float64)
        for fit in self._fits:
            total += self._predict_member(fit, kernels[fit.learner.kernel_key])
        return total / len(self._fits) + self._offset

    def _agreement(self, left: Tokens, right: Tokens) -> npt.NDArray[np.float64]:
        """Normalised one-hot inner product, computed without the encoding.

        The inner product of two one-hot sequences is the number of positions at
        which they agree, so the Gram matrix is an agreement count and the
        ``(L·V)``-dimensional features are never built. Normalising by the length
        keeps it in ``[0, 1]``, which is what makes one default ridge penalty
        meaningful across sequence lengths and across the roster.

        Returns:
            An ``(len(left), len(right))`` array. Every member of the roster is
            an elementwise function of this one matrix, so it is built once per
            call and shared.
        """
        length = self._env.sequence_length
        gram = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
        for start in range(0, left.shape[0], _KERNEL_CHUNK):
            block = left[start : start + _KERNEL_CHUNK]
            counted = (block[:, None, :] == right[None, :, :]).sum(axis=2)
            gram[start : start + block.shape[0]] = counted / length
        return gram

    def _bandwidth(self, agreement: npt.NDArray[np.float64]) -> float:
        """Median pairwise Hamming distance across the training set.

        The median heuristic, and the reason one bandwidth setting transfers
        between an ``L = 8`` toy and an ``L = 256`` protein: the distances
        themselves differ by orders of magnitude, their median does not.

        Args:
            agreement: The square ``(n, n)`` training agreement matrix.

        Returns:
            A strictly positive distance, floored so that a training set of
            identical sequences yields a usable kernel rather than a NaN one.
        """
        upper = agreement[np.triu_indices_from(agreement, k=1)]
        if upper.size == 0:  # pragma: no cover - a fit needs at least two points
            return 1.0
        return max(float(np.median(1.0 - upper)), _DISTANCE_FLOOR)

    def _kernels(self, agreement: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        """Every kernel the roster needs, as transforms of one agreement matrix.

        Args:
            agreement: Normalised agreement, ``(m, n)`` or ``(n, n)``.

        Returns:
            A mapping from :attr:`_Learner.kernel_key` to the matrix that member
            reads. Built per call rather than cached because the training set
            changes every round; keyed so that the six polynomial members share
            two matrices and the four local members share two more.
        """
        built: dict[str, npt.NDArray[np.float64]] = {}
        for learner in self._roster:
            if learner.kernel_key in built:
                continue
            if learner.kind == "local":
                built[learner.kernel_key] = np.exp(
                    -(learner.scale / self._median_distance) * (1.0 - agreement)
                )
            else:
                built[learner.kernel_key] = agreement**learner.degree
        return built
