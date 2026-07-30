"""What a classical baseline may carry when the campaign moves its anchor.

A campaign round searches one Hamming ball. Between rounds
[reanchored][evogfn.env.mutation.MutationEnvironment.reanchored] moves the ball
to the best design measured, and every sampler then has to answer a question the
campaign is not entitled to answer for it: which of my state is still true?

The alternative the campaign falls back on is a rebuild through a factory. That
is always correct and always forgetful -- a rebuilt MLDE has no training set, a
rebuilt annealer is hot again, a rebuilt population is a column of copies of the
new anchor. Under a headline claim that a learned policy transfers across anchors
where classical state does not, a forgetful rebuild is not a neutral default: it
manufactures the very gap the claim is about. So each baseline implements
[reanchored][evogfn.loop.campaign.ReanchorableSampler.reanchored] and carries
everything the architecture genuinely permits.

Three kinds of state, and only one of them is anchor-relative
-------------------------------------------------------------

* **Anchor-free.** Sequences, measured values, a fitted regressor, a temperature,
  a random stream. None of these are expressed relative to the parent, so they
  cross a move untouched. This turns out to be most of what the baselines hold:
  MLDE's whole dataset and ensemble, simulated annealing's entire schedule and
  chain position, hill climbing's incumbent.
* **Anchor-relative but re-projectable.** A population is a set of sequences,
  which is anchor-free, but *membership of the graph* is not: an individual 40
  substitutions from the old anchor may be outside the new anchor's budget.
  [reprojected][evogfn.algorithms.baselines._reanchor.reprojected] pulls it back
  in the way the breeding operators already do, and reports which rows it had to
  change so their stale measurements can be dropped with them.
* **Anchor-relative and not re-projectable.** The pairing between a sampler's
  internal draw and the sequence it decoded to, when the decoder reads the
  anchor. CMA-ES is the case; see its own docstring. The pairing is dropped
  rather than reinterpreted, because reinterpreting it silently attributes a
  score to the wrong draw.

The failure this module prevents is a sampler that carries state which has
quietly stopped meaning what it meant -- a fitness attached to a sequence that
was reverted underneath it, an incumbent the new environment cannot build. Both
would survive every type check and change the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Tokens
    from evogfn.env.mutation import MutationEnvironment


def carried_design(env: MutationEnvironment, design: Tokens) -> Tokens:
    """The design a point-based sampler should stand on inside ``env``.

    Ordinarily the answer is "the one it was already standing on": the campaign
    re-anchors *at* the best design measured, and a hill climber or an annealer
    is usually standing on exactly that, zero mutations from the new anchor.
    But the campaign selects its anchor through the acquisition rule over the
    whole batch, not through the sampler's own bookkeeping, so the two can
    disagree -- and a design far enough from the new anchor is outside the new
    budget, at which point the sampler would be proposing neighbours of a point
    the environment cannot build.

    Args:
        env: The re-anchored environment.
        design: The sequence the sampler currently stands on.

    Returns:
        ``design`` where ``env`` can construct it, and ``env.parent`` otherwise.
        Falling back to the anchor rather than repairing the design is
        deliberate: the anchor is the best measurement the campaign has, so it
        is a better restart than an arbitrary projection of a worse point.
    """
    candidate = np.asarray(design)
    if candidate.shape != (env.sequence_length,):
        return env.parent
    return candidate.copy() if bool(env.is_reachable(candidate[None, :])[0]) else env.parent


def reprojected(
    env: MutationEnvironment,
    population: Tokens,
    rng: np.random.Generator,
) -> tuple[Tokens, npt.NDArray[np.bool_]]:
    """Pull a population back inside the new anchor's mutation budget.

    An individual is a sequence and a sequence does not depend on an anchor, so
    the population itself transfers. What does not transfer is its *legality*:
    the budget is counted from the parent, and the parent moved. Individuals
    already inside the new ball are untouched; the rest have surplus
    substitutions reverted to the new anchor, chosen uniformly at random, which
    is the same projection the breeding operators apply to over-budget
    offspring. Re-using it keeps a re-projected individual indistinguishable
    from one the algorithm could have bred, rather than introducing a second
    notion of what a legal individual is.

    **Why the changed rows are reported.** A population is carried alongside the
    fitness of each member, and a reverted individual is a different sequence
    whose recorded fitness was measured on the old one. Keeping that number
    would let selection promote a design on the strength of a measurement that
    was never taken on it -- an invented result, and one nothing downstream
    could detect. The caller is handed the mask so it can drop those values.

    Args:
        env: The re-anchored environment.
        population: An ``(n, sequence_length)`` array of individuals.
        rng: Draws which surplus substitutions to revert. Passing the sampler's
            own generator is what keeps a whole campaign reproducible from one
            seed across a move.

    Returns:
        The re-projected population and an ``(n,)`` boolean mask that is ``True``
        where the individual came through unchanged, and therefore ``True``
        exactly where its recorded fitness still belongs to it.
    """
    projected = np.array(population, copy=True)
    if projected.size == 0:
        return projected, np.ones(projected.shape[0], dtype=np.bool_)

    anchor = env.parent
    budget = env.max_mutations
    differing = projected != anchor[None, :]
    counts = differing.sum(axis=1)
    intact = np.ones(projected.shape[0], dtype=np.bool_)

    for row in np.flatnonzero(counts > budget):
        positions = np.flatnonzero(differing[row])
        surplus = rng.choice(positions, size=int(counts[row] - budget), replace=False)
        projected[row, surplus] = anchor[surplus]
        intact[row] = False
    return projected, intact
