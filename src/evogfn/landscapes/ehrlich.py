r"""Ehrlich functions: closed-form test landscapes with a known optimum.

Introduced by Stanton et al., *Closed-Form Test Functions for Biophysical
Sequence Optimization Algorithms* (ICML 2024 workshop). They abstract the
geometry of biophysical sequence design -- epistasis, ruggedness, and the fact
that most strings are not constructible at all -- into something that evaluates
in microseconds and whose optimum is known by construction.

The definition, in the paper's notation:

.. math::

    f(x) = \prod_{i=1}^{c} h_q(x, m^{(i)}, s^{(i)})
    \quad\text{if } x \in \mathcal{F},\quad -\infty \text{ otherwise}

.. math::

    h_q(x, m, s) = \max_{\ell} \left(
        \sum_{j=1}^{k} \mathbb{1}\{x_{\ell + s_j} = m_j\}
    \right) // (k/q) / q

A sequence scores by how well it satisfies each of ``c`` *spaced motifs*: an
ordered set of ``k`` tokens at fixed relative offsets, which may appear anywhere
in the sequence. The best-matching placement of each motif is taken, quantised
to ``q`` levels, and the ``c`` results are multiplied -- so missing one motif
entirely zeroes the score regardless of the others. That product is the source of
the epistasis.

Feasibility is separate and harder. The feasible set

.. math:: \mathcal{F} = \{x : A[x_{\ell-1}, x_{\ell}] > 0 \ \forall \ell \ge 2\}

is induced by a discrete Markov process transition matrix ``A`` with some zero
entries: certain token pairs simply cannot be adjacent. Uniformly random
sequences are almost all infeasible for large ``L``, which is what makes naive
search waste its budget.

Two parameters carry most of the difficulty. ``q`` (which must divide ``k``)
controls how sparse the reward signal is: ``q = k`` increments on every extra
matched token, while ``q = 1`` pays nothing until a motif is matched in full.
``c`` and ``k`` set how much has to be satisfied at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.core.types import Alphabet
from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

#: Smallest alphabet that can carry an infeasible transition while staying
#: strongly connected. Below this, every transition matrix is either reducible
#: or fully dense, and feasibility stops meaning anything.
MIN_VOCAB_SIZE = 3


class EhrlichLandscape(FitnessLandscape):
    """A closed-form sequence landscape with a planted, reachable optimum.

    The motifs are carved out of a sequence that is itself feasible, which is
    what guarantees they are jointly satisfiable and that a sequence scoring
    exactly ``1.0`` exists. Verifying that is cheap, so the constructor does it
    rather than trusting the construction.

    Args:
        sequence_length: Length ``L`` of every sequence.
        vocab_size: Alphabet size ``v``. Must be at least
            :data:`MIN_VOCAB_SIZE`.
        n_motifs: Number of motifs ``c`` that must be satisfied simultaneously.
        motif_length: Tokens per motif, ``k``.
        quantization: Number of reward levels per motif, ``q``. Must divide
            ``k``. ``q = k`` gives a dense signal, ``q = 1`` an all-or-nothing
            one.
        max_spacing: Largest gap between consecutive motif positions. Larger
            values spread each motif over more of the sequence.
        transition_density: Fraction of token pairs allowed to be adjacent,
            beyond the cycle that keeps the chain strongly connected. Lower
            values make feasibility harder.
        seed: Seed for the landscape instance. The same seed gives the same
            motifs, transition matrix and optimum.

    Raises:
        ValueError: If the parameters cannot describe a valid instance --
            ``q`` not dividing ``k``, motifs too long to fit in the sequence,
            or a vocabulary too small to constrain.
    """

    def __init__(  # noqa: PLR0913 - a benchmark's parameters are its definition
        self,
        *,
        sequence_length: int = 32,
        vocab_size: int = 20,
        n_motifs: int = 2,
        motif_length: int = 4,
        quantization: int | None = None,
        max_spacing: int = 3,
        transition_density: float = 0.5,
        seed: int = 0,
    ) -> None:
        """Construct the landscape and verify that its optimum is attainable."""
        quantization = motif_length if quantization is None else quantization
        self._validate_parameters(
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            n_motifs=n_motifs,
            motif_length=motif_length,
            quantization=quantization,
            max_spacing=max_spacing,
        )

        self._sequence_length = sequence_length
        self._alphabet = Alphabet.from_string(
            "".join(chr(ord("A") + i) for i in range(vocab_size))
            if vocab_size <= 26  # noqa: PLR2004 - the Latin alphabet has 26 letters
            else "".join(chr(0x100 + i) for i in range(vocab_size))
        )
        self._n_motifs = n_motifs
        self._motif_length = motif_length
        self._quantization = quantization

        rng = np.random.default_rng(seed)
        self._transitions = _random_ergodic_transitions(vocab_size, transition_density, rng)
        self._optimal_sequence = self._sample_feasible(rng)
        self._motifs, self._spacings = self._carve_motifs(self._optimal_sequence, max_spacing, rng)

        # The construction should make the planted sequence score 1.0. Check it,
        # because a silent off-by-one here would invalidate every regret number
        # the landscape ever reports.
        achieved = float(self.evaluate(self._optimal_sequence[None, :])[0, 0])
        if not np.isclose(achieved, 1.0):
            raise RuntimeError(
                f"planted optimum scores {achieved}, not 1.0; the motif construction is wrong"
            )

    @staticmethod
    def _validate_parameters(  # noqa: PLR0913 - mirrors the constructor
        *,
        sequence_length: int,
        vocab_size: int,
        n_motifs: int,
        motif_length: int,
        quantization: int,
        max_spacing: int,
    ) -> None:
        """Reject parameter combinations that cannot describe a valid instance."""
        if vocab_size < MIN_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {MIN_VOCAB_SIZE}, got {vocab_size}")
        if n_motifs < 1 or motif_length < 1:
            raise ValueError("n_motifs and motif_length must both be at least 1")
        if quantization < 1 or motif_length % quantization != 0:
            raise ValueError(
                f"quantization must be at least 1 and divide motif_length, "
                f"got q={quantization}, k={motif_length}"
            )
        if max_spacing < 1:
            raise ValueError(f"max_spacing must be at least 1, got {max_spacing}")
        # Each motif is carved from its own block so that the blocks can be
        # satisfied independently; the widest a motif can span must fit in one.
        block = sequence_length // n_motifs
        span = (motif_length - 1) * max_spacing + 1
        if span > block:
            raise ValueError(
                f"a motif spans up to {span} positions but each of the {n_motifs} blocks is only "
                f"{block} long; reduce motif_length or max_spacing, or lengthen the sequence"
            )

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of every sequence this landscape scores."""
        return self._sequence_length

    @property
    def optimum(self) -> Fitness:
        """The best attainable score, which is ``1.0`` by construction."""
        return np.ones((1,), dtype=np.float64)

    @property
    def optimal_sequence(self) -> Tokens:
        """A sequence achieving :attr:`optimum`.

        There may be others; this is the one the motifs were carved from.
        """
        return self._optimal_sequence.copy()

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        """The transition matrix whose zeros define infeasible adjacencies."""
        return self._transitions.copy()

    @property
    def n_motifs(self) -> int:
        """Number of motifs ``c`` that must be satisfied simultaneously."""
        return self._n_motifs

    @property
    def motif_length(self) -> int:
        """Tokens per motif, ``k``."""
        return self._motif_length

    @property
    def quantization(self) -> int:
        """Reward levels per motif, ``q``."""
        return self._quantization

    @property
    def motifs(self) -> Tokens:
        """The ``(n_motifs, motif_length)`` motif tokens."""
        return self._motifs.copy()

    @property
    def spacings(self) -> Tokens:
        """The ``(n_motifs, motif_length)`` offsets, relative to a placement."""
        return self._spacings.copy()

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences use only permitted adjacent token pairs.

        Args:
            sequences: An ``(n, sequence_length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.

        Raises:
            ValueError: If the input fails validation.
        """
        checked = self._validate(sequences)
        if checked.shape[1] < 2:  # noqa: PLR2004 - a single token has no adjacency to constrain
            return np.ones(checked.shape[0], dtype=np.bool_)
        allowed = self._transitions[checked[:, :-1], checked[:, 1:]] > 0
        return np.all(allowed, axis=1)

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Compute the product of quantised motif satisfactions.

        Infeasible sequences score ``-inf``, following the paper. Callers that
        need a non-negative reward apply a reward transform; keeping the raw
        definition here means the landscape stays checkable against the paper.
        """
        scores = np.ones(sequences.shape[0], dtype=np.float64)
        for motif, spacing in zip(self._motifs, self._spacings, strict=True):
            scores *= self._motif_satisfaction(sequences, motif, spacing)
        scores[~self.is_feasible(sequences)] = -np.inf
        return scores[:, None]

    def _motif_satisfaction(
        self, sequences: Tokens, motif: Tokens, spacing: Tokens
    ) -> npt.NDArray[np.float64]:
        """Best quantised match for one motif over all its valid placements."""
        # Placements where the whole motif still fits inside the sequence.
        last_offset = int(spacing[-1])
        n_placements = self._sequence_length - last_offset
        # (n_placements, k) absolute positions to read for each placement.
        positions = np.arange(n_placements)[:, None] + spacing[None, :]
        # (n, n_placements, k) -> match counts (n, n_placements) -> best (n,).
        gathered = sequences[:, positions]
        matches = (gathered == motif[None, None, :]).sum(axis=2)
        best = matches.max(axis=1)
        # Quantise: floor to one of q levels, then scale into [0, 1].
        step = self._motif_length // self._quantization
        quantised: npt.NDArray[np.float64] = (best // step) / self._quantization
        return quantised

    def feasible_sequence(self, seed: int = 0) -> Tokens:
        """Draw a feasible sequence, for use as a campaign's starting point.

        A directed-evolution campaign starts from a wild type, and on this
        landscape that must be a sequence the DMP admits -- an infeasible parent
        would score minus infinity and give a mutation-based sampler nothing to
        climb from. The draw is independent of the planted optimum, so it leaks
        no information about the answer.

        Args:
            seed: Seeds the walk.

        Returns:
            A feasible sequence of the landscape's length.
        """
        return self._sample_feasible(np.random.default_rng(seed))

    def _sample_feasible(self, rng: np.random.Generator) -> Tokens:
        """Draw a feasible sequence by walking the Markov chain."""
        sequence = np.empty(self._sequence_length, dtype=np.int32)
        sequence[0] = rng.integers(self._alphabet.size)
        for position in range(1, self._sequence_length):
            sequence[position] = rng.choice(
                self._alphabet.size, p=self._transitions[sequence[position - 1]]
            )
        return sequence

    def _carve_motifs(
        self, source: Tokens, max_spacing: int, rng: np.random.Generator
    ) -> tuple[Tokens, Tokens]:
        """Take the motifs out of a sequence that already satisfies them.

        Carving from a known-feasible sequence is what guarantees the motifs are
        jointly satisfiable: the source sequence satisfies all of them at once,
        by construction. Each motif comes from its own block so the blocks stay
        independent.
        """
        motifs = np.empty((self._n_motifs, self._motif_length), dtype=np.int32)
        spacings = np.empty((self._n_motifs, self._motif_length), dtype=np.int32)
        block = self._sequence_length // self._n_motifs

        for index in range(self._n_motifs):
            gaps = rng.integers(1, max_spacing + 1, size=self._motif_length - 1)
            offsets = np.concatenate([[0], np.cumsum(gaps)]).astype(np.int32)
            # Place the motif inside its block, leaving room for its full span.
            latest_start = block - int(offsets[-1])
            start = index * block + int(rng.integers(latest_start))
            motifs[index] = source[start + offsets]
            spacings[index] = offsets
        return motifs, spacings


def _random_ergodic_transitions(
    vocab_size: int, density: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Build a row-stochastic transition matrix with some forbidden transitions.

    Strong connectivity is guaranteed by first laying down a random Hamiltonian
    cycle, so every token remains reachable from every other however sparse the
    rest is. Without that, sampling a feasible sequence could deadlock in a
    token with no permitted successor.

    Args:
        vocab_size: Number of tokens.
        density: Probability of allowing each transition beyond the cycle.
        rng: Source of randomness.

    Returns:
        A ``(vocab_size, vocab_size)`` row-stochastic matrix whose zeros mark
        forbidden adjacencies.
    """
    allowed = rng.random((vocab_size, vocab_size)) < density
    cycle = rng.permutation(vocab_size)
    allowed[cycle, np.roll(cycle, -1)] = True

    weights = rng.uniform(0.5, 1.5, size=(vocab_size, vocab_size)) * allowed
    return weights / weights.sum(axis=1, keepdims=True)
