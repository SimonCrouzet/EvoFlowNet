"""Verify the Ehrlich implementation against an independent transcription.

Every other test in this suite was written by reading the implementation, so
they are self-consistent: if the vectorised scoring had the wrong placement
window or an off-by-one in the spacing offsets, those tests would agree with it
and still pass.

This module removes that circularity. :func:`reference_ehrlich` is a direct,
deliberately naive transcription of the equations in Stanton et al. -- plain
Python loops, no broadcasting, no shared code with the library -- and the tests
assert the two agree across randomly generated landscapes and sequences.

Two independent implementations can of course be wrong in the same way, but only
if the same misreading is made twice from the paper rather than once from the
code, which is a much narrower failure.
"""

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evogfn.landscapes import EhrlichLandscape


def reference_ehrlich(sequence, landscape):
    """Score one sequence by direct transcription of the paper's definition.

    f(x) = prod_i h_q(x, m_i, s_i)  if x in F, else -inf
    h_q(x, m, s) = max_l ( sum_j 1{x[l + s_j] == m_j} ) // (k/q) / q
    F = { x : A[x[l-1], x[l]] > 0 for all l >= 1 }
    """
    transitions = landscape.transition_matrix
    for position in range(1, len(sequence)):
        if transitions[sequence[position - 1], sequence[position]] <= 0:
            return -math.inf

    k = landscape.motif_length
    q = landscape.quantization
    step = k // q

    product = 1.0
    for motif, spacing in zip(landscape.motifs, landscape.spacings, strict=True):
        best = 0
        for start in range(len(sequence)):
            if start + spacing[-1] >= len(sequence):
                continue
            matches = sum(1 for j in range(k) if sequence[start + spacing[j]] == motif[j])
            best = max(best, matches)
        product *= (best // step) / q
    return product


@st.composite
def landscapes(draw):
    """Valid Ehrlich instances, small enough to score with Python loops.

    The length is derived from the motif geometry rather than drawn freely: a
    motif must fit inside its block, so drawing the two independently would
    mostly produce parameter combinations the constructor rightly rejects.
    """
    n_motifs = draw(st.integers(min_value=1, max_value=2))
    motif_length = draw(st.sampled_from([2, 4]))
    max_spacing = draw(st.integers(min_value=1, max_value=2))
    span = (motif_length - 1) * max_spacing + 1
    minimum_length = n_motifs * span
    return EhrlichLandscape(
        sequence_length=draw(st.integers(min_value=minimum_length, max_value=minimum_length + 12)),
        vocab_size=draw(st.integers(min_value=3, max_value=6)),
        n_motifs=n_motifs,
        motif_length=motif_length,
        quantization=draw(st.sampled_from([q for q in (1, 2, 4) if motif_length % q == 0])),
        max_spacing=max_spacing,
        transition_density=draw(st.floats(min_value=0.3, max_value=1.0)),
        seed=draw(st.integers(min_value=0, max_value=2**16)),
    )


class TestAgainstReference:
    @given(landscape=landscapes(), data=st.data())
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_random_sequences_score_identically(self, landscape, data):
        sequences = data.draw(
            st.lists(
                st.lists(
                    st.integers(min_value=0, max_value=landscape.alphabet.size - 1),
                    min_size=landscape.sequence_length,
                    max_size=landscape.sequence_length,
                ),
                min_size=1,
                max_size=8,
            )
        )
        batch = np.array(sequences, dtype=np.int32)
        actual = landscape.evaluate(batch)[:, 0]
        for row, value in zip(batch, actual, strict=True):
            expected = reference_ehrlich(row, landscape)
            if math.isinf(expected):
                assert np.isneginf(value)
            else:
                assert value == pytest.approx(expected)

    @given(landscape=landscapes())
    @settings(max_examples=40, deadline=None)
    def test_the_planted_optimum_agrees_with_the_reference(self, landscape):
        assert reference_ehrlich(landscape.optimal_sequence, landscape) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(6))
    def test_whole_small_space_agrees_with_the_reference(self, seed):
        # Exhaustive rather than sampled: every sequence in the space, including
        # the infeasible ones and the many that score zero.
        landscape = EhrlichLandscape(
            sequence_length=5,
            vocab_size=3,
            n_motifs=1,
            motif_length=2,
            max_spacing=2,
            transition_density=0.6,
            seed=seed,
        )
        every = landscape.enumerate()
        actual = landscape.evaluate(every)[:, 0]
        expected = np.array([reference_ehrlich(row, landscape) for row in every])
        assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
        finite = ~np.isneginf(expected)
        assert np.allclose(actual[finite], expected[finite])


def blank_sequence(landscape, motif):
    """A sequence of a token that appears in no motif.

    Filling with token 0 is a trap: if a motif contains token 0, the background
    itself completes the motif and the sequence scores 1.0 for reasons that have
    nothing to do with what is being tested.
    """
    used = set(landscape.motifs.ravel().tolist())
    filler = next(t for t in range(landscape.alphabet.size) if t not in used)
    sequence = np.full(landscape.sequence_length, filler, dtype=np.int32)
    assert filler not in motif.tolist()
    return sequence


class TestMotifSemantics:
    """Checks on what the scoring actually means, not just that it is consistent."""

    def test_a_motif_scores_wherever_it_appears(self):
        # h_q takes the maximum over placements, so relocating a satisfied motif
        # must not change the score. If the placement window were wrong, this is
        # what would break.
        landscape = EhrlichLandscape(
            sequence_length=16,
            vocab_size=4,
            n_motifs=1,
            motif_length=2,
            max_spacing=1,
            transition_density=1.0,
            seed=1,
        )
        motif = landscape.motifs[0]
        spacing = landscape.spacings[0]
        span = int(spacing[-1]) + 1

        scores = []
        for start in range(landscape.sequence_length - span + 1):
            sequence = blank_sequence(landscape, motif)
            for j, offset in enumerate(spacing):
                sequence[start + int(offset)] = motif[j]
            scores.append(landscape.evaluate(sequence[None, :])[0, 0])

        assert len(set(scores)) == 1, "the same motif scored differently by position"
        assert scores[0] == pytest.approx(1.0)

    def test_right_tokens_at_wrong_spacing_do_not_fully_satisfy(self):
        # The gaps are part of the motif. Tokens in the right order but the wrong
        # positions must not score as a full match.
        # Search for an instance whose motif actually has a gap, rather than
        # skipping when the seed happens to produce an adjacent one -- a test
        # that silently skips is a test that stops protecting anything.
        landscape = next(
            candidate
            for candidate in (
                EhrlichLandscape(
                    sequence_length=16,
                    vocab_size=4,
                    n_motifs=1,
                    motif_length=2,
                    quantization=1,
                    max_spacing=3,
                    transition_density=1.0,
                    seed=seed,
                )
                for seed in range(50)
            )
            if int(candidate.spacings[0][1]) > 1
        )
        motif = landscape.motifs[0]

        # Place the two tokens adjacently instead of at their true gap.
        sequence = blank_sequence(landscape, motif)
        sequence[0] = motif[0]
        sequence[1] = motif[1]
        assert landscape.evaluate(sequence[None, :])[0, 0] < 1.0

        # ...and confirm the same tokens at the correct gap do score, so the
        # assertion above is about the spacing and not about the tokens.
        correct = blank_sequence(landscape, motif)
        for j, offset in enumerate(landscape.spacings[0]):
            correct[int(offset)] = motif[j]
        assert landscape.evaluate(correct[None, :])[0, 0] == pytest.approx(1.0)

    def test_score_is_the_product_over_motifs(self):
        # Satisfying one of two motifs and not the other scores zero, not a half.
        landscape = EhrlichLandscape(
            sequence_length=24,
            vocab_size=4,
            n_motifs=2,
            motif_length=2,
            quantization=1,
            max_spacing=1,
            transition_density=1.0,
            seed=8,
        )
        motif, spacing = landscape.motifs[0], landscape.spacings[0]
        sequence = blank_sequence(landscape, motif)
        for j, offset in enumerate(spacing):
            sequence[int(offset)] = motif[j]
        score = landscape.evaluate(sequence[None, :])[0, 0]
        assert score in (0.0, 1.0)


class TestTransitionMatrix:
    """The transition matrix has to be a valid, connected Markov chain."""

    @given(
        vocab_size=st.integers(min_value=3, max_value=12),
        density=st.floats(min_value=0.05, max_value=1.0),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=50, deadline=None)
    def test_rows_are_probability_distributions(self, vocab_size, density, seed):
        landscape = EhrlichLandscape(
            sequence_length=12,
            vocab_size=vocab_size,
            n_motifs=1,
            motif_length=2,
            transition_density=density,
            seed=seed,
        )
        matrix = landscape.transition_matrix
        assert (matrix >= 0).all()
        assert np.allclose(matrix.sum(axis=1), 1.0)

    @given(
        vocab_size=st.integers(min_value=3, max_value=12),
        density=st.floats(min_value=0.01, max_value=0.3),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    @settings(max_examples=50, deadline=None)
    def test_every_token_stays_reachable_at_any_density(self, vocab_size, density, seed):
        # The construction lays down a Hamiltonian cycle before sparsifying,
        # precisely so this holds. Without it, sampling a feasible sequence could
        # walk into a token with no permitted successor and deadlock.
        landscape = EhrlichLandscape(
            sequence_length=12,
            vocab_size=vocab_size,
            n_motifs=1,
            motif_length=2,
            transition_density=density,
            seed=seed,
        )
        adjacency = landscape.transition_matrix > 0
        assert adjacency.any(axis=1).all(), "a token has no permitted successor"
        assert _reachable_from(adjacency, 0) == set(range(vocab_size))
        assert _reachable_from(adjacency.T, 0) == set(range(vocab_size))


def _reachable_from(adjacency, start):
    """Every node reachable from ``start`` by following permitted transitions."""
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for candidate in np.flatnonzero(adjacency[node]):
            if int(candidate) not in seen:
                seen.add(int(candidate))
                frontier.append(int(candidate))
    return seen
