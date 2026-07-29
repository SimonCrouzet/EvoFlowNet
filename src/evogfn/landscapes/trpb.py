"""The TrpB four-site fitness landscape.

Johnston, Almhjell, Watkins-Dulaney, Liu, Porter, Yang and Arnold, *A
combinatorially complete epistatic fitness landscape in an enzyme active site*
(PNAS 121(31), 2024) measured the growth fitness of variants at four active-site
positions -- V183, F184, V227 and S228 -- of the β-subunit of tryptophan
synthase from *Thermotoga maritima*. With 20 amino acids at each of 4 positions
the space is 160,000 sequences, of which **159,129 were measured**.

This landscape exists because GB1 alone is not an argument. GB1 is the single
most over-used empirical benchmark in the field, and a method tuned on it is
indistinguishable from a method that works. TrpB is the same *shape* of problem
-- four coupled positions, 20 amino acids, enumerable, near-complete -- measured
on a different protein, by a different assay (pooled growth selection in a
tryptophan auxotroph rather than mRNA display), with a different reward
geometry. A result that holds on both is a result about the method.

Why it is harder than GB1
------------------------

Wild-type is ``VFVS``. On the scale used here it sits at 1.0 by construction and
**99.3% of measured variants score below it** -- against 90% for GB1. The
published analysis is blunter still: only **9,783 of 159,129 variants (6.1%)**
are statistically distinguishable from dead, and the landscape carries **520
local optima**. Almost all of the space is flat and dead, so a sampler that
relies on a fitness gradient to find its way has almost no gradient to follow.

The best measured variant is ``AIKG`` at **2.45**, and like GB1's ``FWAA`` it
differs from wild-type at **all four positions**. The runner-up ``CLKG`` at 2.28
shares three of them, which is the epistasis both datasets are known for.

Fitness scale
-------------

Values are FLIP2's, which normalise **wild-type to 1.0**. The PNAS paper and the
`variationalsearch` redistribution instead normalise the **optimum** to 1.0; the
two differ by the constant factor 2.4505 and are otherwise the same measurement
(Pearson r = 0.9997 across all 159,129 variants). Wild-type-relative is the more
useful convention here and matches :mod:`evogfn.landscapes.gb1`, where 1.0
is likewise wild-type, so reward functions transfer between the two without
rescaling.

The 0.55% that were never measured
----------------------------------

871 combinations are absent from the assay. As in GB1, absent means *not
recovered from the library*, not *unfit*. They are imputed as ``0.0`` by
default -- which is close to the median measured fitness of 0.03, i.e. dead --
and :meth:`TrpBLandscape.is_measured` exists so an analysis can exclude them
instead. :attr:`TrpBLandscape.optimum` is likewise the best *measured* fitness.

What this loader deliberately does not provide
----------------------------------------------

The paper's per-variant **active/inactive** call is not a threshold on fitness
(the lowest active variant scores below the highest inactive one), so it cannot
be reconstructed from fitness values alone, and FLIP2 does not redistribute it.
Anyone needing that flag must go to the primary CaltechDATA deposit. See
``notes/review/07-trpb-provenance.md``.
"""

from __future__ import annotations

import csv
import gzip
from typing import TYPE_CHECKING, Final

import numpy as np

from evogfn.core.types import Alphabet
from evogfn.data.cache import fetch
from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

#: Redistributed by FLIP2 (Didi et al., ICML 2026) under CC-BY 4.0. The Zenodo
#: deposit is used rather than ``flip.protein.properties`` because it is
#: versioned, DOI-addressed and publishes its own per-file checksum. The primary
#: deposit (CaltechDATA ``10.22002/h5rah-5z170``, CC0-1.0) is a 3.3GB archive
#: and is not a reasonable dependency for a unit test.
TRPB_URL = "https://zenodo.org/records/18433203/files/trpb/by_position.csv.gz?download=1"

#: Pinned so that a silently updated remote file fails loudly rather than
#: changing published numbers. Verified against the md5 Zenodo publishes for the
#: same file (``77121ffe9d56362edb64614ff0e34f45``).
TRPB_SHA256 = "924a3f4942d89d123856905caac60c2a6739f2bdd92e879caafe9b5e3619cf17"

TRPB_FILENAME = "trpb_flip2_by_position.csv.gz"

#: Positions mutated in the four-site library, in the numbering of the paper.
TRPB_POSITIONS = (183, 184, 227, 228)

#: The wild-type combination at those positions.
TRPB_WILD_TYPE = "VFVS"

#: Number of variants actually assayed, of the 160,000 possible.
TRPB_N_MEASURED = 159_129

#: Length of the full-length sequences in the FLIP2 file.
TRPB_FULL_LENGTH = 389

#: Total rows in the FLIP2 file: the four-site library plus nine 3-site
#: sub-libraries, which this loader filters out.
_N_ROWS = 228_298

# FLIP2 ships all ten TrpB sub-libraries as full-length sequences in one file,
# so the four-site landscape has to be selected rather than read off. A row
# belongs to it exactly when it is wild-type at every *other* position any
# sub-library varies -- these sixteen, in the paper's numbering. Filtering on
# these alone is equivalent to comparing the whole 389-residue sequence, because
# the file varies at no other position; the pinned checksum is what makes that
# equivalence safe to rely on.
_CONTEXT: Final = (
    (104, "A"),
    (105, "E"),
    (106, "T"),
    (107, "G"),
    (108, "A"),
    (117, "T"),
    (118, "A"),
    (119, "A"),
    (162, "L"),
    (166, "I"),
    (182, "Y"),
    (185, "G"),
    (186, "S"),
    (230, "G"),
    (231, "S"),
    (301, "Y"),
)


class TrpBLandscape(FitnessLandscape):
    """Empirical fitness of TrpB variants at four active-site positions.

    Scoring is a table lookup, so evaluation is effectively free and the whole
    160,000-sequence space can be enumerated -- which is what makes exact regret
    and an exact target distribution available, as for
    :class:`~evogfn.landscapes.gb1.GB1Landscape`.

    Args:
        unmeasured_value: Fitness returned for the 871 combinations absent from
            the assay. Defaults to ``0.0``, which on this scale is near the
            median measured fitness. Pass ``float("nan")`` to make their absence
            propagate visibly instead of being silently treated as dead.
        force_download: Re-download the dataset even if a valid cached copy
            exists.

    Raises:
        ChecksumMismatchError: If the downloaded data does not match its pinned
            checksum.
        ValueError: If the file does not contain the expected number of rows or
            four-site variants, which would mean it is not the dataset this
            class was built against.
    """

    def __init__(self, *, unmeasured_value: float = 0.0, force_download: bool = False) -> None:
        """Load the landscape into a dense lookup table."""
        self._alphabet = Alphabet.protein()
        self._unmeasured_value = unmeasured_value

        path = fetch(TRPB_URL, sha256=TRPB_SHA256, filename=TRPB_FILENAME, force=force_download)
        variants, fitness = _read_variants(path)

        size = self._alphabet.size
        self._table = np.full(size**4, unmeasured_value, dtype=np.float64)
        self._measured = np.zeros(size**4, dtype=np.bool_)

        indices = self._flat_index(self._alphabet.encode_many(variants))
        self._table[indices] = fitness
        self._measured[indices] = True

        self._best_measured = float(fitness.max())
        self._best_variant = variants[int(np.argmax(fitness))]

    @property
    def alphabet(self) -> Alphabet:
        """The 20 standard amino acids."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Four, one per assayed position."""
        return len(TRPB_POSITIONS)

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Name of the single objective."""
        return ("growth_fitness",)

    @property
    def optimum(self) -> Fitness:
        """Best *measured* fitness, 2.45 for ``AIKG``.

        Exact with respect to the assay, not to the full 160,000-sequence space:
        one of the 871 unmeasured combinations could in principle score higher.
        """
        return np.array([self._best_measured], dtype=np.float64)

    @property
    def optimal_variant(self) -> str:
        """The best measured combination, ``AIKG``."""
        return self._best_variant

    @property
    def wild_type(self) -> Tokens:
        """The wild-type combination ``VFVS``, whose fitness is 1.0 by definition."""
        return self._alphabet.encode(TRPB_WILD_TYPE)

    @property
    def n_measured(self) -> int:
        """How many of the 160,000 combinations were assayed."""
        return int(self._measured.sum())

    def is_measured(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences were present in the assay.

        Args:
            sequences: An ``(n, 4)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array. ``False`` means the returned fitness is
            the imputed ``unmeasured_value``, not a measurement.

        Raises:
            ValueError: If the input fails validation.
        """
        checked = self._validate(sequences)
        return self._measured[self._flat_index(checked)]

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Look up fitness for each sequence."""
        return self._table[self._flat_index(sequences)][:, None]

    def _flat_index(self, sequences: Tokens) -> npt.NDArray[np.intp]:
        """Collapse ``(n, 4)`` token indices to positions in the lookup table."""
        size = self._alphabet.size
        weights = size ** np.arange(self.sequence_length - 1, -1, -1)
        return np.asarray(sequences, dtype=np.intp) @ weights


def _read_variants(path: Path) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Extract the four-site sub-landscape from the FLIP2 TrpB file.

    Args:
        path: Path to the downloaded gzipped CSV.

    Returns:
        The four-letter combinations and their fitness values.

    Raises:
        ValueError: If a sequence is not full length, or if the file does not
            yield the expected row and variant counts -- any of which would mean
            the file is not the dataset this landscape was built against.
    """
    four_sites = [position - 1 for position in TRPB_POSITIONS]
    context = [(position - 1, residue) for position, residue in _CONTEXT]

    variants: list[str] = []
    values: list[float] = []
    rows = 0
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            sequence = row["sequence"]
            if len(sequence) != TRPB_FULL_LENGTH:
                raise ValueError(
                    f"expected sequences of length {TRPB_FULL_LENGTH}, found one of "
                    f"length {len(sequence)}"
                )
            if any(sequence[index] != residue for index, residue in context):
                continue
            variants.append("".join(sequence[index] for index in four_sites))
            values.append(float(row["target"]))

    if rows != _N_ROWS:
        raise ValueError(
            f"expected {_N_ROWS} rows across the ten TrpB sub-landscapes, found {rows}; "
            f"the dataset is not the one this landscape was built against"
        )
    if len(variants) != TRPB_N_MEASURED:
        raise ValueError(
            f"expected {TRPB_N_MEASURED} four-site variants, found {len(variants)}; "
            f"the four-site sub-landscape could not be identified in this file"
        )
    return variants, np.asarray(values, dtype=np.float64)
