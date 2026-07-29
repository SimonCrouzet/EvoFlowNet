"""The GB1 four-site fitness landscape.

Wu, Dai, Olson, Lloyd-Smith and Sun, *Adaptation in protein fitness landscapes is
facilitated by indirect paths* (eLife, 2016) measured the binding fitness of
variants at four epistatically coupled positions -- V39, D40, G41 and V54 -- of
the IgG-binding domain of protein G. With 20 amino acids at each of 4 positions
the space is 160,000 sequences, of which **149,361 were measured**.

That near-completeness is why this landscape is here. Almost every empirical
fitness benchmark is a sparse sample, so "best found" is the only reportable
quantity. Here the space is small enough to enumerate and dense enough to
lookup, which makes three otherwise-impossible measurements available: exact
regret against the best known variant, how many distinct optima a sampler
recovers, and the exact target distribution ``p*(x) ∝ R(x)^β`` to compare a
learned policy against.

Shape of the landscape
----------------------

Wild-type is ``VDGV`` with fitness 1.0 by definition. The best measured variant
is ``FWAA`` at **8.76**, and it differs from wild-type at **all four positions** --
no single, double or triple mutant comes close, which is the epistasis this
dataset is known for. Roughly 20% of measured variants are completely dead.

The 6.6% that were never measured
---------------------------------

10,639 combinations are absent from the assay. Absent means *not observed in the
library*, which is not the same as unfit -- a variant can be missing because it
was never sampled. Treating those as zero is the common convention and is the
default here, but it is an imputation, and
[GB1Landscape.is_measured][evogfn.landscapes.gb1.GB1Landscape.is_measured]
exists so that any analysis can exclude them instead.

The same caveat applies to the optimum:
[GB1Landscape.optimum][evogfn.landscapes.gb1.GB1Landscape.optimum] is the best
*measured* fitness. An unmeasured variant could in principle exceed it, so regret
against it is exact only with respect to what was assayed.
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import TYPE_CHECKING

import numpy as np

from evogfn.core.types import Alphabet
from evogfn.data.cache import fetch
from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

#: Redistributed by the FLIP benchmark (Dallago et al., NeurIPS 2021 Datasets
#: and Benchmarks), which is the most stable public location for it.
GB1_URL = "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/four_mutations_full_data.csv.zip"

#: Pinned so that a silently updated remote file fails loudly rather than
#: changing published numbers.
GB1_SHA256 = "85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"

GB1_FILENAME = "gb1_four_mutations_full_data.csv.zip"
_MEMBER = "four_mutations_full_data.csv"

#: Positions mutated in the assay, in the numbering of the original paper.
GB1_POSITIONS = (39, 40, 41, 54)

#: The wild-type combination at those positions.
GB1_WILD_TYPE = "VDGV"

#: Number of variants actually assayed, of the 160,000 possible.
GB1_N_MEASURED = 149_361


class GB1Landscape(FitnessLandscape):
    """Empirical fitness of GB1 variants at four coupled positions.

    Scoring is a table lookup, so evaluation is effectively free and the whole
    space can be enumerated.

    Args:
        unmeasured_value: Fitness returned for the 10,639 combinations absent
            from the assay. Defaults to ``0.0``, the usual convention. Pass
            ``float("nan")`` to make their absence propagate visibly instead of
            being silently treated as dead.
        force_download: Re-download the dataset even if a valid cached copy
            exists.

    Raises:
        ChecksumMismatchError: If the downloaded data does not match its pinned
            checksum.
    """

    def __init__(self, *, unmeasured_value: float = 0.0, force_download: bool = False) -> None:
        """Load the landscape into a dense lookup table."""
        self._alphabet = Alphabet.protein()
        self._unmeasured_value = unmeasured_value

        path = fetch(GB1_URL, sha256=GB1_SHA256, filename=GB1_FILENAME, force=force_download)
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
        return len(GB1_POSITIONS)

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Name of the single objective."""
        return ("binding_fitness",)

    @property
    def optimum(self) -> Fitness:
        """Best *measured* fitness.

        Exact with respect to the assay, not to the full 160,000-sequence space:
        an unmeasured combination could in principle score higher.
        """
        return np.array([self._best_measured], dtype=np.float64)

    @property
    def optimal_variant(self) -> str:
        """The best measured combination, ``FWAA``."""
        return self._best_variant

    @property
    def wild_type(self) -> Tokens:
        """The wild-type combination ``VDGV``, whose fitness is 1.0 by definition."""
        return self._alphabet.encode(GB1_WILD_TYPE)

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
    """Extract the variant column and fitness column from the zipped CSV.

    Args:
        path: Path to the downloaded archive.

    Returns:
        The four-letter combinations and their fitness values.

    Raises:
        ValueError: If the archive does not contain the expected number of rows,
            which would mean the file is not the dataset this class expects.
    """
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(_MEMBER).decode()

    variants: list[str] = []
    values: list[float] = []
    for row in csv.DictReader(io.StringIO(raw)):
        variants.append(row["Variants"])
        values.append(float(row["Fitness"]))

    if len(variants) != GB1_N_MEASURED:
        raise ValueError(
            f"expected {GB1_N_MEASURED} measured variants, found {len(variants)}; "
            f"the dataset is not the one this landscape was built against"
        )
    return variants, np.asarray(values, dtype=np.float64)
