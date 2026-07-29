"""Reading a batch of objective values the way a classical baseline needs them.

Every baseline here ranks designs by one number: annealing accepts on a scalar
delta, CMA-ES sorts on a scalar, a GA sorts its population, hill climbing takes a
maximum. The landscape contract, however, is that ``evaluate`` returns
``(n, n_objectives)``.

Flattening that array with ``reshape(-1)`` is correct for one objective and
silently wrong for more: it yields ``n * n_objectives`` numbers, which then line
up against ``n`` sequences by position and pair every design with somebody else's
score. Nothing raises -- ``zip`` stops at the shorter side and the run completes
with plausible, wrong numbers. This module refuses instead, matching
:func:`evoflownet.metrics.performance._as_flat` and
:func:`evoflownet.rewards.base._single_objective`, because the caller genuinely
has a decision to make: which scalarisation the ranking should use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from evoflownet.core.types import Fitness


def single_objective(values: Fitness) -> npt.NDArray[np.float64]:
    """Accept ``(n,)`` or single-objective ``(n, 1)`` and return ``(n,)``.

    Args:
        values: Objective values for a scored batch.

    Returns:
        An ``(n,)`` array aligned with the sequences that were scored.

    Raises:
        ValueError: If the input carries more than one objective, where the
            ranking these baselines depend on is not defined.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] == 1:  # noqa: PLR2004 - a single objective
        return array[:, 0]
    raise ValueError(
        f"expected shape (n,) or (n, 1), got {array.shape}; a classical baseline ranks "
        f"designs by a single number, so multi-objective values must be scalarised "
        f"before they reach observe"
    )
