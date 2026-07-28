"""Turning a preference vector into something a policy can be conditioned on.

MOGFN-PC (Jain et al., 2023, *Multi-Objective GFlowNets*, ICML) trains a single
model over ``p(x|ω) ∝ R(x|ω)^β``: the preference ``ω`` is resampled from a
Dirichlet every batch, fed to the policy as an extra input, and the reward is
scalarised with the same ``ω``. One trained model then covers the whole Pareto
front, and a new trade-off at inference time costs a forward pass rather than a
retraining run. That is the entire reason a preference has to reach the network
at all.

Why not feed ``ω`` in raw
-------------------------

A preference is a handful of numbers in ``[0, 1]``, entering a network whose
other input is a sequence embedding of a few hundred dimensions. Concatenated
raw, it is a whisper next to a shout: the gradient signal for "which trade-off am
I being asked for" is spread over two or three weights, and the network can fit
the training reward while barely conditioning on ``ω`` at all. The failure is
quiet -- the loss looks fine, and it only shows up as a Pareto front that
collapses to a single point.

Thermometer encoding (Buckman et al., 2018) is the fix used by MOGFN-PC: each
scalar becomes ``n_bins`` monotonically-filling coordinates, so the conditioning
signal is comparable in width to the state representation and nearby preferences
share most of their representation. It is not one-hot: adjacent values overlap,
so the encoding stays a smooth function of the preference and the policy can
interpolate between trade-offs it never saw.

This module is deliberately standalone
--------------------------------------

The encoding is array arithmetic with no parameters, so it is testable on its
own and independent of any particular policy architecture.
:class:`~evoflownet.models.policy.SequencePolicy` is left untouched: a
preference-conditioned policy concatenates ``preference_encoding(...)`` to its
trunk input, and everything that is *hard* about doing so -- widening the trunk,
resampling ``ω`` per batch -- belongs to that policy, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

#: Fewest bins that encode anything. With one bin the encoding is a constant and
#: the conditioning signal vanishes, which is the failure this module exists to
#: prevent rather than one it should be able to reproduce.
MIN_BINS = 2

# A batch of preferences is two-dimensional; a single preference is not.
_MATRIX_NDIM = 2


def sample_preferences(
    n_objectives: int,
    n_samples: int = 1,
    *,
    alpha: npt.ArrayLike = 1.0,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Draw preference vectors from a Dirichlet distribution over the simplex.

    MOGFN-PC samples ``ω ~ Dir(alpha)`` afresh for every training batch, which is
    what spreads the model's capacity over the whole front instead of one
    trade-off. ``alpha`` controls where that capacity goes:

    * ``alpha = 1`` -- uniform over the simplex, the default and the neutral choice;
    * ``alpha < 1`` -- mass near the corners, so most batches ask for a near
      single-objective specialist and the extremes of the front are well covered;
    * ``alpha > 1`` -- mass near the centre, so most batches ask for a balanced
      design and the extremes are visited rarely.

    Args:
        n_objectives: Length of each preference vector. Must be at least 1.
        n_samples: How many vectors to draw. Must be at least 1.
        alpha: Dirichlet concentration, either one value shared by all objectives
            or one per objective. Every entry must be positive.
        seed: Seed for the draw. ``None`` draws from fresh entropy, which is
            correct for training and wrong for anything that must be reproduced.

    Returns:
        An ``(n_samples, n_objectives)`` array whose rows are non-negative and
        sum to one.

    Raises:
        ValueError: If a size is not positive, or if ``alpha`` is not positive or
            does not have one entry per objective.
    """
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be at least 1, got {n_objectives}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")

    concentration = np.asarray(alpha, dtype=np.float64)
    if concentration.ndim == 0:
        concentration = np.full(n_objectives, float(concentration), dtype=np.float64)
    if concentration.ndim != 1 or concentration.shape[0] != n_objectives:
        raise ValueError(
            f"alpha must be a scalar or have {n_objectives} entries, got shape "
            f"{concentration.shape}"
        )
    if not np.isfinite(concentration).all() or (concentration <= 0.0).any():
        raise ValueError(
            f"alpha must be finite and positive, got {concentration}; a non-positive "
            f"concentration makes the Dirichlet undefined"
        )

    rng = np.random.default_rng(seed)
    return np.asarray(rng.dirichlet(concentration, size=n_samples), dtype=np.float64)


def preference_encoding(preference: npt.ArrayLike, *, n_bins: int = 16) -> npt.NDArray[np.float64]:
    """Encode a preference vector as the conditioning input of a policy.

    Thermometer-encodes each weight over ``[0, 1]``, the range a simplex weight
    lives in, and flattens the result so it can be concatenated to a state
    representation. The layout is objective-major: the first ``n_bins`` entries
    encode ``ω_0``, the next ``n_bins`` encode ``ω_1``, and so on. Use
    :func:`encoding_dim` to size the network input rather than recomputing it at
    the call site, so the two cannot drift apart.

    Args:
        preference: An ``(n_objectives,)`` preference vector, or an
            ``(n, n_objectives)`` batch of them. Must be non-negative and sum to
            one along the last axis.
        n_bins: Bins per objective. Must be at least :data:`MIN_BINS`.

    Returns:
        An ``(n_objectives * n_bins,)`` array for a single preference, or an
        ``(n, n_objectives * n_bins)`` array for a batch.

    Raises:
        ValueError: If the preference is not a vector or batch of vectors on the
            simplex, or if ``n_bins`` is below :data:`MIN_BINS`.
    """
    weights = np.asarray(preference, dtype=np.float64)
    if weights.ndim not in (1, _MATRIX_NDIM):
        raise ValueError(
            f"expected a preference of shape (n_objectives,) or (n, n_objectives), "
            f"got ndim {weights.ndim}"
        )
    if weights.shape[-1] == 0:
        raise ValueError("preference must cover at least one objective")
    if not np.isfinite(weights).all():
        raise ValueError("preference must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"preference must be non-negative, got a minimum of {weights.min()}")
    if not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            "preference must sum to 1; an unnormalised preference encodes to a "
            "conditioning vector the policy was never trained on"
        )

    encoded = thermometer_encode(weights, n_bins=n_bins, vmin=0.0, vmax=1.0)
    return encoded.reshape(*weights.shape[:-1], weights.shape[-1] * n_bins)


def thermometer_encode(
    values: npt.ArrayLike,
    *,
    n_bins: int = 16,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Spread scalars over monotonically-filling bins.

    Buckman et al. (2018), in the continuous form used by the MOGFN reference
    implementation. The range is cut into ``n_bins`` bins of width
    ``g = (vmax - vmin) / n_bins`` with left edges ``b_j``, and entry ``j`` of the
    encoding of ``v`` is::

        clip(v - b_j, 0, g) / g

    so the encoding fills up from the left as ``v`` grows: all zeros at ``vmin``,
    all ones at ``vmax``, and a partially-filled bin in between. Two properties
    matter and both follow from that form. It is *monotone*, so ordering
    information survives; and it is *continuous* in ``v``, unlike a one-hot
    binning, so preferences either side of a bin edge get nearly the same vector
    and the policy can interpolate between trade-offs it was never trained on.

    The bin edges stop one gap short of ``vmax``, which is the one deviation from
    the reference implementation: it spaces ``n_bins`` edges over the closed
    interval, so its last edge sits at ``vmax`` and that coordinate is zero for
    every value in range. One of the bins therefore carries no information at
    all, and with the small ``n_bins`` used for a preference vector that is a
    noticeable fraction of the conditioning signal.

    Values outside ``[vmin, vmax]`` saturate rather than raise: a temperature or
    a weight just past the end of its declared range should be encoded as
    "maximal", not rejected.

    Args:
        values: An array of any shape; encoding is applied elementwise and adds a
            trailing axis of length ``n_bins``.
        n_bins: Number of bins. Must be at least :data:`MIN_BINS`.
        vmin: Value encoded as all zeros.
        vmax: Value encoded as all ones. Must exceed ``vmin``.

    Returns:
        An array shaped ``(*values.shape, n_bins)`` with entries in ``[0, 1]``.

    Raises:
        ValueError: If ``n_bins`` is below :data:`MIN_BINS`, if ``vmax`` does not
            exceed ``vmin``, or if any value is not finite.
    """
    if n_bins < MIN_BINS:
        raise ValueError(
            f"n_bins must be at least {MIN_BINS}, got {n_bins}; a single bin encodes "
            f"every value identically and conditions the policy on nothing"
        )
    if not vmax > vmin:
        raise ValueError(f"vmax must exceed vmin, got vmin={vmin} and vmax={vmax}")

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("values to encode must be finite")

    gap = (vmax - vmin) / n_bins
    edges = vmin + gap * np.arange(n_bins, dtype=np.float64)
    filled = np.clip(array[..., None] - edges, 0.0, gap) / gap
    return np.asarray(filled, dtype=np.float64)


def encoding_dim(n_objectives: int, *, n_bins: int = 16) -> int:
    """Width of the conditioning vector :func:`preference_encoding` produces.

    Args:
        n_objectives: Number of objectives being traded off.
        n_bins: Bins per objective, matching the value passed to the encoder.

    Returns:
        The number of conditioning inputs a policy must accept.

    Raises:
        ValueError: If ``n_objectives`` is not positive or ``n_bins`` is below
            :data:`MIN_BINS`.
    """
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be at least 1, got {n_objectives}")
    if n_bins < MIN_BINS:
        raise ValueError(f"n_bins must be at least {MIN_BINS}, got {n_bins}")
    return n_objectives * n_bins
