"""Fast Walsh-Hadamard Transform used by the Hadamard compression path.

The transform is in-place: it mutates the caller's C-contiguous float64
buffer through reshape views and returns the same array.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _fwht_inplace(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized in-place FWHT butterfly, along the last axis.

    Works for a single 1-D vector or a 2-D batch (rows, length) — each
    butterfly level is one NumPy operation over the whole array instead of
    a Python-level loop over individual index pairs. Mutates *x* in-place
    via reshape views (no copy of the full array), matching the original
    per-pair butterfly algorithm exactly. Only the first operand is copied
    per level; the second write reuses the second-half view as its output.

    *x* must be C-contiguous. The reshape views only alias the caller's
    buffer under that condition; a non-contiguous input would force a copy,
    so the in-place contract could not be honored and ``ValueError`` is
    raised instead.

    Parameters
    ----------
    x : ndarray
        C-contiguous float64 array whose last axis has length L (must be a
        power of 2).

    Returns
    -------
    ndarray
        The same array, modified in-place.

    Raises
    ------
    ValueError
        If *x* is not C-contiguous.
    """
    if not x.flags.c_contiguous:
        raise ValueError(
            "fwht requires a C-contiguous array: the in-place contract cannot "
            "be honored through a reshape copy."
        )
    length = x.shape[-1]
    h = 1
    while h < length:
        view = x.reshape(*x.shape[:-1], -1, 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :]
        # First half is disjoint memory from b, so writing it does not disturb
        # the view b still read by the subtract below.
        np.add(a, b, out=view[..., 0, :])
        np.subtract(a, b, out=b)
        h *= 2
    return x


def fwht(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """In-place Fast Walsh-Hadamard Transform (butterfly algorithm).

    Operates in O(L log L) using only additions and subtractions. Vectorized
    via NumPy reshape/broadcast (see `_fwht_inplace`) rather than nested
    Python loops — benchmarked at ~32x faster than the original per-pair
    loop implementation for batched Hadamard compression, with identical
    output (see dev_docs/investigation_compression_acceleration_options.md).

    *x* must be C-contiguous: the transform mutates the caller's buffer
    through reshape views, so it cannot operate on a non-contiguous array
    without a copy. Pass ``np.ascontiguousarray(x)`` if needed.

    Parameters
    ----------
    x : ndarray
        C-contiguous 1-D float64 array of length L (must be a power of 2).

    Returns
    -------
    ndarray
        The same array, modified in-place.

    Raises
    ------
    ValueError
        If *x* is not C-contiguous.
    """
    return _fwht_inplace(x)
