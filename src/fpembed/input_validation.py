"""Public-boundary validation for standalone compression input.

This leaf module implements the shared guard used by ``compress_fingerprint``:
it rejects non-real dtypes before any kernel runs and rejects non-finite
floating values, scanning in bounded slices so a large batch never
materialises a full-sized boolean mask.

Public API
----------
validate_real_finite_input : Reject non-real dtypes and non-finite floats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# NumPy dtype kinds that are real numbers: boolean, signed integer, unsigned
# integer, floating. Everything else (complex, object, string, datetime) is
# rejected rather than silently coerced.
_REAL_NUMERIC_KINDS = frozenset("biuf")

# Bounded scan sizes. The finite check is the only O(size) pass that could
# allocate a mask as large as the input, so it walks slices instead.
_FINITE_CHUNK_ELEMENTS = 1 << 20  # 1,048,576 elements per 1-D slice
_FINITE_CHUNK_ROWS = 1 << 12  # 4,096 rows per 2-D slice

_NONFINITE_MSG = (
    "Input contains non-finite values (NaN or infinity); compression "
    "requires finite real numeric input."
)
_NONREAL_MSG = (
    "Input dtype '{dtype}' is not a real numeric type; boolean, integer, "
    "or floating input is required."
)


def validate_real_finite_input(vector: npt.NDArray[Any]) -> None:
    """Reject non-real dtypes and non-finite floating values.

    Integers and booleans are finite by construction and only receive the
    dtype check. Floating input is walked in bounded slices.

    Raises
    ------
    TypeError
        If the dtype is complex, object, string, or any other non-real
        numeric kind.
    ValueError
        If floating input contains NaN, positive infinity, or negative
        infinity.
    """
    kind = vector.dtype.kind
    if kind not in _REAL_NUMERIC_KINDS:
        raise TypeError(_NONREAL_MSG.format(dtype=vector.dtype))
    if kind == "f":
        _require_finite(vector)


def _require_finite(vector: npt.NDArray[Any]) -> None:
    """Raise ``ValueError`` on the first non-finite value, in bounded slices."""
    if vector.ndim == 1:
        for start in range(0, vector.shape[0], _FINITE_CHUNK_ELEMENTS):
            chunk = vector[start:start + _FINITE_CHUNK_ELEMENTS]
            if not np.isfinite(chunk).all():
                raise ValueError(_NONFINITE_MSG)
        return

    for start in range(0, vector.shape[0], _FINITE_CHUNK_ROWS):
        block = vector[start:start + _FINITE_CHUNK_ROWS]
        if not np.isfinite(block).all():
            raise ValueError(_NONFINITE_MSG)
