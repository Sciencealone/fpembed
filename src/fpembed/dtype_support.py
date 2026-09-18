"""Output-dtype normalisation, validation, and casting helpers.

Leaf module importing only NumPy. ``compression.py`` already imports
``compression_projection.py``, so these helpers must live in a module that
depends on nothing internal to avoid a circular import; every compression
layer and the generator consume them.

Responsibilities
----------------
- Normalise a user-supplied dtype to a canonical ``np.dtype`` from the
  accepted set, turning NumPy's ``TypeError`` into a ``ValueError``.
- Enforce the input-independent (stage 1) rows of the dtype/method
  decision table, including the two distinct ``uint16`` range limits.
- Decide whether an array is binary (the ``uint16`` precondition).
- Cast a computed array to the requested output dtype.

Public API
----------
normalize_dtype          : Canonicalise a dtype from the accepted set.
validate_dtype_for_method: Stage-1 method/dtype/compression compatibility.
is_binary                : Whether every element is 0 or 1.
cast_output              : Cast an array to a requested dtype.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Accepted output dtypes. float64 is the default and preserves current
# behaviour; float32 halves stored size for every method; uint16 carries
# the exact integer code for the block-wise geometric/uniform schemes.
_ALLOWED = (np.float64, np.float32, np.uint16)
_ACCEPTED_NAMES = "float64, float32, uint16"

# Methods that have no exact integer representation, so uint16 is rejected.
_NON_INTEGER_METHODS = frozenset({"linear", "log", "hadamard", "random_projection"})

# Per-scheme upper bound on the block size for which the uint16 code fits.
# geometric stores the C-bit block integer n in [0, 2^C - 1]; uniform stores
# a popcount in [0, C]. Beyond these the value overflows uint16.
_UINT16_MAX_COMPRESSION = {"geometric": 16, "uniform": 65535}


def normalize_dtype(dtype: Any) -> np.dtype:
    """Canonicalise *dtype* to an ``np.dtype`` from the accepted set.

    Accepts strings (``"float32"``), NumPy scalar types (``np.float32``),
    and ``np.dtype`` instances, all normalised through ``np.dtype(...)``.

    Raises
    ------
    ValueError
        If *dtype* is unrecognised by NumPy (``np.dtype`` raises
        ``TypeError``, re-raised here as ``ValueError``) or is a valid
        NumPy dtype outside the accepted set (any lossy type such as
        ``int8``, ``uint8``, or ``float16``).
    """
    try:
        dt = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(
            f"Unsupported dtype {dtype!r}. Accepted dtypes: {_ACCEPTED_NAMES}."
        ) from exc

    if dt.type not in _ALLOWED:
        raise ValueError(
            f"Unsupported dtype '{dt}'. Accepted dtypes: {_ACCEPTED_NAMES}."
        )
    return dt


def validate_dtype_for_method(
    dt: np.dtype, method: str, compression: int | None
) -> None:
    """Check input-independent method/dtype/compression compatibility.

    This is stage 1 of validation: everything that can be decided without
    seeing an input array. float64 and float32 are permitted for every
    method. uint16 is permitted only for the block-wise integer schemes,
    and only while the code fits uint16 range.

    When compression is inactive, no compression method runs and the raw
    binary fingerprint is losslessly representable in every accepted dtype,
    so the method-based uint16 rejection is skipped entirely.

    Parameters
    ----------
    dt : np.dtype
        Already normalised via :func:`normalize_dtype`.
    method : str
        Compression method name.
    compression : int or None
        Compression factor, or ``None``/``0`` when compression is inactive.

    Raises
    ------
    ValueError
        If uint16 is requested for an active-compression method with no exact
        integer form, or for geometric/uniform with a compression exceeding
        its uint16 range. The range message cites integer range and is
        deliberately distinct from the geometric mantissa-limit cap message.
    """
    if dt.type is not np.uint16:
        return

    # Inactive compression stores raw binary directly, which fits every
    # accepted dtype; no method runs, so no method-based rejection applies.
    if not compression:
        return

    if method in _NON_INTEGER_METHODS:
        raise ValueError(
            f"dtype='uint16' has no exact integer representation for method "
            f"'{method}'. Use 'geometric' or 'uniform', or dtype float32/float64."
        )

    limit = _UINT16_MAX_COMPRESSION[method]
    if compression > limit:
        raise ValueError(
            f"dtype='uint16' with method '{method}' requires compression "
            f"<= {limit}; the integer code would exceed uint16 range at "
            f"compression={compression}."
        )


def is_binary(arr: npt.NDArray[Any]) -> bool:
    """Return True if every element of *arr* is 0 or 1.

    Uses a dtype-aware fast path for boolean and unsigned-integer arrays,
    where the only way to fail is a value above 1, so checking the maximum
    suffices. All other dtypes fall back to an explicit membership test.
    Both branches are a single ``O(size)`` pass, not sparsity-scaled.
    """
    if arr.dtype.kind in "bu":
        return arr.size == 0 or int(arr.max()) <= 1
    return bool(((arr == 0) | (arr == 1)).all())


def cast_output(arr: npt.NDArray[Any], dt: np.dtype) -> npt.NDArray[Any]:
    """Cast *arr* to dtype *dt*, avoiding a copy when it already matches."""
    return arr.astype(dt, copy=False)
