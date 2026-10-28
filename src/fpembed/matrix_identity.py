"""Projection-matrix validation and identity digests.

A caller-supplied projection matrix is accepted as a canonical ``(D, L)``
array. fpembed never fits the matrix; it only validates it, identifies it by a
digest, and caches its ``(L, D)`` transpose. PCA, whitening, and any other
linear map are fitted outside fpembed, and the resulting matrix is passed in.

Public API
----------
array_id                  : 16-char digest of an array shape and float64 bytes.
matrix_id                 : Alias of ``array_id`` for projection matrices.
validate_matrix_array     : Reject non-2-D, non-real, and non-finite arrays.
validate_projection_matrix: Validate the canonical ``(D, L)`` shape.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt

from fpembed.docs_links import with_docs

# Real numeric kinds: boolean, signed integer, unsigned integer, floating.
_REAL_KINDS = frozenset("biuf")


def array_id(values: npt.ArrayLike) -> str:
    """Return a stable 16-character digest of an array's shape and values.

    The array is converted to C-contiguous ``float64`` before hashing, so the
    same values supplied as ``float32`` or ``float64`` digest identically. The
    digest covers both the shape and the canonical bytes, so a reshape or a
    value change produces a different result.
    """
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def matrix_id(matrix: npt.ArrayLike) -> str:
    """Return the identity digest for a projection matrix."""
    return array_id(matrix)


def validate_matrix_array(matrix: Any) -> npt.NDArray[Any]:
    """Validate that *matrix* is a 2-D real finite array.

    Raises
    ------
    ValueError
        If the array is not 2-D or contains NaN or infinity.
    TypeError
        If the dtype is not a real numeric kind.
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(
            with_docs(
                f"Projection matrix must be 2-D; got shape {array.shape}.",
                "generator-matrix",
            )
        )
    if array.dtype.kind not in _REAL_KINDS:
        raise TypeError(
            with_docs(
                f"Projection matrix dtype '{array.dtype}' is not a real numeric "
                "type; boolean, integer, or floating input is required.",
                "generator-matrix",
            )
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(
            with_docs("Projection matrix must contain only finite values.", "generator-matrix")
        )
    return array


def validate_projection_matrix(
    matrix: Any, *, fp_size: int, output_dim: int,
) -> npt.NDArray[np.float64]:
    """Validate a canonical ``(D, L)`` projection matrix.

    Parameters
    ----------
    matrix : array_like
        Candidate matrix with output dimension ``D`` on the first axis and
        fingerprint length ``L`` on the second.
    fp_size : int
        Expected feature axis ``L``.
    output_dim : int
        Expected first axis ``D``.

    Returns
    -------
    ndarray of float64
        The validated matrix as a C-contiguous ``float64`` array.
    """
    array = validate_matrix_array(matrix)
    expected = (output_dim, fp_size)
    if array.shape != expected:
        raise ValueError(
            with_docs(
                f"Projection matrix must have shape (D, L) = {expected}; "
                f"got {array.shape}.",
                "compression-matrix",
            )
        )
    return np.ascontiguousarray(array, dtype=np.float64)
