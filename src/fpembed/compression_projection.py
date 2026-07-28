"""Global projection compression methods: SRHT and random projection.

Provides the Fast Walsh-Hadamard Transform (FWHT), Subsampled Randomized
Hadamard Transform (SRHT) compression, and seeded random projection
(dense Gaussian and sparse Achlioptas variants).

Public API
----------
fwht                        : In-place Fast Walsh-Hadamard Transform.
build_srht_signs            : Generate random ±1 sign vector for SRHT.
compress_hadamard           : Compress via SRHT (sign flips → FWHT → truncate).
build_rp_matrix             : Build a random projection matrix (dense or sparse).
compress_random_projection  : Compress via matrix multiplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _fwht_inplace(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized in-place FWHT butterfly, along the last axis.

    Works for a single 1-D vector or a 2-D batch (rows, length) — each
    butterfly level is one NumPy operation over the whole array instead of
    a Python-level loop over individual index pairs. Mutates *x* in-place
    via reshape views (no copy of the full array), matching the original
    per-pair butterfly algorithm exactly.

    Parameters
    ----------
    x : ndarray
        Float64 array whose last axis has length L (must be a power of 2).

    Returns
    -------
    ndarray
        The same array, modified in-place.
    """
    length = x.shape[-1]
    h = 1
    while h < length:
        view = x.reshape(*x.shape[:-1], -1, 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    return x


def fwht(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """In-place Fast Walsh-Hadamard Transform (butterfly algorithm).

    Operates in O(L log L) using only additions and subtractions. Vectorized
    via NumPy reshape/broadcast (see `_fwht_inplace`) rather than nested
    Python loops — benchmarked at ~32x faster than the original per-pair
    loop implementation for batched Hadamard compression, with identical
    output (see dev_docs/investigation_compression_acceleration_options.md).

    Parameters
    ----------
    x : ndarray
        1-D float64 array of length L (must be a power of 2).

    Returns
    -------
    ndarray
        The same array, modified in-place.
    """
    return _fwht_inplace(x)


def build_srht_signs(
    length: int, seed: int
) -> npt.NDArray[np.float64]:
    """Generate a random ±1 sign vector for SRHT.

    Parameters
    ----------
    length : int
        Fingerprint length L.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ndarray of float64
        1-D array of ±1 values, length L.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=length).astype(np.float64)


def compress_hadamard(
    vector: npt.NDArray[Any],
    size: int,
    signs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compress using Subsampled Randomized Hadamard Transform.

    Applies sign flips, FWHT, normalization by 1/√L, and truncation to
    D = L // size dimensions.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` or ``(N, L)``.
    size : int
        Compression factor.
    signs : ndarray
        Precomputed ±1 sign vector of length L.

    Returns
    -------
    ndarray of float64
        Compressed embedding — ``(1, D)`` or ``(N, D)``.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    rows, cols = vector.shape
    output_dim = cols // size
    norm = 1.0 / np.sqrt(cols)

    signed = vector.astype(np.float64) * signs  # broadcasts (rows, cols) * (cols,)
    transformed = _fwht_inplace(signed)
    transformed *= norm
    return transformed[:, :output_dim].copy()


def build_rp_matrix(
    fp_length: int,
    output_dim: int,
    seed: int,
    sparse: bool = False,
) -> npt.NDArray[np.float64]:
    """Build a random projection matrix R of shape ``(output_dim, fp_length)``.

    Dense Gaussian: ``R[i,j] ~ N(0, 1/D)``.
    Sparse Achlioptas: ``R[i,j] ∈ {-1, 0, +1}`` with ``P(0) = 2/3``,
    scaled by ``√(3/D)``.

    Parameters
    ----------
    fp_length : int
        Input dimension L.
    output_dim : int
        Output dimension D.
    seed : int
        Random seed for reproducibility.
    sparse : bool
        If True, use Achlioptas variant.

    Returns
    -------
    ndarray of float64
        2-D array of shape ``(D, L)``.
    """
    rng = np.random.RandomState(seed)

    if sparse:
        # Achlioptas: P(-1) = 1/6, P(0) = 2/3, P(+1) = 1/6
        raw = rng.choice([-1, 0, 0, 0, 0, 1], size=(output_dim, fp_length))
        scale = np.sqrt(3.0 / output_dim)
        return raw.astype(np.float64) * scale
    else:
        # Dense Gaussian: N(0, 1/D)
        std = 1.0 / np.sqrt(output_dim)
        return rng.randn(output_dim, fp_length).astype(np.float64) * std


def compress_random_projection(
    vector: npt.NDArray[Any],
    matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compress using a precomputed random projection matrix.

    Computes ``embedding = vector @ matrix.T``.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` or ``(N, L)``.
    matrix : ndarray
        Projection matrix of shape ``(D, L)``.

    Returns
    -------
    ndarray of float64
        Compressed embedding — ``(1, D)`` or ``(N, D)``.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    embedded = vector.astype(np.float64) @ matrix.T
    return embedded
