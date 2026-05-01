"""Block-wise fingerprint compression with pluggable weight schemes.

Provides weight vector construction for geometric, linear, logarithmic,
and uniform schemes, plus block partitioning (contiguous and interleaved)
with weighted dot-product reduction.

Public API
----------
build_block_weights : Build a normalized weight vector for a given scheme.
compress_blockwise  : Compress a fingerprint using block-wise weighted dot product.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_SUPPORTED_SCHEMES = ("geometric", "linear", "log", "uniform")


def build_block_weights(
    block_size: int, scheme: str = "geometric"
) -> npt.NDArray[np.float64]:
    """Return a normalized weight vector of length *block_size*.

    Parameters
    ----------
    block_size : int
        Number of bits per block (must be >= 1).
    scheme : str
        One of ``"geometric"``, ``"linear"``, ``"log"``, ``"uniform"``.

    Returns
    -------
    ndarray of float64
        1-D array of length *block_size*, summing to 1.0.

    Raises
    ------
    ValueError
        If *scheme* is not recognized.
    """
    if scheme == "geometric":
        # Log-space construction for numerical stability (moved from compression.py)
        log_weights = np.arange(block_size, dtype=np.float64) - block_size
        weights = np.exp2(log_weights)
    elif scheme == "linear":
        weights = np.arange(1, block_size + 1, dtype=np.float64)
    elif scheme == "log":
        weights = np.log2(np.arange(2, block_size + 2, dtype=np.float64))
    elif scheme == "uniform":
        return np.full(block_size, 1.0 / block_size, dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported weight scheme '{scheme}'. "
            f"Supported: {', '.join(_SUPPORTED_SCHEMES)}"
        )
    weights /= weights.sum()
    return weights


def compress_blockwise(
    vector: npt.NDArray[Any],
    size: int,
    scheme: str = "geometric",
    interleave: bool = False,
) -> npt.NDArray[np.float64]:
    """Compress a binary fingerprint using block-wise weighted dot product.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` for single or ``(N, L)`` for batch.
    size : int
        Block size (compression factor).
    scheme : str
        Weight scheme name.
    interleave : bool
        If True, use strided (interleaved) partitioning; if False, contiguous.

    Returns
    -------
    ndarray of float64
        Compressed embedding — ``(1, L//size)`` or ``(N, L//size)``.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    rows, cols = vector.shape
    n_blocks = cols // size

    weights = build_block_weights(size, scheme)

    if interleave:
        # Strided: bit[i] → block[i % n_blocks]
        blocks = vector.reshape(rows, size, n_blocks).transpose(0, 2, 1)
    else:
        # Contiguous: bit[i] → block[i // size]
        blocks = vector.reshape(rows, n_blocks, size)

    embedded: npt.NDArray[np.float64] = np.einsum("ijk,k->ij", blocks, weights)
    return embedded
