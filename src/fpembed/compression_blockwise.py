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

    Notes
    -----
    Geometric weights are precision-safe only within the compression factors
    exposed by the public API, where a cap enforced at the public entry points
    (via ``_check_geometric_cap``) keeps every geometric weight representable in
    float64. This helper itself does not raise above that range: it stays
    callable at large block sizes because the non-geometric callers depend on
    it. At large block sizes, however, the smallest geometric weights halve past
    the float64 subnormal floor and underflow to zero — the returned vector
    still sums to one, but its low-order entries are no longer distinguishable
    from zero. Linear, log, and uniform schemes remain fully representable at
    all supported block sizes.
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


def compress_blockwise_exact_uint16(
    vector: npt.NDArray[Any],
    size: int,
    scheme: str = "geometric",
    interleave: bool = False,
) -> npt.NDArray[np.uint16]:
    """Compress a binary fingerprint to exact integer block codes.

    Computes the raw block integer without the float64 einsum's rounding
    residual. For ``geometric`` the value is ``n = sum(b_i * 2^i)`` with the
    block's first bit least significant; for ``uniform`` it is the block
    popcount. These are the un-rescaled integers, related to the float64
    output by ``n / (2**size - 1)`` and ``popcount / size`` respectively.

    Partitioning reuses the identical reshape/transpose logic as
    :func:`compress_blockwise`, so contiguous and interleaved stay consistent
    by construction.

    Parameters
    ----------
    vector : ndarray
        Binary fingerprint — ``(L,)`` for single or ``(N, L)`` for batch.
    size : int
        Block size (compression factor).
    scheme : str
        ``"geometric"`` (block integer) or ``"uniform"`` (popcount).
    interleave : bool
        If True, use strided (interleaved) partitioning; if False, contiguous.

    Returns
    -------
    ndarray of uint16
        Exact integer codes — ``(1, L//size)`` or ``(N, L//size)``.

    Raises
    ------
    ValueError
        If *scheme* has no exact integer form, or the computed integer exceeds
        the ``uint16`` range.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    rows, cols = vector.shape
    n_blocks = cols // size

    if interleave:
        blocks = vector.reshape(rows, size, n_blocks).transpose(0, 2, 1)
    else:
        blocks = vector.reshape(rows, n_blocks, size)

    # uint32 accumulator chosen explicitly: under NEP 50 integer results are no
    # longer value-inspected to widen automatically.
    if scheme == "geometric":
        place = (2 ** np.arange(size)).astype(np.uint32)
        codes = blocks.astype(np.uint32) @ place
    elif scheme == "uniform":
        codes = blocks.sum(axis=-1, dtype=np.uint32)
    else:
        raise ValueError(
            f"Exact uint16 path supports only 'geometric' and 'uniform', "
            f"got '{scheme}'."
        )

    if codes.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Block code exceeds uint16 range for size={size}, "
            f"scheme='{scheme}'."
        )

    return codes.astype(np.uint16)
