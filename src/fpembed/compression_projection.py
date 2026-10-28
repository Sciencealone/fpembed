"""Global projection compression methods: Hadamard and random projection.

Provides the Fast Walsh-Hadamard Transform (FWHT), a seeded and fixed-prefix
truncated Hadamard projection, and seeded random projection (dense Gaussian and
sparse Achlioptas variants). The Hadamard path applies seeded ±1 sign flips and
keeps the first D outputs in Sylvester order. The retained coordinates are fixed
rather than randomly chosen, so it is not conventional randomly subsampled SRHT,
which additionally draws a random subset of coordinates.

Public API
----------
fwht                        : In-place Fast Walsh-Hadamard Transform,
                              re-exported from :mod:`fpembed.compression_fwht`.
build_srht_signs            : Generate the seeded ±1 sign vector for the Hadamard path.
compress_hadamard           : Compress via sign flips → FWHT → fixed-prefix truncate.
build_rp_matrix             : Build a random projection matrix (dense or sparse).
compress_random_projection  : Compress via matrix multiplication or sparse gather.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import numpy.typing as npt

from fpembed.compression_fwht import _fwht_inplace
from fpembed.compression_projection_cache import get_rp_matrix_t
from fpembed.docs_links import with_docs
from fpembed.dtype_support import is_binary

# Maximum fraction of non-zero elements for which the sparse gather beats the
# dense matmul. The gather touches one matrix row per non-zero, so its cost
# scales with density: measured ECFP density is 2.57% at L=2048 and 0.33% at
# L=16384, both well under this threshold. Above it the gather degenerates
# toward and then past the dense product, so the dense path is used instead.
GATHER_MAX_DENSITY = 0.10


def _is_power_of_two(n: int) -> bool:
    """Return True if *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


@functools.lru_cache(maxsize=32)
def build_srht_signs(
    length: int, seed: int
) -> npt.NDArray[np.float64]:
    """Generate the seeded ±1 sign vector for the Hadamard projection.

    Memoised on ``(length, seed)``: repeated compression calls with identical
    parameters reuse the same array instead of redrawing it. Determinism per
    seed is preserved — the same seed always yields the same values.

    The returned array is **immutable — enforced** via ``write=False`` (no
    longer merely a contract). Because the cache returns the
    *same* object to every caller, an in-place write would corrupt every
    subsequent and concurrent call, so the flag is cleared before the array is
    shared. `compress_hadamard` only reads the signs (its sign-flip fold builds
    a fresh array), so callers must copy first if an in-place transform is
    needed.

    Parameters
    ----------
    length : int
        Fingerprint length L.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ndarray of float64
        1-D array of ±1 values, length L. Immutable (read-only, enforced).
    """
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=length).astype(np.float64)
    signs.setflags(write=False)
    return signs


def compress_hadamard(
    vector: npt.NDArray[Any],
    size: int,
    signs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compress using a seeded, fixed-prefix truncated Hadamard projection.

    Produces the first ``D = L // size`` Walsh-Hadamard outputs directly,
    without computing the discarded ``L - D``. The Sylvester recursion
    ``H_2n = [[H_n, H_n], [H_n, -H_n]]`` makes the first ``n`` outputs equal
    to ``H_n · (x[0:n] + x[n:2n])``, so folding the (sign-flipped) length-L
    vector ``log2(size)`` times down to width ``D`` and transforming that
    yields the same leading ``D`` values as transforming the full length and
    truncating.

    The retained coordinates are the first ``D`` in Sylvester order and the
    sign flips are drawn from ``seed``; no coordinate is randomly subsampled.
    This differs from conventional SRHT, which additionally selects a random
    subset of the coordinates. With the retained ``1/√L`` normalization,
    averaging over the random signs gives an expected squared norm of ``D / L``
    times the input squared norm, so the transform scales distances rather than
    preserving them exactly. Bit-identity with the full transform is a
    correctness property of this transform, not evidence of neighborhood
    preservation.

    The sign flips are fused into the first fold and accumulated in place. The
    two half-length input slices are cast to float64 independently as they are
    combined, so the first materialised float array is ``(N, L/2)`` and at most
    two ``(N, L/2)`` temporaries are ever live — no ``(N, L)`` float array is
    materialised on the pruned path. The independent half-width construction
    does not rely on ``astype(copy=False)`` avoiding a copy (which only returns
    the same object when dtype/order already match). Normalization stays
    ``1/√L`` (the transform length was L before pruning), not ``1/√D``.

    The ``size == 1`` (no-prune) branch is the one exception: it genuinely needs
    the full length and keeps a single explicit full-length float array. It is
    unreachable through the generator (which enforces ``compression <= L/2``),
    and reachable only by a direct ``compress_hadamard`` /
    ``compress_fingerprint`` call with ``size == 1``.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` or ``(N, L)``.
    size : int
        Compression factor. L and D = L // size are powers of two.
    signs : ndarray
        Precomputed ±1 sign vector of length L.

    Returns
    -------
    ndarray of float64
        Compressed embedding — ``(1, D)`` or ``(N, D)``.

    Raises
    ------
    ValueError
        If *size* is not a positive divisor of the feature length, or if the
        feature length or ``D = L // size`` is not a power of two.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    cols = vector.shape[1]

    if size < 1:
        raise ValueError(
            with_docs(
                f"Compression factor must be a positive integer, got {size}.",
                "compression-argument-value",
            )
        )
    if cols % size != 0:
        raise ValueError(
            with_docs(
                f"Feature length ({cols}) is not evenly divisible by size ({size}).",
                "compression-compression-divisible",
            )
        )
    output_dim = cols // size
    if not _is_power_of_two(cols) or not _is_power_of_two(output_dim):
        raise ValueError(
            with_docs(
                f"hadamard requires L ({cols}) and D = L // size ({output_dim}) to "
                f"be powers of two.",
                "compression-hadamard-dimensions",
            )
        )
    norm = 1.0 / np.sqrt(cols)

    if output_dim < cols:
        # First fold, fused with the sign flips and accumulated in place. Each
        # half slice is cast to float64 independently, so the first float array
        # is (N, L/2) and no full-width (N, L) temporary is built. Written as
        # two statements; a single `a * b + c * d` expression would instead hold
        # three (N, L/2) arrays live at once.
        half = cols // 2
        folded = vector[:, :half].astype(np.float64) * signs[:half]
        folded += vector[:, half:].astype(np.float64) * signs[half:]

        # Remaining folds: add the two halves until the width reaches D.
        while folded.shape[1] > output_dim:
            mid = folded.shape[1] // 2
            folded = folded[:, :mid] + folded[:, mid:]
    else:
        # size == 1: no pruning possible, so the full length is genuinely
        # needed. This is the one explicit full-length (N, L) float array;
        # unreachable via the generator's compression <= L/2 cap.
        folded = vector.astype(np.float64) * signs

    transformed = _fwht_inplace(np.ascontiguousarray(folded))
    transformed *= norm
    return transformed


def _draw_rp_matrix(
    fp_length: int,
    output_dim: int,
    seed: int,
    sparse: bool = False,
) -> npt.NDArray[np.float64]:
    """Draw a random projection matrix R of shape ``(output_dim, fp_length)``.

    Dense Gaussian: ``R[i,j] ~ N(0, 1/D)``.
    Sparse Achlioptas: ``R[i,j] ∈ {-1, 0, +1}`` with ``P(0) = 2/3``,
    scaled by ``√(3/D)``.

    Pure — no caching. The exact ``RandomState`` sequence is fixed by ``seed``,
    so the same parameters always yield the same values; caching happens one
    layer up on the ``(L, D)`` orientation.

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


def _rp_matrix_T(
    fp_length: int,
    output_dim: int,
    seed: int,
    sparse: bool = False,
) -> npt.NDArray[np.float64]:
    """Cached C-contiguous ``(fp_length, output_dim)`` projection matrix.

    This is the single canonical cached orientation. Both shipped consumers —
    the sparse gather and the dense ``vector @ mt`` product — want ``(L, D)``,
    so it is the form retained. Caching is byte-budget-bounded (see
    :mod:`fpembed.compression_projection_cache`), keyed on
    ``(fp_length, output_dim, seed, sparse)``. The array is read-only; the same
    object is shared across callers.
    """
    return get_rp_matrix_t(
        fp_length, output_dim, seed, sparse, _draw_rp_matrix
    )


def build_rp_matrix(
    fp_length: int,
    output_dim: int,
    seed: int,
    sparse: bool = False,
) -> npt.NDArray[np.float64]:
    """Random projection matrix R of shape ``(output_dim, fp_length)``.

    Served as a non-owning ``.T`` view of the cached C-contiguous ``(L, D)``
    matrix (:func:`_rp_matrix_T`), so no second owning copy exists. The values
    are identical to a direct ``(D, L)`` draw because transposing the cached
    transpose recovers the original orientation.

    The returned view shares the read-only cached buffer, so it is itself
    read-only; callers must copy first if an in-place transform is needed.

    Parameters mirror :func:`_draw_rp_matrix`. Returns the ``(D, L)`` view.
    """
    return _rp_matrix_T(fp_length, output_dim, seed, sparse).T


def compress_random_projection(
    vector: npt.NDArray[Any],
    matrix: npt.NDArray[np.float64] | None = None,
    matrix_t: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compress using a precomputed random projection matrix.

    Computes ``embedding = vector @ matrix.T``. For binary input below
    ``GATHER_MAX_DENSITY`` the equivalent result is produced by gathering and
    summing the matrix rows selected by each vector's non-zero indices, which
    avoids the dense product entirely. The gather equals ``row @ matrix.T``
    only when the non-zeros are exactly 1.0, so non-binary input (count
    fingerprints, precomputed embeddings) falls back to the dense matmul.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` or ``(N, L)``.
    matrix : ndarray, optional
        Projection matrix of shape ``(D, L)``. Required unless *matrix_t* is
        supplied.
    matrix_t : ndarray, optional
        Pre-transposed C-contiguous ``(L, D)`` form of *matrix* (from
        :func:`_rp_matrix_T`). When supplied it takes precedence over *matrix*
        and is reused for both the gather and the dense path; otherwise
        ``matrix.T`` is used, correct but with no memoisation, matching the
        prior behaviour.

    Returns
    -------
    ndarray of float64
        Compressed embedding — ``(1, D)`` or ``(N, D)``.

    Raises
    ------
    ValueError
        If neither a matrix nor its transpose is supplied, or if the matrix
        feature axis does not match the input length.
    """
    single = vector.ndim == 1
    if single:
        vector = vector[np.newaxis, :]

    n_rows, length = vector.shape

    if matrix_t is not None:
        mt = matrix_t
    elif matrix is not None:
        mt = matrix.T
    else:
        raise ValueError("compress_random_projection requires matrix or matrix_t")

    if mt.shape[0] != length:
        raise ValueError(
            with_docs(
                f"Projection matrix feature axis does not match the input length: "
                f"matrix shape {mt.shape}, input shape {vector.shape}.",
                "compression-matrix",
            )
        )

    if n_rows == 0:
        return np.empty((0, mt.shape[1]), dtype=np.float64)
    density = int(np.count_nonzero(vector)) / (n_rows * length)
    if density <= GATHER_MAX_DENSITY and is_binary(vector):
        out = np.empty((n_rows, mt.shape[1]), dtype=np.float64)
        for i in range(n_rows):
            out[i] = mt[np.flatnonzero(vector[i])].sum(axis=0)
        return out

    return vector.astype(np.float64) @ mt
