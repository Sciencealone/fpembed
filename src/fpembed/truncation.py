"""Prefix truncation of projection embeddings.

A projection embedding computed at a low compression factor can be reused at a
higher factor by keeping a prefix of its coordinates. Both projection methods
in fpembed retain fixed leading coordinates, so this is exact for Hadamard and
exact up to floating-point round-off, after rescaling, for random projection.

Public API
----------
truncate_embedding : Return the first ``target_dim`` coordinates of an embedding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_PREFIX_METHODS = ("hadamard", "random_projection")
_BLOCKWISE_METHODS = frozenset({"geometric", "linear", "log", "uniform"})


def truncate_embedding(
    embedding: npt.ArrayLike,
    target_dim: int,
    *,
    method: str,
    source_dim: int | None = None,
) -> npt.NDArray[Any]:
    """Return a prefix of a projection embedding.

    The helper assumes the prefix comes from the same fingerprint, the same
    method parameters, and the same seed as the source embedding. Under that
    precondition the Hadamard prefix equals the embedding computed directly at
    the higher compression factor, and the scaled random-projection prefix
    equals that direct embedding within floating-point round-off.

    Parameters
    ----------
    embedding : array_like
        Source embedding. The last axis is the feature axis.
    target_dim : int
        Number of leading coordinates to keep. Must be positive and no greater
        than *source_dim*.
    method : {"hadamard", "random_projection"}
        Projection method that produced *embedding*. Block-wise methods are
        rejected because their block boundaries move when the output width
        changes, so their outputs are not prefix-consistent.
    source_dim : int, optional
        Feature dimension of the source embedding. Defaults to
        ``embedding.shape[-1]``.

    Returns
    -------
    ndarray
        The first *target_dim* coordinates, as a copy. A 1-D input yields
        ``(target_dim,)``; a 2-D input yields ``(N, target_dim)``; leading
        dimensions beyond 2 are preserved. Random projection output is scaled
        by ``sqrt(source_dim / target_dim)``.

    Raises
    ------
    TypeError
        If *target_dim* is not an integer or *embedding* has a non-numeric
        dtype.
    ValueError
        If *target_dim* or *source_dim* is invalid, *target_dim* exceeds the
        feature axis of *embedding* or *source_dim*, the method is block-wise,
        or the method is unknown.
    """
    array = np.asarray(embedding)
    if array.ndim == 0:
        raise ValueError("embedding must have at least one dimension")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"embedding must have a numeric dtype, got {array.dtype}"
        )

    if isinstance(target_dim, bool) or not isinstance(
        target_dim, (int, np.integer)
    ):
        raise TypeError(
            f"target_dim must be an integer, got {type(target_dim).__name__}"
        )
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    if source_dim is None:
        source_dim = int(array.shape[-1])
    elif isinstance(source_dim, bool) or not isinstance(
        source_dim, (int, np.integer)
    ):
        raise TypeError(
            f"source_dim must be an integer, got {type(source_dim).__name__}"
        )
    source_dim = int(source_dim)
    if source_dim <= 0:
        raise ValueError(f"source_dim must be positive, got {source_dim}")
    feature_length = int(array.shape[-1])
    if target_dim > feature_length:
        raise ValueError(
            f"target_dim ({target_dim}) exceeds the available feature length "
            f"({feature_length})"
        )
    if target_dim > source_dim:
        raise ValueError(
            f"target_dim ({target_dim}) must not exceed source_dim "
            f"({source_dim})"
        )

    if method in _BLOCKWISE_METHODS:
        raise ValueError(
            f"method '{method}' is block-wise; block-wise outputs are not "
            "prefix-consistent. Use 'hadamard' or 'random_projection'."
        )
    if method not in _PREFIX_METHODS:
        raise ValueError(
            f"Unsupported method '{method}' for truncation. Supported methods: "
            f"{', '.join(_PREFIX_METHODS)}."
        )

    prefix = np.array(array[..., :target_dim], copy=True)
    if method == "random_projection":
        prefix *= np.sqrt(source_dim / target_dim)
    return prefix
