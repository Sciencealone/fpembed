"""Fingerprint compression orchestrator.

Routes ``compress_fingerprint()`` calls to the appropriate sub-module
based on the ``method`` parameter. Supports block-wise weight schemes
(geometric, linear, log, uniform) and global projection methods
(hadamard, random_projection).

The default behavior (``method="geometric"``, no ``method_params``)
produces output identical to the original implementation.

Public API
----------
compress_fingerprint : Compress a fingerprint array.
_is_power_of_two     : Check whether an integer is a power of two.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from fpembed.compression_blockwise import compress_blockwise
from fpembed.compression_projection import (
    build_rp_matrix,
    build_srht_signs,
    compress_hadamard,
    compress_random_projection,
)

_BLOCKWISE_METHODS = {"geometric", "linear", "log", "uniform"}
_PROJECTION_METHODS = {"hadamard", "random_projection"}
_ALL_METHODS = _BLOCKWISE_METHODS | _PROJECTION_METHODS

_VALID_PARAMS: dict[str, set[str]] = {
    "geometric": {"interleave"},
    "linear": {"interleave"},
    "log": {"interleave"},
    "uniform": {"interleave"},
    "hadamard": {"seed"},
    "random_projection": {"seed", "sparse"},
}

_DEFAULT_SEED = 42


def _validate_method_params(method: str, method_params: dict) -> None:
    """Validate method_params keys and value types for the given method."""
    valid_keys = _VALID_PARAMS[method]
    unrecognized = set(method_params) - valid_keys
    if unrecognized:
        raise ValueError(
            f"Unrecognized method_params keys {unrecognized} for method "
            f"'{method}'. Valid keys: {valid_keys}"
        )

    if "seed" in method_params:
        seed = method_params["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(
                f"seed must be a non-negative integer, got {seed!r}"
            )

    if "sparse" in method_params:
        sparse = method_params["sparse"]
        if not isinstance(sparse, bool):
            raise ValueError(
                f"sparse must be a boolean, got {type(sparse).__name__}"
            )

    if "interleave" in method_params:
        interleave = method_params["interleave"]
        if not isinstance(interleave, bool):
            raise ValueError(
                f"interleave must be a boolean, got {type(interleave).__name__}"
            )
        if interleave and method in _PROJECTION_METHODS:
            raise ValueError(
                f"interleave is not applicable to method '{method}'"
            )


def compress_fingerprint(
    vector: npt.NDArray[Any],
    size: int,
    *,
    method: str = "geometric",
    method_params: dict | None = None,
) -> npt.NDArray[np.float64]:
    """Compress a binary fingerprint into a dense float embedding.

    Parameters
    ----------
    vector : ndarray
        Input fingerprint — ``(L,)`` or ``(N, L)``.
    size : int
        Compression factor.
    method : str
        Compression method name.
    method_params : dict or None
        Method-specific parameters.

    Returns
    -------
    ndarray of float64
        Compressed embedding.

    Raises
    ------
    ValueError
        If *size* is invalid, *method* is unsupported, or *method_params*
        contains invalid keys/values.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError(
            "Size must be a positive integer. Try size = 16, or size = 32"
        )

    if method not in _ALL_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported: {', '.join(sorted(_ALL_METHODS))}"
        )

    if method_params is None:
        method_params = {}

    _validate_method_params(method, method_params)

    # Block-wise methods
    if method in _BLOCKWISE_METHODS:
        interleave = method_params.get("interleave", False)
        return compress_blockwise(vector, size, scheme=method, interleave=interleave)

    # Projection methods
    fp_len = vector.shape[-1] if vector.ndim > 1 else vector.shape[0]
    output_dim = fp_len // size

    if method == "hadamard":
        seed = method_params.get("seed", _DEFAULT_SEED)
        signs = build_srht_signs(fp_len, seed)
        return compress_hadamard(vector, size, signs)

    # method == "random_projection"
    seed = method_params.get("seed", _DEFAULT_SEED)
    sparse = method_params.get("sparse", False)
    matrix = build_rp_matrix(fp_len, output_dim, seed, sparse)
    return compress_random_projection(vector, matrix)


def _is_power_of_two(n: int) -> bool:
    """Return True if *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0
