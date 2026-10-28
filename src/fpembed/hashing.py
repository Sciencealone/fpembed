"""Fingerprint parameter hashing for stable cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from fpembed.docs_links import with_docs
from fpembed.matrix_identity import matrix_id


_RESERVED_KEYS = ("fp_type", "method", "method_params", "count", "count_transform")
_COUNT_TRANSFORMS = ("none", "log1p")


def _normalize_value(value: Any) -> Any:
    """Convert NumPy scalars to Python scalars, then bools to ints.

    ``np.generic.item()`` returns the equivalent Python scalar, so NumPy
    integers and booleans become JSON-serializable without changing the
    payload for values that already serialized. The bool-to-int rule is
    applied afterwards, which also covers ``np.bool_``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    return int(value) if isinstance(value, bool) else value


def _normalize_params(params: dict) -> dict:
    """Normalize scalars and replace any array with its identity digest.

    A projection matrix becomes ``matrix_id`` plus ``matrix_shape``; any other
    array (for example a precomputed centering vector) becomes ``<key>_id``.
    No array reaches ``json.dumps``.
    """
    normalized: dict = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            normalized[f"{key}_id"] = matrix_id(value)
            if key == "matrix":
                normalized[f"{key}_shape"] = list(value.shape)
            continue
        normalized[key] = _normalize_value(value)
    return normalized


def fp_params_hash(
    fp_type: str,
    fp_params: dict,
    *,
    method: str | None = None,
    method_params: dict | None = None,
    count: bool = False,
    count_transform: str = "none",
) -> str:
    """Compute a 16-char SHA-256 prefix of a sorted JSON payload.

    The payload covers the fingerprint type, the fingerprint parameters, the
    compression *method* (when it is set and is not the default
    ``"geometric"`` with no parameters), and *method_params*. Count mode adds
    ``"count": 1`` and, when a transform is active, ``"count_transform":
    "log1p"``. Binary configurations without a transform keep the original
    payload byte for byte.

    NumPy scalar values (``np.integer``, ``np.bool_``, and other
    ``np.generic`` instances) are normalized to Python scalars before
    serialization. Bool values are then normalized to integers via ``int(v)``,
    so ``True`` and ``1`` (and ``False`` and ``0``) normalize to the same
    value and hash identically. This equivalence is intentional.

    ``"fp_type"``, ``"method"``, ``"method_params"``, ``"count"``, and
    ``"count_transform"`` are reserved keys: passing any of them inside
    *fp_params* raises ``ValueError``, because the keyword payload would
    otherwise overwrite the caller's value and collide with a different
    configuration.

    When *method* is ``None`` or ``"geometric"`` with no *method_params*,
    the hash is identical to the original implementation (backward compatible).

    This hash is a cache key, not a complete identity for a persisted array or
    a trained model: it does not cover ``fp_size``, the compression factor,
    the output dtype or encoding, the preprocessing policy, or dependency
    versions. ``EmbeddedFingerprintGenerator.representation_manifest()``
    provides that record.

    Args:
        fp_type: Fingerprint type identifier (e.g. ``"ecfp"``).
        fp_params: Type-specific fingerprint parameters.
        method: Compression method name (optional).
        method_params: Method-specific parameters (optional).
        count: Whether the representation is a count fingerprint.
        count_transform: ``"none"`` or ``"log1p"``.

    Returns:
        A 16-character hexadecimal string.

    Raises:
        ValueError: If *fp_params* contains one of the reserved keys, if
            *count_transform* is unknown, or if a transform is requested
            without count mode.
        TypeError: If *count* is not a boolean.
    """
    if not isinstance(count, bool):
        raise TypeError(
            with_docs(
                f"count must be a bool, got {type(count).__name__}",
                "identity-argument-type",
            )
        )
    if count_transform not in _COUNT_TRANSFORMS:
        raise ValueError(
            with_docs(
                f"count_transform must be one of {_COUNT_TRANSFORMS}, "
                f"got {count_transform!r}",
                "identity-count-transform-value",
            )
        )
    if count_transform != "none" and not count:
        raise ValueError(
            with_docs(
                "count_transform requires count=True",
                "identity-count-transform-requires-count",
            )
        )

    reserved = [key for key in _RESERVED_KEYS if key in fp_params]
    if reserved:
        raise ValueError(
            with_docs(
                f"{reserved} reserved key(s) cannot appear in fp_params; the "
                f"reserved keys are {', '.join(_RESERVED_KEYS)}.",
                "identity-fp-params",
            )
        )

    normalized = _normalize_params(fp_params)
    payload: dict = {"fp_type": fp_type, **normalized}
    if count:
        payload["count"] = 1
    if count_transform != "none":
        payload["count_transform"] = count_transform

    effective_params = method_params or {}
    if method is not None and not (method == "geometric" and not effective_params):
        norm_mp = _normalize_params(effective_params)
        payload["method"] = method
        if norm_mp:
            payload["method_params"] = norm_mp

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
