"""Fingerprint parameter hashing for stable cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


_RESERVED_KEYS = ("fp_type", "method", "method_params")


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


def fp_params_hash(
    fp_type: str,
    fp_params: dict,
    *,
    method: str | None = None,
    method_params: dict | None = None,
) -> str:
    """Compute a 16-char SHA-256 prefix of a sorted JSON payload.

    The payload covers the fingerprint type, the fingerprint parameters, the
    compression *method* (when it is set and is not the default
    ``"geometric"`` with no parameters), and *method_params*.

    NumPy scalar values (``np.integer``, ``np.bool_``, and other
    ``np.generic`` instances) are normalized to Python scalars before
    serialization. Bool values are then normalized to integers via ``int(v)``,
    so ``True`` and ``1`` (and ``False`` and ``0``) normalize to the same
    value and hash identically. This equivalence is intentional.

    ``"fp_type"``, ``"method"``, and ``"method_params"`` are reserved keys:
    passing any of them inside *fp_params* raises ``ValueError``, because the
    keyword payload would otherwise overwrite the caller's value and collide
    with a different configuration.

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

    Returns:
        A 16-character hexadecimal string.

    Raises:
        ValueError: If *fp_params* contains one of the reserved keys
            ``"fp_type"``, ``"method"``, or ``"method_params"``.
    """
    reserved = [key for key in _RESERVED_KEYS if key in fp_params]
    if reserved:
        raise ValueError(
            f"{reserved} reserved key(s) cannot appear in fp_params; the "
            f"reserved keys are {', '.join(_RESERVED_KEYS)}."
        )

    normalized = {k: _normalize_value(v) for k, v in fp_params.items()}
    payload: dict = {"fp_type": fp_type, **normalized}

    effective_params = method_params or {}
    if method is not None and not (method == "geometric" and not effective_params):
        norm_mp = {k: _normalize_value(v) for k, v in effective_params.items()}
        payload["method"] = method
        if norm_mp:
            payload["method_params"] = norm_mp

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
