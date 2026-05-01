"""Fingerprint parameter hashing for stable cache keys."""

from __future__ import annotations

import hashlib
import json


def fp_params_hash(
    fp_type: str,
    fp_params: dict,
    *,
    method: str | None = None,
    method_params: dict | None = None,
) -> str:
    """Compute a 16-char SHA-256 prefix of sorted JSON of {fp_type, **fp_params}.

    Bool values in *fp_params* are normalized to Python ``bool`` before
    serialization to prevent ``True``/``1`` collisions.

    Args:
        fp_type: Fingerprint type identifier (e.g. ``"ecfp"``).
        fp_params: Type-specific fingerprint parameters.
        method: Compression method name (optional).
        method_params: Method-specific parameters (optional).

    Returns:
        A 16-character hexadecimal string.
    """
    normalized = {
        k: int(v) if isinstance(v, bool) else v
        for k, v in fp_params.items()
    }
    payload: dict = {"fp_type": fp_type, **normalized}

    effective_params = method_params or {}
    if method is not None and not (method == "geometric" and not effective_params):
        norm_mp = {
            k: int(v) if isinstance(v, bool) else v
            for k, v in effective_params.items()
        }
        payload["method"] = method
        if norm_mp:
            payload["method_params"] = norm_mp

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
