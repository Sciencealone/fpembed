"""Versioned representation manifests for persisted embeddings.

This module provides an identity record separate from ``fp_params_hash``. The
legacy hash stays byte-for-byte compatible for cache keys; the manifest is a
fresh JSON-compatible description intended for stored feature matrices and
deployed models. It records the resolved fingerprint parameters, the effective
compression configuration, the physical output encoding, the built-in
canonicalization policy, an optional caller-declared preprocessing block, and
the versions of the packages that produced the values.

The generator imports this module lazily, so importing it from the compression
and scikit-fingerprints factories cannot form an import cycle.

Public API
----------
build_representation_manifest : Build the manifest mapping for a generator.
"""

from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import numpy as np

from fpembed.compression import _resolve_method_params
from fpembed.skfp_factory import _ALLOWED_FP_PARAMS

_SCHEMA_VERSION = 1

# Physical output encodings. "raw_binary" is an uncompressed 0/1 fingerprint
# in the requested dtype; "normalized_block_value" is the geometric float
# value n/(2**C - 1); "mean_block_value" is the uniform float value
# popcount/C; "weighted_block_value" is the linear/log weighted block sum;
# the two integer labels are the exact uint16 codes, and "projection_value"
# is a Hadamard or random-projection output.
_RAW_ENCODING = "raw_binary"
_PROJECTION_ENCODING = "projection_value"
_INTEGER_ENCODINGS = {"geometric": "integer_code", "uniform": "integer_count"}
_BLOCKWISE_ENCODINGS = {
    "geometric": "normalized_block_value",
    "uniform": "mean_block_value",
    "linear": "weighted_block_value",
    "log": "weighted_block_value",
}

# Distribution names recorded for reproducibility. "fpembed" resolves through
# installed distribution metadata, so a stale editable install cannot report a
# version that disagrees with the deployed artifact.
_DISTRIBUTIONS = (
    "fpembed",
    "numpy",
    "rdkit",
    "scikit-fingerprints",
    "selfies",
)

# Built-in preprocessing. Canonicalization normalizes the SMILES spelling
# (atom ordering and RDKit's aromatic form) so the cache key is
# spelling-invariant. It is not chemical standardization: salts, tautomers,
# protonation and stereochemistry are left as written.
_CANONICALIZATION = {
    "policy": "rdkit_canonical_reparse",
    "applies_to": "smiles_input",
}


def build_representation_manifest(generator: Any, preprocessing: dict | None) -> dict:
    """Return a fresh JSON-compatible identity of *generator*'s representation.

    Fingerprint parameters are read from the constructed backend, so omitted
    and explicitly-default values resolve to the same settings. Method
    parameters likewise resolve to their effective values. The returned
    mapping is rebuilt on every call and is independent of generator state.

    Parameters
    ----------
    generator : EmbeddedFingerprintGenerator
        Constructed generator to describe.
    preprocessing : dict or None
        Optional JSON-serializable mapping describing any transformation
        applied outside fpembed. Recorded verbatim under ``external``; it is
        never executed and is not claimed to be standardization.

    Raises
    ------
    TypeError
        If *preprocessing* is not a JSON-serializable mapping.
    """
    fp_type = generator.fp_type
    resolved_defaults = {
        key: _json_scalar(getattr(generator._skfp_fp, key))
        for key in sorted(_ALLOWED_FP_PARAMS.get(fp_type, ()))
    }

    output_dim = generator._out_dim()
    output_dtype = generator.dtype
    compression_active = generator.compression > 0
    if compression_active:
        method = generator.method
        method_params = _resolve_method_params(method, generator.method_params)
    else:
        method = None
        method_params = {}

    return {
        "schema_version": _SCHEMA_VERSION,
        "fingerprint": {
            "type": fp_type,
            "size": generator.fp_size,
            "encoding": "binary",
            "params": resolved_defaults,
        },
        "compression": {
            "active": compression_active,
            "factor": generator.compression if compression_active else None,
            "method": method,
            "method_params": method_params,
            "output_dim": output_dim,
        },
        "output": {
            "dim": output_dim,
            "dtype": output_dtype.name,
            "encoding": _output_encoding(compression_active, method, output_dtype),
        },
        "preprocessing": {
            "canonicalization": dict(_CANONICALIZATION),
            "external": _copy_external(preprocessing),
        },
        "versions": {name: _distribution_version(name) for name in _DISTRIBUTIONS},
    }


def _output_encoding(compression_active: bool, method: str | None, dtype: np.dtype) -> str:
    if not compression_active:
        return _RAW_ENCODING
    if dtype.type is np.uint16:
        # uint16 is only accepted for geometric/uniform, but fall back to the
        # generic code label rather than raising if that invariant ever drifts.
        return _INTEGER_ENCODINGS.get(method, "integer_code")
    return _BLOCKWISE_ENCODINGS.get(method, _PROJECTION_ENCODING)


def _copy_external(preprocessing: dict | None) -> dict | None:
    """Deep-isolate caller metadata through a JSON round-trip."""
    if preprocessing is None:
        return None
    if not isinstance(preprocessing, dict):
        raise TypeError("preprocessing must be a JSON-serializable mapping")
    try:
        return json.loads(json.dumps(preprocessing))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "preprocessing must be a JSON-serializable mapping"
        ) from exc


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_scalar(value: Any) -> Any:
    """Convert a NumPy scalar to its Python equivalent for JSON encoding."""
    if isinstance(value, np.generic):
        return value.item()
    return value
