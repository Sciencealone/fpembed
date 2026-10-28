"""Count-mode validation, raw-path, and representation-state helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from fpembed.compression import _DEFAULT_SEED
from fpembed.compression_projection import _rp_matrix_T, build_srht_signs
from fpembed.docs_links import with_docs
from fpembed.matrix_identity import matrix_id, validate_projection_matrix

SUPPORTED_FP_TYPES = (
    "ecfp", "atom_pair", "topological_torsion", "rdkit", "layered", "pattern",
    "avalon", "secfp", "mhfp", "map", "lingo", "pharmacophore",
    "pubchem", "laggner", "erg",
)

# Backends whose constructor fixes the width. The caller's ``fp_size`` is the
# padded width, so it must be a power of two at or above the native size.
_FIXED_WIDTH_FP_TYPES = frozenset({"pubchem", "laggner", "erg"})


class RepresentationStateMixin:
    """Read-only representation properties shared by the generator.

    The mixin keeps ``__slots__ = ()`` so slotted generator instances gain no
    ``__dict__``. It reads attributes the concrete generator sets in
    ``__init__``.
    """

    __slots__ = ()

    @property
    def fp_type(self) -> str:
        """Fingerprint type identifier."""
        return self._fp_type

    @property
    def fp_size(self) -> int:
        """Fingerprint bit-vector length."""
        return self._fp_size

    @property
    def compression(self) -> int:
        """Compression factor (0 means no compression)."""
        return self._compression

    @property
    def fp_params(self) -> dict:
        """Type-specific fingerprint parameters (copy)."""
        return dict(self._fp_params)

    @property
    def method(self) -> str:
        """Compression method name."""
        return self._method

    @property
    def method_params(self) -> dict:
        """Method-specific parameters (copy)."""
        return dict(self._method_params)

    @property
    def params_hash(self) -> str:
        """Read-only 16-char hex hash covering fp_type, fp_params, method,
        method_params, and the non-default count representation.

        The hash excludes ``fp_size``, the compression factor, the output
        dtype, preprocessing, and dependency versions; use
        :meth:`representation_manifest` for a complete identity.
        """
        return self._params_hash

    @property
    def dtype(self) -> np.dtype:
        """Normalised output dtype applied to every returned array."""
        return self._dtype

    @property
    def count(self) -> bool:
        """Whether this generator produces count fingerprints."""
        return self._count

    @property
    def count_transform(self) -> str:
        """Count transform, ``"none"`` or ``"log1p"``."""
        return self._count_transform

    @property
    def native_size(self) -> int:
        """Backend width before padding."""
        return self._native_size

    @property
    def padding(self) -> str:
        """Padding rule, ``"none"`` or ``"tail_zeros"``."""
        return self._padding

    def _out_dim(self) -> int:
        return (
            self._fp_size // self._compression
            if self._compression > 0
            else self._fp_size
        )


_COUNT_TRANSFORMS = ("none", "log1p")
_COUNT_SUPPORTED_FP_TYPES = (
    "ecfp", "atom_pair", "topological_torsion", "rdkit", "avalon",
    "mhfp", "map", "lingo", "pharmacophore",
)
_COUNT_SUPPORTED_SET = frozenset(_COUNT_SUPPORTED_FP_TYPES)
_COUNT_FORBIDDEN_METHODS = frozenset({"geometric", "linear", "log"})
_COUNT_ALLOWED_METHODS = ("uniform", "hadamard", "random_projection")


def validate_count_mode(fp_type: str, count: bool, count_transform: str) -> None:
    """Validate the count flag, the transform, and the type pairing.

    Raises
    ------
    TypeError
        If *count* is not a boolean.
    ValueError
        If *count_transform* is unknown, if a transform is requested without
        count mode, or if count mode is requested for an unsupported type.
    """
    if not isinstance(count, bool):
        raise TypeError(
            with_docs(
                f"count must be a bool, got {type(count).__name__}",
                "generator-argument-type",
            )
        )
    if count_transform not in _COUNT_TRANSFORMS:
        raise ValueError(
            with_docs(
                f"count_transform must be one of {_COUNT_TRANSFORMS}, "
                f"got {count_transform!r}",
                "generator-count-transform-value",
            )
        )
    if count_transform != "none" and not count:
        raise ValueError(
            with_docs(
                "count_transform requires count=True",
                "generator-count-transform-requires-count",
            )
        )
    if count and fp_type not in _COUNT_SUPPORTED_SET:
        raise ValueError(
            with_docs(
                f"Count mode is not supported for fp_type '{fp_type}'. "
                f"Supported count types: {', '.join(_COUNT_SUPPORTED_FP_TYPES)}",
                "generator-count-mode-not-supported",
            )
        )


def resolve_method(method: str | None, count: bool, compression: int) -> str:
    """Resolve ``method=None`` and reject methods without a count meaning.

    ``None`` becomes ``"uniform"`` for active count compression and
    ``"geometric"`` otherwise. An explicit ``geometric``, ``linear``, or
    ``log`` still raises when count compression is active.
    """
    if method is None:
        return "uniform" if count and compression else "geometric"
    if count and compression and method in _COUNT_FORBIDDEN_METHODS:
        raise ValueError(
            with_docs(
                f"method '{method}' cannot be used with count fingerprints: "
                f"count compression supports "
                f"{', '.join(_COUNT_ALLOWED_METHODS)}. Geometric reversibility "
                "and integer codes are binary-only.",
                "generator-method",
            )
        )
    return method


def validate_count_dtype(dt: np.dtype, count: bool) -> None:
    """Reject ``uint16`` for count mode before any cast can wrap values."""
    if count and dt.type is np.uint16:
        raise ValueError(
            with_docs(
                "dtype='uint16' is not supported for count fingerprints: uint16 "
                "cannot represent count values and would silently wrap. Use "
                "float64 or float32.",
                "generator-uint16-count",
            )
        )


def reject_count_simulation(count: bool, fp_params: dict) -> None:
    """Reject ``count_simulation`` when count mode is active."""
    if count and "count_simulation" in fp_params:
        raise ValueError(
            with_docs(
                "fp_params['count_simulation'] is a binary-mode option and "
                "cannot be combined with count=True.",
                "generator-count-simulation-binary",
            )
        )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def resolve_native_layout(
    skfp_fp: Any, fp_size: int, fp_type: str,
) -> tuple[int, str]:
    """Return ``(native_size, padding)`` for the constructed backend.

    A fixed-width backend keeps its own width; the caller's ``fp_size`` is the
    padded width and must be a power of two at or above the native size. The
    exact native width is rejected for these types because 881, 307, and 315
    are not powers of two. A variable-width backend must match ``fp_size``, so
    it keeps its existing freedom to use a non-power-of-two width.
    """
    native_size = int(skfp_fp.n_features_out)
    if native_size > fp_size:
        raise ValueError(
            with_docs(
                f"fp_size ({fp_size}) is smaller than the backend native_size "
                f"({native_size}).",
                "generator-fp-size-native",
            )
        )
    if fp_type in _FIXED_WIDTH_FP_TYPES:
        if not _is_power_of_two(fp_size):
            raise ValueError(
                with_docs(
                    f"fp_size ({fp_size}) must be a power of two for fixed-width "
                    f"fp_type '{fp_type}' (native_size {native_size}).",
                    "generator-fp-size-power-of-two",
                )
            )
    elif native_size != fp_size:
        raise ValueError(
            with_docs(
                f"Backend native_size ({native_size}) must equal fp_size "
                f"({fp_size}) for variable-width fp_type '{fp_type}'.",
                "generator-fp-size-native",
            )
        )
    padding = "tail_zeros" if native_size < fp_size else "none"
    return native_size, padding


def raw_fingerprint(
    skfp_fp: Any,
    mol: Any,
    native_size: int,
    fp_size: int,
    count: bool,
    count_transform: str,
) -> npt.NDArray[Any]:
    """Return the uncompressed fingerprint, padded and optionally transformed.

    The backend yields ``uint8`` for binary and ``uint32`` for count mode. A
    real vector shorter than *fp_size* is tail-padded with zeros; this only
    happens for a fixed-width type whose padded ``fp_size`` exceeds
    ``native_size``. The ``log1p`` transform is applied after casting to
    ``float64`` so the result matches the compressed representation.
    """
    raw = skfp_fp.transform([mol]).flatten()
    if raw.size < native_size:
        raise RuntimeError(
            f"backend returned {raw.size} values, fewer than the native "
            f"size {native_size}"
        )
    if raw.size < fp_size:
        raw = np.pad(raw, (0, fp_size - raw.size))
    if count and count_transform == "log1p":
        raw = np.log1p(raw.astype(np.float64))
    return raw


def extract_supplied_matrix(
    method_params: dict, fp_size: int, output_dim: int,
) -> tuple[npt.NDArray[np.float64], dict]:
    """Validate a caller matrix and return it with a sanitized parameter copy.

    The returned parameter mapping replaces the array with ``matrix_id`` and
    ``matrix_shape`` so neither the cache key nor the manifest ever holds the
    matrix contents.
    """
    canonical = validate_projection_matrix(
        method_params["matrix"], fp_size=fp_size, output_dim=output_dim,
    )
    stored = dict(method_params)
    del stored["matrix"]
    stored["matrix_id"] = matrix_id(canonical)
    stored["matrix_shape"] = list(canonical.shape)
    return canonical, stored

def prepare_projection_artifacts(
    effective: int,
    method: str,
    fp_size: int,
    method_params: dict,
    supplied_matrix: npt.NDArray[np.float64] | None,
) -> Any:
    """Build or select the cached projection artifact for a generator."""
    if effective == 0:
        return None
    if method == "hadamard":
        seed = method_params.get("seed", _DEFAULT_SEED)
        return build_srht_signs(fp_size, seed)
    if method == "random_projection":
        if supplied_matrix is not None:
            matrix_t = np.ascontiguousarray(supplied_matrix.T)
            matrix_t.setflags(write=False)
            return matrix_t
        seed = method_params.get("seed", _DEFAULT_SEED)
        sparse = method_params.get("sparse", False)
        return _rp_matrix_T(
            fp_size, fp_size // effective, seed, sparse
        )
    return None
