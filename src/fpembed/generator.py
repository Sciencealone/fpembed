"""EmbeddedFingerprintGenerator — unified fingerprint class backed by skfp."""

from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from fpembed.compression import (
    compress_fingerprint,
    _is_power_of_two,
    _check_geometric_cap,
    _ALL_METHODS,
    _validate_method_params,
)
from fpembed.compression_blockwise import compress_blockwise_exact_uint16
from fpembed.compression_projection import (
    compress_hadamard,
    compress_random_projection,
)
from fpembed.docs_links import with_docs
from fpembed.dtype_support import (
    cast_output,
    is_binary,
    normalize_dtype,
    validate_dtype_for_method,
)
from fpembed.embedding_cache import BoundedCache
from fpembed.generator_smiles import _SmilesSelfiesMixin
from fpembed.generator_support import (
    SUPPORTED_FP_TYPES,
    RepresentationStateMixin,
    extract_supplied_matrix,
    prepare_projection_artifacts,
    raw_fingerprint,
    reject_count_simulation,
    resolve_method,
    resolve_native_layout,
    validate_count_dtype,
    validate_count_mode,
)
from fpembed.hashing import fp_params_hash
from fpembed.skfp_factory import _build_skfp

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
_EMPTY_CACHE_INFO = CacheInfo(hits=0, misses=0, maxsize=0, currsize=0)

class EmbeddedFingerprintGenerator(_SmilesSelfiesMixin, RepresentationStateMixin):
    """Unified fingerprint generator backed by scikit-fingerprints.

    Supports fifteen fp_types: ecfp, atom_pair, topological_torsion,
    rdkit, layered, pattern, avalon, secfp, mhfp, map, lingo, pharmacophore,
    pubchem, laggner, erg. The last three are fixed-width: their native widths
    are 881, 307, and 315, and ``fp_size`` is the power-of-two padded width.

    Raw binary fingerprints are optionally reduced to ``D = L / compression``
    features by a pluggable compression method chosen at construction. Available
    methods are the block-wise weighting schemes (``geometric``, ``linear``,
    ``log``, ``uniform``), which collapse each block of bits into one weighted
    value, and the global projections (``hadamard``, ``random_projection``),
    which mix information across all input bits. The block weights or projection
    matrix depend on the selected method and its parameters, not on a single
    fixed mask. With ``compression`` of ``None`` or ``0`` the raw fingerprint is
    returned uncompressed.

    With ``count=True`` the backend returns occurrence counts instead of bits.
    Count output is stored as ``float64`` or ``float32``; ``float32`` is exact
    for integer counts below ``2**24`` and makes no exactness claim above that
    bound. ``count_transform="log1p"`` applies ``numpy.log1p`` to the whole
    representation, raw and compressed. ``uint16`` is rejected for counts.
    """

    __slots__ = (
        "_fp_type", "_fp_size", "_compression", "_fp_params",
        "_method", "_method_params", "_projection_cache",
        "_skfp_fp", "_params_hash", "_cache_size", "_l1_cache", "_l2_cache",
        "_dtype", "_count", "_count_transform", "_native_size", "_padding",
    )

    def __init__(
        self,
        fp_type: str,
        fp_size: int,
        compression: int | None,
        fp_params: dict | None = None,
        cache_size: int | None = None,
        method: str | None = None,
        method_params: dict | None = None,
        dtype: Any = np.float64,
        *,
        count: bool = False,
        count_transform: str = "none",
    ) -> None:
        if fp_params is None:
            fp_params = {}
        if method_params is None:
            method_params = {}

        if isinstance(fp_size, bool) or not isinstance(fp_size, int):
            raise TypeError(
                with_docs(
                    f"fp_size must be an int, got {type(fp_size).__name__}",
                    "generator-argument-type",
                )
            )
        if fp_size <= 0:
            raise ValueError(
                with_docs(f"fp_size must be positive, got {fp_size}", "generator-fp-size-positive")
            )

        if compression is not None:
            if isinstance(compression, bool) or not isinstance(compression, int):
                raise TypeError(
                    with_docs(
                        "compression must be an int or None, got "
                        f"{type(compression).__name__}",
                        "generator-argument-type",
                    )
                )
            if compression < 0:
                raise ValueError(
                    with_docs(
                        f"compression must be >= 0, got {compression}",
                        "generator-argument-value",
                    )
                )

        if cache_size is not None:
            if isinstance(cache_size, bool) or not isinstance(cache_size, int):
                raise TypeError(
                    with_docs(
                        "cache_size must be an int or None, got "
                        f"{type(cache_size).__name__}",
                        "generator-argument-type",
                    )
                )
            if cache_size <= 0:
                raise ValueError(
                    with_docs(
                        f"cache_size must be positive, got {cache_size}",
                        "generator-cache-size-positive",
                    )
                )

        validate_count_mode(fp_type, count, count_transform)
        if fp_type not in SUPPORTED_FP_TYPES:
            raise ValueError(
                with_docs(
                    f"Unsupported fp_type '{fp_type}'. "
                    f"Supported: {', '.join(SUPPORTED_FP_TYPES)}",
                    "generator-unsupported-fp-type",
                )
            )

        effective: int = 0 if (compression is None or compression == 0) else compression
        method = resolve_method(method, count, effective)
        if method not in _ALL_METHODS:
            raise ValueError(
                with_docs(
                    f"Unsupported method '{method}'. "
                    f"Supported: {', '.join(sorted(_ALL_METHODS))}",
                    "generator-method",
                )
            )
        _validate_method_params(method, method_params)

        if effective != 0:
            if not _is_power_of_two(effective):
                raise ValueError(
                    with_docs(
                        f"compression must be a power of 2, got {effective}",
                        "generator-compression-power-of-two",
                    )
                )
            if effective > fp_size / 2:
                raise ValueError(
                    with_docs(
                        f"compression ({effective}) must be <= fp_size / 2 "
                        f"({fp_size // 2})",
                        "generator-compression-half",
                    )
                )
            if fp_size % effective != 0:
                raise ValueError(
                    with_docs(
                        f"fp_size ({fp_size}) must be evenly divisible by "
                        f"compression ({effective})",
                        "generator-compression-divisible",
                    )
                )
            _check_geometric_cap(method, effective)

        if method == "hadamard" and not _is_power_of_two(fp_size):
            raise ValueError(
                with_docs(
                    "Hadamard method requires fp_size to be a power of 2",
                    "generator-hadamard-fp-size",
                )
            )

        normalized_dtype = normalize_dtype(dtype)
        validate_count_dtype(normalized_dtype, count)
        validate_dtype_for_method(normalized_dtype, method, effective)

        supplied_matrix = None
        stored_method_params = dict(method_params)
        if "matrix" in method_params:
            if effective == 0:
                raise ValueError(
                    with_docs(
                        "method_params['matrix'] requires active compression; "
                        "compression is None or 0.",
                        "generator-method-params",
                    )
                )
            supplied_matrix, stored_method_params = extract_supplied_matrix(
                method_params, fp_size, fp_size // effective,
            )

        self._dtype = normalized_dtype
        self._fp_type = fp_type
        self._fp_size = fp_size
        self._compression = effective
        self._fp_params = dict(fp_params)
        self._method = method
        self._method_params = stored_method_params
        self._cache_size = cache_size
        self._count = count
        self._count_transform = count_transform
        reject_count_simulation(count, fp_params)
        self._skfp_fp: Any = _build_skfp(fp_type, fp_size, fp_params, count=count)
        self._native_size, self._padding = resolve_native_layout(
            self._skfp_fp, fp_size, fp_type
        )

        # Precompute projection artifacts for projection methods
        self._projection_cache: Any = prepare_projection_artifacts(
            effective, method, fp_size, self._method_params, supplied_matrix,
        )

        self._params_hash = fp_params_hash(
            fp_type, fp_params, method=method,
            method_params=stored_method_params,
            count=count, count_transform=count_transform,
        )
        if cache_size is not None:
            # Level 1 memoises raw string -> canonical string; level 2 is the
            # embedding memo keyed exactly as before.
            self._l1_cache: Any = BoundedCache(cache_size)
            self._l2_cache: Any = BoundedCache(cache_size)
        else:
            self._l1_cache = None
            self._l2_cache = None

    def _raw_fp(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Raw (uncompressed) fingerprint, padded and count-transformed."""
        return raw_fingerprint(
            self._skfp_fp, mol, self._native_size, self._fp_size,
            self._count, self._count_transform,
        )

    def GetRawFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Return the raw (uncompressed) fingerprint for a single Mol.

        Count mode returns the count values, or their ``log1p`` form, in the
        requested float dtype. ``float32`` is exact for integer counts below
        ``2**24``.
        """
        if mol is None:
            raise ValueError(with_docs("mol must not be None", "generator-mol-none"))
        raw = self._raw_fp(mol)
        if self._dtype.type is np.uint16 and not is_binary(raw):
            raise ValueError(
                with_docs(
                    "dtype='uint16' requires binary values; count or transformed "
                    "representations cannot be stored as uint16.",
                    "generator-uint16-count",
                )
            )
        return raw.astype(self._dtype)

    def GetFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Return the (optionally compressed) fingerprint for a single Mol."""
        if mol is None:
            raise ValueError(with_docs("mol must not be None", "generator-mol-none"))
        raw = self._raw_fp(mol)
        if self._dtype.type is np.uint16 and not is_binary(raw):
            raise ValueError(
                with_docs(
                    "dtype='uint16' requires binary input (every value 0 or 1); "
                    "the exact integer code is undefined otherwise.",
                    "generator-uint16-count",
                )
            )
        if self._compression == 0:
            # Binary bits and count values are lossless in the float dtypes
            # within the documented range.
            return raw.astype(self._dtype)
        if self._dtype.type is np.uint16:
            interleave = self._method_params.get("interleave", False)
            return compress_blockwise_exact_uint16(
                raw, self._compression, scheme=self._method, interleave=interleave,
            ).flatten()
        # Projection methods cast to float64 internally; pass the raw array through.
        if self._method == "hadamard":
            out = compress_hadamard(raw, self._compression, self._projection_cache)
            return cast_output(out.flatten(), self._dtype)
        if self._method == "random_projection":
            out = compress_random_projection(raw, matrix_t=self._projection_cache)
            return cast_output(out.flatten(), self._dtype)
        # Block-wise methods: the float64 cast is applied here, immediately
        # before the einsum call site, so the einsum receives byte-for-byte the
        # same float64 input as before.
        out = compress_fingerprint(
            raw.astype(np.float64), self._compression,
            method=self._method, method_params=self._method_params,
        ).flatten()
        return cast_output(out, self._dtype)

    def GetFingerprintsAsNumPy(
        self, mols: Sequence[Chem.Mol | None], chunk_size: int | None = None,
    ) -> tuple[npt.NDArray[Any], list[int]]:
        """Batch fingerprints. Returns (embeddings, invalid_indices)."""
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError(
                    with_docs(
                        "chunk_size must be a positive integer or None",
                        "generator-argument-value",
                    )
                )
            warnings.warn(
                "chunk_size has no effect and will be removed in a future release",
                DeprecationWarning,
                stacklevel=2,
            )
        total = len(mols)
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        # Only a None mol is invalid input; computation exceptions propagate.
        for i in range(total):
            mol = mols[i]
            if mol is None:
                invalid.append(i)
                continue
            valid.append(self.GetFingerprintAsNumPy(mol))
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=self._dtype), invalid

    def IterFingerprints(
        self, mols: Iterable[Chem.Mol | None],
    ) -> Iterator[tuple[int, npt.NDArray[Any] | None]]:
        """Yield ``(index, embedding_or_none)`` per mol without materialising."""
        for index, mol in enumerate(mols):
            yield index, (
                None if mol is None else self.GetFingerprintAsNumPy(mol)
            )

    def representation_manifest(self, preprocessing: dict | None = None) -> dict:
        """Return a fresh JSON-compatible description of the representation.

        ``preprocessing`` is declarative caller metadata, recorded not run.
        """
        from fpembed.manifest import build_representation_manifest
        return build_representation_manifest(self, preprocessing)

    def clear_cache(self) -> None:
        """Clear both cache levels. No-op if caching is disabled."""
        if self._l2_cache is not None:
            self._l1_cache.cache_clear()
            self._l2_cache.cache_clear()

    def cache_info(self) -> CacheInfo:
        """Return embedding-cache statistics (all zeros when disabled)."""
        if self._l2_cache is not None:
            raw = self._l2_cache.cache_info()
            return CacheInfo(hits=raw.hits, misses=raw.misses, maxsize=raw.maxsize, currsize=raw.currsize)
        return _EMPTY_CACHE_INFO

    def __repr__(self) -> str:
        comp = self._compression if self._compression > 0 else None
        base = (
            f"EmbeddedFingerprintGenerator(fp_type='{self._fp_type}', "
            f"fp_size={self._fp_size}, compression={comp}, "
            f"fp_params={self._fp_params}"
        )
        default_method = (
            "uniform" if self._count and self._compression else "geometric"
        )
        if self._method != default_method or self._method_params:
            base += f", method='{self._method}'"
            base += f", method_params={self._method_params}"
        if self._count:
            base += ", count=True"
        if self._count_transform != "none":
            base += f", count_transform='{self._count_transform}'"
        if self._dtype.type is not np.float64:
            base += f", dtype='{self._dtype}'"
        base += ")"
        return base
