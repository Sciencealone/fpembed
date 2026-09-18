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
    _DEFAULT_SEED,
)
from fpembed.compression_blockwise import compress_blockwise_exact_uint16
from fpembed.compression_projection import (
    _rp_matrix_T,
    build_srht_signs,
    compress_hadamard,
    compress_random_projection,
)
from fpembed.dtype_support import (
    cast_output,
    is_binary,
    normalize_dtype,
    validate_dtype_for_method,
)
from fpembed.embedding_cache import BoundedCache
from fpembed.generator_smiles import _SmilesSelfiesMixin
from fpembed.hashing import fp_params_hash
from fpembed.skfp_factory import _build_skfp

if TYPE_CHECKING:
    from collections.abc import Sequence

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
_EMPTY_CACHE_INFO = CacheInfo(hits=0, misses=0, maxsize=0, currsize=0)

_SUPPORTED_FP_TYPES = (
    "ecfp", "atom_pair", "topological_torsion", "rdkit", "layered", "pattern",
    "avalon", "secfp", "mhfp", "map",
)


class EmbeddedFingerprintGenerator(_SmilesSelfiesMixin):
    """Unified fingerprint generator backed by scikit-fingerprints.

    Supports ten fp_types: ecfp, atom_pair, topological_torsion,
    rdkit, layered, pattern, avalon, secfp, mhfp, map.

    Raw binary fingerprints are optionally reduced to ``D = L / compression``
    features by a pluggable compression method chosen at construction. Available
    methods are the block-wise weighting schemes (``geometric``, ``linear``,
    ``log``, ``uniform``), which collapse each block of bits into one weighted
    value, and the global projections (``hadamard``, ``random_projection``),
    which mix information across all input bits. The block weights or projection
    matrix depend on the selected method and its parameters, not on a single
    fixed mask. With ``compression`` of ``None`` or ``0`` the raw fingerprint is
    returned uncompressed.
    """

    __slots__ = (
        "_fp_type", "_fp_size", "_compression", "_fp_params",
        "_method", "_method_params", "_projection_cache",
        "_skfp_fp", "_params_hash", "_cache_size", "_l1_cache", "_l2_cache",
        "_dtype",
    )

    def __init__(
        self,
        fp_type: str,
        fp_size: int,
        compression: int | None,
        fp_params: dict | None = None,
        cache_size: int | None = None,
        method: str = "geometric",
        method_params: dict | None = None,
        dtype: Any = np.float64,
    ) -> None:
        if fp_params is None:
            fp_params = {}
        if method_params is None:
            method_params = {}

        if isinstance(fp_size, bool) or not isinstance(fp_size, int):
            raise TypeError(f"fp_size must be an int, got {type(fp_size).__name__}")
        if fp_size <= 0:
            raise ValueError(f"fp_size must be positive, got {fp_size}")

        if compression is not None:
            if isinstance(compression, bool) or not isinstance(compression, int):
                raise TypeError(
                    f"compression must be an int or None, got {type(compression).__name__}"
                )
            if compression < 0:
                raise ValueError(f"compression must be >= 0, got {compression}")

        if cache_size is not None:
            if isinstance(cache_size, bool) or not isinstance(cache_size, int):
                raise TypeError(
                    f"cache_size must be an int or None, got {type(cache_size).__name__}"
                )
            if cache_size <= 0:
                raise ValueError(f"cache_size must be positive, got {cache_size}")

        if fp_type not in _SUPPORTED_FP_TYPES:
            raise ValueError(
                f"Unsupported fp_type '{fp_type}'. "
                f"Supported: {', '.join(_SUPPORTED_FP_TYPES)}"
            )
        if method not in _ALL_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Supported: {', '.join(sorted(_ALL_METHODS))}"
            )
        _validate_method_params(method, method_params)

        effective: int = 0 if (compression is None or compression == 0) else compression
        if effective != 0:
            if not _is_power_of_two(effective):
                raise ValueError(f"compression must be a power of 2, got {effective}")
            if effective > fp_size / 2:
                raise ValueError(
                    f"compression ({effective}) must be <= fp_size / 2 ({fp_size // 2})"
                )
            if fp_size % effective != 0:
                raise ValueError(
                    f"fp_size ({fp_size}) must be evenly divisible by compression ({effective})"
                )

        if method == "hadamard" and not _is_power_of_two(fp_size):
            raise ValueError("Hadamard method requires fp_size to be a power of 2")

        normalized_dtype = normalize_dtype(dtype)
        validate_dtype_for_method(normalized_dtype, method, effective)

        self._dtype = normalized_dtype
        self._fp_type = fp_type
        self._fp_size = fp_size
        self._compression = effective
        self._fp_params = dict(fp_params)
        self._method = method
        self._method_params = dict(method_params)
        self._cache_size = cache_size
        self._skfp_fp: Any = _build_skfp(fp_type, fp_size, fp_params)

        # Precompute projection artifacts for projection methods
        self._projection_cache: Any = None
        if effective != 0 and method == "hadamard":
            seed = self._method_params.get("seed", _DEFAULT_SEED)
            self._projection_cache = build_srht_signs(fp_size, seed)
        elif effective != 0 and method == "random_projection":
            seed = self._method_params.get("seed", _DEFAULT_SEED)
            sparse = self._method_params.get("sparse", False)
            output_dim = fp_size // effective
            # Store the memoised C-contiguous (L, D) transpose so the primary
            # single-molecule path reaches the sparse gather.
            self._projection_cache = _rp_matrix_T(
                fp_size, output_dim, seed, sparse
            )

        self._params_hash = fp_params_hash(
            fp_type, fp_params, method=method, method_params=method_params,
        )
        if cache_size is not None:
            # Level 1 memoises raw string -> canonical string; level 2 is the
            # embedding memo keyed exactly as before.
            self._l1_cache: Any = BoundedCache(cache_size)
            self._l2_cache: Any = BoundedCache(cache_size)
        else:
            self._l1_cache = None
            self._l2_cache = None

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
        """Read-only 16-char hex hash of fp_type + fp_params."""
        return self._params_hash

    @property
    def dtype(self) -> np.dtype:
        """Normalised output dtype applied to every returned array."""
        return self._dtype

    def _raw_fp(self, mol: Chem.Mol) -> npt.NDArray[np.uint8]:
        return self._skfp_fp.transform([mol]).flatten()

    def GetRawFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Return the raw (uncompressed) fingerprint for a single Mol."""
        if mol is None:
            raise ValueError("mol must not be None")
        # Raw fingerprints are binary, so every accepted dtype is lossless.
        return self._raw_fp(mol).astype(self._dtype)

    def GetFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[Any]:
        """Return the (optionally compressed) fingerprint for a single Mol."""
        if mol is None:
            raise ValueError("mol must not be None")
        raw = self._raw_fp(mol)
        if self._dtype.type is np.uint16 and not is_binary(raw):
            raise ValueError(
                "dtype='uint16' requires binary input (every value 0 or 1); "
                "the exact integer code is undefined otherwise."
            )
        if self._compression == 0:
            # Raw fingerprints are binary, so every accepted dtype is lossless.
            return raw.astype(self._dtype)
        if self._dtype.type is np.uint16:
            interleave = self._method_params.get("interleave", False)
            return compress_blockwise_exact_uint16(
                raw, self._compression, scheme=self._method, interleave=interleave,
            ).flatten()
        # Projection methods cast to float64 internally; pass raw uint8 through.
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

    def _out_dim(self) -> int:
        return self._fp_size // self._compression if self._compression > 0 else self._fp_size

    def GetFingerprintsAsNumPy(
        self, mols: Sequence[Chem.Mol | None], chunk_size: int | None = None,
    ) -> tuple[npt.NDArray[Any], list[int]]:
        """Batch fingerprints. Returns (embeddings, invalid_indices)."""
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer or None")
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
        if self._method != "geometric" or self._method_params:
            base += f", method='{self._method}'"
            base += f", method_params={self._method_params}"
        if self._dtype.type is not np.float64:
            base += f", dtype='{self._dtype}'"
        base += ")"
        return base
