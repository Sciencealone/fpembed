"""EmbeddedFingerprintGenerator — unified fingerprint class backed by skfp."""

from __future__ import annotations

import functools
from collections import namedtuple
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from skfp.fingerprints import (
    AtomPairFingerprint,
    AvalonFingerprint,
    ECFPFingerprint,
    LayeredFingerprint,
    MAPFingerprint,
    MHFPFingerprint,
    PatternFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
)

from fpembed.compression import (
    compress_fingerprint,
    _is_power_of_two,
    _ALL_METHODS,
    _BLOCKWISE_METHODS,
    _VALID_PARAMS,
    _validate_method_params,
    _DEFAULT_SEED,
)
from fpembed.compression_projection import (
    build_rp_matrix,
    build_srht_signs,
    compress_hadamard,
    compress_random_projection,
)
from fpembed.hashing import fp_params_hash
from fpembed.smiles_utils import canonicalize_smiles, parse_smiles

if TYPE_CHECKING:
    from collections.abc import Sequence

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
_EMPTY_CACHE_INFO = CacheInfo(hits=0, misses=0, maxsize=0, currsize=0)

_SUPPORTED_FP_TYPES = (
    "ecfp", "atom_pair", "topological_torsion", "rdkit", "layered", "pattern",
    "avalon", "secfp", "mhfp", "map",
)


def _build_skfp(fp_type: str, fp_size: int, fp_params: dict) -> Any:
    """Instantiate the appropriate skfp fingerprint object."""
    if fp_type == "ecfp":
        return ECFPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 2),
            count=False, n_jobs=1,
        )
    if fp_type == "atom_pair":
        return AtomPairFingerprint(
            fp_size=fp_size, min_distance=fp_params.get("min_distance", 1),
            max_distance=fp_params.get("max_distance", 30),
            count=False, n_jobs=1,
        )
    if fp_type == "topological_torsion":
        return TopologicalTorsionFingerprint(
            fp_size=fp_size,
            torsion_atom_count=fp_params.get("torsion_atom_count", 4),
            count=False, n_jobs=1,
        )
    if fp_type == "rdkit":
        return RDKitFingerprint(
            fp_size=fp_size, min_path=fp_params.get("min_path", 1),
            max_path=fp_params.get("max_path", 7), count=False, n_jobs=1,
        )
    if fp_type == "layered":
        return LayeredFingerprint(
            fp_size=fp_size, min_path=fp_params.get("min_path", 1),
            max_path=fp_params.get("max_path", 7), n_jobs=1,
        )
    if fp_type == "avalon":
        return AvalonFingerprint(fp_size=fp_size, count=False, n_jobs=1)
    if fp_type == "secfp":
        return SECFPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 3),
            min_radius=fp_params.get("min_radius", 1), n_jobs=1,
        )
    if fp_type == "mhfp":
        return MHFPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 3),
            min_radius=fp_params.get("min_radius", 1),
            variant="bit", n_jobs=1,
        )
    if fp_type == "map":
        return MAPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 2),
            variant="binary", n_jobs=1,
        )
    return PatternFingerprint(fp_size=fp_size, n_jobs=1)


class EmbeddedFingerprintGenerator:
    """Unified fingerprint generator backed by scikit-fingerprints.

    Supports ten fp_types: ecfp, atom_pair, topological_torsion,
    rdkit, layered, pattern, avalon, secfp, mhfp, map.
    Compression via log-space weighted mask.
    """

    __slots__ = (
        "_fp_type", "_fp_size", "_compression", "_fp_params",
        "_method", "_method_params", "_projection_cache",
        "_skfp_fp", "_params_hash", "_cache_size", "_cached_compute",
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
    ) -> None:
        if fp_params is None:
            fp_params = {}
        if method_params is None:
            method_params = {}
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
            self._projection_cache = build_rp_matrix(fp_size, output_dim, seed, sparse)

        self._params_hash = fp_params_hash(
            fp_type, fp_params, method=method, method_params=method_params,
        )
        if cache_size is not None:
            @functools.lru_cache(maxsize=cache_size)
            def _cached(smiles: str) -> npt.NDArray[np.float64] | None:
                mol = parse_smiles(smiles)
                return None if mol is None else self.GetFingerprintAsNumPy(mol)
            self._cached_compute: Any = _cached
        else:
            self._cached_compute = None

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

    def _raw_fp(self, mol: Chem.Mol) -> npt.NDArray[np.float64]:
        return self._skfp_fp.transform([mol]).flatten().astype(np.float64)

    def GetRawFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[np.float64]:
        """Return the raw (uncompressed) fingerprint for a single Mol."""
        if mol is None:
            raise ValueError("mol must not be None")
        return self._raw_fp(mol)

    def GetFingerprintAsNumPy(self, mol: Chem.Mol) -> npt.NDArray[np.float64]:
        """Return the (optionally compressed) fingerprint for a single Mol."""
        if mol is None:
            raise ValueError("mol must not be None")
        raw = self._raw_fp(mol)
        if self._compression == 0:
            return raw
        # Projection methods: use cached artifacts directly
        if self._method == "hadamard":
            return compress_hadamard(raw, self._compression, self._projection_cache).flatten()
        if self._method == "random_projection":
            return compress_random_projection(raw, self._projection_cache).flatten()
        # Block-wise methods: delegate to orchestrator
        return compress_fingerprint(
            raw, self._compression,
            method=self._method, method_params=self._method_params,
        ).flatten()

    def _out_dim(self) -> int:
        return self._fp_size // self._compression if self._compression > 0 else self._fp_size

    def GetFingerprintsAsNumPy(
        self, mols: Sequence[Chem.Mol | None], chunk_size: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        """Batch fingerprints. Returns (embeddings, invalid_indices)."""
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer or None")
        total = len(mols)
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        cs = chunk_size or total
        for start in range(0, total, cs):
            for i in range(start, min(start + cs, total)):
                mol = mols[i]
                if mol is None:
                    invalid.append(i)
                    continue
                try:
                    valid.append(self.GetFingerprintAsNumPy(mol))
                except Exception:
                    invalid.append(i)
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=np.float64), invalid

    def GetFingerprintFromSmiles(
        self, smiles: str, validate: bool = True, canonicalize: bool = True,
    ) -> npt.NDArray[np.float64] | None:
        """Generate a fingerprint from a SMILES string."""
        if validate:
            if canonicalize:
                smiles = canonicalize_smiles(smiles) or ""
                if not smiles:
                    return None
            elif parse_smiles(smiles) is None:
                return None
        if self._cached_compute is not None:
            return self._cached_compute(smiles)
        mol = parse_smiles(smiles)
        return None if mol is None else self.GetFingerprintAsNumPy(mol)

    def GetFingerprintsFromSmiles(
        self, smiles_list: Sequence[str],
        validate: bool = True, canonicalize: bool = True,
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        """Batch fingerprints from SMILES. Returns (embeddings, invalid_indices)."""
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        for i, smi in enumerate(smiles_list):
            try:
                r = self.GetFingerprintFromSmiles(smi, validate=validate, canonicalize=canonicalize)
            except Exception:
                r = None
            if r is None:
                invalid.append(i)
            else:
                valid.append(r)
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=np.float64), invalid

    def GetFingerprintFromSelfies(
        self, selfies_str: str, validate: bool = True, canonicalize: bool = True,
    ) -> npt.NDArray[np.float64] | None:
        """Generate a fingerprint from a SELFIES string."""
        import selfies as sf  # type: ignore[import-untyped]
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            return None
        if smiles is None:
            return None
        return self.GetFingerprintFromSmiles(smiles, validate=validate, canonicalize=canonicalize)

    def GetFingerprintsFromSelfies(
        self, selfies_list: Sequence[str],
        validate: bool = True, canonicalize: bool = True,
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        """Batch fingerprints from SELFIES. Returns (embeddings, invalid_indices)."""
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        for i, sel in enumerate(selfies_list):
            try:
                r = self.GetFingerprintFromSelfies(sel, validate=validate, canonicalize=canonicalize)
            except Exception:
                r = None
            if r is None:
                invalid.append(i)
            else:
                valid.append(r)
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=np.float64), invalid

    def clear_cache(self) -> None:
        """Clear the LRU cache. No-op if caching is disabled."""
        if self._cached_compute is not None:
            self._cached_compute.cache_clear()

    def cache_info(self) -> CacheInfo:
        """Return cache statistics (all zeros when caching is disabled)."""
        if self._cached_compute is not None:
            raw = self._cached_compute.cache_info()
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
        base += ")"
        return base
