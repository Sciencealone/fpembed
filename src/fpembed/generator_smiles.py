"""SMILES/SELFIES batch and single-molecule wrappers for the generator.

Split out of ``generator.py`` as a mixin to keep that module under the
file-size limit. ``_SmilesSelfiesMixin`` relies on attributes and methods
declared by the concrete ``EmbeddedFingerprintGenerator`` (``_l1_cache``,
``_l2_cache``, ``_dtype``, ``GetFingerprintAsNumPy``, ``_out_dim``); it defines
``__slots__ = ()`` so slotted instances gain no ``__dict__``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from fpembed.embedding_cache import _MISS
from fpembed.smiles_utils import canonicalize_to_mol, parse_smiles

if TYPE_CHECKING:
    from collections.abc import Sequence


class _SmilesSelfiesMixin:
    """SMILES/SELFIES fingerprint wrappers mixed into the generator."""

    __slots__ = ()

    def GetFingerprintFromSmiles(
        self, smiles: str, validate: bool = True, canonicalize: bool = True,
    ) -> npt.NDArray[Any] | None:
        """Generate a fingerprint from a SMILES string.

        Raw-keyed branches (canonicalize=False or validate=False) parse once and
        reuse that Mol. The default branch embeds the Mol re-parsed from the
        canonical SMILES, so differently spelled inputs for the same molecule
        embed identically for every fp_type, including order-sensitive ones.
        With caching a warm hit costs no parse.
        """
        if validate and canonicalize:
            return self._from_smiles_canonical(smiles)
        # Raw-keyed branches: the raw string is today's cache key, so consult
        # the embedding cache before validating — a cached string was already
        # validated once and validation is deterministic.
        return self._from_smiles_raw_keyed(smiles)

    @staticmethod
    def _fresh(cached: npt.NDArray[Any] | None) -> npt.NDArray[Any] | None:
        """Return a fresh writable copy of a cached embedding (None passes through).

        The cache stores an owned read-only array; callers routinely scale or
        normalize the returned vector in place, so every return hands back an
        isolated writable copy via ``np.array`` (never ``astype(copy=False)``,
        which may alias). A cached ``None`` (invalid SMILES) passes through
        untouched.
        """
        return None if cached is None else np.array(cached)

    def _compute_and_cache(self, mol: Chem.Mol | None, key: str) -> npt.NDArray[Any] | None:
        emb = None if mol is None else self.GetFingerprintAsNumPy(mol)
        if self._l2_cache is not None:
            if emb is not None:
                # Own the buffer, then freeze it so a caller's in-place write on
                # the returned copy can never poison the shared cache entry.
                emb = np.array(emb)
                emb.setflags(write=False)
            self._l2_cache.put(key, emb)
        return self._fresh(emb)

    def _from_smiles_canonical(self, smiles: str) -> npt.NDArray[Any] | None:
        # Embed the canonical-parsed Mol (canonicalize_to_mol supplies only the
        # canonical key here) to stay spelling-invariant for order-sensitive
        # fp_types.
        if self._l1_cache is None:
            canon, _ = canonicalize_to_mol(smiles)
            return None if canon is None else self.GetFingerprintAsNumPy(parse_smiles(canon))
        canon = self._l1_cache.get(smiles)
        if canon is _MISS:
            canon, _ = canonicalize_to_mol(smiles)
            self._l1_cache.put(smiles, canon)
        if canon is None:
            return None
        emb = self._l2_cache.get(canon)
        if emb is not _MISS:
            return self._fresh(emb)
        return self._compute_and_cache(parse_smiles(canon), canon)

    def _from_smiles_raw_keyed(self, smiles: str) -> npt.NDArray[Any] | None:
        if self._l2_cache is not None:
            emb = self._l2_cache.get(smiles)
            if emb is not _MISS:
                return self._fresh(emb)
        # One parse both validates and supplies the embedded Mol. An
        # unparseable string yields None, keyed under the raw string so a repeat
        # invalid input skips re-parsing.
        return self._compute_and_cache(parse_smiles(smiles), smiles)

    def GetFingerprintsFromSmiles(
        self, smiles_list: Sequence[str],
        validate: bool = True, canonicalize: bool = True,
    ) -> tuple[npt.NDArray[Any], list[int]]:
        """Batch fingerprints from SMILES. Returns (embeddings, invalid_indices)."""
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        # Only a None return (unparseable input) is invalid; computation
        # exceptions propagate.
        for i, smi in enumerate(smiles_list):
            r = self.GetFingerprintFromSmiles(smi, validate=validate, canonicalize=canonicalize)
            if r is None:
                invalid.append(i)
            else:
                valid.append(r)
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=self._dtype), invalid

    def GetFingerprintFromSelfies(
        self, selfies_str: str, validate: bool = True, canonicalize: bool = True,
    ) -> npt.NDArray[Any] | None:
        """Generate a fingerprint from a SELFIES string."""
        import selfies as sf  # type: ignore[import-untyped]
        # Only an undecodable SELFIES (DecoderError) is invalid input; any other
        # exception propagates so the batch layer never masks a computation bug.
        try:
            smiles = sf.decoder(selfies_str)
        except sf.DecoderError:
            return None
        if smiles is None:
            return None
        return self.GetFingerprintFromSmiles(smiles, validate=validate, canonicalize=canonicalize)

    def GetFingerprintsFromSelfies(
        self, selfies_list: Sequence[str],
        validate: bool = True, canonicalize: bool = True,
    ) -> tuple[npt.NDArray[Any], list[int]]:
        """Batch fingerprints from SELFIES. Returns (embeddings, invalid_indices)."""
        invalid: list[int] = []
        valid: list[npt.NDArray[np.float64]] = []
        # Only a None return (undecodable SELFIES) is invalid; computation
        # exceptions propagate.
        for i, sel in enumerate(selfies_list):
            r = self.GetFingerprintFromSelfies(sel, validate=validate, canonicalize=canonicalize)
            if r is None:
                invalid.append(i)
            else:
                valid.append(r)
        if valid:
            return np.vstack(valid), invalid
        return np.empty((0, self._out_dim()), dtype=self._dtype), invalid
