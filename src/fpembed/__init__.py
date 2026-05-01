"""FPembed — Generalized fingerprint embedding library backed by scikit-fingerprints.

Provides :class:`EmbeddedFingerprintGenerator` for generating compressed
molecular fingerprints from SMILES, SELFIES, or RDKit Mol objects, plus
the standalone :func:`compress_fingerprint` compression function and
lightweight SMILES helpers.

Example:
    >>> from fpembed import EmbeddedFingerprintGenerator
    >>> gen = EmbeddedFingerprintGenerator(
    ...     fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}
    ... )
    >>> emb = gen.GetFingerprintFromSmiles("CCO")
    >>> emb.shape
    (128,)
"""

from fpembed.generator import EmbeddedFingerprintGenerator
from fpembed.compression import compress_fingerprint
from fpembed.smiles_utils import parse_smiles, canonicalize_smiles
from fpembed.hashing import fp_params_hash

__version__ = "0.1.3"

__all__ = [
    "EmbeddedFingerprintGenerator",
    "compress_fingerprint",
    "parse_smiles",
    "canonicalize_smiles",
    "fp_params_hash",
    "__version__",
]
