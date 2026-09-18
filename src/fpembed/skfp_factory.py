"""skfp fingerprint object factory — extracted from generator.py."""

from __future__ import annotations

from typing import Any

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

_ALLOWED_FP_PARAMS: dict[str, set[str]] = {
    "ecfp": {"radius"},
    "atom_pair": {"min_distance", "max_distance"},
    "topological_torsion": {"torsion_atom_count"},
    "rdkit": {"min_path", "max_path"},
    "layered": {"min_path", "max_path"},
    "pattern": set(),
    "avalon": set(),
    "secfp": {"radius", "min_radius"},
    "mhfp": {"radius", "min_radius"},
    "map": {"radius"},
}


def _build_skfp(fp_type: str, fp_size: int, fp_params: dict) -> Any:
    """Instantiate the appropriate skfp fingerprint object.

    Rejects fp_params keys outside the per-type allowlist. A supplied
    ``variant`` is always rejected: the fixed binary/bit output mode is
    enforced so every backend yields fixed-length output compatible with
    the compression pipeline. Where scikit-fingerprints validates a known
    key's value, that error is allowed to surface.
    """
    allowed = _ALLOWED_FP_PARAMS.get(fp_type, set())
    unknown = set(fp_params) - allowed
    if unknown:
        if "variant" in unknown:
            raise ValueError(
                f"'variant' is not accepted for fp_type '{fp_type}': the fixed "
                "binary/bit output mode is enforced and cannot be overridden. "
                f"Accepted fp_params keys for '{fp_type}': {sorted(allowed)}."
            )
        raise ValueError(
            f"Unknown fp_params key(s) for fp_type '{fp_type}': "
            f"{sorted(unknown)}. Accepted keys: {sorted(allowed)}."
        )

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
