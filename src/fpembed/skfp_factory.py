"""skfp fingerprint object factory — extracted from generator.py."""

from __future__ import annotations

from typing import Any

from skfp.fingerprints import (
    AtomPairFingerprint,
    AvalonFingerprint,
    ECFPFingerprint,
    ERGFingerprint,
    LaggnerFingerprint,
    LayeredFingerprint,
    LingoFingerprint,
    MAPFingerprint,
    MHFPFingerprint,
    PatternFingerprint,
    PharmacophoreFingerprint,
    PubChemFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
)

_ALLOWED_FP_PARAMS: dict[str, set[str]] = {
    "ecfp": {
        "radius", "include_chirality", "use_pharmacophoric_invariants",
        "use_bond_types", "count_simulation",
    },
    "atom_pair": {
        "min_distance", "max_distance", "use_pharmacophoric_invariants",
        "count_simulation",
    },
    "topological_torsion": {"torsion_atom_count", "count_simulation"},
    "rdkit": {
        "min_path", "max_path", "use_bond_order", "num_bits_per_feature",
        "linear_paths_only", "use_pharmacophoric_invariants",
    },
    "layered": {"min_path", "max_path", "linear_paths_only"},
    "pattern": set(),
    "avalon": set(),
    "secfp": {"radius", "min_radius", "isomeric_smiles", "kekulize"},
    "mhfp": {"radius", "min_radius", "isomeric_smiles", "sssr_rings"},
    "map": {"radius", "include_chirality", "include_duplicated_shingles"},
    "lingo": {"substring_length"},
    "pharmacophore": {"min_points", "max_points"},
    # Fixed-width backends expose no constructor parameters here.
    "pubchem": set(),
    "laggner": set(),
    "erg": {"max_path"},
}

# Backend values the factory pins that affect bit positions but cannot be set
# through fp_params. The keys are disjoint from _ALLOWED_FP_PARAMS; n_jobs and
# batch_size are excluded as execution policy.
_FIXED_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "rdkit": ("count_simulation",),
    "pharmacophore": ("variant", "use_3D"),
    "erg": ("variant",),
}


def _build_skfp(
    fp_type: str, fp_size: int, fp_params: dict, count: bool = False,
) -> Any:
    """Instantiate the appropriate skfp fingerprint object.

    Rejects fp_params keys outside the per-type allowlist. A supplied
    ``variant`` is always rejected: the output mode is selected by *count*,
    not by the caller, so every backend yields fixed-length output compatible
    with the compression pipeline. Where scikit-fingerprints validates a known
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
            include_chirality=fp_params.get("include_chirality", False),
            use_pharmacophoric_invariants=fp_params.get(
                "use_pharmacophoric_invariants", False
            ),
            use_bond_types=fp_params.get("use_bond_types", True),
            count_simulation=fp_params.get("count_simulation", False),
            count=count, n_jobs=1,
        )
    if fp_type == "atom_pair":
        return AtomPairFingerprint(
            fp_size=fp_size, min_distance=fp_params.get("min_distance", 1),
            max_distance=fp_params.get("max_distance", 30),
            use_pharmacophoric_invariants=fp_params.get(
                "use_pharmacophoric_invariants", False
            ),
            count_simulation=fp_params.get("count_simulation", True),
            count=count, n_jobs=1,
        )
    if fp_type == "topological_torsion":
        return TopologicalTorsionFingerprint(
            fp_size=fp_size,
            torsion_atom_count=fp_params.get("torsion_atom_count", 4),
            count_simulation=fp_params.get("count_simulation", True),
            count=count, n_jobs=1,
        )
    if fp_type == "rdkit":
        return RDKitFingerprint(
            fp_size=fp_size, min_path=fp_params.get("min_path", 1),
            max_path=fp_params.get("max_path", 7),
            use_bond_order=fp_params.get("use_bond_order", True),
            num_bits_per_feature=fp_params.get("num_bits_per_feature", 2),
            linear_paths_only=fp_params.get("linear_paths_only", False),
            use_pharmacophoric_invariants=fp_params.get(
                "use_pharmacophoric_invariants", False
            ),
            count_simulation=fp_params.get("count_simulation", False),
            count=count, n_jobs=1,
        )
    if fp_type == "layered":
        return LayeredFingerprint(
            fp_size=fp_size, min_path=fp_params.get("min_path", 1),
            max_path=fp_params.get("max_path", 7),
            linear_paths_only=fp_params.get("linear_paths_only", False),
            n_jobs=1,
        )
    if fp_type == "avalon":
        return AvalonFingerprint(fp_size=fp_size, count=count, n_jobs=1)
    if fp_type == "secfp":
        return SECFPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 3),
            min_radius=fp_params.get("min_radius", 1),
            isomeric_smiles=fp_params.get("isomeric_smiles", False),
            kekulize=fp_params.get("kekulize", True),
            n_jobs=1,
        )
    if fp_type == "mhfp":
        return MHFPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 3),
            min_radius=fp_params.get("min_radius", 1),
            isomeric_smiles=fp_params.get("isomeric_smiles", False),
            sssr_rings=fp_params.get("sssr_rings", True),
            variant="count" if count else "bit", n_jobs=1,
        )
    if fp_type == "map":
        return MAPFingerprint(
            fp_size=fp_size, radius=fp_params.get("radius", 2),
            include_chirality=fp_params.get("include_chirality", False),
            include_duplicated_shingles=fp_params.get(
                "include_duplicated_shingles", False
            ),
            variant="count" if count else "binary", n_jobs=1,
        )
    if fp_type == "lingo":
        return LingoFingerprint(
            fp_size=fp_size,
            substring_length=fp_params.get("substring_length", 4),
            count=count, n_jobs=1,
        )
    if fp_type == "pharmacophore":
        return PharmacophoreFingerprint(
            fp_size=fp_size, variant="folded", use_3D=False,
            min_points=fp_params.get("min_points", 2),
            max_points=fp_params.get("max_points", 3),
            count=count, n_jobs=1,
        )
    if fp_type == "pubchem":
        return PubChemFingerprint(count=False, n_jobs=1)
    if fp_type == "laggner":
        return LaggnerFingerprint(count=False, n_jobs=1)
    if fp_type == "erg":
        return ERGFingerprint(
            variant="bit", max_path=fp_params.get("max_path", 15), n_jobs=1
        )
    return PatternFingerprint(fp_size=fp_size, n_jobs=1)
