"""Shared helpers for the FPembed agent skill.

Standard library plus numpy only. No network access and no pandas. Scripts import
this module with a plain ``from _common import ...`` because Python puts the
script's own directory first on ``sys.path``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

DOCS_BASE = "https://fpembed.readthedocs.io/en/latest/agent/"
UTM_AGENT = "utm_source=agent&utm_medium=skill"
MIN_VERSION = "0.3.0"
CAMPAIGNS = ("quickstart", "doctor", "featurize", "compress", "qsar", "diversity")

_PROJECTION_METHODS = ("hadamard", "random_projection")


def docs_link(context: str) -> str:
    """Return the agent docs URL for one campaign context."""
    return f"{DOCS_BASE}?{UTM_AGENT}&utm_campaign={context}"


def _json_default(value: Any) -> Any:
    """Convert numpy scalars and arrays to plain JSON types."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def emit(result: dict) -> None:
    """Write one JSON object to stdout as a single line."""
    print(json.dumps(result, default=_json_default))


def fail(message: str, context: str | None = None) -> int:
    """Write a failure message to stderr and return a non-zero exit code."""
    print(message, file=sys.stderr)
    if context is not None:
        print(docs_link(context), file=sys.stderr)
    return 1


def _module_available(name: str) -> bool:
    """Report whether an importable module exists without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _version_tuple(value: str | None) -> tuple[int, int, int] | None:
    """Parse the leading numeric parts of a dotted version string."""
    if value is None:
        return None
    parts: list[int] = []
    for piece in str(value).split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def run_doctor() -> dict:
    """Check the runtime and return a tier summary instead of exiting.

    Missing fpembed is reported in the returned dict, never raised.
    """
    version: str | None = None
    installed = False
    try:
        import fpembed

        installed = True
        version = str(getattr(fpembed, "__version__", "")) or None
    except ImportError:
        installed = False

    parsed = _version_tuple(version)
    meets_minimum = bool(parsed is not None and parsed >= _version_tuple(MIN_VERSION))
    has_numpy = _module_available("numpy")
    has_sklearn = _module_available("sklearn")

    return {
        "fpembed": {
            "installed": installed,
            "version": version,
            "meets_minimum": meets_minimum,
        },
        "numpy": has_numpy,
        "scikit_learn": has_sklearn,
        "tier": 2 if has_sklearn else 1,
        "docs": docs_link("doctor"),
    }


def seed_all(seed: int | None) -> None:
    """Seed numpy and the standard library random module from one value."""
    if seed is None:
        return
    import numpy as np

    np.random.seed(seed)
    random.seed(seed)


def projection_seed(method: str, seed: int | None) -> dict:
    """Return the seed keyword for projection methods, or an empty dict.

    FPembed takes the projection seed through ``method_params["seed"]``, not the
    global numpy RNG, so callers must forward this dict instead of relying on
    :func:`seed_all`.
    """
    if seed is not None and method in _PROJECTION_METHODS:
        return {"seed": seed}
    return {}


def read_smiles_csv(
    path: str | Path,
    smiles_col: str,
    id_col: str | None = None,
) -> tuple[list[str], list[str] | None]:
    """Read SMILES and optional IDs from a CSV with the standard library csv module."""
    structures, ids, _ = load_structures(
        path, "smiles", smiles_col=smiles_col, id_col=id_col
    )
    return structures, ids


def resolve_input_format(path: str | Path, input_format: str) -> str:
    """Return the effective input format, letting file suffixes override the flag."""
    suffix = Path(path).suffix.lower()
    if suffix in (".sdf", ".mol"):
        return "mol"
    if suffix in (".csv", ".smi"):
        if input_format == "mol":
            raise ValueError(
                f"{suffix} input cannot use mol format; use .sdf or .mol instead"
            )
        if input_format not in ("smiles", "selfies"):
            raise ValueError(f"unsupported input format {input_format!r}")
        return input_format
    return input_format


def _maybe_ids(ids: list[str] | None) -> list[str] | None:
    """Drop an ID list when every value is empty."""
    if ids is None:
        return None
    return ids if any(value.strip() for value in ids) else None


def _parse_target(value: str, label: str) -> float:
    """Convert one target cell to a float or raise a clear error."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"target {label} has non-numeric value {value!r}") from None


def _read_csv_rows(
    path: str | Path, smiles_col: str, id_col: str | None,
    target_col: str | None,
) -> tuple[list[str], list[str] | None, list[float] | None]:
    """Read string structures, optional IDs, and optional targets from a CSV."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if smiles_col not in fields:
            raise ValueError(
                f"structure column {smiles_col!r} not found; available columns: {fields}"
            )
        if id_col is not None and id_col not in fields:
            raise ValueError(f"ID column {id_col!r} not found; available columns: {fields}")
        if target_col is not None and target_col not in fields:
            raise ValueError(
                f"target column {target_col!r} not found; available columns: {fields}"
            )
        structures: list[str] = []
        ids: list[str] | None = [] if id_col is not None else None
        targets: list[float] | None = [] if target_col is not None else None
        for row_number, row in enumerate(reader, start=2):
            structures.append(row[smiles_col])
            if ids is not None:
                ids.append(row[id_col])
            if targets is not None:
                targets.append(_parse_target(
                    row[target_col], f"column {target_col!r} at row {row_number}"
                ))
    return structures, _maybe_ids(ids), targets


def _read_smi_rows(
    path: str | Path,
) -> tuple[list[str], list[str] | None, None]:
    """Read one structure per non-empty line from a .smi file."""
    structures: list[str] = []
    ids: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            structures.append(parts[0])
            ids.append(parts[1] if len(parts) > 1 else "")
    return structures, _maybe_ids(ids), None


def _read_mol_rows(
    path: str | Path, suffix: str, id_prop: str | None, target_prop: str | None,
) -> tuple[list[Any], list[str] | None, list[float | None] | None]:
    """Read RDKit Mol objects from .sdf or .mol, keeping unparseable records as None."""
    from rdkit import Chem

    if suffix == ".sdf":
        mols: list[Any] = list(Chem.SDMolSupplier(str(path), removeHs=False))
    else:
        mols = [Chem.MolFromMolFile(str(path), removeHs=False)]

    ids: list[str] = []
    targets: list[float | None] | None = [] if target_prop is not None else None
    for index, mol in enumerate(mols):
        if mol is None:
            ids.append("")
            if targets is not None:
                targets.append(None)
            continue
        ids.append(mol.GetProp(id_prop) if id_prop and mol.HasProp(id_prop) else "")
        if targets is not None:
            if not mol.HasProp(target_prop):
                raise ValueError(
                    f"molecule {index} has no {target_prop!r} property"
                )
            targets.append(_parse_target(
                mol.GetProp(target_prop), f"property {target_prop!r} at molecule {index}"
            ))
    return mols, _maybe_ids(ids), targets


def load_structures(
    path: str | Path,
    input_format: str,
    smiles_col: str = "smiles",
    id_col: str | None = None,
    target_col: str | None = None,
    target_prop: str | None = None,
    id_prop: str = "_Name",
) -> tuple[list, list[str] | None, list | None]:
    """Read structures, optional IDs, and optional targets from one input file."""
    effective = resolve_input_format(path, input_format)
    suffix = Path(path).suffix.lower()
    if effective == "mol":
        if suffix not in (".sdf", ".mol"):
            raise ValueError(
                f"mol input needs a .sdf or .mol file, not {suffix or 'a suffixless path'!r}"
            )
        return _read_mol_rows(path, suffix, id_prop, target_prop)
    if suffix == ".smi":
        return _read_smi_rows(path)
    if suffix == ".csv":
        return _read_csv_rows(path, smiles_col, id_col, target_col)
    raise ValueError(
        f"unsupported input format {suffix!r}; use .csv, .smi, .sdf, or .mol"
    )


def embed_structures(gen: Any, structures: list, input_format: str) -> tuple[Any, list[int]]:
    """Batch-embed structures with the generator method that matches the format."""
    if input_format == "mol":
        return gen.GetFingerprintsAsNumPy(structures)
    if input_format == "selfies":
        return gen.GetFingerprintsFromSelfies(structures)
    return gen.GetFingerprintsFromSmiles(structures)


def iter_structures(gen: Any, structures: list, input_format: str) -> Any:
    """Return the lazy iterator that matches the input format."""
    if input_format == "mol":
        return gen.IterFingerprints(structures)
    if input_format == "selfies":
        return gen.IterFingerprintsFromSelfies(structures)
    return gen.IterFingerprintsFromSmiles(structures)


def load_matrix(path: str | Path):
    """Load an embedding matrix from a .npy file or a numeric .csv file."""
    import numpy as np

    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported matrix format {suffix!r}; use .npy or .csv")
