"""SMILES/molecule helper functions for the FPembed package."""

from __future__ import annotations

from rdkit import Chem
from rdkit import rdBase

# Suppress RDKit warnings for invalid SMILES (handled via return values)
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.*")


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Convert a SMILES string to an RDKit Mol object.

    Returns ``None`` for any input that RDKit cannot parse, without raising
    an exception.

    Args:
        smiles: A SMILES string representing a molecule.

    Returns:
        An RDKit Mol object, or ``None`` if the input is invalid.

    Examples:
        >>> parse_smiles("CCO")  # ethanol
        <rdkit.Chem.rdchem.Mol object at ...>
        >>> parse_smiles("not_a_molecule") is None
        True
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string.

    Parses the input with RDKit and returns the canonical SMILES form.
    Returns ``None`` for any input that cannot be parsed or canonicalized.

    Args:
        smiles: A SMILES string to canonicalize.

    Returns:
        The canonical SMILES string, or ``None`` if the input is invalid.

    Examples:
        >>> canonicalize_smiles("OCC")
        'CCO'
        >>> canonicalize_smiles("c1ccccc1")
        'c1ccccc1'
        >>> canonicalize_smiles("invalid") is None
        True
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
