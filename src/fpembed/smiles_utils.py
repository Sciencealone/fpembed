"""SMILES/molecule helper functions for the fpembed package."""

from __future__ import annotations

from rdkit import Chem


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Convert a SMILES string to an RDKit Mol object.

    Returns ``None`` for a well-formed string that RDKit cannot parse into a
    molecule. Non-string input raises ``TypeError`` rather than being coerced
    or silently misclassified as an invalid molecule.

    Args:
        smiles: A SMILES string representing a molecule.

    Returns:
        An RDKit Mol object, or ``None`` if the string is unparseable.

    Raises:
        TypeError: If ``smiles`` is not a string.

    Examples:
        >>> parse_smiles("CCO")  # ethanol
        <rdkit.Chem.rdchem.Mol object at ...>
        >>> parse_smiles("not_a_molecule") is None
        True
    """
    if not isinstance(smiles, str):
        raise TypeError(f"smiles must be a str, got {type(smiles).__name__}")
    return Chem.MolFromSmiles(smiles)


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string.

    Parses the input with RDKit and returns the canonical SMILES form.
    Returns ``None`` for a well-formed string that cannot be parsed. Non-string
    input raises ``TypeError``.

    Args:
        smiles: A SMILES string to canonicalize.

    Returns:
        The canonical SMILES string, or ``None`` if the string is unparseable.

    Raises:
        TypeError: If ``smiles`` is not a string.

    Examples:
        >>> canonicalize_smiles("OCC")
        'CCO'
        >>> canonicalize_smiles("c1ccccc1")
        'c1ccccc1'
        >>> canonicalize_smiles("invalid") is None
        True
    """
    if not isinstance(smiles, str):
        raise TypeError(f"smiles must be a str, got {type(smiles).__name__}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonicalize_to_mol(
    smiles: str,
) -> tuple[str, Chem.Mol] | tuple[None, None]:
    """Canonicalize a SMILES string and return the parsed Mol alongside it.

    Parses the input with RDKit exactly once and derives the canonical SMILES
    from that same Mol, so callers can obtain both products without a second
    parse. Returns ``(None, None)`` for a well-formed string that cannot be
    parsed. Non-string input raises ``TypeError``.

    Args:
        smiles: A SMILES string to canonicalize.

    Returns:
        A ``(canonical_smiles, mol)`` tuple, or ``(None, None)`` if the string
        is unparseable.

    Raises:
        TypeError: If ``smiles`` is not a string.

    Examples:
        >>> canonical, mol = canonicalize_to_mol("OCC")
        >>> canonical
        'CCO'
        >>> canonicalize_to_mol("invalid")
        (None, None)
    """
    if not isinstance(smiles, str):
        raise TypeError(f"smiles must be a str, got {type(smiles).__name__}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), mol
