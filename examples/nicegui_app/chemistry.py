"""Molecular chemistry helpers: SMILES, descriptors, and image rendering."""

import base64
import logging
from io import BytesIO
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import rdBase
from tqdm import tqdm

from fpembed import canonicalize_smiles as _canonical_smiles

# Disable RDKit warnings
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert a SMILES string to its canonical form using RDKit.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        Canonical SMILES string, or None if the input is invalid.
    """
    return _canonical_smiles(smiles)


def smiles_to_formula(smiles: str) -> str:
    """Convert a SMILES string to a molecular formula.

    Args:
        smiles: SMILES string encoding a molecular structure.

    Returns:
        Molecular formula string (e.g. "C6H12O6"), or "N/A" if unparseable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "N/A"
    return CalcMolFormula(mol)


def smiles_to_image_data_uri(smiles: str) -> str:
    """Convert a SMILES string to a base64-encoded PNG data URI.

    Args:
        smiles: SMILES string encoding a molecular structure.

    Returns:
        A data URI string ``"data:image/png;base64,..."`` (200x150 px),
        or ``"N/A"`` if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "N/A"
    img = Draw.MolToImage(mol, size=(200, 150))
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def custom_descriptors(mol, descriptor_list: list, missingVal) -> dict:
    """Calculate a custom list of molecular descriptors for a molecule.

    Args:
        mol: RDKit Mol object.
        descriptor_list: List of descriptor names to calculate.
        missingVal: Value to use when descriptor calculation fails.

    Returns:
        Dictionary mapping descriptor names to calculated values.
    """
    result = {}
    for name, function in Descriptors._descList:
        if name in descriptor_list:
            try:
                val = function(mol)
            except Exception:
                import traceback
                traceback.print_exc()
                val = missingVal
            result[name] = val
    return result


def get_custom_descriptors(molecular, case: str, descriptor_list: list) -> pd.DataFrame:
    """Calculate all descriptors for a list of molecules.

    Args:
        molecular: List of SMILES strings or RDKit molecule objects.
        case: 'smiles' or 'mol_object'.
        descriptor_list: List of descriptor names to calculate.

    Returns:
        DataFrame where each row is a molecule and each column a descriptor.
    """
    if case == "smiles":
        mols = [Chem.MolFromSmiles(smi) for smi in molecular]
        all_desc = [custom_descriptors(m, descriptor_list, 0) for m in tqdm(mols, desc="Mols")]
    elif case == "mol_object":
        all_desc = [custom_descriptors(m, descriptor_list, 0) for m in tqdm(molecular, desc="Mols")]
    else:
        raise ValueError(f"Unknown case: {case!r}. Use 'smiles' or 'mol_object'.")
    return pd.DataFrame(all_desc)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all columns of a DataFrame to the range [0, 1].

    Args:
        df: Input DataFrame with numerical values.

    Returns:
        DataFrame with values normalized between 0 and 1.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
