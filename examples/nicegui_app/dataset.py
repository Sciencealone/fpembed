"""Dataset loading, filtering, and stratified sampling/splitting."""

import logging
import os
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from compressed CSV file.

    Args:
        dataset_name: Name of dataset (RedDB, NFA, QM9).
        config: Configuration dictionary containing dataset paths.

    Returns:
        pd.DataFrame with loaded and SMILES-canonicalized data.

    Raises:
        ValueError: If dataset_name is unknown or all SMILES are invalid.
        FileNotFoundError: If dataset file is not found.
        KeyError: If required columns (smiles, homo, lumo) are missing.
    """
    from chemistry import canonicalize_smiles

    if dataset_name not in config["datasets"]:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(config['datasets'].keys())}"
        )

    dataset_config = config["datasets"][dataset_name]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, config["paths"]["datasets_dir"])
    file_path = os.path.join(datasets_dir, dataset_config["file"])

    try:
        df = pd.read_csv(file_path, compression="gzip")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Invalid CSV format in {file_path}: {e}")

    required_columns = ["smiles", "homo", "lumo"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {dataset_name}: {missing}")

    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    invalid_mask = df["smiles"].isna()
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.warning(
            "Dropped %d molecules with invalid SMILES from %s",
            n_invalid, dataset_name,
        )
        df = df[~invalid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"All SMILES in {dataset_name} are invalid — no molecules to process."
        )

    return df


def filter_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Filter molecules where lumo > homo and add gap column if missing.

    Args:
        df: Input DataFrame with 'homo' and 'lumo' columns.

    Returns:
        Filtered DataFrame with 'gap' column.
    """
    if "gap" not in df.columns:
        df = df.copy()
        df["gap"] = df["lumo"] - df["homo"]
    return df[df["lumo"] > df["homo"]].copy()


def stratified_sample(
    df: pd.DataFrame,
    percentage: float,
    target_col: str,
    n_bins: int,
    random_seed: int,
) -> pd.DataFrame:
    """Perform stratified sampling based on target value distribution.

    Args:
        df: Input DataFrame.
        percentage: Percentage to sample (1-100).
        target_col: Target column name for stratification.
        n_bins: Number of bins for stratification.
        random_seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    if percentage >= 100:
        return df.copy()

    df = df.copy()
    try:
        df["_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["_bin"] = pd.cut(df[target_col], bins=n_bins, labels=False)

    total_size = len(df)
    sample_size = int(np.ceil(total_size * percentage / 100))

    sampled_dfs = []
    for bin_id in df["_bin"].unique():
        bin_df = df[df["_bin"] == bin_id]
        bin_sample_size = min(
            int(np.ceil(sample_size * len(bin_df) / total_size)),
            len(bin_df),
        )
        sampled_dfs.append(bin_df.sample(n=bin_sample_size, random_state=random_seed))

    return pd.concat(sampled_dfs, ignore_index=True).drop(columns=["_bin"])


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    target_col: str,
    n_bins: int,
    random_seed: int,
) -> tuple:
    """Perform stratified train/validation split.

    Args:
        df: Input DataFrame.
        train_ratio: Fraction for training (0.5-0.95).
        target_col: Target column for stratification.
        n_bins: Number of bins for stratification.
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (train_df, val_df)
    """
    df = df.copy()
    try:
        df["_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["_bin"] = pd.cut(df[target_col], bins=n_bins, labels=False)

    train_dfs, val_dfs = [], []
    for bin_id in df["_bin"].unique():
        bin_df = df[df["_bin"] == bin_id]
        bin_train_size = int(np.floor(len(bin_df) * train_ratio))
        if bin_train_size == len(bin_df) and len(bin_df) > 1:
            bin_train_size = len(bin_df) - 1

        shuffled = bin_df.sample(frac=1, random_state=random_seed)
        train_dfs.append(shuffled.iloc[:bin_train_size])
        if len(shuffled.iloc[bin_train_size:]) > 0:
            val_dfs.append(shuffled.iloc[bin_train_size:])

    train_df = pd.concat(train_dfs, ignore_index=True).drop(columns=["_bin"])
    val_df = (
        pd.concat(val_dfs, ignore_index=True).drop(columns=["_bin"])
        if val_dfs
        else pd.DataFrame()
    )
    return train_df, val_df
