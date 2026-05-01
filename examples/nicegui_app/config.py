"""Configuration loading and CPU utility for the FPembed NiceGUI app."""

import os
from typing import Dict, Any

import yaml


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml.

    Returns:
        dict: Full configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml is not found.
        yaml.YAMLError: If config.yaml has invalid YAML syntax.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_n_jobs() -> int:
    """Calculate number of CPU cores to use (N-1 for N>1, else 1).

    CRITICAL: This is AUTOMATIC and NOT configurable via config files.
    Uses os.cpu_count() at runtime and applies the N-1 rule.

    Returns:
        Number of cores to use.
    """
    n_cores = os.cpu_count() or 1
    return max(1, n_cores - 1) if n_cores > 1 else 1
