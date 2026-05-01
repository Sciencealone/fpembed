"""Serialization helpers for persisting optimization results.

Converts numpy arrays/scalars and NaN values to JSON-compatible Python
types so results can be stored in ``app.storage.general`` and restored
after page reloads.
"""

from __future__ import annotations

import math
import time
from typing import Any


def _sanitize_value(val: Any) -> Any:
    """Recursively convert a value to a JSON-serializable Python type.

    * numpy ndarray  → list (recursed)
    * numpy scalar   → Python float / int
    * float NaN      → None
    * dict / list    → recursed
    * everything else → returned as-is
    """
    # --- numpy array ---
    try:
        import numpy as np

        if isinstance(val, np.ndarray):
            return [_sanitize_value(v) for v in val.tolist()]
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            f = float(val)
            return None if math.isnan(f) else f
        if isinstance(val, np.bool_):
            return bool(val)
    except ImportError:
        pass

    # --- plain float NaN ---
    if isinstance(val, float) and math.isnan(val):
        return None

    # --- containers ---
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(v) for v in val]

    return val


# Keys that are shared across all trials (stored once in shared_data)
_SHARED_KEYS = frozenset({
    "smiles_train", "smiles_val",
    "actuals_train", "actuals_val",
})

# Keys that belong to per-trial data
_TRIAL_KEYS = frozenset({
    "trial_number", "fp_type", "fp_size", "fp_params",
    "compression", "compression_method", "method_params",
    "use_descriptors", "n_estimators", "max_depth", "min_samples_split",
    "R2_train", "R2_val", "MAPE_train", "MAPE_val",
    "MSE_train", "MSE_val", "training_time",
    "predictions_train", "predictions_val",
})


def serialize_results(
    all_results: list[dict],
    widget_config: dict | None = None,
    **metadata: Any,
) -> dict:
    """Package and sanitize optimization results in normalized format.

    Produces a ``shared_data`` section (SMILES, actuals, dataset metadata,
    widget config, reproducibility) stored once, plus a ``trials`` list
    with per-trial data, and a top-level ``metric`` key.

    Parameters
    ----------
    all_results:
        List of per-trial result dicts (may contain numpy arrays/scalars).
    widget_config:
        Optional dict of widget configuration values used for the run.
    **metadata:
        Keyword arguments stored alongside results.  Expected keys:
        ``ds_name``, ``metric``, ``target_col``, ``train_ratio``,
        ``reproducibility``.

    Returns
    -------
    dict
        A normalized JSON-serializable dict ready for
        ``app.storage.general["optimization_results"]``.
    """
    # --- Build shared_data from first trial + metadata ---
    shared: dict[str, Any] = {}
    if all_results:
        first = all_results[0]
        for key in _SHARED_KEYS:
            if key in first:
                shared[key] = _sanitize_value(first[key])

    # Metadata fields that belong in shared_data
    for key in ("ds_name", "target_col", "train_ratio"):
        if key in metadata:
            shared[key] = _sanitize_value(metadata.pop(key))

    shared["widget_config"] = _sanitize_value(widget_config or {})

    if "reproducibility" in metadata:
        shared["reproducibility"] = _sanitize_value(metadata.pop("reproducibility"))

    # --- Build per-trial list ---
    trials: list[dict] = []
    for r in all_results:
        trial: dict[str, Any] = {}
        for k, v in r.items():
            if k in _SHARED_KEYS:
                continue
            trial[k] = _sanitize_value(v)
        trials.append(trial)

    result: dict[str, Any] = {"shared_data": shared, "trials": trials}

    # metric lives at top level
    if "metric" in metadata:
        result["metric"] = _sanitize_value(metadata.pop("metric"))

    # Any remaining metadata at top level
    for k, v in metadata.items():
        result[k] = _sanitize_value(v)

    return result


def deserialize_results(stored: dict) -> dict:
    """Read stored results and return them in the format expected by
    render functions.

    Supports two formats:

    * **Normalized (047+)**: ``shared_data`` + ``trials`` list.
      Reconstructs full per-trial dicts by merging shared data back.
    * **Legacy (pre-047)**: ``top10_results`` flat list.
      Falls back to returning the data as-is for backward compatibility.

    Parameters
    ----------
    stored:
        The dict previously produced by :func:`serialize_results` and
        retrieved from ``app.storage.general["optimization_results"]``.

    Returns
    -------
    dict
        Keys: ``all_results`` (list of full trial dicts),
        ``widget_config``, ``ds_name``, ``metric``, ``target_col``,
        ``train_ratio``.
    """
    # --- Normalized format (047+) ---
    if "shared_data" in stored:
        shared = stored["shared_data"]
        trials_raw = stored.get("trials", [])

        all_results: list[dict] = []
        for trial in trials_raw:
            merged = dict(shared)
            # Remove non-trial keys from the merge
            merged.pop("widget_config", None)
            merged.pop("reproducibility", None)
            merged.pop("ds_name", None)
            merged.pop("target_col", None)
            merged.pop("train_ratio", None)
            merged.update(trial)
            all_results.append(merged)

        return {
            "all_results": all_results,
            "widget_config": shared.get("widget_config", {}),
            "ds_name": shared.get("ds_name"),
            "metric": stored.get("metric"),
            "target_col": shared.get("target_col"),
            "train_ratio": shared.get("train_ratio"),
        }

    # --- Legacy format (pre-047) ---
    return {
        "all_results": stored.get("top10_results", []),
        "widget_config": stored.get("widget_config", {}),
        "ds_name": stored.get("ds_name"),
        "metric": stored.get("metric"),
        "target_col": stored.get("target_col"),
        "train_ratio": stored.get("train_ratio"),
    }


def serialize_progress(
    progress_snapshot: dict,
    widget_config: dict,
    ds_name: str,
    metric: str,
    target_col: str,
) -> dict:
    """Package progress state for JSON file storage.

    Parameters
    ----------
    progress_snapshot:
        Current in-progress optimization state (trial count, elapsed
        time, top results).
    widget_config:
        Widget configuration values used for the run.
    ds_name:
        Dataset name.
    metric:
        Optimization metric name.
    target_col:
        Target column name.

    Returns
    -------
    dict
        A JSON-serializable dict with ``version: 1``, a timestamp, all
        metadata fields, and the sanitized progress and widget config.
    """
    return {
        "version": 1,
        "timestamp": time.time(),
        "ds_name": _sanitize_value(ds_name),
        "metric": _sanitize_value(metric),
        "target_col": _sanitize_value(target_col),
        "widget_config": _sanitize_value(widget_config),
        "progress": _sanitize_value(progress_snapshot),
    }


def deserialize_progress(stored: dict) -> dict:
    """Read stored progress and return in expected format.

    Parameters
    ----------
    stored:
        The dict previously produced by :func:`serialize_progress`.

    Returns
    -------
    dict
        Keys: ``progress``, ``widget_config``, ``ds_name``, ``metric``,
        ``target_col``.
    """
    return {
        "progress": stored.get("progress", {}),
        "widget_config": stored.get("widget_config", {}),
        "ds_name": stored.get("ds_name"),
        "metric": stored.get("metric"),
        "target_col": stored.get("target_col"),
    }
