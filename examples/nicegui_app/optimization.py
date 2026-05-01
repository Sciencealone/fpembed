"""Optimization runner, result selection, DataFrame builder, and JSON export."""

import json
import logging
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from objective import UnifiedObjective

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.INFO)


def run_optimization(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    cache,
    optimization_metric: str,
    user_fp_param_ranges: dict,
    size_bounds: tuple,
    enabled_fp_types: list[str],
    rf_bounds: dict,
    n_trials: int,
    time_limit_seconds: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    stop_flag: Optional[list] = None,
    study_holder: Optional[list] = None,
    descriptors_enabled: bool = True,
    enabled_compression_methods: list[str] | None = None,
    interleave_enabled: bool = False,
) -> optuna.Study:
    """Create and run the unified Optuna study.

    Args:
        train_df: Training data.
        val_df: Validation data.
        config: Full config dict.
        cache: SQLiteCacheDB instance.
        optimization_metric: 'MSE', 'MAPE', or 'R2'.
        user_fp_param_ranges: User-edited per-FP-type parameter ranges dict.
        size_bounds: (min_size, max_size).
        enabled_fp_types: List of enabled FP type strings.
        rf_bounds: RF hyperparameter bounds dict.
        n_trials: Maximum number of trials.
        time_limit_seconds: Max time in seconds (None or 0 = no limit).
        progress_callback: Study-level callback from make_progress_callback().
        stop_flag: Mutable list [bool]; set [True] to stop after current trial.
        study_holder: Optional mutable list to receive study object early.
        descriptors_enabled: Whether Optuna may toggle descriptors per trial.
        enabled_compression_methods: List of enabled compression method names.
        interleave_enabled: Whether to enable bit interleaving for block-wise methods.

    Returns:
        The completed Optuna study object.
    """
    direction = "maximize" if optimization_metric == "R2" else "minimize"
    study = optuna.create_study(direction=direction)

    if study_holder is not None:
        study_holder.clear()
        study_holder.append(study)

    objective = UnifiedObjective(
        train_df=train_df,
        val_df=val_df,
        config=config,
        cache=cache,
        optimization_metric=optimization_metric,
        user_fp_param_ranges=user_fp_param_ranges,
        size_bounds=size_bounds,
        enabled_fp_types=enabled_fp_types,
        rf_bounds=rf_bounds,
        n_trials=n_trials,
        descriptors_enabled=descriptors_enabled,
        enabled_compression_methods=enabled_compression_methods,
        interleave_enabled=interleave_enabled,
    )

    callbacks = []

    if progress_callback is not None:
        callbacks.append(progress_callback)

    if time_limit_seconds and time_limit_seconds > 0:
        import time as _time
        _start = _time.time()

        def _time_limit_callback(study, trial):
            if _time.time() - _start >= time_limit_seconds:
                study.stop()

        callbacks.append(_time_limit_callback)

    if stop_flag is not None:
        def _stop_flag_callback(study, trial):
            if stop_flag[0]:
                study.stop()

        callbacks.append(_stop_flag_callback)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=callbacks)
    return study


def select_all_results(
    study: optuna.Study,
    optimization_metric: str,
) -> list[dict]:
    """Select ALL completed trials ranked by the chosen metric.

    For R2 the best trials have the highest values; for MAPE and MSE
    the best trials have the lowest values.

    Args:
        study: Completed Optuna study object.
        optimization_metric: One of 'R2', 'MAPE', or 'MSE'.

    Returns:
        List of all completed trial_result dicts, ordered best-to-worst.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    metric_key = f"{optimization_metric}_val"
    reverse = optimization_metric == "R2"

    completed.sort(
        key=lambda t: t.user_attrs.get("metrics", {}).get(
            metric_key, float("-inf") if reverse else float("inf")
        ),
        reverse=reverse,
    )

    results = []
    for t in completed:
        metrics = t.user_attrs.get("metrics", {})
        results.append({
            "trial_number": t.number,
            "fp_type": t.user_attrs.get("fp_type"),
            "fp_size": t.user_attrs.get("fp_size"),
            "fp_params": t.user_attrs.get("fp_params", {}),
            "compression": t.user_attrs.get("compression"),
            "n_estimators": t.user_attrs.get("n_estimators"),
            "max_depth": t.user_attrs.get("max_depth"),
            "min_samples_split": t.user_attrs.get("min_samples_split"),
            "use_descriptors": t.user_attrs.get("use_descriptors"),
            "compression_method": t.user_attrs.get("compression_method"),
            "method_params": t.user_attrs.get("method_params", {}),
            "R2_train": metrics.get("R2_train"),
            "R2_val": metrics.get("R2_val"),
            "MAPE_train": metrics.get("MAPE_train"),
            "MAPE_val": metrics.get("MAPE_val"),
            "MSE_train": metrics.get("MSE_train"),
            "MSE_val": metrics.get("MSE_val"),
            "training_time": metrics.get("Training_Time"),
            "predictions_train": np.array(t.user_attrs.get("predictions_train", [])),
            "predictions_val": np.array(t.user_attrs.get("predictions_val", [])),
            "actuals_train": np.array(t.user_attrs.get("actuals_train", [])),
            "actuals_val": np.array(t.user_attrs.get("actuals_val", [])),
            "smiles_train": t.user_attrs.get("smiles_train", []),
            "smiles_val": t.user_attrs.get("smiles_val", []),
        })
    return results


def build_results_dataframe(top10_results: list[dict]) -> pd.DataFrame:
    """Build a display DataFrame from top-10 results for the Results_Table."""
    rows = []
    for r in top10_results:
        compression = r.get("compression", 0)
        use_desc = r.get("use_descriptors")
        rows.append({
            "Trial": r.get("trial_number"),
            "Type": "FP" if compression == 0 else "eFP",
            "FP Type": r.get("fp_type", "?"),
            "Descriptors": "On" if use_desc else "Off",
            "Method": r.get("compression_method", "none"),
            "Size": r.get("fp_size"),
            "Compression": compression,
            "n_estimators": r.get("n_estimators"),
            "max_depth": r.get("max_depth"),
            "min_samples_split": r.get("min_samples_split"),
            "R2_val": r.get("R2_val"),
            "MAPE_val": r.get("MAPE_val"),
            "MSE_val": r.get("MSE_val"),
        })
    return pd.DataFrame(rows)


def export_results_json(
    results: list[dict],
    reproducibility: dict | None = None,
) -> str:
    """Serialize top-10 results to a valid JSON string.

    SMILES lists (smiles_train, smiles_val) are identical across all trials
    (same train/val split), so they are hoisted into a shared top-level
    ``shared_data`` section to avoid duplication.

    Args:
        results: List of trial_result dicts.
        reproducibility: Optional reproducibility metadata dict.

    Returns:
        JSON string.
    """
    # Keys that are identical across trials — write once at top level
    _SHARED_KEYS = {"smiles_train", "smiles_val", "actuals_train", "actuals_val"}

    # Extract shared data from the first result
    shared_data: dict = {}
    if results:
        first = results[0]
        for key in _SHARED_KEYS:
            val = first.get(key)
            if val is not None:
                shared_data[key] = val.tolist() if isinstance(val, np.ndarray) else val

    serializable = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if k in _SHARED_KEYS:
                continue
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif isinstance(v, np.integer):
                entry[k] = int(v)
            elif isinstance(v, np.floating):
                entry[k] = float(v)
            else:
                entry[k] = v
        serializable.append(entry)

    payload: dict = {}
    if reproducibility is not None:
        payload["reproducibility"] = reproducibility
    if shared_data:
        payload["shared_data"] = shared_data
    payload["trials"] = serializable

    return json.dumps(payload, indent=2)
