"""Background optimization thread management and module-level state.

Simplified design: no SQLite run persistence,
no recovery logic, no cache.save_run_state() calls.  Module-level
_run_state survives naturally across browser interactions in NiceGUI.
"""

import logging
import threading
import time
import traceback as _tb

import pandas as pd

from dataset import stratified_split
from objective import make_progress_callback
from optimization import run_optimization, select_all_results
from progress_file import delete_progress, save_progress

logger = logging.getLogger(__name__)

_run_state: dict = {
    "thread": None,
    "stop_flag": [False],
    "progress": {},
    "status": "idle",
    "error": None,
    "widget_config": {},
    "ds_name": None,
    "metric": None,
    "target_col": None,
    "last_persist_time": 0.0,
}


def get_run_state() -> dict:
    """Return a reference to the module-level ``_run_state`` dict."""
    return _run_state


def is_optimization_running() -> bool:
    """True only when status is 'running' AND the thread is alive."""
    t = _run_state["thread"]
    return _run_state["status"] == "running" and t is not None and t.is_alive()


def _build_progress_snapshot(
    trial_number: int,
    total_trials: int,
    elapsed: float,
    top10: list[dict],
    time_limit_seconds: int,
) -> dict:
    """Build a complete progress snapshot (never partial).

    Args:
        trial_number: Count of ALL trials (completed + pruned + failed).
        total_trials: Target n_trials.
        elapsed: Seconds since optimization start.
        top10: Current ranked results (all completed trials).
        time_limit_seconds: Time limit (0 = no limit).

    Returns:
        Dict with all required progress fields.
    """
    return {
        "trial_number": trial_number,
        "total_trials": total_trials,
        "elapsed_seconds": elapsed,
        "time_limit_seconds": time_limit_seconds,
        "top10_results": list(top10),
        "timestamp": time.time(),
    }


def _maybe_persist_progress(app_dir: str, persist_interval: float) -> None:
    """Write progress to disk if enough time has elapsed since last write.

    Called from the persistence-wrapping callback after each trial.
    Only the worker thread calls this, so no locking is needed for
    ``last_persist_time``.

    Args:
        app_dir: Application directory for the progress file.
        persist_interval: Minimum seconds between writes.
    """
    now = time.time()
    if now - _run_state["last_persist_time"] < persist_interval:
        return
    progress = _run_state.get("progress")
    if not progress:
        return
    save_progress(
        app_dir=app_dir,
        progress_snapshot=progress,
        widget_config=_run_state.get("widget_config", {}),
        ds_name=_run_state.get("ds_name", ""),
        metric=_run_state.get("metric", ""),
        target_col=_run_state.get("target_col", ""),
    )
    _run_state["last_persist_time"] = now


def start_optimization_thread(
    df_sampled: pd.DataFrame,
    config: dict,
    cache,
    optimization_metric: str,
    user_fp_param_ranges: dict,
    size_bounds: tuple,
    enabled_fp_types: list[str],
    rf_bounds: dict,
    train_ratio: float,
    n_trials: int,
    time_limit_seconds: int,
    target_col: str,
    descriptors_enabled: bool = True,
    enabled_compression_methods: list[str] | None = None,
    interleave_enabled: bool = False,
    widget_config: dict | None = None,
    ds_name: str | None = None,
    metric: str | None = None,
    app_dir: str | None = None,
) -> None:
    """Stop any existing thread, reset state, spawn a new daemon thread.

    Args:
        df_sampled: Sampled DataFrame ready for splitting.
        config: Full config dict.
        cache: SQLiteCacheDB instance.
        optimization_metric: 'MSE', 'MAPE', or 'R2'.
        user_fp_param_ranges: User-edited per-FP-type parameter ranges.
        size_bounds: (min_size, max_size).
        enabled_fp_types: List of enabled FP type strings.
        rf_bounds: RF hyperparameter bounds dict.
        train_ratio: Train/validation split ratio.
        n_trials: Maximum number of trials.
        time_limit_seconds: Time limit in seconds (0 = no limit).
        target_col: Target column name.
        descriptors_enabled: Whether Optuna may toggle descriptors per trial.
        enabled_compression_methods: List of enabled compression method names.
        interleave_enabled: Whether to enable bit interleaving.
        widget_config: Widget configuration snapshot for persistence.
        ds_name: Dataset name for progress file metadata.
        metric: Optimization metric name for progress file metadata.
        app_dir: Application directory for progress file writes.
    """
    if _run_state["thread"] is not None and _run_state["thread"].is_alive():
        _run_state["stop_flag"][0] = True
        _run_state["thread"].join(timeout=5)

    _run_state["stop_flag"] = [False]
    _run_state["progress"] = {}
    _run_state["status"] = "running"
    _run_state["error"] = None
    _run_state["widget_config"] = widget_config or {}
    _run_state["ds_name"] = ds_name
    _run_state["metric"] = metric or optimization_metric
    _run_state["target_col"] = target_col
    _run_state["last_persist_time"] = 0.0

    persist_cfg = config.get("session_persistence", {})
    persist_interval = persist_cfg.get("progress_persist_interval", 30)

    kwargs = dict(
        df_sampled=df_sampled,
        config=config,
        cache=cache,
        optimization_metric=optimization_metric,
        user_fp_param_ranges=user_fp_param_ranges,
        size_bounds=size_bounds,
        enabled_fp_types=enabled_fp_types,
        rf_bounds=rf_bounds,
        train_ratio=train_ratio,
        n_trials=n_trials,
        time_limit_seconds=time_limit_seconds,
        target_col=target_col,
        descriptors_enabled=descriptors_enabled,
        enabled_compression_methods=enabled_compression_methods,
        interleave_enabled=interleave_enabled,
        app_dir=app_dir,
        persist_interval=persist_interval,
    )
    t = threading.Thread(target=_optimization_worker, kwargs=kwargs, daemon=True)
    _run_state["thread"] = t
    t.start()


def stop_optimization_thread(wait: bool = False) -> None:
    """Set stop_flag for graceful termination; optionally join with timeout.

    Args:
        wait: If True, block until the thread finishes (5 s timeout).
    """
    _run_state["stop_flag"][0] = True
    if wait and _run_state["thread"] is not None:
        _run_state["thread"].join(timeout=5)


def _optimization_worker(
    df_sampled: pd.DataFrame,
    config: dict,
    cache,
    optimization_metric: str,
    user_fp_param_ranges: dict,
    size_bounds: tuple,
    enabled_fp_types: list[str],
    rf_bounds: dict,
    train_ratio: float,
    n_trials: int,
    time_limit_seconds: int,
    target_col: str,
    descriptors_enabled: bool = True,
    enabled_compression_methods: list[str] | None = None,
    interleave_enabled: bool = False,
    app_dir: str | None = None,
    persist_interval: float = 30,
) -> None:
    """Background thread target.  MUST NOT call any NiceGUI UI APIs.

    Performs data splitting, creates the study-level progress callback,
    runs the Optuna study, and updates _run_state atomically.
    Periodically persists progress to disk when *app_dir* is set.
    """
    try:
        train_df, val_df = stratified_split(
            df_sampled,
            train_ratio=train_ratio,
            target_col=target_col,
            n_bins=config["split_defaults"]["n_bins"],
            random_seed=config["random_seed"],
        )
    except Exception as e:
        _run_state["status"] = "error"
        _run_state["error"] = {
            "message": f"Error splitting data: {e}",
            "traceback": _tb.format_exc(),
        }
        return

    start_time = time.time()
    study_holder: list = []

    base_callback = make_progress_callback(
        run_state=_run_state,
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        start_time=start_time,
        time_limit_seconds=time_limit_seconds,
    )

    def _persisting_callback(study, trial):
        """Wrap the base progress callback with periodic disk persistence."""
        base_callback(study, trial)
        if app_dir:
            _maybe_persist_progress(app_dir, persist_interval)

    try:
        study = run_optimization(
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
            time_limit_seconds=(
                time_limit_seconds if time_limit_seconds > 0 else None
            ),
            progress_callback=_persisting_callback,
            stop_flag=_run_state["stop_flag"],
            study_holder=study_holder,
            descriptors_enabled=descriptors_enabled,
            enabled_compression_methods=enabled_compression_methods,
            interleave_enabled=interleave_enabled,
        )

        final_top10 = select_all_results(study, optimization_metric)
        all_trials = len(study.trials)
        elapsed = time.time() - start_time

        _run_state["progress"] = _build_progress_snapshot(
            trial_number=all_trials,
            total_trials=n_trials,
            elapsed=elapsed,
            top10=final_top10,
            time_limit_seconds=time_limit_seconds,
        )

        _run_state["status"] = "completed"
        if app_dir:
            delete_progress(app_dir)

    except Exception as e:
        _run_state["status"] = "error"
        _run_state["error"] = {
            "message": str(e),
            "traceback": _tb.format_exc(),
        }
