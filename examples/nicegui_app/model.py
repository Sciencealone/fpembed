"""Model training, evaluation metrics, and formatting utilities."""

import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from config import calculate_n_jobs


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple:
    """Normalize features to [0, 1] using MinMaxScaler fitted on training data.

    Args:
        X_train: Training features (2D numpy array).
        X_val: Validation features (2D numpy array).

    Returns:
        tuple: (normalized_X_train, normalized_X_val)
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
    min_samples_split: float,
    random_state: int,
) -> tuple:
    """Train a Random Forest regressor and measure training time.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (or None for unlimited).
        min_samples_split: Minimum samples required to split (fraction).
        random_state: Random seed.

    Returns:
        tuple: (trained_model, training_time_seconds)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=calculate_n_jobs(),
    )
    start = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - start


def calculate_mape(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Formula: mean(|actual - predicted| / |actual|) x 100, computed only
    over samples where actual != 0.  Returns 0.0 when all actuals are zero.

    Args:
        y_actual: Array of actual values.
        y_predicted: Array of predicted values.

    Returns:
        MAPE as a percentage value (0.0 when all actuals are zero).
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)
    non_zero = y_actual != 0
    if not np.any(non_zero):
        return 0.0
    ape = np.abs((y_actual[non_zero] - y_predicted[non_zero]) / y_actual[non_zero])
    return float(np.mean(ape) * 100)


def calculate_metrics(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_time: float,
) -> dict:
    """Calculate all performance metrics for a trained model.

    Args:
        model: Trained RandomForestRegressor.
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        training_time: Time taken to train (seconds).

    Returns:
        dict with R2_train, R2_val, MAPE_train, MAPE_val,
        MSE_train, MSE_val, Training_Time.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    return {
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_val": r2_score(y_val, y_val_pred),
        "MAPE_train": calculate_mape(y_train, y_train_pred),
        "MAPE_val": calculate_mape(y_val, y_val_pred),
        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE_val": mean_squared_error(y_val, y_val_pred),
        "Training_Time": training_time,
    }


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric values for display.

    R2/MSE -> 4 decimal places; MAPE -> 2 decimal places + "%";
    Training_Time -> 2 decimal places + "s".
    """
    if metric_name == "Training_Time":
        return f"{value:.2f}s"
    elif "MAPE" in metric_name:
        return f"{value:.2f}%"
    return f"{value:.4f}"


def format_hyperparameter_value(value, param_name: str) -> str:
    """Format hyperparameter values for display.

    Args:
        value: Hyperparameter value.
        param_name: Name of the hyperparameter.

    Returns:
        Formatted string.
    """
    if value is None and param_name == "max_depth_optimized":
        return "None"
    if pd.isna(value):
        return "N/A"
    if param_name == "n_estimators_optimized":
        return str(int(value))
    elif param_name == "max_depth_optimized":
        return str(int(value))
    elif param_name == "min_samples_split_optimized":
        return f"{value:.4f}"
    return str(value)


def format_rmse_auto(value: float) -> str:
    """Format RMSE/MSE values to 4 significant digits.

    Automatically chooses between regular float notation (e.g., ``0.1234``)
    and scientific notation (e.g., ``1.23e-06``) depending on magnitude.
    """
    return f"{value:.4g}"


def format_mse_label(value: float) -> str:
    """Format MSE value for bar chart labels.

    Delegates to :func:`format_rmse_auto` for consistent formatting.
    """
    return format_rmse_auto(value)


def format_trial_log_entry(
    trial_number: int,
    metrics: dict,
    params: dict,
    use_descriptors: bool | None = None,
    compression_method: str | None = None,
) -> str:
    """Format a single trial's results for the Trial_Log.

    Args:
        trial_number: 1-based trial number.
        metrics: Dict with R2_val, MAPE_val, MSE_val keys.
        params: Dict with fp_type, fp_size, fp_params, compression,
                n_estimators, max_depth, min_samples_split.
        use_descriptors: Optional boolean; when not None, appends
                         desc=on or desc=off to the log line.
        compression_method: Optional compression method name to display.

    Returns:
        Formatted log string.
    """
    fp_label = "FP" if params.get("compression", 0) == 0 else "eFP"
    max_depth_str = (
        str(params["max_depth"]) if params.get("max_depth") is not None else "None"
    )
    fp_type = params.get("fp_type", "?")
    fp_params = params.get("fp_params", {})

    type_params_parts = [f"{k}={v}" for k, v in fp_params.items()]
    type_params_str = ", ".join(type_params_parts) if type_params_parts else ""

    rmse_str = format_rmse_auto(math.sqrt(metrics["MSE_val"]))
    line = (
        f"Trial {trial_number}: "
        f"R2_val={metrics['R2_val']:.3f}, "
        f"MAPE_val={metrics['MAPE_val']:.2f}%, "
        f"RMSE={rmse_str} | "
        f"fp_type={fp_type}, "
        f"fp_size={params['fp_size']}, "
        f"compression={params['compression']} ({fp_label})"
    )
    if type_params_str:
        line += f" | {type_params_str}"
    line += (
        f" | n_estimators={params['n_estimators']}, "
        f"max_depth={max_depth_str}, "
        f"min_samples_split={params['min_samples_split']}"
    )
    if use_descriptors is not None:
        line += f" | desc={'on' if use_descriptors else 'off'}"
    if compression_method is not None:
        line += f" | method={compression_method}"
    return line
