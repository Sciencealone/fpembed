"""UnifiedObjective class for Optuna optimization trials.

Bug fix: trial_callback removed from __call__ success path.
Progress counting moved to study-level callback via make_progress_callback()
which counts ALL trial states (COMPLETE + PRUNED + FAIL).
"""

import logging
import time

import numpy as np
import optuna
import pandas as pd

from fpembed import EmbeddedFingerprintGenerator
from model import normalize_features, train_random_forest, calculate_metrics
from parameters import filter_compressions_by_granularity

logger = logging.getLogger(__name__)

_FP_TYPES = [
    "ecfp", "atom_pair", "topological_torsion", "rdkit", "layered", "pattern",
    "avalon", "secfp", "mhfp", "map",
]


class UnifiedObjective:
    """Callable objective for the unified Optuna study.

    Searches the combined parameter space of fingerprint parameters
    (fp_type, fp_size, compression, per-type params) and Random Forest
    hyperparameters in a single study.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: dict,
        cache,
        optimization_metric: str,
        user_fp_param_ranges: dict | None = None,
        size_bounds: tuple = (),
        enabled_fp_types: list[str] | None = None,
        rf_bounds: dict | None = None,
        n_trials: int = 100,
        descriptors_enabled: bool = True,
        enabled_compression_methods: list[str] | None = None,
        interleave_enabled: bool = False,
    ):
        self.config = config
        self.cache = cache
        self.optimization_metric = optimization_metric
        self.granularity_pct = 100
        self.n_trials = n_trials
        self.descriptors_enabled = descriptors_enabled
        _ALL_METHODS = [
            "geometric", "linear", "log", "uniform", "hadamard", "random_projection",
        ]
        self.enabled_compression_methods = (
            enabled_compression_methods if enabled_compression_methods else _ALL_METHODS
        )
        self.interleave_enabled = interleave_enabled

        dataset_keys = list(config["datasets"].keys())
        self.target_col = config["datasets"][dataset_keys[0]]["target"]

        self.train_smiles = train_df["smiles"].tolist()
        self.val_smiles = val_df["smiles"].tolist()
        self.y_train = train_df[self.target_col].values
        self.y_val = val_df[self.target_col].values

        all_sizes = sorted(config["fp_parameters"]["sizes"])
        self.valid_sizes = [s for s in all_sizes if size_bounds[0] <= s <= size_bounds[1]]
        self.enabled_fp_types = enabled_fp_types if enabled_fp_types else list(_FP_TYPES)
        if user_fp_param_ranges:
            self.fp_param_ranges = user_fp_param_ranges
        else:
            self.fp_param_ranges = config["fp_parameters"]

        self.rf_bounds = rf_bounds
        self.min_comp = config["fp_constraints"]["min_compression"]
        self.max_divisor = config["fp_constraints"]["max_compression_divisor"]

        self.all_compressions = sorted(set().union(*(
            filter_compressions_by_granularity(
                s, self.granularity_pct, self.min_comp, self.max_divisor
            )
            for s in self.valid_sizes
        )))

        self.random_state = config.get("random_seed", 42)

    def _sample_fp_params(self, trial: optuna.Trial, fp_type: str) -> dict:
        """Sample per-FP-type parameters conditionally."""
        ranges = self.fp_param_ranges.get(fp_type, {})
        params = {}
        if fp_type == "ecfp" and "radius" in ranges:
            lo, hi = ranges["radius"]
            params["radius"] = trial.suggest_int("ecfp_radius", lo, hi)
        elif fp_type == "atom_pair":
            if "min_distance" in ranges:
                lo, hi = ranges["min_distance"]
                params["min_distance"] = trial.suggest_int("ap_min_distance", lo, hi)
            if "max_distance" in ranges:
                lo, hi = ranges["max_distance"]
                params["max_distance"] = trial.suggest_int("ap_max_distance", lo, hi)
        elif fp_type == "topological_torsion" and "torsion_atom_count" in ranges:
            lo, hi = ranges["torsion_atom_count"]
            params["torsion_atom_count"] = trial.suggest_int("tt_torsion_atom_count", lo, hi)
        elif fp_type == "rdkit":
            if "min_path" in ranges:
                lo, hi = ranges["min_path"]
                params["min_path"] = trial.suggest_int("rdkit_min_path", lo, hi)
            if "max_path" in ranges:
                lo, hi = ranges["max_path"]
                params["max_path"] = trial.suggest_int("rdkit_max_path", lo, hi)
        elif fp_type == "layered":
            if "min_path" in ranges:
                lo, hi = ranges["min_path"]
                params["min_path"] = trial.suggest_int("layered_min_path", lo, hi)
            if "max_path" in ranges:
                lo, hi = ranges["max_path"]
                params["max_path"] = trial.suggest_int("layered_max_path", lo, hi)
        elif fp_type == "secfp" and "radius" in ranges:
            lo, hi = ranges["radius"]
            params["radius"] = trial.suggest_int("secfp_radius", lo, hi)
        elif fp_type == "mhfp" and "radius" in ranges:
            lo, hi = ranges["radius"]
            params["radius"] = trial.suggest_int("mhfp_radius", lo, hi)
        elif fp_type == "map" and "radius" in ranges:
            lo, hi = ranges["radius"]
            params["radius"] = trial.suggest_int("map_radius", lo, hi)
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute a single Optuna trial.

        NOTE: trial_callback is NOT called here (bug fix).
        Progress counting is handled by the study-level callback.
        """
        try:
            fp_type = trial.suggest_categorical("fp_type", self.enabled_fp_types)
            fp_size = trial.suggest_categorical("fp_size", self.valid_sizes)
            compression = trial.suggest_categorical("compression", self.all_compressions)

            valid_for_size = filter_compressions_by_granularity(
                fp_size, self.granularity_pct, self.min_comp, self.max_divisor,
            )
            if compression not in valid_for_size:
                raise optuna.TrialPruned(
                    f"compression {compression} invalid for fp_size {fp_size}"
                )

            fp_params = self._sample_fp_params(trial, fp_type)

            if compression > 0:
                compression_method = trial.suggest_categorical(
                    "compression_method", self.enabled_compression_methods,
                )
                _BLOCKWISE = {"geometric", "linear", "log", "uniform"}
                method_params = {}
                if compression_method in _BLOCKWISE:
                    method_params["interleave"] = self.interleave_enabled
                else:
                    method_params["seed"] = self.random_state

                # Hadamard requires fp_size to be power of 2
                if compression_method == "hadamard" and (
                    fp_size <= 0 or (fp_size & (fp_size - 1)) != 0
                ):
                    raise optuna.TrialPruned(
                        f"Hadamard requires fp_size to be power of 2, got {fp_size}"
                    )
            else:
                compression_method = "none"
                method_params = {}

            generator = EmbeddedFingerprintGenerator(
                fp_type=fp_type,
                fp_size=fp_size,
                compression=compression,
                fp_params=fp_params,
                method=compression_method if compression > 0 else "geometric",
                method_params=method_params,
            )

            n_est_min, n_est_max = self.rf_bounds["n_estimators"]
            n_estimators = trial.suggest_int("n_estimators", n_est_min, n_est_max)

            md_min, md_max, include_none = self.rf_bounds["max_depth"]
            if include_none:
                use_none = trial.suggest_categorical("max_depth_none", [True, False])
                max_depth = None if use_none else trial.suggest_int("max_depth", md_min, md_max)
            else:
                max_depth = trial.suggest_int("max_depth", md_min, md_max)

            mss_min, mss_max = self.rf_bounds["min_samples_split"]
            min_samples_split = trial.suggest_float("min_samples_split", mss_min, mss_max)

            if self.descriptors_enabled:
                use_descriptors = trial.suggest_categorical(
                    "use_descriptors", [True, False],
                )
            else:
                use_descriptors = False

            fp_train = self.cache.get_fingerprints_batch(self.train_smiles, generator)
            fp_val = self.cache.get_fingerprints_batch(self.val_smiles, generator)

            if use_descriptors:
                desc_train = self.cache.get_descriptors_batch(self.train_smiles)
                desc_val = self.cache.get_descriptors_batch(self.val_smiles)
                X_train = np.hstack([desc_train.values, fp_train])
                X_val = np.hstack([desc_val.values, fp_val])
            else:
                X_train = fp_train
                X_val = fp_val

            X_train_norm, X_val_norm = normalize_features(X_train, X_val)

            model, training_time = train_random_forest(
                X_train_norm, self.y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
            )

            metrics = calculate_metrics(
                model, X_train_norm, self.y_train, X_val_norm, self.y_val, training_time,
            )

            predictions_train = model.predict(X_train_norm)
            predictions_val = model.predict(X_val_norm)

            for key, val in [
                ("predictions_train", predictions_train.tolist()),
                ("predictions_val", predictions_val.tolist()),
                ("actuals_train", self.y_train.tolist()),
                ("actuals_val", self.y_val.tolist()),
                ("smiles_train", self.train_smiles),
                ("smiles_val", self.val_smiles),
                ("metrics", metrics),
                ("fp_type", fp_type),
                ("fp_size", fp_size),
                ("fp_params", fp_params),
                ("compression", compression),
                ("n_estimators", n_estimators),
                ("max_depth", max_depth),
                ("min_samples_split", min_samples_split),
                ("use_descriptors", use_descriptors),
                ("compression_method", compression_method),
                ("method_params", method_params),
            ]:
                trial.set_user_attr(key, val)

            return metrics[f"{self.optimization_metric}_val"]

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return float("-inf") if self.optimization_metric == "R2" else float("inf")


def make_progress_callback(
    run_state: dict,
    n_trials: int,
    optimization_metric: str,
    start_time: float,
    time_limit_seconds: int,
) -> callable:
    """Create an Optuna study-level callback that fires after EVERY trial.

    Counts all trials (COMPLETE + PRUNED + FAIL) via len(study.trials)
    for accurate progress. Builds a progress snapshot and updates
    run_state atomically via single reference assignment.

    Args:
        run_state: Module-level _run_state dict to update.
        n_trials: Target number of trials.
        optimization_metric: Metric name for result ranking.
        start_time: time.time() when optimization started.
        time_limit_seconds: Time limit in seconds (0 = no limit).

    Returns:
        Callback function with signature (study, trial).
    """
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        from optimization import select_all_results

        all_trial_count = len(study.trials)
        all_results = select_all_results(study, optimization_metric)

        snapshot = {
            "trial_number": all_trial_count,
            "total_trials": n_trials,
            "elapsed_seconds": time.time() - start_time,
            "time_limit_seconds": time_limit_seconds,
            "top10_results": all_results,
            "timestamp": time.time(),
        }
        run_state["progress"] = snapshot

    return callback
