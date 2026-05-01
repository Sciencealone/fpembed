"""Helper functions for app.py — dataset caching and molecule counting."""

from __future__ import annotations

import logging

import pandas as pd

from nicegui import app

from dataset import filter_molecules, load_dataset, stratified_sample
from optimization import build_results_dataframe

logger = logging.getLogger(__name__)

_dataset_cache: dict[str, pd.DataFrame] = {}

def _load_and_filter_dataset(ds_name: str, config: dict) -> pd.DataFrame:
    """Pure, picklable: load dataset + filter molecules, return DataFrame."""
    df = load_dataset(ds_name, config)
    df = filter_molecules(df)
    return df

def _sample_and_count(df: pd.DataFrame, pct: float, target_col: str, config: dict) -> int:
    """Pure, picklable: stratified-sample the DataFrame and return count."""
    if pct < 100:
        df = stratified_sample(
            df, pct, target_col,
            n_bins=config["subset_defaults"]["n_bins"],
            random_seed=config["random_seed"],
        )
    return len(df)

def _compute_molecule_count(ds_name: str, config: dict, pct: float) -> int:
    """Pure, picklable function for run.cpu_bound() — loads and counts molecules."""
    df = _load_and_filter_dataset(ds_name, config)
    target_col = config["datasets"][ds_name]["target"]
    return _sample_and_count(df, pct, target_col, config)

def _apply_filters(
    all_rows: list[dict],
    filter_state: dict[str, set[str]],
) -> list[dict]:
    """Return rows matching all column filters (AND logic)."""
    return [
        row for row in all_rows
        if all(str(row.get(c, "")) in a for c, a in filter_state.items())
    ]

def _update_table_rows(
    table_state: dict,
    new_results: list[dict],
    metric: str,
) -> None:
    """Update the results table with new trial data, preserving UI state."""
    display_df = build_results_dataframe(new_results)
    metric_col = f"{metric}_val"
    if metric_col in display_df.columns:
        display_df = display_df.sort_values(
            by=metric_col, ascending=(metric != "R2"),
        )

    # Build prepared rows (rounded floats, None → "None")
    rows = display_df.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 6)
            elif v is None:
                row[k] = "None"

    # Update master row list
    table_state["all_rows"] = rows

    # Detect new categorical values and add them as checked
    filter_state = table_state["filter_state"]
    for col in table_state["categorical_cols"]:
        existing = filter_state.get(col, set())
        for row in rows:
            val = str(row.get(col, ""))
            if val not in existing:
                existing.add(val)
        filter_state[col] = existing

    # Apply current filters and update table rows (preserves UI state)
    filtered = _apply_filters(rows, filter_state)
    table_state["table"].update_rows(filtered)
    # Refresh filter dropdown templates so new values appear as checkboxes
    from ui_results import _refresh_filter_slots
    _refresh_filter_slots(table_state["table"], table_state)
    persist_table_ui_state(table_state)

def _set_widgets_enabled(w: dict, enabled: bool) -> None:
    """Enable or disable all parameter widgets during optimization."""
    m = "enable" if enabled else "disable"
    for key in ("dataset_select", "subset_slider", "min_size_select",
                "max_size_select", "ne_min", "ne_max", "md_min", "md_max",
                "md_none_checkbox", "mss_min", "mss_max", "train_slider",
                "metric_select", "trials_slider", "time_slider",
                "descriptor_checkbox"):
        wgt = w.get(key)
        if wgt:
            getattr(wgt, m)()
    for cb in w.get("fp_checkboxes", {}).values():
        getattr(cb, m)()
    for cb in w.get("compression_checkboxes", {}).values():
        getattr(cb, m)()
    interleave_cb = w.get("interleave_checkbox")
    if interleave_cb:
        getattr(interleave_cb, m)()
    for fp_params in w.get("fp_param_inputs", {}).values():
        for minmax in fp_params.values():
            getattr(minmax["min"], m)()
            getattr(minmax["max"], m)()

def _restore_widget_config(w: dict, widget_config: dict, config: dict | None = None) -> None:
    """Restore widget values from a stored widget_config dict.

    Skips missing keys gracefully for backward compatibility with
    stored data that lacks widget_config.
    """
    if not widget_config:
        return

    _simple_keys = {
        "dataset_name": "dataset_select",
        "subset_pct": "subset_slider",
        "min_size": "min_size_select",
        "max_size": "max_size_select",
        "ne_min": "ne_min",
        "ne_max": "ne_max",
        "md_min": "md_min",
        "md_max": "md_max",
        "md_none": "md_none_checkbox",
        "mss_min": "mss_min",
        "mss_max": "mss_max",
        "train_ratio": "train_slider",
        "metric": "metric_select",
        "trials": "trials_slider",
        "time_hours": "time_slider",
    }
    for cfg_key, widget_key in _simple_keys.items():
        if cfg_key in widget_config and widget_key in w:
            w[widget_key].value = widget_config[cfg_key]

    # Restore descriptor checkbox — fallback reads from config.yaml, not hardcoded
    if "descriptor_checkbox" in w:
        _desc_fallback = config.get("descriptors_enabled", False) if config else False
        w["descriptor_checkbox"].value = widget_config.get(
            "descriptors_enabled", _desc_fallback)

    fp_enabled = widget_config.get("fp_enabled", {})
    for fp_key, enabled in fp_enabled.items():
        cb = w.get("fp_checkboxes", {}).get(fp_key)
        if cb is not None:
            cb.value = enabled

    cm_enabled = widget_config.get("compression_methods_enabled", {})
    for method_key, enabled in cm_enabled.items():
        cb = w.get("compression_checkboxes", {}).get(method_key)
        if cb is not None:
            cb.value = enabled
    if "interleave" in widget_config and "interleave_checkbox" in w:
        w["interleave_checkbox"].value = widget_config["interleave"]

    fp_ranges = widget_config.get("fp_param_ranges", {})
    for fp_key, params in fp_ranges.items():
        fp_inputs = w.get("fp_param_inputs", {}).get(fp_key, {})
        for pname, bounds in params.items():
            mm = fp_inputs.get(pname)
            if mm is None:
                continue
            if "min" in bounds and "min" in mm:
                mm["min"].value = bounds["min"]
            if "max" in bounds and "max" in mm:
                mm["max"].value = bounds["max"]

def _collect_widget_config(w: dict) -> dict:
    """Extract current widget values into a JSON-serializable config dict."""
    return {
        "dataset_name": w["dataset_select"].value,
        "subset_pct": w["subset_slider"].value,
        "descriptors_enabled": w["descriptor_checkbox"].value,
        "fp_enabled": {
            fp_key: cb.value
            for fp_key, cb in w["fp_checkboxes"].items()
        },
        "compression_methods_enabled": {
            method_key: cb.value
            for method_key, cb in w.get("compression_checkboxes", {}).items()
        },
        "interleave": w["interleave_checkbox"].value if "interleave_checkbox" in w else False,
        "min_size": w["min_size_select"].value,
        "max_size": w["max_size_select"].value,
        "fp_param_ranges": {
            fp_key: {
                pname: {"min": mm["min"].value, "max": mm["max"].value}
                for pname, mm in params.items()
            }
            for fp_key, params in w["fp_param_inputs"].items()
        },
        "ne_min": w["ne_min"].value,
        "ne_max": w["ne_max"].value,
        "md_min": w["md_min"].value,
        "md_max": w["md_max"].value,
        "md_none": w["md_none_checkbox"].value,
        "mss_min": w["mss_min"].value,
        "mss_max": w["mss_max"].value,
        "train_ratio": w["train_slider"].value,
        "metric": w["metric_select"].value,
        "trials": w["trials_slider"].value,
        "time_hours": w["time_slider"].value,
    }

def _build_default_widget_config(config: dict) -> dict:
    """Extract default widget values from the config dict (config.yaml).

    Returns a dict in the same format as _collect_widget_config, but with
    values sourced from config.yaml defaults instead of current widget state.
    """
    hp = config["optuna_settings"]["hyperparameter_ranges"]
    fp_params = config["fp_parameters"]
    sizes = fp_params["sizes"]

    fp_type_keys = [k for k in fp_params if k != "sizes"]
    fp_defaults = config.get("fp_defaults", {})
    fp_enabled = {
        k: fp_defaults.get(k, {}).get("enabled", True)
        for k in fp_type_keys
    }

    cm_defaults = config.get("compression_methods", {})
    _CM_KEYS = ["geometric", "linear", "log", "uniform", "hadamard", "random_projection"]
    compression_methods_enabled = {
        k: cm_defaults.get(k, {}).get("enabled", True)
        for k in _CM_KEYS
    }
    interleave_default = cm_defaults.get("interleave_default", False)

    fp_param_ranges = {
        k: {
            pname: {"min": bounds[0], "max": bounds[1]}
            for pname, bounds in fp_params[k].items()
        }
        for k in fp_type_keys
        if fp_params[k]
    }

    include_none_depth = (
        config.get("optuna_settings", {})
        .get("hyperparameter_ranges", {})
        .get("include_none_depth", True)
    )

    return {
        "dataset_name": next(iter(config["datasets"])),
        "subset_pct": config["subset_defaults"]["percentage"],
        "descriptors_enabled": config.get("descriptors_enabled", False),
        "fp_enabled": fp_enabled,
        "compression_methods_enabled": compression_methods_enabled,
        "interleave": interleave_default,
        "min_size": sizes[0],
        "max_size": sizes[-1],
        "fp_param_ranges": fp_param_ranges,
        "ne_min": hp["n_estimators"][0],
        "ne_max": hp["n_estimators"][1],
        "md_min": hp["max_depth"][0],
        "md_max": hp["max_depth"][1],
        "md_none": include_none_depth,
        "mss_min": hp["min_samples_split"][0],
        "mss_max": hp["min_samples_split"][1],
        "train_ratio": config["split_defaults"]["train_ratio"],
        "metric": config["optuna_settings"]["optimization_metric"],
        "trials": config["stopping_defaults"]["default_trials"],
        "time_hours": config["stopping_defaults"]["default_time_limit"] / 3600,
    }

def persist_table_ui_state(table_state: dict) -> None:
    """Persist Table_UI_State (filter_state, pagination) to storage."""
    if not table_state or "table" not in table_state:
        return
    try:
        fs = table_state.get("filter_state", {})
        serialized_filters = {c: sorted(v) for c, v in fs.items()}
        pagination: dict = {}
        tbl = table_state["table"]
        if hasattr(tbl, "_props") and "pagination" in tbl._props:
            p = tbl._props["pagination"]
            pagination = {
                "rowsPerPage": p.get("rowsPerPage", 10),
                "sortBy": p.get("sortBy", ""),
                "descending": p.get("descending", True),
                "page": p.get("page", 1),
            }
        app.storage.general["table_ui_state"] = {
            "filter_state": serialized_filters,
            "pagination": pagination,
        }
    except Exception:
        logger.debug("Failed to persist table UI state", exc_info=True)

def restore_table_ui_state(table_state: dict) -> None:
    """Restore persisted Table_UI_State from storage and apply to table."""
    if not table_state or "table" not in table_state:
        return
    stored = app.storage.general.get("table_ui_state")
    if not stored or not isinstance(stored, dict):
        return
    try:
        stored_filters = stored.get("filter_state", {})
        if stored_filters and isinstance(stored_filters, dict):
            cur = table_state.get("filter_state", {})
            for col, vals in stored_filters.items():
                if col in cur and isinstance(vals, list):
                    cur[col] = cur[col] & set(vals)
        stored_pag = stored.get("pagination", {})
        if stored_pag and isinstance(stored_pag, dict):
            tbl = table_state["table"]
            if hasattr(tbl, "_props") and "pagination" in tbl._props:
                pag = tbl._props["pagination"]
                for key in ("rowsPerPage", "sortBy", "descending", "page"):
                    if key in stored_pag:
                        pag[key] = stored_pag[key]
                tbl.update()
        filtered = _apply_filters(
            table_state["all_rows"], table_state["filter_state"])
        table_state["table"].update_rows(filtered)
    except Exception:
        logger.debug("Failed to restore table UI state", exc_info=True)

def clear_table_ui_state() -> None:
    """Remove persisted Table_UI_State from storage (used on reset)."""
    app.storage.general.pop("table_ui_state", None)


def persist_trial_selector_state(selected_labels: list[str]) -> None:
    """Persist the Trial_Selector selection to app.storage.general."""
    try:
        app.storage.general["trial_selector_state"] = {"selected_labels": list(selected_labels)}
    except Exception:
        logger.debug("Failed to persist trial selector state", exc_info=True)

def restore_trial_selector_state(valid_labels: list[str], default_label: str) -> list[str]:
    """Restore persisted selection, validating against current labels."""
    try:
        stored = app.storage.general.get("trial_selector_state")
        if stored and isinstance(stored, dict):
            labels = stored.get("selected_labels", [])
            if labels and all(lb in valid_labels for lb in labels):
                return list(labels)
    except Exception:
        logger.debug("Failed to restore trial selector state", exc_info=True)
    return [default_label]

def clear_trial_selector_state() -> None:
    """Remove persisted Trial_Selector state from storage."""
    app.storage.general.pop("trial_selector_state", None)
