"""NiceGUI UI render functions for configuration sections."""

from nicegui import ui

# All supported FP types with human-readable labels and descriptions
_FP_TYPE_LABELS = {
    "ecfp": {
        "label": "ECFP",
        "description": "Extended-Connectivity Fingerprint (circular, Morgan algorithm)",
    },
    "atom_pair": {
        "label": "Atom Pair",
        "description": "Atom Pair Fingerprint (pairs of atoms + shortest path distance)",
    },
    "topological_torsion": {
        "label": "Topological Torsion",
        "description": "Topological Torsion Fingerprint (four consecutively bonded atoms)",
    },
    "rdkit": {
        "label": "RDKit",
        "description": "RDKit Topological Fingerprint (path-based, Daylight-like)",
    },
    "layered": {
        "label": "Layered",
        "description": "Layered Fingerprint (experimental path-based, multiple layers)",
    },
    "pattern": {
        "label": "Pattern",
        "description": "Pattern Fingerprint (substructure screening, no type-specific params)",
    },
}


def render_header() -> None:
    """Display application title and description."""
    ui.label("FPembed Optimization Application").classes("text-h4 text-weight-bold")
    ui.label(
        "Compare Fingerprints (FP) and Embedded Fingerprints (eFP) for molecular"
        " property prediction using Random Forest models. Uses unified Optuna"
        " optimization across six fingerprint types."
    ).classes("text-body1 text-grey-8")
    ui.separator()


def render_dataset_section(config: dict) -> dict:
    """Render dataset selection and subset percentage controls.
    Returns dict of widget refs: dataset_select, subset_slider,
    molecule_count_label, target_label.
    """
    ui.label("Dataset Configuration").classes("text-h6")
    dataset_options = list(config["datasets"].keys())
    dataset_select = ui.select(
        label="Select Dataset", options=dataset_options, value=dataset_options[0],
    ).classes("w-64")
    target_col = config["datasets"][dataset_options[0]]["target"]
    target_label = ui.label(f"Target feature: {target_col}").classes(
        "text-caption text-blue-8"
    )
    default_pct = config["subset_defaults"]["percentage"]
    subset_slider = ui.slider(min=1, max=100, value=default_pct, step=1).props(
        "label-always"
    ).classes("q-mt-lg")
    ui.label().bind_text_from(
        subset_slider, "value", backward=lambda v: f"Subset: {v}%"
    )
    molecule_count_label = ui.label("Molecules after filtering: —").classes(
        "text-caption"
    )
    ui.separator()
    return {
        "dataset_select": dataset_select,
        "subset_slider": subset_slider,
        "molecule_count_label": molecule_count_label,
        "target_label": target_label,
    }


def render_fp_type_selector(config: dict) -> dict:
    """Render FP type selector with size bounds, type checkboxes, per-type params.
    Returns dict: min_size_select, max_size_select, fp_checkboxes, fp_param_inputs,
    warning_label, expansions.
    """
    ui.label("Fingerprint Configuration").classes("text-h6")
    available_sizes = config["fp_parameters"]["sizes"]
    size_options = {s: str(s) for s in available_sizes}
    with ui.row().classes("gap-4"):
        min_size_select = ui.select(
            label="FP Size - Lower Bound", options=size_options,
            value=available_sizes[0],
        ).classes("w-48")
        max_size_select = ui.select(
            label="FP Size - Upper Bound", options=size_options,
            value=available_sizes[-1],
        ).classes("w-48")
    # FP type checkboxes
    ui.label("Enabled Fingerprint Types").classes("text-subtitle2 q-mt-sm")
    fp_defaults = config.get("fp_defaults", {})
    fp_checkboxes = {}
    with ui.row().classes("gap-4 flex-wrap"):
        for fp_key, fp_info in _FP_TYPE_LABELS.items():
            default_enabled = fp_defaults.get(fp_key, {}).get("enabled", True)
            fp_checkboxes[fp_key] = ui.checkbox(
                fp_info["label"], value=default_enabled
            ).tooltip(fp_info["description"])
    warning_label = ui.label(
        "⚠ At least one fingerprint type must be enabled."
    ).classes("text-negative text-caption")
    warning_label.set_visibility(False)
    # Per-type parameter ranges in collapsible sections
    fp_params_cfg = config["fp_parameters"]
    fp_param_inputs = {}
    expansions = {}
    for fp_key, fp_info in _FP_TYPE_LABELS.items():
        ranges = fp_params_cfg.get(fp_key, {})
        if not ranges:
            continue
        exp = ui.expansion(
            f"{fp_info['label']} Parameters", icon="tune"
        ).classes("w-full")
        expansions[fp_key] = exp
        fp_param_inputs[fp_key] = {}
        with exp:
            for param_name, (default_min, default_max) in ranges.items():
                with ui.row().classes("gap-4 items-center"):
                    fmt = "%.3f" if isinstance(default_min, float) else None
                    min_input = ui.number(
                        label=f"{param_name} min", value=default_min, format=fmt,
                    ).classes("w-36")
                    max_input = ui.number(
                        label=f"{param_name} max", value=default_max, format=fmt,
                    ).classes("w-36")
                fp_param_inputs[fp_key][param_name] = {
                    "min": min_input, "max": max_input,
                }
    ui.separator()
    return {
        "min_size_select": min_size_select,
        "max_size_select": max_size_select,
        "fp_checkboxes": fp_checkboxes,
        "fp_param_inputs": fp_param_inputs,
        "warning_label": warning_label,
        "expansions": expansions,
    }


_COMPRESSION_METHOD_LABELS = {
    "geometric": {
        "label": "Geometric",
        "description": "Geometric weighting (2^i), highest dynamic range (65536:1)",
    },
    "linear": {
        "label": "Linear",
        "description": "Linear weighting (i+1), moderate dynamic range (S:1)",
    },
    "log": {
        "label": "Logarithmic",
        "description": "Logarithmic weighting (log₂(i+2)), low dynamic range (~4.1:1)",
    },
    "uniform": {
        "label": "Uniform",
        "description": "Uniform weighting (1/S), mean pooling with 1:1 dynamic range",
    },
    "hadamard": {
        "label": "Hadamard",
        "description": "SRHT global projection — every output depends on every input bit",
    },
    "random_projection": {
        "label": "Random Projection",
        "description": "Johnson-Lindenstrauss random projection with distance preservation",
    },
}

_BLOCKWISE_METHODS = {"geometric", "linear", "log", "uniform"}


def render_compression_method_selector(config: dict) -> dict:
    """Render compression method checkboxes, interleave toggle, and warning.

    Returns dict with keys: compression_checkboxes, interleave_checkbox,
    compression_warning_label.
    """
    ui.label("Compression Methods").classes("text-h6")
    cm_defaults = config.get("compression_methods", {})
    compression_checkboxes = {}
    with ui.row().classes("gap-4 flex-wrap"):
        for method_key, method_info in _COMPRESSION_METHOD_LABELS.items():
            default_enabled = cm_defaults.get(method_key, {}).get("enabled", True)
            compression_checkboxes[method_key] = ui.checkbox(
                method_info["label"], value=default_enabled,
            ).tooltip(method_info["description"])
    interleave_default = cm_defaults.get("interleave_default", False)
    interleave_checkbox = ui.checkbox(
        "Interleave bits", value=interleave_default,
    ).tooltip(
        "Strided block partitioning for block-wise methods — "
        "each block samples uniformly across the entire fingerprint"
    )
    compression_warning_label = ui.label(
        "⚠ At least one compression method must be enabled."
    ).classes("text-negative text-caption")
    compression_warning_label.set_visibility(False)
    ui.separator()
    return {
        "compression_checkboxes": compression_checkboxes,
        "interleave_checkbox": interleave_checkbox,
        "compression_warning_label": compression_warning_label,
    }


def render_descriptor_toggle(config: dict) -> dict:
    """Render the descriptor toggle checkbox.

    Reads default from config['descriptors_enabled'] (falls back to True).
    Returns dict with key 'descriptor_checkbox'.
    """
    default = config.get("descriptors_enabled", True)
    descriptor_checkbox = ui.checkbox(
        "Include Molecular Descriptors", value=default
    ).tooltip(
        "When enabled, Optuna decides per-trial whether to include "
        "73 RDKit molecular descriptors alongside fingerprints. "
        "When disabled, all trials use fingerprints only."
    )
    ui.separator()
    return {"descriptor_checkbox": descriptor_checkbox}


def render_rf_bounds(config: dict) -> dict:
    """Render RF hyperparameter bound controls.
    Returns dict: ne_min, ne_max, md_min, md_max, md_none_checkbox, mss_min, mss_max.
    """
    ui.label("Random Forest Bounds").classes("text-h6")
    ranges = config["optuna_settings"]["hyperparameter_ranges"]
    ne_range = ranges["n_estimators"]
    with ui.row().classes("gap-4"):
        ne_min = ui.number(
            label="n_estimators - Min",
            value=ne_range[0], min=ne_range[0], max=ne_range[1], step=10,
        ).classes("w-48")
        ne_max = ui.number(
            label="n_estimators - Max",
            value=ne_range[1], min=ne_range[0], max=ne_range[1], step=10,
        ).classes("w-48")
    md_range = ranges["max_depth"]
    with ui.row().classes("gap-4"):
        md_min = ui.number(
            label="max_depth - Min",
            value=md_range[0], min=md_range[0], max=md_range[1], step=1,
        ).classes("w-48")
        md_max = ui.number(
            label="max_depth - Max",
            value=md_range[1], min=md_range[0], max=md_range[1], step=1,
        ).classes("w-48")
    include_none_depth = (
        config.get("optuna_settings", {})
        .get("hyperparameter_ranges", {})
        .get("include_none_depth", True)
    )
    md_none_checkbox = ui.checkbox(
        "Include unlimited depth (None)", value=include_none_depth
    ).tooltip("Allow Optuna to try unlimited tree depth")
    mss_range = ranges["min_samples_split"]
    with ui.row().classes("gap-4"):
        mss_min = ui.number(
            label="min_samples_split - Min",
            value=mss_range[0], min=mss_range[0], max=mss_range[1],
            step=0.001, format="%.3f",
        ).classes("w-48")
        mss_max = ui.number(
            label="min_samples_split - Max",
            value=mss_range[1], min=mss_range[0], max=mss_range[1],
            step=0.001, format="%.3f",
        ).classes("w-48")
    ui.separator()
    return {
        "ne_min": ne_min, "ne_max": ne_max,
        "md_min": md_min, "md_max": md_max,
        "md_none_checkbox": md_none_checkbox,
        "mss_min": mss_min, "mss_max": mss_max,
    }


def render_split_configuration(config: dict) -> dict:
    """Render train/validation split configuration.
    Returns dict: train_slider, train_pct_label, val_pct_label.
    """
    ui.label("Train/Validation Split").classes("text-h6")
    default_ratio = config["split_defaults"]["train_ratio"]
    train_slider = ui.slider(
        min=0.5, max=0.95, value=default_ratio, step=0.05
    ).props("label-always").classes("q-mt-lg")
    with ui.row().classes("gap-8"):
        train_pct_label = ui.label(
            f"Train: {default_ratio * 100:.1f}%"
        ).classes("text-weight-medium")
        val_pct_label = ui.label(
            f"Validation: {(1 - default_ratio) * 100:.1f}%"
        ).classes("text-weight-medium")

    def _on_ratio_change(e):
        r = e.value if hasattr(e, "value") else e.args
        train_pct_label.set_text(f"Train: {r * 100:.1f}%")
        val_pct_label.set_text(f"Validation: {(1 - r) * 100:.1f}%")

    train_slider.on("update:model-value", _on_ratio_change)
    ui.separator()
    return {
        "train_slider": train_slider,
        "train_pct_label": train_pct_label,
        "val_pct_label": val_pct_label,
    }


def render_optimization_metric(config: dict) -> dict:
    """Render metric selector (MSE, MAPE, R2). Returns dict: metric_select."""
    ui.label("Optimization Metric").classes("text-h6")
    default_metric = config.get("optuna_settings", {}).get(
        "optimization_metric", "R2"
    )
    options = ["MSE", "MAPE", "R2"]
    metric_select = ui.select(
        label="Metric to optimize", options=options,
        value=default_metric if default_metric in options else "R2",
    ).classes("w-48").tooltip("MSE and MAPE are minimized; R2 is maximized.")
    ui.separator()
    return {"metric_select": metric_select}
