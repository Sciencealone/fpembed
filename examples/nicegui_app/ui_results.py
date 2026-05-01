"""NiceGUI UI components for results table, prediction analysis, and export."""

import numpy as np
from nicegui import ui

from api_endpoints import clear_prediction_store, populate_prediction_store
from optimization import build_results_dataframe, export_results_json
from ui_prediction import (
    build_prediction_data,
    create_prediction_chart,
    render_vega_chart,
)

# 10 distinct colors for up to 10 trials in comparison plots
_TRIAL_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#8D6A9F", "#3891A6",
]

_CATEGORICAL_COLS = ["Type", "FP Type", "Descriptors", "Method"]
_METRIC_WIDTH = "width: 100px"
_DEFAULT_ROWS_PER_PAGE = 10


def _get_rows_per_page(config: dict | None) -> int:
    """Read rows_per_page from config, falling back to default."""
    if config is None:
        return _DEFAULT_ROWS_PER_PAGE
    rt = config.get("results_table", {})
    val = rt.get("rows_per_page", _DEFAULT_ROWS_PER_PAGE)
    return val if isinstance(val, int) and val > 0 else _DEFAULT_ROWS_PER_PAGE


def _build_column_defs(display_df) -> list[dict]:
    """Build column definitions for the NiceGUI table."""
    columns = []
    for col in display_df.columns:
        col_def: dict = {
            "name": col, "label": col, "field": col,
            "sortable": True, "align": "left",
        }
        if col.endswith("_val"):
            col_def["style"] = _METRIC_WIDTH
            col_def["headerStyle"] = _METRIC_WIDTH
        columns.append(col_def)
    return columns


def _prepare_rows(display_df) -> list[dict]:
    """Convert DataFrame to list of row dicts with rounded floats."""
    rows = display_df.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 6)
            elif v is None:
                row[k] = "None"
    return rows


def _build_initial_filter_state(
    rows: list[dict],
    categorical_cols: list[str],
) -> dict[str, set[str]]:
    """Build initial filter state with all values checked."""
    state: dict[str, set[str]] = {}
    for col in categorical_cols:
        values = {str(row.get(col, "")) for row in rows}
        state[col] = values
    return state


def _make_filter_template(col_name: str, values: list[str]) -> str:
    """Build a Vue/Quasar template for a categorical column filter panel.

    The template renders the column label with a filter-icon dropdown
    containing one checkbox per distinct value.  Checkbox toggles emit
    ``filter_toggle`` events back to the Python server.
    """
    checkboxes = ""
    for val in sorted(values):
        safe = val.replace("'", "\\'")
        checkboxes += (
            f'<q-item dense clickable>'
            f'<q-item-section>'
            f'<q-checkbox dense '
            f':model-value="true" '
            f'label="{safe}" '
            f'@update:model-value="(v) => $parent.$emit(\'filter_toggle\', '
            f'{{col: \'{safe}\', colName: \'{col_name}\', checked: v}})" '
            f'/>'
            f'</q-item-section>'
            f'</q-item>'
        )
    return (
        '<q-th :props="props">'
        '  <div class="row items-center no-wrap">'
        '    <span>{{ props.col.label }}</span>'
        '    <q-btn flat dense round icon="filter_list" size="xs" class="q-ml-xs" @click.stop>'
        '      <q-menu>'
        '        <q-list dense style="min-width: 120px">'
        f'          {checkboxes}'
        '        </q-list>'
        '      </q-menu>'
        '    </q-btn>'
        '  </div>'
        '</q-th>'
    )


def _add_filter_panel_slots(table, table_state: dict) -> None:
    """Attach filter panel header slots and wire toggle handler."""
    filter_state = table_state["filter_state"]

    _refresh_filter_slots(table, table_state)

    def _on_filter_toggle(e):
        """Handle a filter checkbox toggle from the Vue template."""
        args = e.args
        col_name = args.get("colName", "")
        val = args.get("col", "")
        checked = args.get("checked", True)
        if col_name not in filter_state:
            return
        if checked:
            filter_state[col_name].add(val)
        else:
            filter_state[col_name].discard(val)
        filtered = _apply_client_filter(table_state["all_rows"], filter_state)
        table.update_rows(filtered)
        from app_helpers import persist_table_ui_state
        persist_table_ui_state(table_state)

    table.on("filter_toggle", _on_filter_toggle)


def _refresh_filter_slots(table, table_state: dict) -> None:
    """Rebuild filter dropdown templates from current filter_state values."""
    filter_state = table_state["filter_state"]
    for col_name in table_state["categorical_cols"]:
        values = sorted(filter_state.get(col_name, set()))
        template = _make_filter_template(col_name, values)
        table.add_slot(f"header-cell-{col_name}", template)
    table.update()


def _apply_client_filter(
    all_rows: list[dict],
    filter_state: dict[str, set[str]],
) -> list[dict]:
    """Return rows matching all column filters (AND logic)."""
    result = []
    for row in all_rows:
        match = True
        for col, allowed in filter_state.items():
            if str(row.get(col, "")) not in allowed:
                match = False
                break
        if match:
            result.append(row)
    return result


def render_results_table(
    all_results: list[dict],
    metric: str,
    container,
    config: dict | None = None,
) -> dict | None:
    """Render all results as a paginated NiceGUI table with filter panels.

    Returns a table_state dict containing the table reference, all_rows,
    filter_state, and categorical_cols — or None if no results.
    """
    if not all_results:
        with container:
            ui.label("No results to display.").classes("text-caption text-grey-7")
        return None

    rows_per_page = _get_rows_per_page(config)
    display_df = build_results_dataframe(all_results)
    metric_col = f"{metric}_val"
    if metric_col in display_df.columns:
        display_df = display_df.sort_values(
            by=metric_col, ascending=(metric != "R2"),
        )

    columns = _build_column_defs(display_df)
    all_rows = _prepare_rows(display_df)

    categorical_cols = _CATEGORICAL_COLS
    filter_state = _build_initial_filter_state(all_rows, categorical_cols)

    with container:
        ui.label("Results").classes("text-h6 q-mt-md")
        table = ui.table(
            columns=columns,
            rows=all_rows,
            pagination={
                "rowsPerPage": rows_per_page,
                "sortBy": metric_col,
                "descending": metric == "R2",
            },
        ).props("dense").classes("w-full")

    table_state: dict = {
        "table": table,
        "all_rows": list(all_rows),
        "filter_state": filter_state,
        "categorical_cols": list(categorical_cols),
    }

    _add_filter_panel_slots(table, table_state)
    return table_state


def build_trial_option_labels(
    all_results: list[dict],
    metric: str,
) -> tuple[list[str], dict[str, int]]:
    """Build Trial_Option labels and a label→index mapping.

    Args:
        all_results: Trial result dicts, sorted best-to-worst.
        metric: Goal metric name ('R2', 'MAPE', or 'MSE').

    Returns:
        Tuple of (labels list, trial_map dict mapping label → index).
    """
    labels: list[str] = []
    trial_map: dict[str, int] = {}
    for idx, result in enumerate(all_results):
        n = result.get("trial_number", 0)
        val = result.get(f"{metric}_val", 0.0)
        if metric == "R2":
            label = f"Trial {n}: R2={val:.4f}"
        elif metric == "MAPE":
            label = f"Trial {n}: MAPE={val:.2f}%"
        else:
            label = f"Trial {n}: MSE={val:.4g}"
        labels.append(label)
        trial_map[label] = idx
    return labels, trial_map


def render_trial_selector(
    all_results: list[dict],
    metric: str,
    container,
) -> dict | None:
    """Render the unified trial multi-select widget.

    Args:
        all_results: All completed trial result dicts, sorted best-to-worst.
        metric: Goal metric name ('R2', 'MAPE', or 'MSE').
        container: NiceGUI container to render into.

    Returns:
        Dict with keys 'selector', 'trial_map', 'all_results', 'metric',
        or None if no results.
    """
    if not all_results:
        return None

    labels, trial_map = build_trial_option_labels(all_results, metric)
    default_selection = [labels[0]] if labels else []

    with container:
        selector = ui.select(
            label="Select trials to display",
            options=labels,
            value=default_selection,
            multiple=True,
        ).classes("w-96")

    return {
        "selector": selector,
        "trial_map": trial_map,
        "all_results": all_results,
        "metric": metric,
    }


def render_scatter_chart(
    selected_results: list[dict],
    cache_db,
    target_col: str,
    container,
    chart_height: int = 500,
) -> None:
    """Render the Actual vs Predicted scatter chart for selected trials.

    Args:
        selected_results: Trial result dicts to display.
        cache_db: SQLiteCacheDB instance.
        target_col: Target column name.
        container: NiceGUI container to render into.
        chart_height: Chart height in pixels.
    """
    clear_prediction_store()
    trial_ids = [t.get("trial_number", 0) for t in selected_results]
    trial_labels = [f"Trial {tid}" for tid in trial_ids]
    colors = _TRIAL_COLORS[: len(selected_results)]
    df, all_vals = build_prediction_data(
        selected_results, colors, cache_db, target_col,
    )
    populate_prediction_store(trial_ids, df)
    spec = create_prediction_chart(
        trial_ids, all_vals, colors, trial_labels, target_col,
        height=chart_height,
    )
    render_vega_chart(spec, container)


def render_export_section(
    top10_results: list[dict],
    config: dict,
    dataset_name: str,
    train_ratio: float,
    container,
    descriptors_enabled: bool = True,
    get_selected_trials: callable = None,
) -> None:
    """Render Export All and Export Selected buttons using ui.download()."""
    if not top10_results:
        with container:
            ui.label("No results available for export.").classes(
                "text-caption text-grey-7"
            )
        return

    import sklearn
    from rdkit.Chem import rdBase as _rdBase
    import optuna

    reproducibility: dict = {
        "dataset_name": dataset_name,
        "random_seed": config.get("random_seed"),
        "train_val_split_ratio": train_ratio,
        "descriptors_enabled": descriptors_enabled,
        "rdkit_version": _rdBase.rdkitVersion,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "optuna_version": optuna.__version__,
        "descriptors": config.get("descriptors", []),
    }

    with container:
        ui.label("Export Results").classes("text-h6 q-mt-md")

        def _export_all():
            json_data = export_results_json(
                top10_results, reproducibility=reproducibility,
            )
            ui.download(json_data.encode("utf-8"), "optimization_results.json")

        ui.button("Export All (JSON)", on_click=_export_all, icon="download")

        if get_selected_trials is not None:
            def _export_selected():
                selected = get_selected_trials()
                if not selected:
                    ui.notify("Select at least one trial to export")
                    return
                json_data = export_results_json(
                    selected, reproducibility=reproducibility,
                )
                if len(selected) == 1:
                    fname = f"trial_{selected[0].get('trial_number', 0)}.json"
                else:
                    fname = "selected_trials.json"
                ui.download(json_data.encode("utf-8"), fname)

            ui.button(
                "Export Selected (JSON)",
                on_click=_export_selected,
                icon="download",
            )
