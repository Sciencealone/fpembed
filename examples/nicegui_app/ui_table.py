"""Pure table helpers for the results table (pagination, filters, rows)."""

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
