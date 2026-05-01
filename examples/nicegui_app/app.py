"""NiceGUI molecular optimization application — main entry point."""

import asyncio
import logging
import os, sys
from pathlib import Path

from nicegui import app, run, ui, Client, context

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from cache_db import SQLiteCacheDB
from config import load_config
from dataset import stratified_sample
from optimization_thread import (
    get_run_state, is_optimization_running,
    start_optimization_thread, stop_optimization_thread)
from ui_charts import render_top10_bar_charts
from ui_config import (
    render_dataset_section, render_descriptor_toggle, render_fp_type_selector,
    render_header, render_optimization_metric, render_rf_bounds,
    render_split_configuration, render_compression_method_selector)
from ui_controls import (
    render_optimization_controls, render_progress_display,
    render_stopping_criteria, update_progress_labels)
from ui_results import (
    render_export_section, render_results_table, render_trial_selector,
    render_scatter_chart)
from app_helpers import (
    _dataset_cache, _load_and_filter_dataset, _sample_and_count,
    _update_table_rows, _set_widgets_enabled, _restore_widget_config,
    _collect_widget_config, _build_default_widget_config, clear_table_ui_state,
    persist_trial_selector_state, clear_trial_selector_state)
from storage_helpers import serialize_results
from api_endpoints import register_api_endpoints
from tab_manager import on_client_connect, on_client_disconnect, is_primary
from app_restore import restoration_priority_chain, restore_completed_results

logger = logging.getLogger(__name__)
_config = load_config()
_cache_db_path = os.path.join(_app_dir, _config["paths"]["cache_db"])
_cache = SQLiteCacheDB(_cache_db_path, _config.get("descriptors", []))
register_api_endpoints(_cache)
_grace_period = _config.get("session_persistence", {}).get(
    "primary_tab_grace_period", 5)

def _handle_connect(client: Client) -> None:
    on_client_connect(client.id, _grace_period)

def _handle_disconnect(client: Client) -> None:
    on_client_disconnect(client.id, _grace_period)

app.on_connect(_handle_connect)
app.on_disconnect(_handle_disconnect)
async def _collect_params(w: dict) -> dict:
    """Read all widget values and return kwargs for start_optimization_thread."""
    ds_name = w["dataset_select"].value
    target_col = _config["datasets"][ds_name]["target"]
    if ds_name in _dataset_cache:
        df = _dataset_cache[ds_name].copy()
    else:
        df = await run.cpu_bound(_load_and_filter_dataset, ds_name, _config)
        _dataset_cache[ds_name] = df
        df = df.copy()
    pct = w["subset_slider"].value
    if pct < 100:
        df = stratified_sample(
            df, pct, target_col, n_bins=_config["subset_defaults"]["n_bins"],
            random_seed=_config["random_seed"])
    enabled = [k for k, cb in w["fp_checkboxes"].items() if cb.value] or ["ecfp"]
    enabled_cm = [k for k, cb in w["compression_checkboxes"].items() if cb.value]
    if not enabled_cm:
        enabled_cm = ["geometric"]
    interleave = w["interleave_checkbox"].value
    user_ranges: dict = {}
    for fp_key, params in w["fp_param_inputs"].items():
        user_ranges[fp_key] = {
            pname: (minmax["min"].value, minmax["max"].value)
            for pname, minmax in params.items()
        }
    time_hours = w["time_slider"].value
    return dict(
        df_sampled=df, config=_config, cache=_cache,
        optimization_metric=w["metric_select"].value, user_fp_param_ranges=user_ranges,
        size_bounds=(w["min_size_select"].value, w["max_size_select"].value),
        enabled_fp_types=enabled, rf_bounds={
            "n_estimators": (int(w["ne_min"].value), int(w["ne_max"].value)),
            "max_depth": (int(w["md_min"].value), int(w["md_max"].value),
                          w["md_none_checkbox"].value),
            "min_samples_split": (w["mss_min"].value, w["mss_max"].value)},
        train_ratio=w["train_slider"].value, n_trials=int(w["trials_slider"].value),
        time_limit_seconds=int(time_hours * 3600) if time_hours > 0 else 0,
        target_col=target_col,
        descriptors_enabled=w["descriptor_checkbox"].value,
        enabled_compression_methods=enabled_cm,
        interleave_enabled=interleave)

async def _async_update_molecule_count(w: dict, show_toast: bool = True) -> None:
    """Async version: offloads dataset loading to a separate process, uses cache."""
    n = ui.notification("Loading dataset\u2026", type="ongoing", timeout=None) if show_toast else None
    try:
        ds_name = w["dataset_select"].value
        target_col = _config["datasets"][ds_name]["target"]
        w["target_label"].set_text(f"Target feature: {target_col}")
        w["molecule_count_label"].set_text("Loading dataset\u2026")
        was_cache_miss = ds_name not in _dataset_cache
        if was_cache_miss:
            df = await run.cpu_bound(_load_and_filter_dataset, ds_name, _config)
            _dataset_cache[ds_name] = df
        cached_df = _dataset_cache[ds_name]
        count = _sample_and_count(cached_df, w["subset_slider"].value, target_col, _config)
        w["molecule_count_label"].set_text(f"Molecules after filtering: {count}")
        if n is not None:
            n.dismiss()
        if show_toast and was_cache_miss:
            ui.notify("Dataset loaded", type="positive", timeout=2000)
    except Exception as exc:
        w["molecule_count_label"].set_text(f"Error: {exc}")
        if n is not None:
            n.dismiss()

@ui.page("/")
def main_page():
    """Build the complete application UI and wire all event handlers."""
    timer_ref: list = [None]
    _restoring: list = [False]
    table_ref = [None]
    with ui.column().classes("w-full q-pa-sm gap-2"):
        render_header()
        with ui.row().classes("w-full items-center gap-2"):
            config_status_label = ui.label("New configuration")
            reset_btn = ui.button("Reset Configuration")
            reset_btn.set_visibility(False)
        w_ds = render_dataset_section(_config)
        w_fp = render_fp_type_selector(_config)
        w_cm = render_compression_method_selector(_config)
        w_desc = render_descriptor_toggle(_config)
        w_rf = render_rf_bounds(_config)
        w_split = render_split_configuration(_config)
        w_metric = render_optimization_metric(_config)
        w_stop = render_stopping_criteria(_config)
        w_ctrl = render_optimization_controls()
        w_prog = render_progress_display()
        results_container = ui.column().classes("w-full")
        banner_container = ui.column().classes("w-full")

    w = {**w_ds, **w_fp, **w_cm, **w_desc, **w_rf, **w_split, **w_metric, **w_stop, **w_ctrl}
    w["config_status_label"] = config_status_label
    w["reset_btn"] = reset_btn
    on_client_connect(context.client.id, _grace_period, from_page_handler=True)
    _client_id = context.client.id
    _is_secondary = not is_primary(_client_id)
    if _is_secondary:
        with banner_container:
            ui.label(
                "📡 Read-only view — another tab is controlling the optimization"
            ).classes("text-h6 text-info q-pa-sm bg-blue-1 full-width")
        _set_widgets_enabled(w, False)
        w["run_btn"].disable()
        w["stop_btn"].disable()
    ui.timer(0, lambda: _async_update_molecule_count(w), once=True)
    def _check_fp_selection():
        any_on = any(cb.value for cb in w["fp_checkboxes"].values())
        w["warning_label"].set_visibility(not any_on)
    for cb in w["fp_checkboxes"].values():
        cb.on("update:model-value", lambda _: _check_fp_selection())
    def _check_compression_selection():
        any_on = any(cb.value for cb in w["compression_checkboxes"].values())
        w["compression_warning_label"].set_visibility(not any_on)
    for cb in w["compression_checkboxes"].values():
        cb.on("update:model-value", lambda _: _check_compression_selection())

    async def _on_dataset_change(_evt):
        if _restoring[0]:
            return
        app.storage.general.pop("optimization_results", None)
        _dataset_cache.clear()
        if is_optimization_running():
            stop_optimization_thread()
            w_prog["status_label"].set_text("Stopped — dataset changed.")
        results_container.clear()
        await _async_update_molecule_count(w)
    w["dataset_select"].on("update:model-value", _on_dataset_change)
    w["subset_slider"].on(
        "update:model-value", lambda _: _async_update_molecule_count(w, show_toast=False))
    async def _poll_progress():
        state = get_run_state()
        status = state["status"]
        progress = state.get("progress", {})
        if progress:
            update_progress_labels(
                w_prog, progress.get("trial_number", 0), progress.get("total_trials", 0),
                progress.get("elapsed_seconds", 0), progress.get("time_limit_seconds", 0))
            top10 = progress.get("top10_results", [])
            if top10:
                if table_ref[0] is None:
                    table_ref[0] = render_results_table(
                        top10, w["metric_select"].value, results_container,
                        config=_config)
                else:
                    _update_table_rows(table_ref[0], top10, w["metric_select"].value)
        if status == "completed":
            timer_ref[0].deactivate()
            get_run_state()["status"] = "idle"
            await _on_optimization_done(progress)
        elif status == "error":
            _on_optimization_error(state.get("error"))

    async def _on_optimization_done(progress: dict):
        get_run_state()["status"] = "idle"
        n = ui.notification("Preparing results\u2026", type="ongoing", timeout=None)
        try:
            if timer_ref[0]:
                timer_ref[0].deactivate(); timer_ref[0] = None
            w_prog["status_label"].set_text("\u2705 Optimization completed.")
            w["run_btn"].enable(); w["stop_btn"].disable()
            _set_widgets_enabled(w, True)
            top10 = progress.get("top10_results", [])
            results_container.clear()
            table_ref[0] = None
            ds_name = w["dataset_select"].value
            target_col = _config["datasets"][ds_name]["target"]
            # (1) Results table
            render_results_table(
                top10, w["metric_select"].value, results_container, config=_config)
            await asyncio.sleep(0)
            # (2) Export section with get_selected_trials + (3) Trial_Selector
            selector_state = render_trial_selector(
                top10, w["metric_select"].value, results_container)

            def _get_selected_from_selector():
                if not selector_state:
                    return []
                chosen = selector_state["selector"].value or []
                tm, ar = selector_state["trial_map"], selector_state["all_results"]
                return [ar[tm[lbl]] for lbl in chosen if lbl in tm]

            render_export_section(
                top10, _config, ds_name, w["train_slider"].value,
                results_container, descriptors_enabled=w["descriptor_checkbox"].value,
                get_selected_trials=_get_selected_from_selector)
            await asyncio.sleep(0)
            # (4) Chart_Container
            with results_container:
                chart_container = ui.column().classes("w-full")
            try:
                viewport_width = await ui.run_javascript('return window.innerWidth')
            except Exception:
                viewport_width = 900
            chart_height = int(viewport_width / 3)

            def _render_charts():
                chart_container.clear()
                if not selector_state:
                    return
                chosen = selector_state["selector"].value or []
                if not chosen:
                    with chart_container:
                        ui.label("Select at least one trial").classes(
                            "text-caption text-grey-7")
                    return
                tm, ar = selector_state["trial_map"], selector_state["all_results"]
                sel = [ar[tm[lbl]] for lbl in chosen if lbl in tm]
                if not sel:
                    with chart_container:
                        ui.label("Select at least one trial").classes(
                            "text-caption text-grey-7")
                    return
                render_top10_bar_charts(sel, chart_container, chart_height=chart_height)
                render_scatter_chart(
                    sel, _cache, target_col, chart_container, chart_height=chart_height)
                persist_trial_selector_state(chosen)

            if selector_state:
                selector_state["selector"].on(
                    "update:model-value", lambda _: _render_charts())
            _render_charts()
            await asyncio.sleep(0)
            metric, train_ratio = w["metric_select"].value, w["train_slider"].value
            widget_config = _collect_widget_config(w)
            stored = serialize_results(
                top10, widget_config=widget_config, ds_name=ds_name,
                metric=metric, target_col=target_col, train_ratio=train_ratio)
            app.storage.general["optimization_results"] = stored
            n.dismiss()
            ui.notify("Results ready \u2705", type="positive", timeout=3000)
        except Exception as exc:
            n.dismiss()
            logger.exception("Error rendering optimization results")
            ui.notify(f"Error rendering results: {exc}", type="negative")

    def _on_optimization_error(error_info):
        if timer_ref[0]:
            timer_ref[0].deactivate(); timer_ref[0] = None
        info = error_info or {}
        w_prog["status_label"].set_text(f"❌ Error: {info.get('message', 'Unknown error')}")
        w["run_btn"].enable(); w["stop_btn"].disable()
        _set_widgets_enabled(w, True)
        if info.get("traceback"):
            with results_container:
                with ui.expansion("Error Traceback", icon="error").classes("w-full"):
                    ui.label(info["traceback"]).classes("text-caption font-mono text-negative")

    async def _on_run_click():
        try:
            params = await _collect_params(w)
        except Exception as exc:
            ui.notify(f"Parameter error: {exc}", type="negative")
            return
        app.storage.general.pop("optimization_results", None)
        results_container.clear()
        table_ref[0] = None
        clear_trial_selector_state()
        _set_widgets_enabled(w, False)
        w["run_btn"].disable(); w["stop_btn"].enable()
        w_prog["container"].set_visibility(True)
        w_prog["status_label"].set_text("⏳ Optimization running…")
        widget_config = _collect_widget_config(w)
        ds_name = w["dataset_select"].value
        metric = w["metric_select"].value
        start_optimization_thread(
            **params, widget_config=widget_config,
            ds_name=ds_name, metric=metric, app_dir=_app_dir)
        timer_ref[0] = ui.timer(2.0, _poll_progress)
    w["run_btn"].on_click(_on_run_click)

    def _on_stop_click():
        stop_optimization_thread()
        w_prog["status_label"].set_text("⏸ Stopping after current trial…")
        w["stop_btn"].disable()
    w["stop_btn"].on_click(_on_stop_click)

    async def _on_reset_click():
        app.storage.general.pop("optimization_results", None)
        clear_table_ui_state()
        clear_trial_selector_state()
        results_container.clear()
        table_ref[0] = None
        w_prog["status_label"].set_text(""); w_prog["container"].set_visibility(False)
        w["config_status_label"].set_text("New configuration")
        w["reset_btn"].set_visibility(False); _restoring[0] = True
        try:
            _restore_widget_config(w, _build_default_widget_config(_config), _config)
        finally:
            _restoring[0] = False
        _dataset_cache.clear(); await _async_update_molecule_count(w)
    w["reset_btn"].on_click(_on_reset_click)

    ui.timer(0, lambda: restoration_priority_chain(
        app_dir=_app_dir, w=w, w_prog=w_prog,
        timer_ref=timer_ref, table_ref=table_ref,
        results_container=results_container,
        _restoring=_restoring, poll_progress_fn=_poll_progress,
        config=_config, cache=_cache,
        banner_container=banner_container,
        restore_completed_fn=lambda: restore_completed_results(
            w, w_prog, _restoring, results_container, _config, _cache),
    ), once=True)

app.add_static_files('/vendor', Path(__file__).parent / 'static' / 'vendor')
_reconnect_timeout = _config.get("session_persistence", {}).get("reconnect_timeout", 3600)
ui.run(title="FP-embed Optimization", port=8080, reconnect_timeout=_reconnect_timeout, host="127.0.0.1")
