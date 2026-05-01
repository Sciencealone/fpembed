"""Restoration helpers for app.py — priority chain, silent reconnect, recovery banner.
"""

from __future__ import annotations

import logging
from datetime import datetime

from nicegui import app, ui

from optimization_thread import get_run_state, is_optimization_running
from progress_file import load_progress, delete_progress
from storage_helpers import deserialize_results, deserialize_progress
from app_helpers import (
    _restore_widget_config, _set_widgets_enabled,
    restore_table_ui_state, restore_trial_selector_state,
    persist_trial_selector_state,
)
from ui_controls import update_progress_labels
from ui_results import (
    render_results_table, render_export_section, render_trial_selector,
    render_scatter_chart,
)
from ui_charts import render_top10_bar_charts

logger = logging.getLogger(__name__)


def _silent_reconnect(
    w: dict, w_prog: dict, timer_ref: list, table_ref: list,
    results_container,
    _restoring: list, poll_progress_fn,
    config: dict,
) -> None:
    """Reconnect UI to a running optimization — no banner, no prompt.

    Restores progress display, results table, and widget config
    from the in-memory ``_run_state``, then starts the polling timer.
    """
    state = get_run_state()
    progress = state.get("progress", {})
    widget_config = state.get("widget_config", {})

    # Restore widget config
    if widget_config:
        _restoring[0] = True
        try:
            _restore_widget_config(w, widget_config, config)
        finally:
            _restoring[0] = False

    # Disable controls — optimization is running
    _set_widgets_enabled(w, False)
    w["run_btn"].disable()
    w["stop_btn"].enable()
    w_prog["container"].set_visibility(True)
    w_prog["status_label"].set_text("⏳ Optimization in progress — reconnected")

    # Restore progress display
    if progress:
        update_progress_labels(
            w_prog, progress.get("trial_number", 0),
            progress.get("total_trials", 0),
            progress.get("elapsed_seconds", 0),
            progress.get("time_limit_seconds", 0))
        top10 = progress.get("top10_results", [])
        if top10:
            table_ref[0] = render_results_table(
                top10, w["metric_select"].value, results_container,
                config=config)
            if table_ref[0]:
                restore_table_ui_state(table_ref[0])

    # Config status
    w["config_status_label"].set_text("Previous Configuration restored")
    w["reset_btn"].set_visibility(True)

    # Start polling timer for real-time updates
    timer_ref[0] = ui.timer(2.0, poll_progress_fn)
    logger.info("Silent reconnect to running optimization")


def _show_recovery_banner(
    app_dir: str, progress_data: dict,
    w: dict, w_prog: dict, _restoring: list,
    results_container,
    config: dict, cache, banner_container,
) -> None:
    """Show a recovery banner for an interrupted optimization run.

    The user must explicitly choose to view recovered results or discard.
    """
    data = deserialize_progress(progress_data)
    progress = data.get("progress", {})
    trial_count = progress.get("trial_number", 0)
    total_trials = progress.get("total_trials", "?")
    ds_name = data.get("ds_name", "Unknown")
    ts = progress_data.get("timestamp", 0)
    ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"

    with banner_container:
        banner = ui.card().classes(
            "w-full bg-amber-1 q-pa-md q-mb-md")
        with banner:
            ui.label("⚠️ Interrupted optimization detected").classes(
                "text-h6 text-warning")
            ui.label(
                f"Found partial results: {trial_count}/{total_trials} trials "
                f"on dataset '{ds_name}' (last saved {ts_str})"
            ).classes("text-body2")
            with ui.row().classes("q-mt-sm gap-2"):
                view_btn = ui.button(
                    "View Recovered Results", icon="restore",
                ).props("color=primary")
                discard_btn = ui.button(
                    "Discard & Start Fresh", icon="delete",
                ).props("color=negative outline")

    def _on_view():
        banner_container.clear()
        _restore_from_progress(
            data, w, w_prog, _restoring, results_container,
            config, cache)

    def _on_discard():
        banner_container.clear()
        delete_progress(app_dir)
        logger.info("User discarded recovered progress")

    view_btn.on_click(_on_view)
    discard_btn.on_click(_on_discard)


def _restore_from_progress(
    data: dict, w: dict, w_prog: dict, _restoring: list,
    results_container,
    config: dict, cache,
) -> None:
    """Restore partial results from deserialized progress data."""
    progress = data.get("progress", {})
    widget_config = data.get("widget_config", {})
    top10 = progress.get("top10_results", [])
    metric = data.get("metric", "R2")
    ds_name = data.get("ds_name")
    trial_count = progress.get("trial_number", 0)
    total_trials = progress.get("total_trials", "?")

    if widget_config:
        _restoring[0] = True
        try:
            _restore_widget_config(w, widget_config, config)
        finally:
            _restoring[0] = False

    results_container.clear()
    if top10:
        table_state = render_results_table(
            top10, metric, results_container, config=config)
        if table_state:
            restore_table_ui_state(table_state)

    w_prog["status_label"].set_text(
        f"⚠️ Partial results recovered — "
        f"{trial_count}/{total_trials} trials completed before interruption")
    w_prog["container"].set_visibility(True)
    w["config_status_label"].set_text("Previous Configuration restored")
    w["reset_btn"].set_visibility(True)
    logger.info("Restored partial results from progress file")


async def restoration_priority_chain(
    app_dir: str, w: dict, w_prog: dict,
    timer_ref: list, table_ref: list,
    results_container,
    _restoring: list, poll_progress_fn,
    config: dict, cache,
    banner_container,
    restore_completed_fn,
) -> None:
    """Evaluate restoration sources in priority order.

    Priority: (a) running thread, (b) Storage_General, (c) Progress_File, (d) blank.
    """
    # (a) Running optimization → silent reconnect
    if is_optimization_running():
        _silent_reconnect(
            w, w_prog, timer_ref, table_ref,
            results_container, _restoring, poll_progress_fn,
            config)
        return

    # (b) Completed results in Storage_General
    stored = app.storage.general.get("optimization_results")
    if stored:
        await restore_completed_fn()
        return

    # (c) Progress_File on disk
    try:
        progress_data = load_progress(app_dir)
    except Exception:
        logger.exception("Error loading progress file")
        progress_data = None
    if progress_data:
        _show_recovery_banner(
            app_dir, progress_data, w, w_prog, _restoring,
            results_container, config, cache,
            banner_container)
        return

    # (d) Blank page — nothing to restore
    logger.debug("No restoration source found — blank page")


async def restore_completed_results(
    w: dict, w_prog: dict, _restoring: list,
    results_container, config: dict, cache,
) -> None:
    """Restore completed optimization results from app.storage.general.

    Extracted from the original ``_restore_results`` closure in ``main_page()``.
    """
    try:
        stored = app.storage.general.get("optimization_results")
        if not stored:
            return
        data = deserialize_results(stored)
        all_results = data.get("all_results", [])
        if not all_results:
            return
        ds_name, metric = data.get("ds_name"), data.get("metric")
        target_col = data.get("target_col", "target")
        train_ratio = data.get("train_ratio", 0.8)
        results_container.clear()

        # (1) Results table (unchanged)
        table_state = render_results_table(
            all_results, metric, results_container, config=config)
        if table_state:
            restore_table_ui_state(table_state)

        # (2) Trial_Selector
        selector_state = render_trial_selector(
            all_results, metric, results_container)

        # Restore persisted selection
        if selector_state:
            labels = list(selector_state["trial_map"].keys())
            default_label = labels[0] if labels else ""
            restored_labels = restore_trial_selector_state(
                labels, default_label)
            _restoring[0] = True
            try:
                selector_state["selector"].value = restored_labels
            finally:
                _restoring[0] = False

        # (3) Export section with get_selected_trials
        def _get_selected_from_selector():
            if not selector_state:
                return []
            chosen = selector_state["selector"].value or []
            tm = selector_state["trial_map"]
            ar = selector_state["all_results"]
            return [ar[tm[lbl]] for lbl in chosen if lbl in tm]

        render_export_section(
            all_results, config, ds_name, train_ratio, results_container,
            descriptors_enabled=w["descriptor_checkbox"].value,
            get_selected_trials=_get_selected_from_selector)

        # (4) Chart_Container
        try:
            viewport_width = await ui.run_javascript(
                'return window.innerWidth')
        except Exception:
            viewport_width = 900
        chart_height = int(viewport_width / 3)

        with results_container:
            chart_container = ui.column().classes("w-full")

        def _render_charts():
            if _restoring[0]:
                return
            chart_container.clear()
            if not selector_state:
                return
            chosen = selector_state["selector"].value or []
            if not chosen:
                with chart_container:
                    ui.label("Select at least one trial").classes(
                        "text-caption text-grey-7")
                return
            tm = selector_state["trial_map"]
            ar = selector_state["all_results"]
            sel = [ar[tm[lbl]] for lbl in chosen if lbl in tm]
            if not sel:
                with chart_container:
                    ui.label("Select at least one trial").classes(
                        "text-caption text-grey-7")
                return
            render_top10_bar_charts(
                sel, chart_container, chart_height=chart_height)
            render_scatter_chart(
                sel, cache, target_col, chart_container,
                chart_height=chart_height)
            persist_trial_selector_state(chosen)

        if selector_state:
            selector_state["selector"].on(
                "update:model-value", lambda _: _render_charts())
        _render_charts()

        w_prog["status_label"].set_text("\u2705 Optimization completed.")
        w_prog["container"].set_visibility(True)
        w["run_btn"].enable()
        w["stop_btn"].disable()
        widget_config = data.get("widget_config", {})
        if widget_config:
            _restoring[0] = True
            try:
                _restore_widget_config(w, widget_config, config)
            finally:
                _restoring[0] = False
        logger.info("Restored optimization results from storage")
        w["config_status_label"].set_text("Previous Configuration restored")
        w["reset_btn"].set_visibility(True)
    except Exception:
        app.storage.general.pop("optimization_results", None)
        logger.exception("Failed to restore results from storage")
