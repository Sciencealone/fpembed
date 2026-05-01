"""NiceGUI UI render functions for optimization controls and progress display."""

from nicegui import ui


def render_stopping_criteria(config: dict) -> dict:
    """Render trial count slider and time limit control.
    Returns dict: trials_slider, time_slider, time_limit_label.
    """
    ui.label("Stopping Criteria").classes("text-h6")

    stopping = config.get("stopping_defaults", {})
    default_trials = stopping.get("default_trials", 100)
    max_trials = stopping.get("max_trials", 10000)
    max_time_hours = stopping.get("max_time_hours", 24)

    trials_slider = ui.slider(
        min=10, max=max_trials, value=default_trials, step=10
    ).props("label-always").classes("q-mt-lg")
    ui.label().bind_text_from(
        trials_slider, "value", backward=lambda v: f"Max trials: {v}"
    )

    time_slider = ui.slider(
        min=0.0, max=float(max_time_hours), value=0.0, step=0.5
    ).props("label-always").classes("q-mt-lg")
    time_limit_label = ui.label("Time limit: ∞ (no limit)").classes("text-caption")

    def _on_time_change(e):
        hours = e.value if hasattr(e, "value") else e.args
        if hours == 0:
            time_limit_label.set_text("Time limit: ∞ (no limit)")
        else:
            h = int(hours)
            m = int((hours - h) * 60)
            time_limit_label.set_text(f"Time limit: {h}h {m}m")

    time_slider.on("update:model-value", _on_time_change)
    ui.separator()

    return {
        "trials_slider": trials_slider,
        "time_slider": time_slider,
        "time_limit_label": time_limit_label,
    }


def render_optimization_controls() -> dict:
    """Render Run and Stop buttons.
    Returns dict: run_btn, stop_btn.
    """
    with ui.row().classes("gap-4 q-mt-md"):
        run_btn = ui.button(
            "🚀 Run Optimization", color="primary"
        ).classes("text-weight-bold")
        stop_btn = ui.button(
            "⏹️ Stop", color="grey"
        )
        stop_btn.disable()

    return {"run_btn": run_btn, "stop_btn": stop_btn}


def render_progress_display() -> dict:
    """Create progress labels for timer updates.
    Returns dict: trial_label, time_label, status_label, container.
    """
    container = ui.column().classes("w-full q-mt-sm")
    with container:
        trial_label = ui.label("Trial — / —").classes("text-weight-medium")
        time_label = ui.label("Time: 00:00 / ∞").classes("text-caption")
        status_label = ui.label("").classes("text-caption text-blue-8")
    container.set_visibility(False)

    return {
        "trial_label": trial_label,
        "time_label": time_label,
        "status_label": status_label,
        "container": container,
    }


def format_elapsed(elapsed_seconds: float, time_limit_seconds: int) -> str:
    """Format elapsed time and limit into 'HH:MM / HH:MM' or 'HH:MM / ∞'."""
    elapsed_m, _ = divmod(int(elapsed_seconds), 60)
    elapsed_h, elapsed_m = divmod(elapsed_m, 60)
    elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}"
    if time_limit_seconds and time_limit_seconds > 0:
        limit_m, _ = divmod(int(time_limit_seconds), 60)
        limit_h, limit_m = divmod(limit_m, 60)
        limit_str = f"{limit_h:02d}:{limit_m:02d}"
    else:
        limit_str = "∞"
    return f"Time: {elapsed_str} / {limit_str}"


def update_progress_labels(
    widgets: dict,
    trial_number: int,
    total_trials: int,
    elapsed_seconds: float,
    time_limit_seconds: int = 0,
) -> None:
    """Update progress display labels with current values."""
    widgets["trial_label"].set_text(f"Trial {trial_number} / {total_trials}")
    widgets["time_label"].set_text(
        format_elapsed(elapsed_seconds, time_limit_seconds)
    )
