"""Altair bar chart rendering for top-10 optimization results via vega-embed."""

import logging
import uuid
import warnings

warnings.filterwarnings("ignore", message=".*is_pandas_dataframe.*", category=UserWarning)

import altair as alt
import pandas as pd
from nicegui import ui

from model import format_mse_label

logger = logging.getLogger(__name__)

# Local vendor paths for vega-embed rendering (served via app.add_static_files)
_VEGA_LOCAL = "/vendor/vega.min.js"
_VEGA_LITE_LOCAL = "/vendor/vega-lite.min.js"
_VEGA_EMBED_LOCAL = "/vendor/vega-embed.min.js"

_METRIC_TITLES = {"R2": "R² Score", "MAPE": "MAPE (%)", "MSE": "MSE"}


def _vega_poll_js() -> str:
    """Return JS defining global waitForVegaEmbed(callback, chartId).

    Polls typeof vegaEmbed !== 'undefined' every 100ms with 15s timeout.
    Sends diagnostic events to /api/chart_diagnostic at each stage.
    """
    return (
        "if (typeof window.waitForVegaEmbed === 'undefined') {\n"
        "  window.waitForVegaEmbed = function(callback, chartId) {\n"
        "    var interval = 100, timeout = 15000, elapsed = 0;\n"
        "    fetch('/api/chart_diagnostic', {method:'POST',\n"
        "      headers:{'Content-Type':'application/json'},\n"
        "      body:JSON.stringify({event:'poll_start',chart_id:chartId})\n"
        "    }).catch(function(){});\n"
        "    var timer = setInterval(function() {\n"
        "      if (typeof vegaEmbed !== 'undefined' && document.getElementById(chartId) !== null) {\n"
        "        clearInterval(timer);\n"
        "        fetch('/api/chart_diagnostic', {method:'POST',\n"
        "          headers:{'Content-Type':'application/json'},\n"
        "          body:JSON.stringify({event:'poll_success',chart_id:chartId,\n"
        "            elapsed_ms:elapsed})\n"
        "        }).catch(function(){});\n"
        "        callback(null);\n"
        "        return;\n"
        "      }\n"
        "      elapsed += interval;\n"
        "      if (elapsed >= timeout) {\n"
        "        clearInterval(timer);\n"
        "        fetch('/api/chart_diagnostic', {method:'POST',\n"
        "          headers:{'Content-Type':'application/json'},\n"
        "          body:JSON.stringify({event:'poll_timeout',chart_id:chartId,\n"
        "            elapsed_ms:elapsed})\n"
        "        }).catch(function(){});\n"
        "        callback(new Error('vegaEmbed not available after 15s'));\n"
        "      }\n"
        "    }, interval);\n"
        "  };\n"
        "}\n"
    )


def ensure_vega_head_scripts() -> None:
    """Dynamically load local vega/vega-lite/vega-embed scripts via JS.

    Uses ui.run_javascript() with document.createElement('script') to
    load scripts in dependency order (vega → vega-lite → vega-embed).
    A global guard prevents redundant loading on repeated calls.
    """
    ui.run_javascript(
        "if (window.__vegaScriptsLoading || (typeof vegaEmbed !== 'undefined')) { return; }\n"
        "window.__vegaScriptsLoading = true;\n"
        "\n"
        "function loadScript(src, name, onDone) {\n"
        "  var s = document.createElement('script');\n"
        "  s.src = src;\n"
        "  s.onload = function() {\n"
        "    console.log('[vega-loader] Loaded ' + name);\n"
        "    if (onDone) onDone();\n"
        "  };\n"
        "  s.onerror = function(error) {\n"
        "    console.error('[vega-loader] Failed to load ' + name + ':', error);\n"
        "  };\n"
        "  document.head.appendChild(s);\n"
        "}\n"
        "\n"
        f"loadScript('{_VEGA_LOCAL}', 'vega', function() {{\n"
        f"  loadScript('{_VEGA_LITE_LOCAL}', 'vega-lite', function() {{\n"
        f"    loadScript('{_VEGA_EMBED_LOCAL}', 'vega-embed', function() {{\n"
        "      window.__vegaScriptsLoaded = true;\n"
        "    });\n"
        "  });\n"
        "});\n"
    )


def create_top10_bar_chart(top10_results: list[dict], metric: str, height: int = 400) -> alt.Chart:
    """Build an Altair bar chart spec for one metric (R2, MAPE, or MSE).

    Shows train and validation bars per trial, sorted best-to-worst.
    Includes enhanced tooltips with FP params, RF params, and all metrics.
    """
    if not top10_results:
        empty_df = pd.DataFrame({"x": [0], "y": [0], "text": ["No results yet"]})
        return (
            alt.Chart(empty_df)
            .mark_text(fontSize=14)
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="text:N")
            .properties(width="container", height=height)
        )

    reverse = metric == "R2"
    val_key = f"{metric}_val"
    default = float("-inf") if reverse else float("inf")
    sorted_results = sorted(
        top10_results,
        key=lambda r: default if r.get(val_key) is None else r.get(val_key),
        reverse=reverse,
    )

    rows = []
    for r in sorted_results:
        trial_label = f"Trial {r.get('trial_number', '?')}"
        compression = r.get("compression", 0)
        max_depth = r.get("max_depth")
        common = {
            "Trial": trial_label,
            "fp_type": str(r.get("fp_type", "?")),
            "fp_size": str(r.get("fp_size", "?")),
            "compression": compression,
            "type_label": "eFP" if compression > 0 else "FP",
            "n_estimators": str(r.get("n_estimators", "?")),
            "max_depth": str(max_depth) if max_depth is not None else "None",
            "min_samples_split": str(r.get("min_samples_split", "?")),
            "R2_val": round(r.get("R2_val", 0), 4),
            "MAPE_val": round(r.get("MAPE_val", 0), 2),
            "MSE_val": r.get("MSE_val", 0),
        }
        rows.append({**common, "Set": "Train", "Value": r.get(f"{metric}_train") or 0})
        rows.append({**common, "Set": "Validation", "Value": r.get(val_key) or 0})

    df = pd.DataFrame(rows)
    df["label"] = (
        df["Value"].apply(format_mse_label)
        if metric == "MSE"
        else df["Value"].apply(lambda v: f"{v:.4f}")
    )

    trial_order = [f"Trial {r.get('trial_number', '?')}" for r in sorted_results]

    tooltip_fields = [
        alt.Tooltip("Trial:N"), alt.Tooltip("Set:N"),
        alt.Tooltip("Value:Q", format=".4f"),
        alt.Tooltip("fp_type:N", title="FP Type"),
        alt.Tooltip("fp_size:N", title="Size"),
        alt.Tooltip("compression:Q", title="Compression"),
        alt.Tooltip("type_label:N", title="Type"),
        alt.Tooltip("n_estimators:N", title="n_estimators"),
        alt.Tooltip("max_depth:N", title="max_depth"),
        alt.Tooltip("min_samples_split:N", title="min_samples_split"),
        alt.Tooltip("R2_val:Q", title="R2_val", format=".4f"),
        alt.Tooltip("MAPE_val:Q", title="MAPE_val", format=".2f"),
        alt.Tooltip("MSE_val:Q", title="MSE_val", format=".4g"),
    ]

    bars = (
        alt.Chart(df).mark_bar().encode(
            x=alt.X("Trial:N", sort=trial_order, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color(
                "Set:N",
                scale=alt.Scale(
                    domain=["Train", "Validation"], range=["#2E86AB", "#A23B72"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            xOffset="Set:N",
            tooltip=tooltip_fields,
        )
    )
    text = (
        alt.Chart(df).mark_text(dy=-8, fontSize=10).encode(
            x=alt.X("Trial:N", sort=trial_order),
            y=alt.Y("Value:Q"),
            text="label:N",
            xOffset="Set:N",
        )
    )
    return (bars + text).properties(
        title=_METRIC_TITLES.get(metric, metric), width="container", height=height,
    )


def render_altair_chart(chart: alt.Chart, container) -> ui.html:
    """Convert Altair chart to Vega-Lite JSON and render via vega-embed.

    Uses waitForVegaEmbed polling to wait for CDN scripts, wraps vegaEmbed
    in try/catch, and logs chart ID and spec size at INFO level.
    """
    ensure_vega_head_scripts()
    spec_json = chart.to_json()
    vis_id = f"vis-{uuid.uuid4().hex[:12]}"
    logger.info("render_altair_chart: chart_id=%s spec_size=%d chart_type=bar", vis_id, len(spec_json))
    with container:
        el = ui.html(f'<div id="{vis_id}" style="width:100%;"></div>', sanitize=False).classes("w-full")
    ui.run_javascript(
        f"{_vega_poll_js()}\n"
        f"waitForVegaEmbed(function(err) {{\n"
        f"  if (err) {{ console.error('vegaEmbed poll timeout for {vis_id}:', err); return; }}\n"
        f"  try {{\n"
        f"    vegaEmbed('#{vis_id}', {spec_json}, {{actions: true}})\n"
        f"      .catch(function(e) {{ console.error('vegaEmbed async error:', e); }});\n"
        f"  }} catch (e) {{\n"
        f"    console.error('vegaEmbed sync error for {vis_id}:', e);\n"
        f"  }}\n"
        f"}}, '{vis_id}')"
    )
    return el


def render_top10_bar_charts(
    top10_results: list[dict],
    container,
    chart_height: int = 400,
) -> None:
    """Render all three bar charts (R2, MAPE, MSE) into the given container."""
    for metric in ("R2", "MAPE", "MSE"):
        chart = create_top10_bar_chart(top10_results, metric, height=chart_height)
        render_altair_chart(chart, container)
