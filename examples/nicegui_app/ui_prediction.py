"""Prediction chart builders: scatter plot data and Vega-Lite spec construction."""

import json
import logging
import uuid

import numpy as np
import pandas as pd
from nicegui import ui

from ui_charts import ensure_vega_head_scripts, _vega_poll_js

logger = logging.getLogger(__name__)


def build_prediction_data(
    selected_trials: list[dict],
    colors: list[str],
    cache_db=None,
    target_col: str = "target",
) -> tuple[pd.DataFrame, list[float]]:
    """Build scatter plot DataFrame from selected trials."""
    rows: list[dict] = []
    all_vals: list[float] = []
    point_idx = 0

    for trial, color in zip(selected_trials, colors):
        label = f"Trial {trial.get('trial_number', '?')}"
        pred_train = np.asarray(trial.get("predictions_train", []))
        act_train = np.asarray(trial.get("actuals_train", []))
        pred_val = np.asarray(trial.get("predictions_val", []))
        act_val = np.asarray(trial.get("actuals_val", []))
        smiles_train = trial.get("smiles_train", [])
        smiles_val = trial.get("smiles_val", [])

        for i in range(len(act_train)):
            actual, predicted = float(act_train[i]), float(pred_train[i])
            all_vals.extend([actual, predicted])
            smi = smiles_train[i] if i < len(smiles_train) else ""
            rows.append({
                "Actual": actual, "Predicted": predicted,
                "Set": "Train", "Trial": label, "Color": color,
                "point_index": point_idx, "smiles": smi,
            })
            point_idx += 1

        for i in range(len(act_val)):
            actual, predicted = float(act_val[i]), float(pred_val[i])
            all_vals.extend([actual, predicted])
            smi = smiles_val[i] if i < len(smiles_val) else ""
            rows.append({
                "Actual": actual, "Predicted": predicted,
                "Set": "Val", "Trial": label, "Color": color,
                "point_index": point_idx, "smiles": smi,
            })
            point_idx += 1

    cols = ["Actual", "Predicted", "Set", "Trial", "Color", "point_index", "smiles"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df, all_vals


def create_prediction_chart(
    trial_ids: list[int],
    all_vals: list[float],
    colors: list[str],
    trial_labels: list[str],
    target_col: str = "target",
    height: int = 500,
) -> dict:
    """Build a Vega-Lite spec dict with URL-based data source."""
    ids_param = ",".join(str(t) for t in sorted(trial_ids))
    data_url = f"/api/prediction_data?trial_ids={ids_param}"

    scatter_layer = {
        "data": {"url": data_url, "format": {"type": "json"}},
        "mark": {"type": "point", "filled": True, "size": 60},
        "params": [{"name": "grid", "select": "interval", "bind": "scales"}],
        "encoding": {
            "x": {
                "field": "Actual", "type": "quantitative",
                "title": f"Actual Values ({target_col})",
            },
            "y": {
                "field": "Predicted", "type": "quantitative",
                "title": f"Predicted Values ({target_col})",
            },
            "color": {
                "field": "Trial", "type": "nominal",
                "scale": {
                    "domain": list(dict.fromkeys(trial_labels)),
                    "range": list(dict.fromkeys(colors)),
                },
                "legend": {"title": "Trial"},
            },
            "shape": {
                "field": "Set", "type": "nominal",
                "scale": {
                    "domain": ["Train", "Val"],
                    "range": ["circle", "triangle-up"],
                },
                "legend": {"title": "Set"},
            },
        },
    }

    layers = [scatter_layer]

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05 if hi > lo else 1.0
        line_layer = {
            "data": {
                "values": [
                    {"x": lo - margin, "y": lo - margin},
                    {"x": hi + margin, "y": hi + margin},
                ],
            },
            "mark": {
                "type": "line", "color": "red",
                "strokeDash": [6, 4], "strokeWidth": 2,
            },
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
            },
        }
        layers.append(line_layer)

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Predicted vs Actual",
        "width": "container",
        "height": height,
        "layer": layers,
    }
    return spec


def render_vega_chart(spec: dict, container) -> None:
    """Render a Vega-Lite spec via vegaEmbed with polling, try/catch, and tooltips."""
    ensure_vega_head_scripts()
    vis_id = f"vis-{uuid.uuid4().hex[:12]}"
    spinner_id = f"spinner-{vis_id}"
    spec_json = json.dumps(spec)
    logger.info("render_vega_chart: chart_id=%s spec_size=%d chart_type=prediction", vis_id, len(spec_json))

    with container:
        ui.html(
            f'<div id="{vis_id}" style="width:100%; position:relative;">'
            f'<div id="{spinner_id}" style="display:flex; align-items:center;'
            f' justify-content:center; height:500px;">'
            f'<span style="font-size:1.2em; color:#888;">Loading chart\u2026</span>'
            f"</div></div>",
            sanitize=False,
        ).classes("w-full")

    js_code = _build_vega_embed_js(vis_id, spinner_id, spec_json)
    ui.run_javascript(js_code)


def _build_vega_embed_js(vis_id: str, spinner_id: str, spec_json: str) -> str:
    """Build JS code for vegaEmbed with polling, try/catch, and error display."""
    es = ("color:red; display:flex; align-items:center; "
          "justify-content:center; height:500px; font-size:1.1em;")
    diag = ("function _diag(evt,err){"
            "fetch('/api/chart_diagnostic',{method:'POST',"
            "headers:{'Content-Type':'application/json'},"
            f"body:JSON.stringify({{event:evt,chart_id:'{vis_id}',"
            "error:err||'',chart_type:'prediction'})"
            "}).catch(function(){});}")
    return (
        f"(function() {{\n"
        f"  {_vega_poll_js()}\n"
        f"  {diag}\n"
        f"  var spec = {spec_json};\n"
        f"  waitForVegaEmbed(function(err) {{\n"
        f"    var sp = document.getElementById('{spinner_id}');\n"
        f"    if (err) {{\n"
        f"      if (sp) sp.innerHTML = '<div style=\"{es}\">"
        f"\u26A0\uFE0F Timeout: vegaEmbed not available after 15s</div>';\n"
        f"      _diag('vega_error','timeout: '+err.message); return;\n"
        f"    }}\n"
        f"    try {{\n"
        f"      vegaEmbed('#{vis_id}', spec, {{actions: true}}).then(function(result) {{\n"
        f"        if (sp) sp.remove();\n"
        f"        _diag('vega_success','');\n"
        f"        {_tooltip_handler_js(vis_id)}\n"
        f"      }}).catch(function(asyncErr) {{\n"
        f"        if (sp) sp.innerHTML = '<div style=\"{es}\">"
        f"\u26A0\uFE0F Chart error: '+asyncErr.message+'</div>';\n"
        f"        console.error('vegaEmbed async error:', asyncErr);\n"
        f"        _diag('vega_error','async: '+asyncErr.message);\n"
        f"      }});\n"
        f"    }} catch (e) {{\n"
        f"      if (sp) sp.innerHTML = '<div style=\"{es}\">"
        f"\u26A0\uFE0F '+e.name+': '+e.message+'</div>';\n"
        f"      console.error('vegaEmbed sync error for {vis_id}:', e);\n"
        f"      _diag('vega_error','sync: '+e.name+': '+e.message);\n"
        f"    }}\n"
        f"  }}, '{vis_id}');\n"
        f"}})()"
    )


def _tooltip_handler_js(vis_id: str) -> str:
    """Return JS code for debounced molecule image tooltip with LRU cache and text labels."""
    return (
        "var view = result.view;\n"
        "var ttDiv = document.createElement('div');\n"
        "ttDiv.style.cssText = 'position:fixed; pointer-events:none; "
        "background:#fff; border:1px solid #ccc; border-radius:4px; "
        "padding:4px; box-shadow:0 2px 8px rgba(0,0,0,0.15); "
        "z-index:10000; display:none;';\n"
        "document.body.appendChild(ttDiv);\n"
        "\n"
        "var LRU_MAX = 200;\n"
        "var imgCache = new Map();\n"
        "function lruGet(k) {\n"
        "  if (!imgCache.has(k)) return null;\n"
        "  var v = imgCache.get(k); imgCache.delete(k); imgCache.set(k, v);\n"
        "  return v;\n"
        "}\n"
        "function lruSet(k, v) {\n"
        "  if (imgCache.has(k)) imgCache.delete(k);\n"
        "  imgCache.set(k, v);\n"
        "  if (imgCache.size > LRU_MAX) {\n"
        "    imgCache.delete(imgCache.keys().next().value);\n"
        "  }\n"
        "}\n"
        "\n"
        "function buildTextHtml(datum) {\n"
        "  return '<div style=\"font-size:12px; line-height:1.4;\">'"
        " + '<b>' + (datum.Trial || '') + '</b> &middot; ' + (datum.Set || '') + '<br>'"
        " + 'Actual: ' + (typeof datum.Actual === 'number' ? datum.Actual.toFixed(4) : '') + '<br>'"
        " + 'Predicted: ' + (typeof datum.Predicted === 'number' ? datum.Predicted.toFixed(4) : '')"
        " + '</div>';\n"
        "}\n"
        "\n"
        "var debounceTimer = null;\n"
        "var abortCtrl = null;\n"
        "\n"
        f"var container = document.getElementById('{vis_id}');\n"
        "container.addEventListener('mousemove', function(evt) {\n"
        "  ttDiv.style.left = (evt.clientX + 12) + 'px';\n"
        "  ttDiv.style.top = (evt.clientY + 12) + 'px';\n"
        "});\n"
        "\n"
        "view.addEventListener('mouseover', function(evt, item) {\n"
        "  if (!item || !item.datum || !item.datum.smiles) {\n"
        "    ttDiv.style.display = 'none';\n"
        "    if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }\n"
        "    if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }\n"
        "    return;\n"
        "  }\n"
        "  var d = item.datum;\n"
        "  var smi = d.smiles;\n"
        "  var textHtml = buildTextHtml(item.datum);\n"
        "  var cached = lruGet(smi);\n"
        "  if (cached) {\n"
        "    ttDiv.innerHTML = textHtml + '<div style=\"margin-top:4px;\">'"
        " + '<img src=\"' + cached + '\" "
        "style=\"max-width:200px; max-height:200px;\"></div>';\n"
        "    ttDiv.style.display = 'block';\n"
        "    return;\n"
        "  }\n"
        "  ttDiv.innerHTML = textHtml + '<div class=\"tt-img\"></div>';\n"
        "  ttDiv.style.display = 'block';\n"
        "  if (debounceTimer) clearTimeout(debounceTimer);\n"
        "  if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }\n"
        "  debounceTimer = setTimeout(function() {\n"
        "    abortCtrl = new AbortController();\n"
        "    var b64 = btoa(smi).replace(/\\+/g, '-')"
        ".replace(/\\//g, '_').replace(/=+$/, '');\n"
        "    fetch('/api/mol_image/' + b64, {signal: abortCtrl.signal})\n"
        "      .then(function(r) { return r.ok ? r.blob() : null; })\n"
        "      .then(function(blob) {\n"
        "        if (!blob) return;\n"
        "        var url = URL.createObjectURL(blob);\n"
        "        lruSet(smi, url);\n"
        "        var imgDiv = ttDiv.querySelector('.tt-img');\n"
        "        if (imgDiv) {\n"
        "          imgDiv.innerHTML = '<img src=\"' + url + '\" "
        "style=\"max-width:200px; max-height:200px;\">';\n"
        "          imgDiv.style.marginTop = '4px';\n"
        "        }\n"
        "      })\n"
        "      .catch(function(e) {\n"
        "        if (e.name !== 'AbortError') console.warn('tooltip fetch:', e);\n"
        "      });\n"
        "  }, 180);\n"
        "});\n"
        "\n"
        "view.addEventListener('mouseout', function() {\n"
        "  ttDiv.style.display = 'none';\n"
        "  ttDiv.innerHTML = '';\n"
        "  if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null; }\n"
        "  if (abortCtrl) { abortCtrl.abort(); abortCtrl = null; }\n"
        "});\n"
    )
