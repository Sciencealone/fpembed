"""Progress file persistence for crash recovery.

Handles atomic read/write/delete of the JSON progress file so that
partial optimization results survive server crashes.  All public
functions are safe to call from any thread — failures are logged,
never raised.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any

from storage_helpers import _sanitize_value

logger = logging.getLogger(__name__)

PROGRESS_FILENAME = "_optimization_progress.json"


def get_progress_file_path(app_dir: str) -> str:
    """Return the full path to the progress file.

    Parameters
    ----------
    app_dir:
        The application directory where the progress file is stored.

    Returns
    -------
    str
        Absolute path to ``_optimization_progress.json`` inside *app_dir*.
    """
    return os.path.join(app_dir, PROGRESS_FILENAME)


def save_progress(
    app_dir: str,
    progress_snapshot: dict,
    widget_config: dict,
    ds_name: str,
    metric: str,
    target_col: str,
) -> None:
    """Atomically write progress state to disk.

    Uses a temporary file + ``os.replace()`` so the progress file is
    either fully written or absent — never half-written.

    Logs a warning on any failure and never raises.

    Parameters
    ----------
    app_dir:
        Application directory for the progress file.
    progress_snapshot:
        Current ``Progress_Snapshot`` dict from ``_run_state["progress"]``.
    widget_config:
        Widget configuration captured at optimization start.
    ds_name:
        Dataset name.
    metric:
        Optimization metric (e.g. ``"R2"``).
    target_col:
        Target column name.
    """
    path = get_progress_file_path(app_dir)
    payload: dict[str, Any] = {
        "version": 1,
        "timestamp": time.time(),
        "ds_name": _sanitize_value(ds_name),
        "metric": _sanitize_value(metric),
        "target_col": _sanitize_value(target_col),
        "widget_config": _sanitize_value(widget_config),
        "progress": _sanitize_value(progress_snapshot),
    }
    try:
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            dir=app_dir,
            delete=False,
        )
        try:
            json.dump(payload, fd)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
            os.replace(fd.name, path)
        except BaseException:
            fd.close()
            # Clean up the temp file on failure
            try:
                os.unlink(fd.name)
            except OSError:
                pass
            raise
    except Exception:
        logger.warning("Failed to write progress file %s", path, exc_info=True)


def load_progress(app_dir: str) -> dict | None:
    """Load and validate the progress file.

    Returns ``None`` when the file is missing, corrupt, or lacks
    required keys.  Corrupt files are deleted and a warning is logged.

    Parameters
    ----------
    app_dir:
        Application directory containing the progress file.

    Returns
    -------
    dict | None
        The parsed progress dict, or ``None``.
    """
    path = get_progress_file_path(app_dir)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt progress file %s: %s — deleting", path, exc)
        _safe_delete(path)
        return None

    # Validate required keys
    required_keys = {"version", "timestamp", "progress", "widget_config"}
    if not isinstance(data, dict) or not required_keys.issubset(data):
        logger.warning(
            "Invalid progress file %s: missing keys %s — deleting",
            path,
            required_keys - (data.keys() if isinstance(data, dict) else set()),
        )
        _safe_delete(path)
        return None

    return data


def delete_progress(app_dir: str) -> None:
    """Delete the progress file if it exists.

    Silently ignores missing files.

    Parameters
    ----------
    app_dir:
        Application directory containing the progress file.
    """
    _safe_delete(get_progress_file_path(app_dir))


def _safe_delete(path: str) -> None:
    """Remove *path* if it exists; silently ignore ``FileNotFoundError``."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
