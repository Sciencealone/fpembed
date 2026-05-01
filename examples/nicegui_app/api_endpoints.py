"""FastAPI endpoints for prediction chart data and molecule images.

Serves prediction data via HTTP GET (bypassing WebSocket size limits)
and molecule images on-demand with caching headers.
"""

import base64
import json
import logging
from typing import Optional

import pandas as pd
from nicegui import app
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

MAX_CHART_POINTS = 2000


def _downsample_records(records: list[dict], max_points: int = MAX_CHART_POINTS) -> list[dict]:
    """Stratified downsample records by (Trial, Set) if count exceeds max_points."""
    if len(records) <= max_points:
        return records
    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    group_cols = [c for c in ("Trial", "Set") if c in df.columns]
    if not group_cols:
        return df.sample(n=max_points, random_state=42).to_dict(orient="records")
    frac = max_points / len(df)
    indices: list[int] = []
    for _, grp in df.groupby(group_cols):
        n = max(1, int(len(grp) * frac))
        indices.extend(grp.sample(n=n, random_state=42).index.tolist())
    sampled = df.loc[sorted(indices[:max_points])]
    return sampled.to_dict(orient="records")


class _DownsamplingStore(dict):
    """Dict subclass that applies stratified downsampling on get()."""

    def get(self, key, default=None):
        raw = super().get(key, default)
        if raw is None or not isinstance(raw, list):
            return raw
        return _downsample_records(raw)


# Module-level store: prediction data keyed by comma-separated trial IDs
_prediction_store: dict = _DownsamplingStore()

# Module-level cache_db reference, set via register_api_endpoints()
_cache_db = None


def register_api_endpoints(cache_db) -> None:
    """Store cache_db reference and register API endpoints on the NiceGUI app.

    Args:
        cache_db: SQLiteCacheDB instance with get_images_batch() method.
    """
    global _cache_db
    _cache_db = cache_db

    @app.get("/api/prediction_data")
    async def get_prediction_data(trial_ids: str = "") -> JSONResponse:
        """Return prediction data as JSON array for the requested trial IDs.

        Query params:
            trial_ids: comma-separated trial numbers (e.g. "1,3,5")

        Returns:
            JSONResponse with list of {Actual, Predicted, Set, Trial,
            Color, point_index, smiles} records.
        """
        key = trial_ids.strip()
        raw_records = dict.get(_prediction_store, key, [])
        raw_count = len(raw_records)
        records = _downsample_records(raw_records)
        response_bytes = len(json.dumps(records).encode("utf-8"))
        logger.info(
            "prediction_data key=%s requested=%d returned=%d size_bytes=%d",
            key, raw_count, len(records), response_bytes,
        )
        return JSONResponse(content=records)

    @app.get("/api/mol_image/{smiles_b64}")
    async def get_mol_image(smiles_b64: str) -> Response:
        """Return PNG image bytes for a base64url-encoded SMILES string.

        Path params:
            smiles_b64: base64url-encoded SMILES (standard base64 with
                        + → -, / → _, no padding required)

        Returns:
            PNG image with Cache-Control headers, or 404 on failure.
        """
        try:
            padded = smiles_b64 + "=" * (-len(smiles_b64) % 4)
            smiles = base64.urlsafe_b64decode(padded).decode("utf-8")
        except Exception:
            return Response(
                content=b"Invalid SMILES encoding",
                status_code=404,
                media_type="text/plain",
            )

        if not smiles:
            return Response(
                content=b"Empty SMILES",
                status_code=404,
                media_type="text/plain",
            )

        png_bytes = _resolve_mol_image(smiles)
        if png_bytes is None:
            return Response(
                content=b"Could not generate image",
                status_code=404,
                media_type="text/plain",
            )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/chart_diagnostic")
    async def post_chart_diagnostic(request_body: dict) -> Response:
        """Log a chart diagnostic event and return 204 No Content."""
        event = request_body.get("event", "unknown")
        chart_id = request_body.get("chart_id", "unknown")
        elapsed_ms = request_body.get("elapsed_ms")
        error = request_body.get("error")
        chart_type = request_body.get("chart_type")
        logger.info(
            "chart_diagnostic event=%s chart_id=%s elapsed_ms=%s error=%s chart_type=%s",
            event, chart_id, elapsed_ms, error, chart_type,
        )
        return Response(status_code=204)


def _resolve_mol_image(smiles: str) -> Optional[bytes]:
    """Look up or generate a molecule PNG image for the given SMILES.

    Tries cache_db first, falls back to direct generation.
    Returns raw PNG bytes or None on failure.
    """
    data_uri = None

    if _cache_db is not None:
        try:
            results = _cache_db.get_images_batch([smiles])
            if results and results[0] and results[0] != "N/A":
                data_uri = results[0]
        except Exception:
            logger.debug("Cache lookup failed for %s", smiles, exc_info=True)

    if data_uri is None:
        try:
            from chemistry import smiles_to_image_data_uri
            data_uri = smiles_to_image_data_uri(smiles)
        except Exception:
            logger.debug("Image generation failed for %s", smiles, exc_info=True)
            return None

    if data_uri is None or data_uri == "N/A":
        return None

    return _extract_png_bytes(data_uri)


def _extract_png_bytes(data_uri: str) -> Optional[bytes]:
    """Extract raw PNG bytes from a data:image/png;base64,... URI."""
    prefix = "data:image/png;base64,"
    if not data_uri.startswith(prefix):
        return None
    try:
        return base64.b64decode(data_uri[len(prefix):])
    except Exception:
        return None


def populate_prediction_store(trial_ids: list[int], df) -> None:
    """Store prediction DataFrame records for the API endpoint.

    Args:
        trial_ids: List of trial numbers included in the data.
        df: pandas DataFrame with columns Actual, Predicted, Set,
            Trial, Color, point_index, smiles.
    """
    key = ",".join(str(t) for t in sorted(trial_ids))
    records = df.to_dict(orient="records")
    clean = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if hasattr(v, "item"):
                row[k] = v.item()
            elif v is None:
                row[k] = None
            else:
                row[k] = v
        clean.append(row)
    _prediction_store[key] = clean


def populate_prediction_store_from_records(
    trial_ids: list[int], records: list[dict]
) -> None:
    """Store pre-cleaned prediction records for the API endpoint.

    Unlike :func:`populate_prediction_store` which accepts a DataFrame,
    this function accepts records that are already serialized (e.g. from
    ``app.storage.general`` after a page reload).

    Args:
        trial_ids: List of trial numbers included in the data.
        records: List of dicts with keys Actual, Predicted, Set,
            Trial, Color, point_index, smiles.
    """
    key = ",".join(str(t) for t in sorted(trial_ids))
    _prediction_store[key] = list(records)


def clear_prediction_store() -> None:
    """Remove all entries from the prediction store."""
    _prediction_store.clear()
