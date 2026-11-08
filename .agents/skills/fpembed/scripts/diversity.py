"""Explore an embedding matrix with exemplars, outliers, or neighbor queries.

Distances are computed in NumPy only. Exemplar picking is deterministic
max-min selection, outlier scoring ranks rows by their k-th nearest neighbor
distance, and neighbor mode answers queries against the reference matrix.
There is no random path, so ties always break by the lowest row index.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from _common import emit, fail, load_matrix, run_doctor


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Pick exemplars, score outliers, or query neighbors in embeddings."
    )
    parser.add_argument("--input", help=".npy or .csv embeddings matrix")
    parser.add_argument("--ids", help="one-column ID file or first CSV column")
    parser.add_argument(
        "--mode", choices=("exemplars", "outliers", "neighbors"), help="analysis to run"
    )
    parser.add_argument("--n-exemplars", type=int, help="exemplar count")
    parser.add_argument("--k", type=int, help="neighbor count")
    parser.add_argument(
        "--metric", choices=("euclidean", "cosine"), default="euclidean"
    )
    parser.add_argument("--query", help=".npy or .csv query matrix (neighbors only)")
    parser.add_argument("--output", help="optional CSV output path")
    parser.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    return parser


def _as_2d(array: Any) -> np.ndarray:
    """Return a float64 2D matrix, treating a single vector as one row."""
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embedding input must be a 2D matrix")
    return matrix


def _pairwise(left: np.ndarray, right: np.ndarray, metric: str) -> np.ndarray:
    """Return the pairwise distance matrix between two row sets."""
    if metric == "euclidean":
        left_sq = np.einsum("ij,ij->i", left, left)[:, None]
        right_sq = np.einsum("ij,ij->i", right, right)[None, :]
        squared = left_sq + right_sq - 2.0 * (left @ right.T)
        np.clip(squared, 0.0, None, out=squared)
        return np.sqrt(squared)

    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    left_safe = np.where(left_norm == 0.0, 1.0, left_norm)
    right_safe = np.where(right_norm == 0.0, 1.0, right_norm)
    similarity = (left @ right.T) / (left_safe[:, None] * right_safe[None, :])
    similarity[left_norm == 0.0, :] = 0.0
    similarity[:, right_norm == 0.0] = 0.0
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _select_exemplars(
    matrix: np.ndarray, count: int, metric: str
) -> tuple[list[int], list[float]]:
    """Pick max-min exemplars, seeding from the point farthest from the centroid."""
    centroid_distance = _pairwise(matrix.mean(axis=0, keepdims=True), matrix, metric)[0]
    distances = _pairwise(matrix, matrix, metric)
    selected = [int(np.argmax(centroid_distance))]
    scores = [float(centroid_distance[selected[0]])]
    nearest = distances[selected[0]].copy()
    nearest[selected[0]] = -np.inf
    while len(selected) < count:
        candidate = int(np.argmax(nearest))
        selected.append(candidate)
        scores.append(float(nearest[candidate]))
        nearest = np.minimum(nearest, distances[candidate])
        nearest[candidate] = -np.inf
    return selected, scores


def _outlier_scores(matrix: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Return each row's distance to its k-th nearest neighbor."""
    distances = _pairwise(matrix, matrix, metric)
    np.fill_diagonal(distances, np.inf)
    return np.sort(distances, axis=1)[:, k - 1]


def _neighbors(
    reference: np.ndarray, query: np.ndarray, k: int, metric: str
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return each query's k nearest reference indices and distances."""
    distances = _pairwise(query, reference, metric)
    results: list[tuple[int, np.ndarray, np.ndarray]] = []
    for row in range(distances.shape[0]):
        order = np.argsort(distances[row], kind="stable")[:k]
        results.append((row, order, distances[row, order]))
    return results


def _load_ids(path: str | Path) -> list[str]:
    """Read IDs from a .npy file, one per line, or the first CSV column."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return [str(value) for value in np.ravel(np.load(path))]
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            return [row[0] for row in csv.reader(handle) if row]
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _write_csv(path: str | Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write rows to a CSV file, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _envelope(mode: str, n: int, metric: str) -> dict[str, Any]:
    """Return the shared result fields for one mode."""
    return {"status": "ok", "command": "diversity", "mode": mode, "n": n, "metric": metric}


def _score_rows(
    indices: list[int], scores: list[float], ids: list[str] | None
) -> list[list[Any]]:
    """Return index, optional ID, score rows for a CSV export."""
    if ids is None:
        return [[index, score] for index, score in zip(indices, scores)]
    return [[index, ids[index], score] for index, score in zip(indices, scores)]


def _run_exemplars(
    matrix: np.ndarray, args: argparse.Namespace, ids: list[str] | None
) -> int:
    """Pick exemplars and emit the result."""
    n = matrix.shape[0]
    count = args.n_exemplars
    if count is None:
        return fail("--n-exemplars is required for exemplars mode", "diversity")
    if not 1 <= count <= n:
        return fail(f"--n-exemplars must be between 1 and {n}", "diversity")

    indices, scores = _select_exemplars(matrix, count, args.metric)
    payload = _envelope("exemplars", n, args.metric)
    payload["n_exemplars"] = count
    payload["exemplar_indices"] = indices
    payload["exemplar_scores"] = scores
    if ids is not None:
        payload["exemplar_ids"] = [ids[index] for index in indices]
    if args.output is not None:
        header = ["index", "score"] if ids is None else ["index", "id", "score"]
        _write_csv(args.output, header, _score_rows(indices, scores, ids))
        payload["output"] = str(args.output)
    emit(payload)
    return 0


def _run_outliers(
    matrix: np.ndarray, args: argparse.Namespace, ids: list[str] | None
) -> int:
    """Rank rows by their k-th nearest neighbor distance."""
    n = matrix.shape[0]
    k = args.k
    if k is None:
        return fail("--k is required for outliers mode", "diversity")
    if not 1 <= k <= n - 1:
        return fail(f"--k must be between 1 and {n - 1}", "diversity")

    scores = _outlier_scores(matrix, k, args.metric)
    order = np.argsort(-scores, kind="stable")
    indices = [int(index) for index in order]
    ranked = [float(scores[index]) for index in order]
    payload = _envelope("outliers", n, args.metric)
    payload["k"] = k
    payload["outlier_indices"] = indices
    payload["outlier_scores"] = ranked
    if ids is not None:
        payload["outlier_ids"] = [ids[index] for index in indices]
    if args.output is not None:
        header = ["index", "score"] if ids is None else ["index", "id", "score"]
        _write_csv(args.output, header, _score_rows(indices, ranked, ids))
        payload["output"] = str(args.output)
    emit(payload)
    return 0


def _run_neighbors(
    matrix: np.ndarray, args: argparse.Namespace, ids: list[str] | None
) -> int:
    """Return the k nearest reference rows for every query row."""
    n = matrix.shape[0]
    if args.query is None:
        return fail("--query is required for neighbors mode", "diversity")
    k = args.k
    if k is None:
        return fail("--k is required for neighbors mode", "diversity")
    if not 1 <= k <= n:
        return fail(f"--k must be between 1 and {n}", "diversity")

    query = _as_2d(load_matrix(args.query))
    if query.shape[1] != matrix.shape[1]:
        return fail(
            f"query dimension {query.shape[1]} does not match input "
            f"dimension {matrix.shape[1]}",
            "diversity",
        )

    neighbors: list[dict[str, Any]] = []
    csv_rows: list[list[Any]] = []
    for query_index, order, distances in _neighbors(matrix, query, k, args.metric):
        indices = [int(index) for index in order]
        entry: dict[str, Any] = {
            "query_index": query_index,
            "indices": indices,
            "distances": [float(value) for value in distances],
        }
        if ids is not None:
            entry["ids"] = [ids[index] for index in indices]
        neighbors.append(entry)
        for rank, (index, distance) in enumerate(zip(indices, distances), start=1):
            row: list[Any] = [query_index, rank, index]
            if ids is not None:
                row.append(ids[index])
            csv_rows.append(row + [float(distance)])

    payload = _envelope("neighbors", n, args.metric)
    payload["n_queries"] = query.shape[0]
    payload["k"] = k
    payload["neighbors"] = neighbors
    if args.output is not None:
        header = ["query_index", "rank", "index"]
        if ids is not None:
            header.append("id")
        _write_csv(args.output, header + ["distance"], csv_rows)
        payload["output"] = str(args.output)
    emit(payload)
    return 0


def _run(args: argparse.Namespace) -> int:
    """Load the matrix and dispatch on the requested mode."""
    matrix = _as_2d(load_matrix(args.input))
    ids = None
    if args.ids is not None:
        ids = _load_ids(args.ids)
        if len(ids) != matrix.shape[0]:
            return fail(
                f"--ids has {len(ids)} values but the matrix has {matrix.shape[0]} rows",
                "diversity",
            )

    if args.mode == "exemplars":
        return _run_exemplars(matrix, args, ids)
    if args.mode == "outliers":
        return _run_outliers(matrix, args, ids)
    return _run_neighbors(matrix, args, ids)


def main(argv: list[str] | None = None) -> int:
    """Run the diversity CLI and return the exit code."""
    args = _build_parser().parse_args(argv)

    if args.doctor:
        doctor = run_doctor()
        installed = bool(doctor["fpembed"]["installed"])
        emit({"status": "ok" if installed else "error", "command": "diversity", **doctor})
        return 0 if installed else 1

    if args.input is None:
        return fail("--input is required unless --doctor is given", "diversity")
    if args.mode is None:
        return fail("--mode is required unless --doctor is given", "diversity")

    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary preserves the library message
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
