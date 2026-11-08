"""Fit and apply a scikit-learn QSAR surrogate on FPembed embeddings.

``fit`` trains Ridge or LogisticRegression on compressed fingerprints,
``predict`` rebuilds the generator from the saved config, and invalid SMILES
are reported rather than silently dropped. The generator is never pickled.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

from _common import (
    embed_structures,
    emit,
    fail,
    load_structures,
    projection_seed,
    resolve_input_format,
    run_doctor,
)


def _sklearn_available() -> bool:
    """Report whether scikit-learn is importable without importing it."""
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):
        return False


def _add_embedding_flags(parser: argparse.ArgumentParser) -> None:
    """Add the fingerprint embedding flags shared with featurize."""
    parser.add_argument("--fp-type", default="ecfp", help="fingerprint type")
    parser.add_argument("--fp-size", type=int, default=2048, help="fingerprint length")
    parser.add_argument("--compression", type=int, default=16, help="compression factor")
    parser.add_argument("--count", action="store_true", help="count mode")
    parser.add_argument("--count-transform", choices=("none", "log1p"),
                        default="none", help="count transform")
    parser.add_argument("--method", help="compression method (library default if omitted)")
    parser.add_argument("--dtype", choices=("float64", "float32", "uint16"),
                        default="float64", help="output dtype")


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Fit or apply a scikit-learn QSAR surrogate on FPembed embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="train a surrogate and save a joblib bundle")
    fit.add_argument("--smiles-csv", help="CSV or .sdf/.mol input with structures and targets")
    fit.add_argument("--smiles-col", default="smiles", help="SMILES or SELFIES column name")
    fit.add_argument("--target-col", help="target column name (CSV input)")
    fit.add_argument("--target-prop", help="target property name (mol input)")
    fit.add_argument("--id-col", help="optional ID column name")
    fit.add_argument("--id-prop", default="_Name", help="SDF property holding IDs")
    fit.add_argument(
        "--input-format", choices=("smiles", "selfies", "mol"), default="smiles",
        help="input entity format; .sdf and .mol suffixes imply mol",
    )
    fit.add_argument("--task", choices=("regression", "classification"),
                     default="regression", help="model task")
    fit.add_argument("--split", choices=("random", "scaffold"), default="random",
                     help="train/test split strategy")
    fit.add_argument("--seed", type=int, default=0, help="random seed")
    fit.add_argument("--test-size", type=float, default=0.2, help="test fraction")
    fit.add_argument("--cv", type=int, help="optional k-fold cross-validation")
    fit.add_argument("--model-out", default="surrogate.joblib", help="bundle output path")
    fit.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    _add_embedding_flags(fit)

    predict = subparsers.add_parser(
        "predict", help="apply a saved surrogate and write predictions")
    predict.add_argument("--model-in", help="joblib bundle from fit")
    predict.add_argument("--smiles-csv", help="CSV, .smi, .sdf, or .mol input")
    predict.add_argument("--smiles-col", default="smiles", help="SMILES or SELFIES column name")
    predict.add_argument("--id-col", help="optional ID column name")
    predict.add_argument("--id-prop", default="_Name", help="SDF property holding IDs")
    predict.add_argument(
        "--input-format", choices=("smiles", "selfies", "mol"), default=None,
        help="input entity format; defaults to the format stored in the model",
    )
    predict.add_argument("--output-csv", default="predictions.csv",
                         help="predictions output path")
    predict.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    return parser


def _build_generator(args: argparse.Namespace) -> Any:
    """Construct the embedding generator from the parsed fit flags."""
    from fpembed import EmbeddedFingerprintGenerator

    return EmbeddedFingerprintGenerator(
        fp_type=args.fp_type, fp_size=args.fp_size, compression=args.compression,
        fp_params={}, cache_size=None, method=args.method,
        method_params=projection_seed(args.method, args.seed),
        dtype=args.dtype, count=args.count, count_transform=args.count_transform)


def _fp_config(gen: Any) -> dict[str, Any]:
    """Return the constructor kwargs that rebuild the generator exactly."""
    return {
        "fp_type": gen.fp_type, "fp_size": gen.fp_size,
        "compression": gen.compression, "fp_params": {}, "cache_size": None,
        "method": gen.method, "method_params": gen.method_params,
        "dtype": str(gen.dtype), "count": gen.count,
        "count_transform": gen.count_transform,
    }


def _random_split(X: Any, y: Any, task: str, args: argparse.Namespace):
    """Split rows with train_test_split, stratifying classification labels."""
    from sklearn.model_selection import train_test_split

    stratify = y if task == "classification" else None
    return train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify)


def _scaffold_split(structures: list, test_size: float) -> tuple[Any, Any]:
    """Assign whole Bemis-Murcko scaffold groups to train or test rows."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(structures):
        mol = Chem.MolFromSmiles(entry) if isinstance(entry, str) else entry
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=mol), []).append(index)

    target = math.floor(float(test_size) * len(structures))
    test_indices: list[int] = []
    train_indices: list[int] = []
    for key in sorted(groups):
        indices = groups[key]
        if len(test_indices) + len(indices) <= target:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not test_indices or not train_indices:
        raise ValueError(
            "scaffold split produced an empty train or test set; "
            "add more scaffolds or change --test-size")
    return (np.sort(np.asarray(train_indices, dtype=int)),
            np.sort(np.asarray(test_indices, dtype=int)))


def _make_model(task: str, seed: int) -> Any:
    """Return an unfitted scikit-learn estimator for the task."""
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(max_iter=1000, random_state=seed)


def _metrics(task: str, model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    """Compute the task-specific metrics on the held-out split."""
    if task == "regression":
        from sklearn.metrics import (
            mean_absolute_error,
            r2_score,
            root_mean_squared_error,
        )

        predicted = model.predict(X_test)
        return {
            "r2": float(r2_score(y_test, predicted)),
            "rmse": float(root_mean_squared_error(y_test, predicted)),
            "mae": float(mean_absolute_error(y_test, predicted)),
        }

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    predicted = model.predict(X_test)
    result: dict[str, Any] = {"roc_auc": None}
    try:
        probabilities = model.predict_proba(X_test)[:, -1]
        result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    except (AttributeError, ValueError):
        pass
    result["f1"] = float(f1_score(y_test, predicted))
    result["accuracy"] = float(accuracy_score(y_test, predicted))
    return result


def _cross_validate(
    task: str, seed: int, X: Any, y: Any, folds: int,
) -> dict[str, Any]:
    """Run shuffled k-fold cross-validation on the full dataset."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    if task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = "accuracy"
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scoring = "r2"

    scores = cross_val_score(_make_model(task, seed), X, y, cv=splitter, scoring=scoring)
    return {
        "folds": folds, "scoring": scoring, "mean": float(scores.mean()),
        "std": float(scores.std()), "scores": [float(value) for value in scores],
    }


def _write_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    """Persist the model bundle, creating parent directories."""
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)


def _run_fit(args: argparse.Namespace) -> int:
    """Train a surrogate, save the bundle, and emit the JSON envelope."""
    import numpy as np

    import fpembed

    input_format = resolve_input_format(args.smiles_csv, args.input_format)
    structures, ids, targets = load_structures(
        args.smiles_csv, input_format, smiles_col=args.smiles_col,
        id_col=args.id_col, target_col=args.target_col,
        target_prop=args.target_prop, id_prop=args.id_prop,
    )
    gen = _build_generator(args)
    X, invalid = embed_structures(gen, structures, input_format)

    invalid_set = set(invalid)
    kept = [index for index in range(len(structures)) if index not in invalid_set]
    y = np.asarray([targets[index] for index in kept], dtype=float)
    if y.shape[0] < 2:
        raise ValueError("at least two valid rows with targets are required to fit")
    kept_structures = [structures[index] for index in kept]

    if args.split == "scaffold":
        train_indices, test_indices = _scaffold_split(kept_structures, args.test_size)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        X_train, X_test, y_train, y_test = _random_split(X, y, args.task, args)

    model = _make_model(args.task, args.seed)
    model.fit(X_train, y_train)
    metrics = _metrics(args.task, model, X_test, y_test)

    bundle = {
        "fp_config": _fp_config(gen), "manifest": gen.representation_manifest(),
        "model": model, "task": args.task, "target_col": args.target_col,
        "input_format": input_format, "fpembed_version": fpembed.__version__,
    }
    _write_bundle(args.model_out, bundle)

    result: dict[str, Any] = {
        "status": "ok", "command": "qsar", "subcommand": "fit",
        "task": args.task, "n_input": len(structures), "n_valid": int(y.shape[0]),
        "n_dropped": len(invalid), "invalid_indices": sorted(invalid),
        "train_size": int(X_train.shape[0]), "test_size": int(X_test.shape[0]),
        "metrics": metrics, "model_out": str(args.model_out),
        "fpembed_version": fpembed.__version__,
    }
    if ids is not None:
        result["invalid_ids"] = [ids[index] for index in sorted(invalid)]
    if args.cv is not None:
        result["cv"] = _cross_validate(args.task, args.seed, X, y, args.cv)
    emit(result)
    return 0


def _display_structures(structures: list, input_format: str) -> list[str]:
    """Return text to write in the prediction CSV, converting Mols to SMILES."""
    if input_format != "mol":
        return list(structures)
    from rdkit import Chem

    return ["" if value is None else Chem.MolToSmiles(value) for value in structures]


def _write_predictions(
    path: str | Path, smiles: list[str], ids: list[str] | None,
    kept: list[int], predictions: Any, probabilities: Any,
) -> None:
    """Write valid predictions with optional IDs, SMILES, and probabilities."""
    header = (["id"] if ids is not None else []) + ["smiles", "prediction"]
    if probabilities is not None:
        header.append("probability")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, source in enumerate(kept):
            values = ([ids[source]] if ids is not None else []) + [
                smiles[source], predictions[row]]
            if probabilities is not None:
                values.append(float(probabilities[row]))
            writer.writerow(values)


def _run_predict(args: argparse.Namespace) -> int:
    """Apply a saved bundle and emit the JSON envelope."""
    import joblib

    import fpembed
    from fpembed import EmbeddedFingerprintGenerator

    bundle = joblib.load(args.model_in)
    gen = EmbeddedFingerprintGenerator(**bundle["fp_config"])
    input_format = args.input_format or bundle.get("input_format", "smiles")
    structures, ids, _ = load_structures(
        args.smiles_csv, input_format, smiles_col=args.smiles_col,
        id_col=args.id_col, id_prop=args.id_prop,
    )
    X, invalid = embed_structures(gen, structures, input_format)

    invalid_set = set(invalid)
    kept = [index for index in range(len(structures)) if index not in invalid_set]
    model = bundle["model"]
    task = bundle["task"]

    predictions: Any = []
    probabilities: Any = None
    if kept:
        predictions = model.predict(X)
        if task == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, -1]

    display = _display_structures(structures, input_format)
    _write_predictions(args.output_csv, display, ids, kept, predictions, probabilities)

    result: dict[str, Any] = {
        "status": "ok", "command": "qsar", "subcommand": "predict",
        "n_input": len(structures), "n_valid": len(kept), "n_dropped": len(invalid),
        "invalid_indices": sorted(invalid), "output_csv": str(args.output_csv),
        "task": task, "fpembed_version": fpembed.__version__,
    }
    if ids is not None:
        result["invalid_ids"] = [ids[index] for index in sorted(invalid)]
    emit(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the qsar CLI and return the exit code."""
    args = _build_parser().parse_args(argv)

    if args.doctor:
        doctor = run_doctor()
        installed = bool(doctor["fpembed"]["installed"])
        emit({"status": "ok" if installed else "error", "command": "qsar", **doctor})
        return 0 if installed else 1

    if not _sklearn_available():
        return fail("scikit-learn is required for QSAR; pip install scikit-learn", "qsar")

    try:
        if args.command == "fit":
            if not args.smiles_csv:
                return fail("--smiles-csv is required for fit", "qsar")
            input_format = resolve_input_format(args.smiles_csv, args.input_format)
            if input_format == "mol":
                if not args.target_prop:
                    return fail("--target-prop is required for mol input", "qsar")
            elif not args.target_col:
                return fail("--target-col is required for CSV input", "qsar")
            return _run_fit(args)
        if not args.model_in:
            return fail("--model-in is required for predict", "qsar")
        if not args.smiles_csv:
            return fail("--smiles-csv is required for predict", "qsar")
        return _run_predict(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary preserves the library message
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
