"""Command-line interface for FPembed.

The ``demo`` subcommand featurizes a small built-in set of drug-like SMILES
with the default ECFP configuration and prints one JSON object describing the
run. When scikit-learn is importable it also fits deterministic Ridge and
LogisticRegression models on synthetic targets derived from RDKit descriptors
and reports their in-sample metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from fpembed import EmbeddedFingerprintGenerator, __version__

FP_TYPE = "ecfp"
FP_SIZE = 2048
COMPRESSION = 16

DEMO_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CCO",  # ethanol
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
    "O=C(O)c1ccccc1O",  # salicylic acid
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # salbutamol
    "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",  # naproxen
    "CN1CCC[C@H]1c1ccncc1",  # nicotine
]


def _sklearn_available() -> bool:
    """Return whether scikit-learn can be imported."""
    return importlib.util.find_spec("sklearn") is not None


def _embed(smiles: Sequence[str]) -> np.ndarray:
    """Featurize every SMILES and return the embedding matrix."""
    generator = EmbeddedFingerprintGenerator(
        fp_type=FP_TYPE,
        fp_size=FP_SIZE,
        compression=COMPRESSION,
        fp_params={},
    )
    embeddings, invalid = generator.GetFingerprintsFromSmiles(list(smiles))
    if invalid:
        raise RuntimeError(f"built-in demo SMILES failed to parse at indices {invalid}")
    return embeddings


def _regression_targets(smiles: Sequence[str]) -> np.ndarray:
    """Derive a deterministic synthetic target from RDKit descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    values = []
    for smiles_str in smiles:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise RuntimeError(f"built-in demo SMILES failed to parse: {smiles_str}")
        values.append(0.05 * Descriptors.MolWt(mol) + 0.5 * Descriptors.MolLogP(mol))
    return np.round(np.asarray(values, dtype=float), 4)


def _fit_qsar(embeddings: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Fit the demo regression and classification models and score them."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    regression = Ridge()
    regression.fit(embeddings, targets)
    predictions = regression.predict(embeddings)

    labels = (targets >= np.median(targets)).astype(int)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(embeddings, labels)
    probabilities = classifier.predict_proba(embeddings)[:, 1]
    class_predictions = classifier.predict(embeddings)

    return {
        "regression": {
            "r2": float(r2_score(targets, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
            "mae": float(mean_absolute_error(targets, predictions)),
        },
        "classification": {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "f1": float(f1_score(labels, class_predictions)),
            "accuracy": float(accuracy_score(labels, class_predictions)),
        },
    }


def _run_demo() -> dict[str, Any]:
    """Run the demo and return its JSON-compatible payload."""
    embeddings = _embed(DEMO_SMILES)
    payload: dict[str, Any] = {
        "status": "ok",
        "command": "demo",
        "n": int(embeddings.shape[0]),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "fp_type": FP_TYPE,
        "fp_size": FP_SIZE,
        "compression": COMPRESSION,
        "fpembed_version": __version__,
    }
    if _sklearn_available():
        payload["tier"] = 2
        payload["qsar"] = _fit_qsar(embeddings, _regression_targets(DEMO_SMILES))
    else:
        payload["tier"] = 1
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="fpembed",
        description="FPembed command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "demo",
        help="Featurize a built-in set of SMILES and optionally fit demo QSAR models.",
    )
    parser.parse_args(argv)

    print(json.dumps(_run_demo(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
