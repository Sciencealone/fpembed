---
name: fpembed
description: FPembed: ultra-fast molecular fingerprint embedding and QSAR modeling from SMILES, SELFIES, or RDKit Mols. Use when predicting molecular properties (regression or classification), when a dataset is large enough that full-length molecular fingerprints are too slow or memory-heavy to model directly, or when compressing external embeddings and exploring chemical space (exemplars, outliers).
license: MIT
---

# FPembed

## When to use

- Predicting a molecular property, regression or classification (QSAR), from a table of SMILES or SELFIES, or from RDKit Mols in an SDF or MOL file.
- A dataset large enough that full-length molecular fingerprints are too slow or too memory-heavy to model directly.
- Compressing external embeddings or exploring chemical space: exemplar picking or outlier scoring.

## First step

Run the doctor before anything else. The commands in this file run from this skill's folder, so script paths are relative to it.

```
python scripts/featurize.py --doctor
```

The doctor reports whether the fpembed package imports, its version against the minimum of 0.3.0, whether numpy and scikit-learn are present, and the available tier. Tier 1 needs FPembed and numpy and covers featurize, compress, and diversity. Tier 2 adds scikit-learn and covers qsar.

## Scripts

| Script | Purpose | Tier |
|---|---|---|
| `featurize.py` | Turn SMILES, SELFIES, or RDKit Mols into dense fingerprint embeddings, batch or streaming, with an invalid-row audit. | 1 |
| `compress.py` | Compress an external embedding matrix or a raw fingerprint array. | 1 |
| `qsar_engine.py` | Fit and predict regression or classification models from SMILES, SELFIES, or RDKit Mols. | 2 |
| `diversity.py` | Pick max-min exemplars, score outliers, or query nearest neighbors. | 1 |

## Quick start

Run these three commands from this skill's folder, in order. The bundled demo fixture has a `smiles` column and a regression `target` column.

```
python scripts/featurize.py --doctor
python scripts/featurize.py --input resources/demo.csv --output embeddings.npy
python scripts/qsar_engine.py fit --smiles-csv resources/demo.csv --target-col target --model-out surrogate.joblib
```

Each script prints one JSON object to stdout. Diagnostics go to stderr, and a failure exits non-zero. Every random path accepts `--seed`; the block-wise compression methods are deterministic.

## Detail

See `references/` for install and auto-discovery paths, the full flag tables, the supported fingerprint types, and the step-by-step quickstart. The docs page is at https://fpembed.readthedocs.io/en/latest/agent.html?utm_source=agent&utm_medium=skill&utm_campaign=quickstart.
