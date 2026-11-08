# Quickstart

These commands run from the skill folder, the directory that contains `SKILL.md`. They use the bundled fixture at `resources/demo.csv`, which has columns `smiles`, `id`, `target`, and `activity`.

## 1. Check the environment

```
python scripts/featurize.py --doctor
```

The doctor prints a JSON summary. Confirm that `fpembed.installed` is true, `fpembed.meets_minimum` is true, and `numpy` is true. `scikit_learn` and `tier` tell you whether the QSAR step will run. The same `--doctor` flag exists on every script.

## 2. Featurize the fixture

```
python scripts/featurize.py --input resources/demo.csv --output embeddings.npy
```

The JSON result reports `n_input`, `n_valid`, `n_invalid`, the invalid indices, the output path, and the fingerprint settings. The default settings are ECFP with `fp_size` 2048 and compression 16, which gives a 128-dimensional embedding per molecule.

## 3. Fit a QSAR model

```
python scripts/qsar_engine.py fit --smiles-csv resources/demo.csv --target-col target --model-out surrogate.joblib
```

This trains a Ridge regression on the `target` column, reports `r2`, `rmse`, and `mae`, and writes the model bundle. For the binary target, replace `--target-col target` with `--target-col activity` and add `--task classification`. Add `--cv 5` for k-fold cross-validation.

## 4. Predict

```
python scripts/qsar_engine.py predict --model-in surrogate.joblib --smiles-csv resources/demo.csv --output-csv predictions.csv
```

`predict` rebuilds the embedding generator from the saved configuration, so it never unpickles a generator. The JSON result reports the valid and invalid counts, and the CSV holds the predictions.

## Other input formats

The same flow works with SELFIES and with RDKit Mols.

For SELFIES, store the strings in a column and select the format:

```
python scripts/featurize.py --input molecules.csv --smiles-col selfies --input-format selfies --output embeddings.npy
python scripts/qsar_engine.py fit --smiles-csv molecules.csv --smiles-col selfies --input-format selfies --target-col target --model-out surrogate.joblib
```

For Mol input, use an `.sdf` or `.mol` file. The suffix implies `mol`, so `--input-format mol` is optional. Use `--id-prop` to read IDs from an SDF property, and `--target-prop` on `qsar_engine.py fit` to read the target:

```
python scripts/featurize.py --input molecules.sdf --id-prop _Name --output embeddings.npy
python scripts/qsar_engine.py fit --smiles-csv molecules.sdf --target-prop pIC50 --model-out surrogate.joblib
```

When the input format matches, `predict` reuses the format stored in the model, so `--input-format` is only needed if the prediction file differs from the training file.

For a wider reference, the agent documentation is at https://fpembed.readthedocs.io/en/latest/agent.html?utm_source=agent&utm_medium=skill&utm_campaign=quickstart.
