# Command-line flags

All four scripts print one JSON object to stdout, send diagnostics to stderr, and exit non-zero on failure. Every script accepts `--doctor`, which prints the environment check and exits. Paths below are relative to the skill folder.

## featurize.py

Turns SMILES, SELFIES, or RDKit Mols into compressed FPembed fingerprint embeddings.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | CSV, `.smi`, `.sdf`, or `.mol` input path |
| `--smiles-col` | `smiles` | SMILES or SELFIES column name |
| `--id-col` | none | optional ID column in a CSV |
| `--id-prop` | `_Name` | SDF property holding IDs, Mol input only |
| `--input-format` | `smiles` | one of `smiles`, `selfies`, `mol`; `.sdf` and `.mol` suffixes imply `mol` |
| `--fp-type` | `ecfp` | one of the 15 supported fingerprint types |
| `--fp-size` | `2048` | fingerprint length |
| `--compression` | `16` | compression factor, a power of two that divides `--fp-size` |
| `--count` | off | count mode |
| `--count-transform` | `none` | `none` or `log1p` |
| `--method` | library default | one of the six compression methods; the library picks `geometric`, or `uniform` for count mode |
| `--dtype` | `float64` | `float64`, `float32`, or `uint16` |
| `--cache-size` | none | bounded cache size for repeat queries |
| `--stream` | off | iterate and write chunks instead of materializing; requires `--output-format csv` |
| `--raw` | off | export the uncompressed fingerprint |
| `--dedupe` | off | canonicalize and deduplicate before embedding |
| `--manifest` | off | include the representation manifest and installed version in the JSON |
| `--output` | `embeddings.npy` | output path |
| `--output-format` | `npy` | `npy` or `csv` |
| `--doctor` | off | run the self-check and exit |
| `--seed` | none | seeds `random_projection` only |

Invalid rows are reported as indices and, when an ID column or ID property is available, as IDs. No row is dropped silently. `uint16` is accepted only for binary fingerprints; the library rejects it otherwise and the message passes through unchanged.

## compress.py

Compresses an external embedding matrix or a raw fingerprint array.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | `.npy` or numeric `.csv` matrix, or a single vector |
| `--mode` | `external` | `external` for `PrecomputedEmbeddingCompressor`, `fingerprint` for `compress_fingerprint` |
| `--compression` | required | compression factor |
| `--method` | mode default | external: `random_projection` or `hadamard`; fingerprint: any of the six methods; the external default is `random_projection`, the fingerprint default is `geometric` |
| `--dtype` | `float64` | `float64`, `float32`, or `uint16`; `uint16` is fingerprint-only |
| `--source-id` | required for external | non-empty model or checkpoint name |
| `--normalize` | `none` | `none` or `l2`, external only |
| `--center` | none | path to a `.npy` center vector, external only |
| `--truncate-to` | none | keep this many leading dimensions via `truncate_embedding`; `hadamard` and `random_projection` only |
| `--output` | `compressed.npy` | output path |
| `--doctor` | off | run the self-check and exit |
| `--seed` | none | seeds `random_projection` only |

## qsar_engine.py

Two subcommands, `fit` and `predict`. Both share the embedding flags and accept `--doctor`.

### fit

| Flag | Default | Meaning |
|---|---|---|
| `--smiles-csv` | required | CSV, `.smi`, `.sdf`, or `.mol` input with structures and targets |
| `--smiles-col` | `smiles` | SMILES or SELFIES column name |
| `--target-col` | required for CSV | target column name |
| `--target-prop` | required for Mol | target property name in an SDF |
| `--id-col` | none | optional ID column |
| `--id-prop` | `_Name` | SDF property holding IDs |
| `--input-format` | `smiles` | one of `smiles`, `selfies`, `mol`; `.sdf` and `.mol` suffixes imply `mol` |
| `--task` | `regression` | `regression` or `classification` |
| `--split` | `random` | `random` or `scaffold` (Bemis-Murcko) |
| `--seed` | `0` | random seed for the split, cross-validation, and the estimator |
| `--test-size` | `0.2` | held-out test fraction |
| `--cv` | none | optional integer k for k-fold cross-validation |
| `--model-out` | `surrogate.joblib` | saved model bundle path |
| `--doctor` | off | run the self-check and exit |
| `--fp-type` | `ecfp` | fingerprint type |
| `--fp-size` | `2048` | fingerprint length |
| `--compression` | `16` | compression factor |
| `--count` | off | count mode |
| `--count-transform` | `none` | `none` or `log1p` |
| `--method` | library default | compression method |
| `--dtype` | `float64` | `float64`, `float32`, or `uint16` |

### predict

| Flag | Default | Meaning |
|---|---|---|
| `--model-in` | required | bundle written by `fit` |
| `--smiles-csv` | required | CSV, `.smi`, `.sdf`, or `.mol` input |
| `--smiles-col` | `smiles` | SMILES or SELFIES column name |
| `--id-col` | none | optional ID column |
| `--id-prop` | `_Name` | SDF property holding IDs |
| `--input-format` | stored in the model | one of `smiles`, `selfies`, `mol`; omit to reuse the fit format |
| `--output-csv` | `predictions.csv` | predictions output path |
| `--doctor` | off | run the self-check and exit |

`fit` never pickles the embedding generator. It stores the constructor configuration in the bundle, and `predict` rebuilds the generator from it.

## diversity.py

Explores an embedding matrix with pure NumPy distance math. There is no random path, so there is no seed flag.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | `.npy` or numeric `.csv` embeddings matrix |
| `--ids` | none | one-column ID file (`.npy`, `.csv`, or plain text) matching the rows |
| `--mode` | required | `exemplars`, `outliers`, or `neighbors` |
| `--n-exemplars` | required for exemplars | number of max-min exemplars |
| `--k` | required for outliers and neighbors | neighbor count |
| `--metric` | `euclidean` | `euclidean` or `cosine` |
| `--query` | required for neighbors | `.npy` or numeric `.csv` query matrix |
| `--output` | none (stdout JSON) | optional CSV output path |
| `--doctor` | off | run the self-check and exit |
