# FPembed - Generalized Molecular Fingerprint Embeddings

A lightweight Python package for generating compressed molecular fingerprint embeddings, backed by [scikit-fingerprints](https://github.com/scikit-fingerprints/scikit-fingerprints). Supports ten binary fingerprint types through a single unified class.

FPembed compresses standard molecular fingerprints using weighted binary masking, producing compact float vectors suitable for machine-learning models. The package accepts SMILES, SELFIES, and RDKit Mol objects as input.

The concept of compressing molecular fingerprints via weighted binary masking was originally introduced for Morgan fingerprints in the eMFP paper:

> Nuñez-Andrade, E. A., Vidal-Daza, I., Gomez-Bombarelli, R., Ryan, J. W., & Martin-Martinez, F. J. (2025).
> Embedded Morgan Fingerprints for more efficient molecular property predictions with machine learning.
> *ChemRxiv* (preprint). https://doi.org/10.26434/chemrxiv-2025-6hfp8

```bibtex
@article{nunez2025emfp,
  author  = {Nu{\~n}ez-Andrade, Emilio A. and Vidal-Daza, Isaac and Gomez-Bombarelli, Rafael and Ryan, James W. and Martin-Martinez, Francisco J.},
  title   = {Embedded {Morgan} Fingerprints for more efficient molecular property predictions with machine learning},
  journal = {ChemRxiv},
  year    = {2025},
  doi     = {10.26434/chemrxiv-2025-6hfp8},
  note    = {Preprint}
}
```

Original concept repository: [MMLabCodes/eMFP](https://github.com/MMLabCodes/eMFP)

## Supported Fingerprint Types

| Type | `fp_type` | Type-specific params |
|------|-----------|---------------------|
| Extended Connectivity (ECFP) | `ecfp` | `radius` (default 2) |
| Atom Pair | `atom_pair` | `min_distance` (1), `max_distance` (30) |
| Topological Torsion | `topological_torsion` | `torsion_atom_count` (4) |
| RDKit | `rdkit` | `min_path` (1), `max_path` (7) |
| Layered | `layered` | `min_path` (1), `max_path` (7) |
| Pattern | `pattern` | (none) |
| Avalon | `avalon` | (none) |
| SECFP | `secfp` | `radius` (default 3) |
| MHFP | `mhfp` | `radius` (default 3) |
| MAP4 | `map` | `radius` (default 2) |

## Compression Methods

FPembed supports six compression methods, selectable via the `method` parameter on `EmbeddedFingerprintGenerator`. The default is `geometric`.

### Method Reference

| Method (`method` value) | Category | `method_params` | Dynamic Range / Distance Preservation | Complexity |
|-------------------------|----------|-----------------|---------------------------------------|------------|
| `geometric` | block-wise | `interleave` (bool) | 65,536:1 dynamic range | O(L)       |
| `linear` | block-wise | `interleave` (bool) | S:1 dynamic range | O(L)       |
| `log` | block-wise | `interleave` (bool) | ~4.1:1 dynamic range | O(L)       |
| `uniform` | block-wise | `interleave` (bool) | 1:1 (mean pooling) | O(L)       |
| `hadamard` | global | `seed` (int) | orthogonal projection | O(L log L) |
| `random_projection` | global | `seed` (int) + `sparse` (bool) | JL distance preservation | O(L D)     |

### The `method` Parameter

Pass `method` to the Generator constructor to select a compression strategy.

### Method-Specific Parameters (`method_params`)

- **Block-wise methods** (`geometric`, `linear`, `log`, `uniform`): accept `interleave` (bool, default `False`). When `True`, bits are assigned to blocks by stride (`bit[i] -> block[i % n_blocks]`) instead of contiguous partitioning, breaking hash clustering artifacts.
- **`hadamard`**: accepts `seed` (int, default `42`). Controls the random sign flips applied before the Fast Walsh-Hadamard Transform.
- **`random_projection`**: accepts `seed` (int, default `42`) and `sparse` (bool, default `False`). The `sparse` option uses the Achlioptas variant with approximately 2/3 zero entries for faster computation at comparable quality.

Seed-based methods (`hadamard`, `random_projection`) are fully deterministic given the same seed and NumPy version. The default seed is `42`.

### Code Examples

```python
from fpembed import EmbeddedFingerprintGenerator

# Geometric (default)
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2})

# Linear weights
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}, method="linear")

# Logarithmic weights
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}, method="log")

# Uniform weights (mean pooling)
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}, method="uniform")

# Hadamard (SRHT)
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}, method="hadamard")

# Random projection
gen = EmbeddedFingerprintGenerator(fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}, method="random_projection")
```

Bit-interleaving with a block-wise method:

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=16,
    fp_params={"radius": 2}, method="linear",
    method_params={"interleave": True}
)
```

Standalone `compress_fingerprint` with a non-default method:

```python
from fpembed import compress_fingerprint
import numpy as np

fp = np.random.randint(0, 2, size=2048).astype(np.float64)
emb = compress_fingerprint(fp, size=16, method="hadamard", method_params={"seed": 42})
print(emb.shape)  # (1, 128)
```

### Choosing a Method

Block-wise methods (`geometric`, `linear`, `log`, `uniform`) are fast (O(L)) and simple - use them when speed matters or compression ratios are modest. Among these, `geometric` preserves the most dynamic range while `uniform` treats all bits equally (mean pooling). Global projection methods (`hadamard`, `random_projection`) mix information across all input bits, which helps retain more information at high compression ratios. `hadamard` is efficient (O(L log L)) and requires power-of-2 fingerprint sizes; `random_projection` offers the strongest theoretical distance-preservation guarantees (Johnson-Lindenstrauss lemma) at the cost of O(L D) complexity.

### Performance Characteristics

All methods produce the same output dimensionality (D = L / compression) but differ in speed and memory:

| Method | Speed                                    | Precomputed Memory                   | Best For |
|--------|------------------------------------------|--------------------------------------|----------|
| Block-wise (all four) | Fastest - single vectorized einsum, O(L) | Negligible (C-length weight vector)  | Default choice; large batches |
| Random projection | Fast - BLAS matmul, O(L D)               | DxL matrix (~2 MB for L=2048, D=128) | Best theoretical guarantees (JL lemma) |
| Hadamard (SRHT) | Slowest - pure-Python FWHT, O(L log L)   | L-length sign vector (~16 KB)        | Small-scale experiments; future optimization |

Block-wise methods are ~2–5x faster than random projection and orders of magnitude faster than Hadamard in practice. Random projection's memory cost grows quadratically with fingerprint size.

## Why Use Embedded Fingerprints?

Predictive accuracy is one axis of comparison between raw and embedded fingerprints - and the gap can be narrow, especially on large datasets where raw fingerprints have enough data to exploit all 2048 bits directly. However, accuracy is not the only metric that matters. Embedded fingerprints offer substantial, guaranteed advantages on every operational dimension: storage, speed, memory, and sample efficiency.

The core argument is not "embedded fingerprints are always more accurate" but rather "embedded fingerprints achieve comparable accuracy at a fraction of the computational cost."

### Storage Size

This is the most clear-cut advantage. The compression ratio is deterministic and independent of dataset, model, or method:

| Representation | Per-molecule (L=2048) | Per-molecule (L=4096) | 100K molecules (L=2048) |
|---|---|---|---|
| Raw binary FP (float64) | 16 KB | 32 KB | ~1.6 GB |
| Embedded, C=16 (float64) | 1 KB | 2 KB | ~100 MB |
| Embedded, C=32 (float64) | 512 B | 1 KB | ~50 MB |

A 16x reduction at C=16 applies unconditionally - it does not depend on the dataset, the ML model, or the compression method chosen. This matters for storing precomputed fingerprints on disk or in a database, transmitting embeddings over a network, loading datasets into memory for training, and caching repeated lookups via the built-in LRU cache.

### ML Training and Inference Speed

The downstream ML model operates on the feature vector. Fewer features means faster training and prediction:

- **Tree-based models (Random Forest, XGBoost)**: Feature splitting cost is proportional to the number of features. Going from 2048 to 128 features means each tree split considers ~16x fewer candidates. For hyperparameter searches (e.g., Optuna with hundreds of trials), this compounds into significant wall-clock savings.
- **Neural networks**: The first dense layer's weight matrix shrinks from `(2048 x hidden)` to `(128 x hidden)` - 16x fewer parameters and 16x fewer multiply-adds per forward pass.
- **Distance-based methods (k-NN, similarity search)**: Pairwise distance computation is O(N² x D). Reducing D from 2048 to 128 gives a direct 16x speedup.

### Memory During ML Training

During model training, the feature matrix for N=100K molecules occupies `(100000, 2048)` float64 = ~1.6 GB for raw fingerprints, versus `(100000, 128)` = ~100 MB for embedded. Tree-based models create internal copies and histograms proportional to feature count. GPU-based models benefit from smaller input tensors that allow larger batch sizes and better hardware utilization.

### Compression Overhead

The compression step itself is negligible for block-wise methods (~1 ms per 1000 molecules). The total pipeline cost is:

- **Raw**: `skfp generation time`
- **Embedded**: `skfp generation time + ~1 ms per 1000 molecules` (block-wise)

The downstream ML speedup from 128 vs 2048 features far exceeds this overhead.

### Sample Efficiency

High-dimensional spaces (2048 binary features) suffer from the curse of dimensionality - distances become less meaningful and models need exponentially more data to fill the space. Compressing to 128 dense, information-rich features acts as a form of regularization. Empirically, embedded fingerprints reach good predictive performance with fewer training samples than raw fingerprints. This is particularly valuable when labeled molecular data is scarce or expensive to obtain.

### Summary of Advantages

| Metric | Raw FP (L=2048)         | Embedded FP (D=128)             | Advantage             |
|---|-------------------------|---------------------------------|-----------------------|
| Feature matrix memory (100K mols) | ~1.6 GB                 | ~100 MB                         | 16x smaller           |
| Per-molecule storage | 16 KB                   | 1 KB                            | 16x smaller           |
| Tree model training speed | Baseline                | ~16x fewer split candidates     | Faster                |
| Neural net first-layer params | 2048 x H                | 128 x H                         | 16x fewer             |
| Pairwise distance computation | O(N² x 2048)            | O(N² x 128)                     | 16x faster            |
| Small-dataset accuracy | Baseline                | Often superior (regularization) | Better generalization |
| Large-dataset accuracy | Slightly higher ceiling | Comparable                      | Marginal tradeoff     |

The choice between raw and embedded fingerprints is a classic accuracy-vs-efficiency tradeoff. Embedded fingerprints sacrifice a small amount of information for dramatic improvements in storage, speed, and memory - making them the practical default for most molecular ML workflows.

## Project Structure

```
fpembed/
├── src/fpembed/                # pip-distributable package
│   ├── __init__.py
│   ├── generator.py            # EmbeddedFingerprintGenerator
│   ├── compression.py          # compress_fingerprint (orchestrator)
│   ├── compression_blockwise.py # block-wise weight schemes
│   ├── compression_projection.py # Hadamard SRHT + random projection
│   ├── smiles_utils.py         # parse_smiles, canonicalize_smiles
│   ├── hashing.py              # fp_params_hash
│   └── py.typed                # PEP 561 marker
├── examples/
│   ├── quickstart.ipynb        # usage notebook
│   ├── datasets/               # molecular datasets (RedDB, NFA, QM9)
│   └── nicegui_app/            # NiceGUI demo application
├── pyproject.toml
├── environment.yml
└── README.md
```

## Installation

Install the core package (rdkit, numpy, selfies, scikit-fingerprints):

```bash
pip install fpembed
```

Install with demo app dependencies (nicegui, optuna, pandas, scikit-learn, etc.):

```bash
pip install fpembed[app]
```

For development (editable install):

```bash
pip install -e .
```

### Conda Environment

A full conda environment is provided for reproducibility:

```bash
conda env create -f environment.yml
conda activate fpembed
```

This installs all dependencies and the `fpembed` package in editable mode.

## Quick Start

### Single Molecule (SMILES)

```python
from fpembed import EmbeddedFingerprintGenerator

gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}
)

# Generate compressed fingerprint from SMILES
emb = gen.GetFingerprintFromSmiles("CCO")
print(emb.shape)  # (128,)
```

### Different Fingerprint Types

```python
# Atom Pair fingerprint
gen_ap = EmbeddedFingerprintGenerator(
    fp_type="atom_pair", fp_size=2048, compression=16,
    fp_params={"min_distance": 1, "max_distance": 30}
)

# Topological Torsion fingerprint
gen_tt = EmbeddedFingerprintGenerator(
    fp_type="topological_torsion", fp_size=2048, compression=16,
    fp_params={"torsion_atom_count": 4}
)
```

### Single Molecule (SELFIES)

```python
emb = gen.GetFingerprintFromSelfies("[C][C][O]")
print(emb.shape)  # (128,)
```

### Batch Processing

```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "invalid_smiles"]

embeddings, invalid_indices = gen.GetFingerprintsFromSmiles(smiles_list)
print(embeddings.shape)    # (3, 128) - 3 valid molecules
print(invalid_indices)      # [3] - index of invalid SMILES
```

### Raw Fingerprint (No Compression)

```python
gen_raw = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=None, fp_params={"radius": 2}
)
fp = gen_raw.GetFingerprintFromSmiles("CCO")
print(fp.shape)  # (2048,)
```

### Standalone Compression Function

```python
import numpy as np
from fpembed import compress_fingerprint

fp = np.random.randint(0, 2, size=2048).astype(np.float64)
emb = compress_fingerprint(fp, size=16)
print(emb.shape)  # (1, 128)
```

### Parameter Hashing

```python
from fpembed import fp_params_hash

h = fp_params_hash("ecfp", {"radius": 2})
print(h)  # 16-char hex string, stable across sessions
```

### Caching for Repeated Lookups

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=16,
    fp_params={"radius": 2}, cache_size=1024
)

# First call computes and caches
emb = gen.GetFingerprintFromSmiles("CCO")

# Second call returns cached result
emb2 = gen.GetFingerprintFromSmiles("CCO")

print(gen.cache_info())  # CacheInfo(hits=1, misses=1, maxsize=1024, currsize=1)
gen.clear_cache()
```

## Running the Demo App

The NiceGUI demo app provides an interactive UI for optimizing fingerprint embeddings. The examples are not included in the `pip install fpembed` package - clone the repository to access them.

**Warning**: the demo app uses a cache to speed up the calculations. Please provide at least 100 GB of free disk space before the evaluation. The obsolete cache file `examples/nicegui_app/cache.db` can be deleted manually afterward.

```bash
git clone https://github.com/Sciencealone/fpembed.git
cd fpembed

# Install the core package with app dependencies
pip install fpembed[app]

# Or install pinned versions from requirements.txt
pip install -r requirements.txt

# Run the NiceGUI app
cd examples/nicegui_app
python app.py
```

A Jupyter notebook with quick-start examples is also available at `examples/quickstart.ipynb`.

## Datasets

The following datasets are included in `examples/datasets/` (obtained from their original sources):

| Dataset | DOI |
|---|---|
| RedDB Database | https://doi.org/10.1038/s41597-022-01832-2 |
| Non-Fullerene Acceptors Database | https://doi.org/10.1016/j.joule.2017.10.006 |
| QM9 Database | https://doi.org/10.1038/sdata.2014.22 |

## License

This project is licensed under the terms of the MIT open source license. Please refer to the [LICENSE](LICENSE) for the full terms.

## Legal & Trademark Notice

The source code of this project is licensed under the permissive **MIT License**. You are free to use, modify, and distribute the code, even for commercial purposes, as long as the original copyright notice is retained.

However, the name **"FPembed"**, its branding, and identifiers are the intellectual property of the project author (@Sciencealone). 
* **Commercial Branding:** The unauthorized use of the name "FPembed" to brand, market, or promote commercial software, corporate AI engines, or proprietary services within the fields of bioinformatics, chemoinformatics, and AI drug discovery is strictly prohibited.
* **Community Protection:** This notice is established to prevent public confusion and to protect the open-source community from misleading corporate misrepresentations. For licensing queries regarding the project name, please open an issue or contact the author directly.

## AI disclosure

AI usage during project development is declared in [aidecl.yaml](aidecl.yaml) following the [AI Declaration Format](https://ai-declaration.org/).

## Support
This project is provided as-is, and may be updated over time. If you have questions, please open an issue.