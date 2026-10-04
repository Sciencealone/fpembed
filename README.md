# FPembed - Generalized Molecular Fingerprint Embeddings

[![PyPI version](https://badge.fury.io/py/fpembed.svg)](https://badge.fury.io/py/fpembed)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.21447996-blue)](https://doi.org/10.5281/zenodo.21447996)


## Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Conda Environment](#conda-environment)
- [Quick Start](#quick-start)
- [Why Use Embedded Fingerprints?](#why-use-embedded-fingerprints)
  - [Storage Size](#storage-size)
  - [ML Training and Inference Speed](#ml-training-and-inference-speed)
  - [Memory During ML Training](#memory-during-ml-training)
  - [Compression Overhead](#compression-overhead)
  - [Sample Efficiency](#sample-efficiency)
  - [Summary of Advantages](#summary-of-advantages)
- [Supported Fingerprint Types](#supported-fingerprint-types)
- [Compression Methods](#compression-methods)
  - [Method Reference](#method-reference)
  - [The `method` Parameter](#the-method-parameter)
  - [Method-Specific Parameters (`method_params`)](#method-specific-parameters-method_params)
  - [Code Examples](#code-examples)
  - [Choosing a Method](#choosing-a-method)
  - [Output Dtype](#output-dtype)
  - [Representation Manifest](#representation-manifest)
  - [Compression Limits](#compression-limits)
  - [Performance Characteristics](#performance-characteristics)
- [Project Structure](#project-structure)
- [Running the Demo App](#running-the-demo-app)
- [Datasets](#datasets)
- [License](#license)
- [Legal & Trademark Notice](#legal--trademark-notice)
- [Commercial & Research Services](#commercial--research-services)
- [AI disclosure](#ai-disclosure)
- [Support](#support)

## Introduction

A Python package for generating compressed molecular fingerprint embeddings, backed by [scikit-fingerprints](https://github.com/scikit-fingerprints/scikit-fingerprints). Supports ten binary fingerprint types through a single unified class.

FPembed compresses standard molecular fingerprints into compact vectors suitable for machine-learning models, using a choice of pluggable compression methods — block-wise weighting schemes (including the default weighted binary masking) or global projections (see [Compression Methods](#compression-methods)). The package accepts SMILES, SELFIES, and RDKit Mol objects as input.

At a glance: a feature vector reduced from L to D = L / C dimensions (e.g. 16x fewer features at C=16), a drop-in fit for any scikit-learn model, and — in the evaluated workloads — accuracy comparable to raw fingerprints. The dimensionality reduction is exact and guaranteed; the downstream runtime, memory, and accuracy effects depend on your model, hardware, and dataset (see [Why Use Embedded Fingerprints?](#why-use-embedded-fingerprints) for what is guaranteed versus workload-dependent). If you just want to try it, jump to [Quick Start](#quick-start).

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

## Installation

The package is installed from the working tree with an editable install — clone the repository, then install `fpembed` and its core dependencies (rdkit, numpy, selfies, scikit-fingerprints) in one step:

```bash
git clone https://github.com/Sciencealone/fpembed.git
cd fpembed
pip install -e .
```

Add the demo app dependencies (nicegui, optuna, pandas, scikit-learn, etc.) with the `app` extra:

```bash
pip install -e ".[app]"
```

> **Future consumer path.** Once a release is published to PyPI, `pip install fpembed` (or `pip install fpembed[app]`) will install the packaged wheel. That path is not the repository-setup route above and is listed here only for eventual published-package users; the working tree is always installed editable.

### Conda Environment

A full conda environment is provided for reproducibility. It is authoritative for the development setup and installs `fpembed` editable (`-e .`) from the working tree:

```bash
conda env create -f environment.yml
conda activate fpembed
```

This installs all dependencies and the `fpembed` package in editable mode.

### Dependency Files

Three files describe dependencies, each with a distinct role — do not confuse the tested lock with the metadata ranges:

- **`pyproject.toml`** — package metadata. Declares the version ranges of `fpembed`'s own dependencies (core and the `app` extra) that a consumer resolves against.
- **`requirements.txt`** — the third-party application lock: exact pins of the app's runtime dependencies, tested together. It does **not** self-pin `fpembed`; the package itself is installed editable via `pip install -e .`.
- **`environment.yml`** — the authoritative conda environment. It recreates the full development environment and installs `fpembed` editable (`-e .`). `requirements.txt` and `environment.yml` are kept in sync, with `environment.yml` as the source of truth.

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

### Bounded-Memory Iteration

The batch methods above collect every valid embedding before stacking them into one matrix. At scale that peak allocation dominates: one million molecules at D=1024 needs about 4.1 GB as float32 or 8.2 GB as float64 for the final matrix alone, before counting the individual arrays held while it is built. When the input does not fit in memory, iterate instead:

```python
for index, emb in gen.IterFingerprintsFromSmiles(smiles_iterable):
    if emb is None:
        continue  # normal invalid input at this index
    store(index, emb)  # consume or write one row at a time
```

`IterFingerprints`, `IterFingerprintsFromSmiles`, and `IterFingerprintsFromSelfies` accept any iterable, pull one input at a time, and yield `(index, embedding_or_none)`. Indices match input order and include invalid entries, which yield `None`; for `IterFingerprints`, a `None` molecule also yields `None`. Unexpected computation errors surface when the item is consumed, matching the single-molecule methods. The iterator never materializes the input or accumulates rows, so peak memory is yours to bound.

**Memory contract.** These APIs are not streaming in the sense of a total process cap. The `random_projection` matrix cache is bounded by retained bytes, but a matrix referenced by a live generator stays resident, so the cache budget is not a process-wide limit. Embeddings you keep are yours to release, and `dtype` reduces the returned and stored size, not the peak transient memory during compression.

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

**Input contract.** `compress_fingerprint` accepts 1-D `(L,)` and 2-D `(N, L)` arrays of real numeric dtype: boolean, signed or unsigned integer, or floating. Non-real dtypes (complex, object, string, datetime) raise `TypeError`; floating input containing NaN or either infinity raises `ValueError`. Integer output (`dtype="uint16"`) additionally requires binary input and represents integer codes or counts, not normalized floats. Neither validation nor compression mutates the array you pass in.

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

### SMILES Canonicalization and Chemical Standardization

FPembed canonicalizes SMILES spelling before keying and before parsing, so the same molecule written two ways produces the same embedding: the pipeline works with the canonical form (atom ordering and RDKit's aromatic form). That makes results spelling-invariant for atom-order-sensitive backends, and it is why the cache key can be built from the canonical string.

Canonicalization is not chemical standardization. FPembed does not remove salts, normalize tautomers, pick a protonation state, or impose a stereochemistry policy, because those decisions change scientific meaning and are the caller's to make. Two inputs that differ by a counter-ion or a tautomer stay distinct. If your workflow needs standardization, apply it before calling FPembed and describe it with `representation_manifest(preprocessing=...)` (see [Representation Manifest](#representation-manifest)).

## Why Use Embedded Fingerprints?

Predictive accuracy is one axis of comparison between raw and embedded fingerprints - and the gap can be narrow, especially on large datasets where raw fingerprints have enough data to exploit all 2048 bits directly. However, accuracy is not the only metric that matters. The one guaranteed advantage is dimensionality: an embedded vector has D = L / C features instead of L, and the stored byte count follows directly from D and the chosen dtype (calculated exactly in [Storage Size](#storage-size)). The operational benefits that follow — training and inference speed, memory during training, sample efficiency — are consequences of fewer features, but their magnitude is workload-, model-, and hardware-dependent, not a fixed guarantee.

The core argument is not "embedded fingerprints are always more accurate" but rather "embedded fingerprints can achieve comparable accuracy at a fraction of the feature count, and therefore of the downstream computational cost."

### Storage Size

The output dimensionality (D = L / C) is deterministic and independent of dataset, model, or method; the byte counts below follow from D and the stated dtype. The size figures are per representation under the named dtype assumptions (float64 = 8 bytes/value, uint16 = 2 bytes/value):

| Representation | Per-molecule (L=2048) | Per-molecule (L=4096) | 100K molecules (L=2048) |
|---|---|---|---|
| Raw binary FP (float64, one byte-inflated bit per element) | 16 KB | 32 KB | ~1.6 GB |
| Raw binary FP (bit-packed, 1 bit per element) | 256 B | 512 B | ~25 MB |
| Embedded, C=16 (float64) | 1 KB | 2 KB | ~100 MB |
| Embedded, C=16 (uint16) | 256 B | 512 B | ~25 MB |
| Embedded, C=32 (float64, `uniform`/projection only) | 512 B | 1 KB | ~50 MB |

A word on the baseline. Comparing an embedding against a raw fingerprint *stored as float64* overstates the storage gain, because a raw fingerprint is binary and would sensibly be bit-packed. Against **bit-packed** raw fingerprints, the `geometric` embedding at C=16 in float64 is actually *larger* (1 KB vs 256 B per molecule at L=2048): geometric at C=16 is a reversible re-encoding — it packs each 16-bit block into a single wide value, carrying the same information in fewer but larger numbers. Storing that embedding as `uint16` (see [Output Dtype](#output-dtype)) brings it back to parity with bit-packed input.

So raw byte savings are **not** the reason to embed. The genuine, unconditional advantage is the **downstream feature count**: the embedded vector has D = L / C features instead of L. Downstream training, inference, and the in-memory feature matrix then shrink roughly in proportion to the feature count, and that reduction is what the rest of this section quantifies. Keep the scaling directional rather than universal: similarity on bit-packed fingerprints works on packed words, tree learners subsample features and pay fixed costs, and approximate-neighbor indexes change the comparison, so none of these costs follows the feature count exactly.

### ML Training and Inference Speed

The downstream ML model operates on the feature vector. Fewer features means faster training and prediction:

- **Tree-based models (Random Forest, XGBoost)**: Feature splitting cost is proportional to the number of features. Going from 2048 to 128 features means each tree split considers ~16x fewer candidates. For hyperparameter searches (e.g., Optuna with hundreds of trials), this compounds into significant wall-clock savings.
- **Neural networks**: The first dense layer's weight matrix shrinks from `(2048 x hidden)` to `(128 x hidden)` - 16x fewer parameters and 16x fewer multiply-adds per forward pass.
- **Distance-based methods (k-NN, similarity search)**: Pairwise distance computation is O(N² x D). Reducing D from 2048 to 128 shrinks the distance computation itself by 16x; index build and query overhead, memory traffic, and candidate re-ranking sit on top of that and reduce the realized speedup.

### Memory During ML Training

During model training, the feature matrix for N=100K molecules occupies `(100000, 2048)` float64 = ~1.6 GB for raw fingerprints, versus `(100000, 128)` = ~100 MB for embedded. Tree-based models create internal copies and histograms proportional to feature count. GPU-based models benefit from smaller input tensors that allow larger batch sizes and better hardware utilization.

### Compression Overhead

The compression step itself is small for block-wise methods, but its cost is not universal. Under warm-cache conditions, and on the machine where the project measured it, block-wise compression added on the order of 1 ms per 1000 molecules. Treat that as an order-of-magnitude indication with `L`, dtype, and hardware attached, not as a constant. Cold-cache cost is higher: the projection methods build or fetch a projection artifact on first use, and `random_projection` additionally pays to draw and store its `(L, D)` matrix.

The total pipeline cost is:

- **Raw**: `skfp generation time`
- **Embedded**: `skfp generation time + compression time`

Measure both phases on the target hardware before relying on the overhead being negligible. The downstream ML saving from 128 versus 2048 features is usually larger, but that too is workload-dependent.

### Sample Efficiency

High-dimensional spaces (2048 binary features) suffer from the curse of dimensionality - distances become less meaningful and models need more data to fill the space. Compressing to 128 dense features can act as a form of regularization. In practice, embedded fingerprints have reached good predictive performance with fewer training samples than raw fingerprints on the datasets we examined; this is a workload- and model-dependent effect, not a guarantee, and it is most likely to matter when labeled molecular data is scarce or expensive to obtain.

### Summary of Advantages

| Metric | Raw FP (L=2048)         | Embedded FP (D=128)             | Advantage             |
|---|-------------------------|---------------------------------|-----------------------|
| Feature matrix memory (100K mols) | ~1.6 GB                 | ~100 MB                         | 16x smaller           |
| Per-molecule storage vs float64 raw | 16 KB                | 1 KB (256 B as uint16)          | 16x smaller (64x)     |
| Per-molecule storage vs bit-packed raw | 256 B              | 256 B as uint16 (1 KB float64)  | Parity, not a win (see [Storage Size](#storage-size)) |
| Tree model training speed | Baseline                | ~16x fewer split candidates     | Workload-dependent    |
| Neural net first-layer params | 2048 x H                | 128 x H                         | 16x fewer             |
| Pairwise distance computation | O(N² x 2048)            | O(N² x 128)                     | ~16x fewer ops        |
| Small-dataset accuracy | Baseline                | Can be superior (regularization) | Model-dependent       |
| Large-dataset accuracy | Slightly higher ceiling | Comparable                      | Marginal tradeoff     |

The guaranteed columns are the feature count and its exact consequences (parameter counts, per-molecule byte counts under a named dtype). The speed and accuracy columns are directional: they follow from fewer features but their realized magnitude depends on the model, hardware, and dataset. The choice between raw and embedded fingerprints is a classic accuracy-vs-efficiency tradeoff — embedded fingerprints trade a small amount of information for a smaller, denser feature vector, which is a practical default for many molecular ML workflows.

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
| SECFP | `secfp` | `radius` (3), `min_radius` (1) |
| MHFP | `mhfp` | `radius` (3), `min_radius` (1) |
| MAP4 | `map` | `radius` (default 2) |

## Compression Methods

FPembed supports six compression methods, selectable via the `method` parameter on `EmbeddedFingerprintGenerator`. The default is `geometric`.

### Method Reference

| Method (`method` value) | Category | `method_params` | Dynamic Range / Distance Preservation | Complexity |
|-------------------------|----------|-----------------|---------------------------------------|------------|
| `geometric` | block-wise | `interleave` (bool) | (2**C − 1):1 dynamic range (e.g. 65,535:1 at C=16) | O(L)       |
| `linear` | block-wise | `interleave` (bool) | C:1 dynamic range | O(L)       |
| `log` | block-wise | `interleave` (bool) | ~log2(C+1):1 (~4.1:1 at C=16) | O(L)       |
| `uniform` | block-wise | `interleave` (bool) | 1:1 (mean pooling) | O(L)       |
| `hadamard` | global | `seed` (int) | seeded, truncated Hadamard projection (expected squared norm D/L) | O(L log L) |
| `random_projection` | global | `seed` (int) + `sparse` (bool) | JL-style distance preservation for the projection itself | O(L D)     |

### The `method` Parameter

Pass `method` to the Generator constructor to select a compression strategy.

### Method-Specific Parameters (`method_params`)

- **Block-wise methods** (`geometric`, `linear`, `log`, `uniform`): accept `interleave` (bool, default `False`). When `True`, bits are assigned to blocks by stride (`bit[i] -> block[i % n_blocks]`) instead of contiguous partitioning, breaking hash clustering artifacts.
- **`hadamard`**: accepts `seed` (int, default `42`). Controls the random sign flips applied before the Fast Walsh-Hadamard Transform.
- **`random_projection`**: accepts `seed` (int, default `42`) and `sparse` (bool, default `False`). `sparse=True` draws the Achlioptas distribution, whose entries are about 2/3 zero, but the matrix is stored densely, so it saves no memory. On the binary-input fast path the work is set by the number of set bits in the input row, not by the zeros in the matrix, so `sparse` does not automatically make that path faster; any arithmetic saving applies to the dense-product fallback used for non-binary input. It also changes the projection values, so compare the two settings on your task.

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

Block-wise methods (`geometric`, `linear`, `log`, `uniform`) are fast (O(L)) and simple, so reach for them when speed or modest compression ratios matter most. `geometric` keeps the widest code range per block, while `uniform` treats every bit equally (mean pooling). Global projection methods (`hadamard`, `random_projection`) mix information across all input bits, which tends to retain more information at high compression ratios. `hadamard` is efficient (O(L log L)) and requires power-of-2 fingerprint sizes; `random_projection` carries the strongest theoretical distance-preservation statement (Johnson-Lindenstrauss) at the cost of O(L D) work.

Method rankings depend on the downstream model, so treat any single recommendation as a starting point rather than an optimum. The table below records the direction found in the project's own experiments (see `dev_docs/investigation_foundation_model_embedding_compression.md`); those measurements were taken on specific datasets and models and do not transfer as a guarantee. The recorded configuration ECFP with `fp_size=16384` and `C=16` is a starting candidate, not a proven optimum, and it does not change the default method.

| Application | Starting point | What to check before relying on it |
|---|---|---|
| Tree-based molecular property prediction | `geometric`, `C=16` | Compare against raw fingerprints and other block-wise methods; tree feature subsampling changes the realized speedup |
| Linear prediction | `random_projection` | Re-tune regularization after changing the representation |
| Approximate similarity, clustering, or retrieval | `random_projection` | Measure neighborhood preservation and `recall@k` under the intended metric; cosine correlation with a reference embedding is not enough |
| Exact binary similarity (Tanimoto) | Bit-packed raw fingerprints | Compressed-space Euclidean distance is not binary Tanimoto |
| Dense continuous embeddings | Projection methods | The geometric binary-code argument no longer applies |

Retrieval and similarity use cases need task-specific retrieval evaluation. A high correlation with a reference embedding does not establish that nearest neighbors are preserved.

### Output Dtype

Both `EmbeddedFingerprintGenerator` (constructor argument `dtype`, applied to every returned array) and the standalone `compress_fingerprint()` (keyword-only `dtype`) accept an output dtype. Accepted values are given either as strings or NumPy types:

| `dtype` | Bytes/value | Semantics |
|---|---|---|
| `float64` (default) | 8 | The values existing callers already get, computed exactly as before. For `geometric`, the value equals the integer code divided by `2**C - 1`, apart from the float64 round-off of the accumulation path. |
| `float32` | 4 | **Information-preserving** for `geometric` at `C <= 16` (see below). ~1e-7 relative error for the projection methods. Permitted for every method. |
| `uint16` | 2 | The **exact integer code**. Permitted only for `geometric` and `uniform`, and only for binary input. |

`dtype` reduces the **returned and stored** size of the embedding. It does **not** reduce peak transient memory during compression: the float paths still compute internally in `float64` and cast on return, so the working set is unchanged — only the array you keep is smaller.

`dtype` is a representation of identical information, not a different fingerprint, so it does not enter `fp_params_hash()` and does not affect cache keys.

#### `float32` — information-preserving, not bit-equal

For `geometric` at `C <= 16`, the `float32` output is *information-preserving*: the underlying block integer `n` is recovered **exactly** by

```
n = rint(value * (2**C - 1))
```

The stored `float32` value is **not** bit-equal to the `float64` output — it differs by about 0.002 of one code step at `C = 16`. That difference is far below the 0.5 needed to misround, so recovery of `n` is reliable, but you must not compare `float32` and `float64` outputs for equality. This is why the dtype is described as "information-preserving" rather than "lossless".

#### `uint16` — the exact integer code

`uint16` carries the raw integer code directly, not a rescaled `[0, 1]` value, and is computed exactly (it does not inherit the tiny residual the float64 einsum carries):

| Method | Stored value | Range | Recover the float64 embedding by |
|---|---|---|---|
| `geometric` | block integer `n = Σ bᵢ·2ⁱ` (first bit least significant) | `0 .. 2**C - 1` | `value / (2**C - 1)` |
| `uniform` | block popcount | `0 .. C` | `value / C` |

`interleave=True` is supported identically — interleaving changes which bits form a block, not the recovery formula.

#### Reversibility is not distance preservation

At supported precision, `geometric` recovers the original bits exactly, but that does not mean the decoded values preserve the geometry of those bits, or that a model can reach them equally easily. A two-bit counterexample, with `C=2` so `2**C - 1 = 3`:

- Codes `1` and `2` differ in both bits, yet their decoded values differ by only `1/3`.
- Codes `0` and `2` differ in one bit, yet their decoded values differ by `2/3`.

Ranking pairs by Euclidean distance in the decoded space therefore does not reproduce ranking by Hamming distance. Higher-order bits also map to larger positional values than lower-order bits, which makes them easier for a threshold-based model to reach; injectivity alone does not make all bits equally learnable. Exact binary similarity (Tanimoto on the original bits) stays a separate operation from comparing compressed vectors.

Pooling methods (`linear`, `log`, `uniform`) additionally collide distinct blocks, so their suitability has to be measured on the task rather than inferred from the weight range.

Keep the sources of loss separate: fingerprint hashing collisions come from the fingerprint itself, block pooling from the compression step, float rounding from the stored dtype, and any accuracy change from the downstream model. Only pooling and rounding are compression concerns.

**Which combinations raise `ValueError`:**

- `uint16` with `linear`, `log`, `hadamard`, or `random_projection` — these have no exact integer form.
- `uint16` + `geometric` with `C > 16` — the block integer would exceed `uint16` range. (`uint16` + `uniform` requires `C <= 65535`, which is unreachable in practice but still enforced.)
- `uint16` with **non-binary** input — the integer identity requires every element to be 0 or 1; count fingerprints or arbitrary float input would be silently corrupted, so this is rejected rather than accepted.
- Any dtype outside `{float64, float32, uint16}` (including lossy types such as `int8`, `uint8`, `float16`) — rejected with a message naming the accepted set.

```python
import numpy as np
from fpembed import compress_fingerprint

fp = np.random.randint(0, 2, size=2048).astype(np.uint8)

# Exact integer code, 4x smaller than float64
codes = compress_fingerprint(fp, size=16, dtype="uint16")

# Recover the geometric float64 value (the code / (2**16 - 1), up to float64 round-off)
recovered = codes / (2**16 - 1)
```

### Representation Manifest

`fp_params_hash()` is a cache key: it covers the fingerprint type, its parameters, and the compression method. It is deliberately narrow and must not change, so it is not a complete identity for an array you persist. `EmbeddedFingerprintGenerator.representation_manifest()` returns a fresh JSON-compatible mapping for that purpose:

```python
import json

manifest = gen.representation_manifest()
print(json.dumps(manifest, indent=2))
```

It records the resolved fingerprint type, width, and parameters (omitted and explicitly-default values resolve to the same settings), whether compression is active together with its factor, method and effective method parameters including `seed` and `interleave`, the output dimension, dtype, and physical encoding, the built-in canonicalization policy, an optional declarative `preprocessing` mapping you supply, and the installed `fpembed`, `numpy`, `rdkit`, `scikit-fingerprints`, and `selfies` versions. `schema_version` describes the manifest format and moves independently of the package version.

The `output.encoding` field distinguishes what the values physically are: `raw_binary` for an uncompressed fingerprint, `normalized_block_value` and `mean_block_value` for geometric and uniform float output, `integer_code` and `integer_count` for the exact `uint16` codes, `weighted_block_value` for linear and log, and `projection_value` for Hadamard and random projection. A `uint16` geometric array and its `float64` counterpart are therefore not interchangeable: the recovery formula relates them, but only the integer form is exact.

Built-in canonicalization and external preprocessing are reported separately. The manifest stores a caller-supplied `preprocessing` mapping verbatim under `preprocessing.external` and never executes it; supplying one does not make the pipeline standardized. The manifest is additive: it does not change `fp_params_hash`, cache keys, or returned arrays. Implementations may change the layout between versions, so re-derive a manifest rather than comparing its bytes across `fpembed` releases.

### Compression Limits

`geometric` compression is capped at `C <= 32`; requesting `compression > 32` with `geometric` (at either entry point) raises `ValueError`. Geometric block weights span `2**-C`, so the block integer survives only while it fits the float64 mantissa. The einsum accumulates roughly `C · 2**-52` relative error against a code spacing of `2**-C`, so recovery starts failing well before the naive 53-bit limit — at `C = 64` the low ~11 bits of every block are silently absorbed. `C <= 32` is the largest power of two comfortably inside the safe range (verified exact for `C <= 16`). A `ValueError`, not a warning, is used, because warnings can be suppressed in evaluation runs — precisely where silent bit loss does the most damage.

The cap applies only to `geometric` and only when compression is active (`compression=None` or `0` is exempt). The other methods (`linear`, `log`, `uniform`, `hadamard`, `random_projection`) have bounded weight ratios or use no block weights, so none approaches the mantissa limit even at large `C` and none is capped.

### Performance Characteristics

All methods produce the same output dimensionality (D = L / compression) but differ in speed and memory:

| Method | Speed                                                                              | Precomputed Memory                   | Best For |
|--------|------------------------------------------------------------------------------------|--------------------------------------|----------|
| Block-wise (all four) | Fastest - single vectorized einsum, O(L)                                           | Negligible (C-length weight vector)  | Default choice; large batches |
| Hadamard (seeded, truncated) | Fast: pruned fold-then-FWHT, where the input is folded down to length D *before* transforming, so the discarded outputs are never computed. The retained first-D outputs are bit-identical to truncating the full transform for binary input. Correctness against that reference does not by itself prove neighborhood or distance preservation. | L-length sign vector (~16 KB) | High compression ratios |
| Random projection | Fast - for binary input, gathers and sums only the matrix rows selected by the set bits (exploiting fingerprint sparsity) instead of a full dense matmul; non-binary input falls back to the dense product. The gather result is mathematically equivalent to the dense product **within float64 round-off** (the two sum the same terms in a different order), not bit-identical. One canonical C-contiguous `(L, D)` matrix is cached per parameter set. | (L, D) matrix (~2 MB for L=2048, D=128) | High compression ratios; JL-style guarantee for the projection itself |

Both projection methods gained fast paths — the Hadamard fold is bit-identical to the full transform for binary input, and the random-projection gather is equivalent to the dense product within float64 round-off. Measured wall-time improvements (L=16384, D=1024, on one machine):

- **Hadamard** (pruned fold-then-FWHT vs full-length transform then truncate): roughly **3x** at N=1 and **6x** at N=16.
- **Random projection** (binary row-gather vs dense matmul, at ECFP-like sparsity): roughly **28x** at N=1 and **2.5x** at N=16.

These are measured wall times on one machine, not a theoretical claim: the pruned FWHT reduces butterfly *operation count* by about 9x, but the code is already vectorized and largely bandwidth-bound, so the wall-time gain is materially less than the operation-count reduction. Block-wise methods remain the fastest overall choice. Random projection's matrix memory still grows with `L × D`. Note that `sparse=True` (the Achlioptas variant) stores its matrix **densely** and therefore saves no memory despite the name; it trades roughly 2/3 zero entries for a denser arithmetic pattern in the dense-product path, not a smaller footprint, and the binary-input gather selects matrix rows by the input's set bits rather than by the matrix nonzeros.

**Hadamard specifics.** The implementation applies seeded random sign flips and keeps the first `D` outputs in Sylvester order. It does not randomly choose which `D` of the `L` coordinates to keep, so it is a seeded, truncated Hadamard projection rather than conventional randomly subsampled SRHT. With the retained `1/sqrt(L)` normalization, averaging over the random signs gives an expected squared norm of `D / L` times the input squared norm; a distance-scale correction would multiply the outputs by `sqrt(L / D)`, but that does not remove the fixed-coordinate selection. Read the fold's bit-identity with the full transform as correctness of this transform, not as evidence of neighborhood preservation.

**Projection matrix cache.** The random-projection matrix is cached to avoid redrawing it on every call. Only a **single canonical orientation** is retained per parameter set — one C-contiguous `(L, D)` matrix; the `(D, L)` orientation is served as a non-owning `.T` view of the same buffer, so no second owning copy exists. The cache is bounded by **total retained bytes**, not entry count: the budget defaults to 512 MiB (`RP_CACHE_BYTE_BUDGET` in `compression_projection_cache.py`), with a small entry-count floor so a single oversized matrix cannot evict itself. Least-recently-used matrices are dropped once the accumulated bytes exceed the budget.

## Project Structure

```
fpembed/
├── src/fpembed/                       # importable package (installed editable)
│   ├── __init__.py                    # public API and __version__
│   ├── generator.py                   # EmbeddedFingerprintGenerator
│   ├── generator_smiles.py            # SMILES/SELFIES/Mol input mixin
│   ├── compression.py                 # compress_fingerprint (orchestrator)
│   ├── compression_blockwise.py       # block-wise weight schemes
│   ├── compression_projection.py      # seeded truncated Hadamard + random projection
│   ├── compression_projection_cache.py # byte-bounded projection-matrix cache
│   ├── dtype_support.py               # output dtype validation and casting
│   ├── embedding_cache.py             # BoundedCache for embedding lookups
│   ├── hashing.py                     # fp_params_hash
│   ├── input_validation.py            # real/finite input boundary guard
│   ├── manifest.py                    # representation manifest builder
│   ├── skfp_factory.py                # scikit-fingerprints backend factory
│   ├── smiles_utils.py                # parse_smiles, canonicalize_smiles
│   └── py.typed                       # PEP 561 marker
├── examples/
│   ├── quickstart.ipynb               # usage notebook
│   ├── datasets/                      # molecular datasets (RedDB, NFA, QM9)
│   └── nicegui_app/                   # NiceGUI demo application
├── pyproject.toml                     # package metadata and dependency ranges
├── requirements.txt                   # third-party app lock (no FPembed self-pin)
├── environment.yml                    # authoritative conda env (installs -e .)
└── README.md
```

## Running the Demo App

The NiceGUI demo app provides an interactive UI for optimizing fingerprint embeddings. The examples live in the repository working tree, not in the installed package - clone the repository to access them.

```bash
git clone https://github.com/Sciencealone/fpembed.git
cd fpembed

# Install FPembed (editable) plus the app dependencies from the working tree
pip install -e ".[app]"

# Alternatively, recreate the full pinned environment (installs -e . itself)
conda env create -f environment.yml
conda activate fpembed

# Run the NiceGUI app
cd examples/nicegui_app
python app.py
```

The `-e ".[app]"` install and `environment.yml` both provide `fpembed` from the working tree; neither pulls a `fpembed` wheel from PyPI. `requirements.txt` pins only the third-party app dependencies and can be installed on top of the editable package (`pip install -r requirements.txt`) if you want the exact tested versions.

### Demo App Cache Lifecycle

The demo app writes a cache database at `examples/nicegui_app/cache.db` to avoid recomputing fingerprints and embeddings across runs.

- **Purpose & growth.** The cache accumulates computed results as you evaluate molecules. It can grow large during extended optimization runs — provide at least **100 GB of free disk space** before a large evaluation.
- **Safe deletion.** `cache.db` is a pure regenerable artifact. You may delete it at any time while the app is **not running**; the app recreates an empty cache on the next launch. Stop the app before deleting the file to avoid removing a database with open handles.
- **Data-loss consequences.** Deleting the cache loses only saved computation, never source data or results you have exported — every entry can be recomputed from the input molecules. The trade-off is a slower first run afterward while the cache refills. Deleting it while the app is running is unsafe and can leave a partially written file; if that happens, stop the app and delete the leftover file before restarting.

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

The name **"FPembed"**, its branding, and identifiers are the intellectual property of the project author (@Sciencealone). 
* **Commercial Branding:** The unauthorized use of the name "FPembed" to brand, market, or promote commercial software, corporate AI engines, or proprietary services within the fields of bioinformatics, chemoinformatics, and AI drug discovery is strictly prohibited.
* **Community Protection:** This notice is established to prevent public confusion and to protect the open-source community from misleading corporate misrepresentations. For licensing queries regarding the project name, please open an issue or contact the author directly.

## Commercial & Research Services

For the project onboarding, corporate deployments, high-performance compute scaling (HPC / Databricks / Ray), or custom model fine-tuning for pharma R&D and research grants (Horizon Europe / Art. 60 LOSU / AEI) as well as other related tasks, official technical support and contract services are available.

**[View Full Service Offerings & Contact Details](COMMERCIAL_SUPPORT.md)**

## AI disclosure

AI usage during project development is declared in [aidecl.yaml](aidecl.yaml) following the [AI Declaration Format](https://ai-declaration.org/).

## Support
This project is provided as-is, and may be updated over time. If you have questions, please open an issue.