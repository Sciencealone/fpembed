# FPembed User Manual

## Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Molecular fingerprints](#molecular-fingerprints)
  - [Fingerprint length and native width](#fingerprint-length-and-native-width)
  - [Compression factor and output dimension](#compression-factor-and-output-dimension)
  - [Input handling and canonicalization](#input-handling-and-canonicalization)
  - [Identity and reproducibility](#identity-and-reproducibility)
- [Why Embedded Fingerprints](#why-embedded-fingerprints)
  - [Storage size](#storage-size)
  - [Training and inference speed](#training-and-inference-speed)
  - [Memory during training](#memory-during-training)
  - [Compression overhead](#compression-overhead)
  - [Sample efficiency](#sample-efficiency)
  - [Summary](#summary)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install from source](#install-from-source)
  - [Install from PyPI](#install-from-pypi)
  - [Conda](#conda)
  - [Verification](#verification)
- [Quick Start](#quick-start)
  - [Binary fingerprint from SMILES](#binary-fingerprint-from-smiles)
  - [Count fingerprint](#count-fingerprint)
  - [Batch input and label alignment](#batch-input-and-label-alignment)
  - [Representation manifest](#representation-manifest)
- [Working with the Generator](#working-with-the-generator)
  - [SELFIES and RDKit Mol input](#selfies-and-rdkit-mol-input)
  - [Raw fingerprint](#raw-fingerprint)
  - [Streaming input](#streaming-input)
  - [Invalid input and label alignment](#invalid-input-and-label-alignment)
  - [Standalone compression](#standalone-compression)
  - [Output dtype](#output-dtype)
  - [Choosing a configuration](#choosing-a-configuration)
- [Fingerprint Types](#fingerprint-types)
  - [Choosing a fingerprint type](#choosing-a-fingerprint-type)
  - [Catalog](#catalog)
  - [Count support](#count-support)
  - [Lingo](#lingo)
  - [Folded pharmacophore](#folded-pharmacophore)
  - [Fixed-width types](#fixed-width-types)
- [Compression Methods](#compression-methods)
  - [How compression works](#how-compression-works)
  - [Method reference](#method-reference)
  - [Method parameters](#method-parameters)
  - [Method resolution](#method-resolution)
  - [Count compression rules](#count-compression-rules)
  - [Geometric limit](#geometric-limit)
  - [Caller-supplied projection matrix](#caller-supplied-projection-matrix)
  - [Choosing a method](#choosing-a-method)
- [Output Dtype and Exactness](#output-dtype-and-exactness)
  - [Accepted dtypes](#accepted-dtypes)
  - [float32 exactness](#float32-exactness)
  - [uint16 exact integer codes](#uint16-exact-integer-codes)
  - [Count output dtype](#count-output-dtype)
  - [Reversibility is not distance preservation](#reversibility-is-not-distance-preservation)
- [Identity, Manifests, and Caching](#identity-manifests-and-caching)
  - [What the parameter hash covers](#what-the-parameter-hash-covers)
  - [Matrix and center digests](#matrix-and-center-digests)
  - [Manifest schema 2](#manifest-schema-2)
  - [Output encoding labels](#output-encoding-labels)
  - [What to store](#what-to-store)
  - [Caching](#caching)
- [Precomputed Embeddings](#precomputed-embeddings)
  - [When to use](#when-to-use)
  - [Compressor configuration](#compressor-configuration)
  - [Compression and normalization](#compression-and-normalization)
  - [Identity and manifest](#identity-and-manifest)
  - [CLAMP recipe](#clamp-recipe)
  - [Persisting embeddings](#persisting-embeddings)
- [Recipes](#recipes)
  - [scikit-learn pipeline](#scikit-learn-pipeline)
  - [Count with a projection](#count-with-a-projection)
  - [Fixed-width PubChem](#fixed-width-pubchem)
  - [PCA and whitening fitted outside FPembed](#pca-and-whitening-fitted-outside-fpembed)
  - [Reuse one projection at several widths](#reuse-one-projection-at-several-widths)
- [Performance and Memory](#performance-and-memory)
  - [Complexity](#complexity)
  - [Peak memory](#peak-memory)
  - [Projection memory](#projection-memory)
  - [Parallelism](#parallelism)
  - [Pharmacophore cost](#pharmacophore-cost)
- [Limitations](#limitations)
- [API Reference](#api-reference)
  - [EmbeddedFingerprintGenerator](#embeddedfingerprintgenerator)
  - [compress_fingerprint](#compress_fingerprint)
  - [truncate_embedding](#truncate_embedding)
  - [PrecomputedEmbeddingCompressor](#precomputedembeddingcompressor)
  - [fp_params_hash](#fp_params_hash)
  - [SMILES helpers](#smiles-helpers)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
  - [Common errors](#common-errors)
  - [FAQ](#faq)
- [Compatibility](#compatibility)
  - [Python and dependencies](#python-and-dependencies)
  - [API stability](#api-stability)
  - [Manifest and hash compatibility](#manifest-and-hash-compatibility)
- [Glossary](#glossary)
- [Original Concept (eMFP)](#original-concept-emfp)
  - [Software and methods](#software-and-methods)
  - [Citing FPembed](#citing-fpembed)
- [License and Support](#license-and-support)


## Introduction

FPembed turns molecular fingerprints into dense vectors. It generates
fingerprints, compresses them, validates input, tracks identity, and caches
results, then returns plain NumPy arrays. Use it when you want standard
fingerprints in a smaller, model-ready form and need the fingerprint definition
to stay explicit.

| Entry point | Input | Use when |
|---|---|---|
| `EmbeddedFingerprintGenerator` | SMILES, SELFIES, RDKit Mol | You start from molecules |
| `PrecomputedEmbeddingCompressor` | 1-D or 2-D external embeddings | You already have embeddings from another model |
| `compress_fingerprint()` | 1-D or 2-D fingerprint array | You already have fingerprints and want only compression |

## Core Concepts

### Molecular fingerprints

A molecular fingerprint maps a molecule to a numeric vector. A binary
fingerprint holds 0 and 1. A count fingerprint holds non-negative occurrence
counts.

Select count mode with `count=True`. Count mode is a representation setting on
the generator; `fp_type` stays the same. Counts are non-negative integers, and
FPembed returns `float64` or `float32`. Binary and count configurations have
separate identities, so they produce different parameter hashes and manifest
encodings.

### Fingerprint length and native width

- `L` is the fingerprint length, named `fp_size` in the API.
- A variable-width fingerprint type produces exactly `L` values.
- A fixed-width fingerprint type has its own native width: PubChem 881, Laggner
  307, and ERG 315. For those types, `fp_size` is the padded width. It must be a
  power of two and at least the native width, and values beyond the native width
  are zeros.

### Compression factor and output dimension

- `C` is the compression factor, named `compression`.
- `D` is the output dimension. It is `L / C` when compression is active and `L`
  when it is not.

The generator and `PrecomputedEmbeddingCompressor` require a power-of-two factor
that divides the input width and is at most half of it.
`compress_fingerprint()` only requires the factor to divide the feature length.
With `compression=None` or `0`, no compression runs.

A block-wise method splits the fingerprint into blocks of `C` positions and
reduces each block to one value. A projection method combines all positions.
The methods are described in [Compression Methods](#compression-methods).

A smaller `fp_size` narrows the fingerprint itself. Hashed fingerprints then
have more collisions, because more substructures map to the same positions.
Compression leaves the fingerprint at its original width and reduces the output
afterward. The two choices differ, so compare them on your validation split.

### Input handling and canonicalization

The generator accepts SMILES, SELFIES, and RDKit `Mol` objects. By default,
FPembed canonicalizes SMILES input before keying and parsing. The canonical form
makes the cache spelling-invariant and makes order-sensitive fingerprint types
return the same embedding for different spellings of the same molecule.

Canonicalization normalizes SMILES spelling, atom ordering, and RDKit's aromatic
form. It stops short of chemical standardization: FPembed leaves salts,
tautomers, protonation state, and stereochemistry as written. Those choices
change scientific meaning and belong to the caller. If a workflow needs
standardization, apply it before calling FPembed and describe it in
`representation_manifest(preprocessing=...)`.

The `canonicalize` flag selects the canonical branch. FPembed accepts `validate`
for compatibility, but parsing always validates, so the flag does not change
which branch runs.

### Identity and reproducibility

`params_hash` is a 16-character cache key. It covers the fingerprint type,
fingerprint parameters, method, method parameters, and the non-default count
representation. It excludes `fp_size`, the compression factor, the output
dtype, preprocessing, and dependency versions.

`representation_manifest()` returns schema 2, which records those excluded
fields. Persist the manifest alongside an embedding matrix or a deployed model;
the hash is only a cache key.

Seed-based projection methods are deterministic given the same seed and NumPy
version. The default seed is 42.

## Why Embedded Fingerprints

A fingerprint encodes a molecule as a vector of fixed length. A 2048-bit ECFP
hashes each substructure into a position and leaves the rest at zero, so few of
the 2048 values are nonzero. Training and inference pay for the full
width: a tree scans a threshold per position, a dense layer holds a weight per
position, and a distance adds a term per position.

An embedding is a shorter, dense vector that carries the fingerprint information
in the form the model consumes. The compression factor `C` sets the ratio
`D = L / C`. At `C=16`, a 2048-bit fingerprint becomes 128 values. A block-wise
method makes each output summarize 16 positions, and `geometric` turns that block
into one integer scaled to `[0, 1]`, while `uniform` turns it into the fraction
of set bits. A projection method mixes all 2048 positions into each of the 128
outputs. The model reads 128 numbers in place of 2048.

You can shrink the input at two stages. A smaller `fp_size` changes the
fingerprint itself: the hashing step maps more substructures onto the same
positions, so collisions rise and resolution falls before a model sees it. You
get a narrower fingerprint without a separate compression step, and you accept
more collisions. Compression keeps the fingerprint at full width and reduces the
vector after generation, so the fingerprint keeps its resolution and you pay one
extra step.

Compare the two on a validation split. A narrower fingerprint and a compressed
one can reach the same width and differ in accuracy, because they lose different
information. Which one wins depends on your substructures and your model.

Some quantities follow from the configuration. Width is `D = L / C`, so `C=16`
turns 2048 positions into 128 features, about 16 times fewer inputs. Stored
bytes are `D * itemsize`. Accuracy, training time, memory savings, and sample
efficiency depend on the model, the data, and the hardware: a gradient-boosted
tree and a linear model read the same columns and respond in different ways.

FPembed's maps fall into two families. Block-wise methods reduce each block of
`C` positions to one value. With the default `interleave=False`, output `i`
depends on contiguous inputs `i*C` through `i*C + C - 1`, so nearby bits stay
together and local patterns survive. Projection methods mix all `L` positions
into every output, which can retain more information at high `C`. They cost more
than a block-wise method, from `O(L log L)` for Hadamard to `O(L * D)` for random
projection.

The maps are fixed and data-independent. A molecule gives the same embedding
alone or in a batch of a million, during training or at inference. Seeded
projections reproduce the map from the same seed and NumPy version, so a model
trained on one run reads the same features in the next.

Principal component analysis (PCA) is a fitted alternative. It centers the data
and finds the orthogonal directions that carry the most variance: directions at
right angles, so each component carries variance the others do not repeat. It
orders those directions by variance and keeps the leading ones, which carry most
of the variance. When your data sits near a low-dimensional subspace, a fit can
keep more structure than a fixed map because it chooses directions from your
sample. If two fingerprint positions always flip together, PCA can fold them
into one component and keep the same information in fewer columns. Dropping
low-variance directions can remove noise from distance calculations, though it
can also remove signal.

You must fit PCA on training data, so the map depends on that sample. PCA works
on centered data, scaling changes the components, and a few outliers can rotate
the leading directions. A large component count can overfit the training sample.
If you refit on new data, the components can rotate, and a model trained on the
old map reads different features.

FPembed does not fit PCA. Fit it outside on training data and pass the
components as a `(D, L)` matrix through
`method_params={"matrix": pca.components_}` with `method="random_projection"`.
The generator applies `M @ x` without centering, so that path is a linear
projection. For the full transform, subtract `pca.mean_` and call
`compress_fingerprint()`. The
[PCA recipe](#pca-and-whitening-fitted-outside-fpembed) shows both calls.

Consider PCA for linear models and for distances on dense, correlated features,
when you have enough rows for a stable fit and preprocessing you can reproduce
at inference. Skip it when the dataset is small, when a tree model does the
work, or when you need one portable map with no fitted state.

### Storage size

Output dimension is deterministic. `D = L / C` gives 128 features at `C=16`, and
`bytes = D * itemsize` gives the stored size, with `itemsize` at 8 bytes for
float64, 4 for float32, and 2 for uint16.

| Representation | Per molecule (L=2048) | Per molecule (L=4096) | 100K molecules (L=2048) |
|---|---:|---:|---:|
| Raw binary, float64 | 16 KB | 32 KB | about 1.6 GB |
| Raw binary, bit-packed | 256 B | 512 B | about 25 MB |
| Embedded, C=16, float64 | 1 KB | 2 KB | about 100 MB |
| Embedded, C=16, uint16 | 256 B | 512 B | about 25 MB |
| Embedded, C=32, float64 | 512 B | 1 KB | about 50 MB |

Read the first two rows together. Raw float64 costs 16 KB per molecule because
the array holds 2048 values of 8 bytes, and most of them are zero. Bit-packing
the same bits costs 256 B, 64 times less, because one byte carries eight
positions.

A `geometric` embedding at C=16 in float64 costs 1 KB, which is larger than the
bit-packed raw fingerprint. The embedding packs 16 bits into one 8-byte value:
128 values of 8 bytes carry the same 2048 bits that fit in 256 packed bytes.
Storing the embedding as uint16 returns it to 256 B.

The feature count is the reliable gain. The model reads 128 inputs instead of
2048 at every step. Choose bit-packed raw when disk or network dominates and the
model can consume sparse binary input. Choose the embedding when model input
width dominates, and use uint16 when you want byte parity with bit-packed raw.

### Training and inference speed

- A tree model searches for a split by scanning candidate thresholds for each
  feature. With `D` features instead of `L`, that scan touches about `C` times
  fewer candidates. Feature and row subsampling, fixed costs per tree, and the
  histogram build for each node limit the gain, so measure the wall clock.
- A neural network's first dense layer holds a weight matrix of shape
  `(L, hidden)`. At `D` inputs the matrix becomes `(D, hidden)`, and its
  parameter count and multiply-adds fall by about `C`. The later layers keep
  their size, so the saving applies to one layer and its incoming activations.
- A distance method computes a distance by summing over dimensions. At `D`
  dimensions the inner loop runs about `C` times shorter. Building and
  maintaining an index, moving data through memory, and reranking candidates
  after the search keep their full-width cost.

You can bound the realized gain with a budget. If split search takes 40 percent
of training time and you cut its work by 16 times, that 40 percent becomes 2.5
percent, so total time falls by 1.6 times. The remaining 60 percent sets the
floor; measure your own split.

### Memory during training

A 100K by 2048 float64 feature matrix needs about 1.6 GB. The same 100K rows at
128 float32 features need about 51 MB. Two factors compound: 16 times fewer
columns and half the bytes per value, so the example uses about 32 times less
memory.

The training matrix scales with row count, width, and dtype. Tree models keep
histograms and internal copies that follow the feature count, so the saving
reaches those buffers. Smaller input tensors can allow a larger mini-batch on a
GPU, and the gain compounds when device memory runs out before the dataset does.

`dtype` changes stored bytes and leaves peak transient memory where it was. The
float paths compute in float64 and cast on return, so during compression the raw
fingerprint and the float64 working buffers exist at full width. Plan peak memory
from the input width and the batch size; plan disk from the output dtype.

### Compression overhead

Warm-cache block-wise compression added about 1 ms per 1000 molecules in our
measurements, or about 1 second per million. Treat the figure as an order of
magnitude; the cost varies with the method, the width, and the hardware.

Cold-cache projection calls build a sign vector or draw a projection matrix
before the first row, so the first batch costs more than later ones.
A caller-supplied matrix skips the draw and pays validation on the first call.

Generation cost varies by fingerprint type and can exceed compression.
Pharmacophore costs more per molecule than the topological types, while
compression of its output costs the same per row. You pay compression once per
molecule, and the narrower matrix saves work at training and at every inference
call. Time both steps on your hardware before you weigh them.

### Sample efficiency

Distance concentration describes what happens to pairwise distances as dimension
grows. In a high-dimensional space the nearest and farthest of a set of random
points land at similar distances, because each added dimension adds noise to
every pair. A model that separates molecules by distance then needs more rows to
find a boundary that holds.

A shorter dense vector can act as a regularizer. The model works from 128
summaries instead of 2048 positions. Each summary mixes a block of bits, or all
positions for a projection, so a split on one column draws on many substructures
at once. Distances between rows can spread out when the representation is shorter
and smoother, which gives a distance-based model more to work with.

Both effects can reverse. A map that discards bits your labels depend on
loses accuracy, and a block that sums unrelated substructures can hide the one
that matters. With enough rows the wider input fits the same structure and the
gain shrinks. Test this benefit with cross-validation; it is the least certain.

### Summary

| Metric | Raw fingerprint (L=2048) | Embedded (D=128) | Caveat |
|---|---|---|---|
| Feature matrix, 100K molecules | about 1.6 GB float64 | about 100 MB float64 | Exact from shape and dtype |
| Per molecule versus bit-packed raw | 256 B | 256 B uint16, 1 KB float64 | Byte parity |
| Tree split candidates | Baseline | About 16x fewer | Realized speedup is workload-dependent |
| First dense layer | `2048 x H` | `128 x H` | Exact parameter count |
| Pairwise distance work | `O(N^2 x 2048)` | `O(N^2 x 128)` | Index overhead remains |
| Small-dataset accuracy | Baseline | Can improve | Model-dependent |
| Large-dataset accuracy | Slightly higher ceiling | Comparable | Empirical |

Feature count, parameter count, and byte count follow from shape and dtype, so
you can compute them before you run a benchmark. The speed and accuracy columns
are directional: they depend on the model, the hardware, and the dataset.

Start from `ecfp` at 2048 bits with `compression=16` and `dtype="float64"`. That
gives 128 features. Change one axis at a time: method, compression factor, dtype,
fingerprint type. Measure accuracy, training time, and peak memory on a
validation split after each change; the three move in different directions.

For a storage constraint, bit-pack the raw fingerprint or store the embedding as
uint16, which reaches byte parity at C=16. For a linear model or similarity
search, try `random_projection`: it mixes positions and keeps distances closer
to the full-width ones. For a tree model, try `geometric`, which keeps block
structure and the widest code range for binary input. Pass a fitted PCA matrix
if you trained it on your data and need a learned map.

## Installation

### Requirements

FPembed requires Python 3.10 or newer, below 3.14. Runtime dependencies are
NumPy 1.26.4 or newer below 3, RDKit 2023.9.6 or newer, scikit-fingerprints 2.1
or newer below 3, and SELFIES 2.2 or newer.

Count fingerprints, fixed-width fingerprint types, and precomputed embeddings
require FPembed 0.3.0 or newer.

### Install from source

```bash
git clone https://github.com/Sciencealone/fpembed.git
cd fpembed
python -m pip install -e .
```

### Install from PyPI

```bash
python -m pip install fpembed
```

Use the source install when the PyPI release is older than 0.3.0.

### Conda

```bash
conda create -n fpembed python=3.12
conda activate fpembed
python -m pip install fpembed
```

### Verification

```python
import fpembed

print(fpembed.__version__)
```

The distribution metadata and `fpembed.__version__` should agree. `pip show
fpembed` prints the installed metadata.

## Quick Start

### Binary fingerprint from SMILES

```python
from fpembed import EmbeddedFingerprintGenerator

gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp",        # ECFP (Morgan) fingerprint
    fp_size=2048,
    compression=16,
    fp_params={"radius": 2},
)

embedding = gen.GetFingerprintFromSmiles("CCO")
print(embedding.shape)  # (128,)
```

**Recommended starting point:** `ecfp`, `fp_size=2048`, `compression=16`, the
default method, and `dtype=float64`, which gives 128 features. From there,
change one axis at a time.

### Count fingerprint

```python
count_gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=16,
    fp_params={"radius": 2},
    count=True,
    count_transform="log1p",
)

counts = count_gen.GetFingerprintFromSmiles("CCO")
print(counts.shape)  # (128,)
```

With `compression=16`, `count=True`, and no explicit method, the method resolves
to `uniform`. Each output value is the mean `log1p` count over a block of 16
positions.

### Batch input and label alignment

```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "not_a_molecule"]
embeddings, invalid = gen.GetFingerprintsFromSmiles(smiles_list)
print(embeddings.shape)  # (3, 128)
print(invalid)           # [3]

y = [label for i, label in enumerate(labels) if i not in set(invalid)]
```

`embeddings` is a 2-D float array, and batch methods remove invalid SMILES, so
filter the labels with the same indices.

### Representation manifest

```python
import json

manifest = gen.representation_manifest()
print(json.dumps(manifest, indent=2))
```

The manifest is a fresh JSON-compatible mapping; the library does not store it.

## Working with the Generator

### SELFIES and RDKit Mol input

```python
embedding = gen.GetFingerprintFromSelfies("[C][C][O]")
print(embedding.shape)  # (128,)
```

```python
from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
embedding = gen.GetFingerprintAsNumPy(mol)
print(embedding.shape)  # (128,)
```

### Raw fingerprint

```python
gen_raw = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=None,
    fp_params={"radius": 2},
)

raw = gen_raw.GetFingerprintFromSmiles("CCO")
print(raw.shape)  # (2048,)
```

### Streaming input

Batch methods collect every valid embedding before stacking them. For inputs
that do not fit in memory, iterate:

```python
for index, embedding in gen.IterFingerprintsFromSmiles(smiles_iterable):
    if embedding is None:
        continue
    store(index, embedding)
```

`IterFingerprints`, `IterFingerprintsFromSmiles`, and
`IterFingerprintsFromSelfies` pull one input at a time and yield
`(index, embedding_or_none)`. Indices match input order. The iterator does not
materialize the input or accumulate rows.

### Invalid input and label alignment

Single-molecule methods return `None` for invalid input. Batch methods return
the valid rows plus the indices of invalid inputs, as shown in
[Quick Start](#batch-input-and-label-alignment). An all-invalid batch returns an
empty `(0, D)` array.

Filter labels with the returned indices before training. Computation errors
raise instead of appearing as invalid input.

### Standalone compression

```python
import numpy as np
from fpembed import compress_fingerprint

fp = np.random.randint(0, 2, size=2048)
embedding = compress_fingerprint(fp, size=16)
print(embedding.shape)  # (1, 128)
```

### Output dtype

```python
gen16 = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=16,
    fp_params={"radius": 2},
    dtype="float32",
)
print(gen16.GetFingerprintFromSmiles("CCO").dtype)  # float32
```

`dtype` shrinks the returned and stored array. Peak transient memory stays the
same, because the float paths compute in float64 and cast on return. `dtype` is
not part of `params_hash`.
[Output Dtype and Exactness](#output-dtype-and-exactness) covers the exactness
rules.

### Choosing a configuration

| Axis | Starting point | Change it when |
|---|---|---|
| Fingerprint type | `ecfp` with `radius=2` | Your task needs longer-range or substructure relationships |
| `fp_size` | 2048 | You need a wider fingerprint or fewer hashing collisions |
| Compression factor | 16 | Lower it if the model underfits with 128 features; raise it if storage or speed matters more |
| Method | Default for the mode | You need distance preservation at high compression |
| Dtype | `float64` | You need smaller arrays and accept the precision limits |

Change one axis at a time and compare on a validation split. The default method
is `geometric` for binary input and `uniform` for count input.

## Fingerprint Types

### Choosing a fingerprint type

Start with `ecfp` for general-purpose property prediction and similarity.
`atom_pair` and `topological_torsion` capture longer-range relationships.
`rdkit`, `layered`, `pattern`, and `avalon` are path- or substructure-based.
`secfp`, `mhfp`, and `map` are hashed circular variants. `lingo` works on SMILES
substrings. `pharmacophore` uses pharmacophore points and is slower. `pubchem`,
`laggner`, and `erg` are fixed-width legacy fingerprints.

### Catalog

| Type | `fp_type` | Count mode | Native width | Main parameters and defaults |
|---|---|:---:|---:|---|
| Extended Connectivity (Morgan) | `ecfp` | Yes | `fp_size` | `radius=2`, `include_chirality=False`, `use_pharmacophoric_invariants=False`, `use_bond_types=True`, `count_simulation=False` |
| Atom Pair | `atom_pair` | Yes | `fp_size` | `min_distance=1`, `max_distance=30`, `use_pharmacophoric_invariants=False`, `count_simulation=True` |
| Topological Torsion | `topological_torsion` | Yes | `fp_size` | `torsion_atom_count=4`, `count_simulation=True` |
| RDKit | `rdkit` | Yes | `fp_size` | `min_path=1`, `max_path=7`, `use_bond_order=True`, `num_bits_per_feature=2`, `linear_paths_only=False`, `use_pharmacophoric_invariants=False` |
| Layered | `layered` | No | `fp_size` | `min_path=1`, `max_path=7`, `linear_paths_only=False` |
| Pattern | `pattern` | No | `fp_size` | None |
| Avalon | `avalon` | Yes | `fp_size` | None |
| SECFP | `secfp` | No | `fp_size` | `radius=3`, `min_radius=1`, `isomeric_smiles=False`, `kekulize=True` |
| MHFP | `mhfp` | Yes | `fp_size` | `radius=3`, `min_radius=1`, `isomeric_smiles=False`, `sssr_rings=True` |
| MAP4 | `map` | Yes | `fp_size` | `radius=2`, `include_chirality=False`, `include_duplicated_shingles=False` |
| Lingo | `lingo` | Yes | `fp_size` | `substring_length=4` |
| Pharmacophore | `pharmacophore` | Yes | `fp_size` | `min_points=2`, `max_points=3`; folded 2D only |
| PubChem | `pubchem` | No | 881 | No type parameters; use padded `fp_size=1024` |
| Laggner | `laggner` | No | 307 | No type parameters; use padded `fp_size=512` |
| ERG | `erg` | No | Follows `max_path` (315 at 15) | `max_path=15`; fixed bit variant, so the native width follows `max_path`; use a padded `fp_size` at or above it |

Pass all parameter values through `fp_params`. Omitted and explicitly default
values produce the same fingerprint configuration and manifest parameters.
`fp_params` rejects `random_state` and `variant`.

### Count support

`secfp`, `layered`, `pattern`, `pubchem`, `laggner`, and `erg` reject
`count=True` with a `ValueError` that names the supported count types and the
requested type. The other nine types support count mode.

`ecfp`, `atom_pair`, and `topological_torsion` also accept `count_simulation`,
which is a binary-mode option: combining it with `count=True` raises
`ValueError`. The catalog lists the per-type defaults, `False` for `ecfp` and
`True` for `atom_pair` and `topological_torsion`. Changing the option changes
bit positions, so fingerprints and models built with the old setting are not
interchangeable. `rdkit` does not accept it.

### Lingo

`lingo` produces a substring fingerprint over canonical SMILES. The only
parameter for this type is `substring_length`, default 4. Lingo supports count
mode, and the canonical-SMILES branch keeps different spellings of the same
molecule identical.

### Folded pharmacophore

`pharmacophore` selects the folded 2D variant with `variant="folded"` and
`use_3D=False`. Set the point range with `min_points` (default 2) and
`max_points` (default 3). You cannot select `raw_bits`, a 3D mode, or a
different variant. Pharmacophore supports count mode.

Pharmacophore is much slower than the topological types. See
[Pharmacophore cost](#pharmacophore-cost) for numbers.

### Fixed-width types

PubChem and Laggner have a fixed native width. ERG has a native width that
follows `max_path`: 315 at the default 15. `fp_size` is the padded width you
request. The rules are:

- `fp_size` must be a power of two and at least the native width, including
  when it equals the native width. The exact native widths 881 and 307 are not
  powers of two and are rejected; ERG's configured width is treated the same
  way.
- Values are placed at their native indices and the tail is filled with zeros.
- The raw and compressed shapes follow the padded `fp_size`.
- A variable-width type accepts any positive `fp_size`, and its native width
  equals `fp_size`.

The compression behavior at C=16 follows from the padded width:

| Type | Native | Padded `fp_size` | Live output blocks | Dead blocks | Real positions in tail block |
|---|---:|---:|---:|---:|---:|
| `pubchem` | 881 | 1024 | 56 of 64 | 8 of 64 (12.5%) | 1 of 16 |
| `laggner` | 307 | 512 | 20 of 32 | 12 of 32 (37.5%) | 3 of 16 |
| `erg` | 315 | 512 | 20 of 32 | 12 of 32 (37.5%) | 11 of 16 |

A dead block is entirely padding and always reduces to zero. The tail block is
only partly populated. These costs follow from the fixed native width and the
power-of-two padded width.

## Compression Methods

### How compression works

Block-wise methods split the fingerprint into blocks of `C` positions and
reduce each block to one number. With `geometric`, each block is read as a
`C`-bit integer. For `C=4`, the block `1011` is 11, and the stored value is
`11/15`. The mapping is reversible for binary input because every block integer
is distinct. `uniform` stores the block mean: the fraction of set bits for
binary input, or the mean count for count input. `linear` and `log` use
increasing and logarithmic weights instead of powers of two. Projection methods
combine all positions; they cost more but can preserve distances better at high
compression.

Six methods are available: four block-wise schemes and two projections.

### Method reference

| Method | Category | `method_params` | Effect |
|---|---|---|---|
| `geometric` | Block-wise | `interleave` | Positional integer code per block; reversible for binary input |
| `linear` | Block-wise | `interleave` | Weighted sum per block |
| `log` | Block-wise | `interleave` | Logarithmically weighted sum per block |
| `uniform` | Block-wise | `interleave` | Mean of the block; block mean count for counts |
| `hadamard` | Global | `seed` | Seeded, truncated Hadamard projection |
| `random_projection` | Global | `seed`, `sparse`, optional `matrix` | Dense or sparse random projection; Johnson-Lindenstrauss style |

All methods produce `D = L / C` output values. Complexity and retained memory
are summarized in [Performance and Memory](#performance-and-memory).

### Method parameters

- Block-wise methods accept `interleave`, default `False`. When true, block
  membership is by stride rather than contiguous ranges. Interleaving changes
  which positions share a block, while the output shape stays the same.
- `hadamard` accepts `seed`, default 42. The seed controls the random sign flips
  applied before the transform.
- `random_projection` accepts `seed` (default 42) and `sparse` (default
  `False`). `sparse=True` draws the Achlioptas distribution and changes the
  projection values. The matrix is still stored densely, so `sparse` does not
  reduce memory.
- `random_projection` also accepts a caller-supplied `matrix`. See
  [Caller-supplied projection matrix](#caller-supplied-projection-matrix).

Unknown `method_params` keys raise `ValueError`. Seeds must be non-negative
integers, and `interleave` and `sparse` must be booleans.

### Method resolution

`method=None` resolves by mode:

| State | Resolved method |
|---|---|
| Compression inactive | `geometric` (no method runs) |
| Binary, compression active | `geometric` |
| Count, compression active | `uniform` |

An explicit forbidden method still raises.

### Count compression rules

With active count compression, only `uniform`, `hadamard`, and
`random_projection` are allowed. `geometric`, `linear`, and `log` raise
`ValueError` because their block value is defined on bits or on weighted
sums, neither of which has a count interpretation.

With `uniform`, each output is the mean count in its block, or the mean
`log1p` count when the transform is active. With `hadamard` and
`random_projection`, each output is a projected count value. Random projection
uses the dense product for counts because the sparse path assumes binary
non-zeros.

The standalone `compress_fingerprint()` accepts counts for all six methods.
The restricted method set applies to the generator.

When count compression is inactive, the count-specific method rejection does
not apply, because no compression runs. Checks such as the Hadamard
power-of-two rule still apply.

### Geometric limit

`geometric` compression is capped at `C <= 32`. Above that, the block integer
starts to lose low bits in the float64 accumulation. The cap applies only to
`geometric` and only when compression is active. The other methods are not
capped.

### Caller-supplied projection matrix

`random_projection` accepts a canonical matrix in
`method_params={"matrix": M}` with shape `(D, L)`, where `L` is the fingerprint
length and `D = L // compression`. FPembed validates the shape, requires a real
finite array, converts it to float64, and digests it. It applies the matrix as
given rather than fitting it.

The digest, `matrix_id`, and `matrix_shape` enter the parameter hash and the
manifest. The matrix contents stay out of the cache key. See
[Identity, Manifests, and Caching](#identity-manifests-and-caching) and the
[Reuse one projection at several widths](#reuse-one-projection-at-several-widths)
recipe.

### Choosing a method

Block-wise methods are fast and simple. `geometric` retains the widest code
range for binary input. `uniform` treats every position equally and is the
count default. Projection methods mix information across positions and can
retain more information at high compression ratios. `hadamard` requires
power-of-two dimensions. `random_projection` carries a distance-preservation
statement for the projection itself and costs more.

Method rankings depend on the downstream model. A configuration that works for
a tree model may not be optimal for a linear model. Treat any single
recommendation as a starting point.

| Use case | Starting point | Check before relying on it |
|---|---|---|
| Tree-based property prediction | `geometric`, C=16 | Compare against raw fingerprints and other methods |
| Linear prediction | `random_projection` | Retune regularization |
| Similarity or retrieval | `random_projection` | Measure recall at k and neighborhood preservation |
| Exact binary similarity | Bit-packed raw fingerprints | Compressed Euclidean distance differs from Tanimoto |
| Dense external embeddings | Projection methods | The binary code argument does not apply |

PCA is a learned linear map fitted on data. FPembed's methods are fixed and do
not look at the data. To use PCA, fit it outside FPembed and pass the matrix;
the generator applies the matrix without centering. The corrected recipe is in
[PCA and whitening fitted outside FPembed](#pca-and-whitening-fitted-outside-fpembed).

## Output Dtype and Exactness

### Accepted dtypes

The generator and `compress_fingerprint()` accept `float64`, `float32`, and
`uint16`. The precomputed compressor accepts only `float64` and `float32`.
Strings such as `"float32"` and NumPy types such as `np.float32` are accepted.

| `dtype` | Bytes per value | Semantics |
|---|---:|---|
| `float64` | 8 | Default. Geometric value is `n / (2**C - 1)`, up to accumulation round-off. |
| `float32` | 4 | Information-preserving for geometric at C <= 16. About 1e-7 relative error for projections. |
| `uint16` | 2 | Exact integer code for geometric or uniform, binary input only. |

`dtype` shrinks the returned and stored array. Peak transient memory stays the
same, because the float paths compute in float64 and cast on return.

`dtype` represents the same information in a different storage type. It stays
out of `params_hash` and does not change cache keys.

With compression inactive, `uint16` stores the raw 0/1 fingerprint. Method
restrictions apply only when compression is active.

### float32 exactness

For `geometric` at C <= 16, a float32 value differs from the float64 value by
about 0.002 of one code step at C=16. The error is far below the 0.5 needed to
misround, so you can recover the block integer exactly with:

```text
n = rint(value * (2**C - 1))
```

The dtype therefore preserves the information for geometric, even when the bits
differ from the float64 form.

For count output, float32 is exact for integer counts below `2**24`. Above that
bound, float32 cannot represent every integer and the manual makes no exactness
claim.

### uint16 exact integer codes

`uint16` stores the integer code directly.

| Method | Stored value | Range | Recover the float64 value with |
|---|---|---|---|
| `geometric` | Block integer `sum(b_i * 2**i)`, first bit least significant | `0 .. 2**C - 1` | `value / (2**C - 1)` |
| `uniform` | Block popcount | `0 .. C` | `value / C` |

`interleave=True` changes which positions form a block; the recovery formula
stays the same.

`uint16` raises `ValueError` in these cases:

- `linear`, `log`, `hadamard`, or `random_projection`, which have no exact
  integer form.
- `geometric` with C > 16, where the block integer exceeds uint16 range.
- `uniform` with C > 65535, which is unreachable in practice but still checked.
- Non-binary input, because the integer identity requires every value to be 0
  or 1.
- Count mode, at construction. Counts cannot be stored in uint16.
- Any dtype outside the accepted set, such as `int8`, `uint8`, or `float16`.

### Count output dtype

Count output accepts `float64` and `float32` and rejects `uint16` because
counts cannot be represented exactly.

Uncompressed count output has shape `(fp_size,)`. Compressed count output has
shape `(fp_size // compression,)`. With no transform, the float64 values are
exact integers. With `log1p`, they are exact `numpy.log1p` values.

### Reversibility is not distance preservation

Geometric recovers the original bits at supported precision, but the decoded
values do not preserve Hamming geometry. With C=2, so `2**C - 1 = 3`:

- Codes 1 and 2 differ in both bits, yet their decoded values differ by 1/3.
- Codes 0 and 2 differ in one bit, yet their decoded values differ by 2/3.

Ranking pairs by Euclidean distance in the decoded space therefore differs from
ranking by Hamming distance. Exact binary similarity stays a separate
operation. Pooling methods also collide distinct blocks, so measure their
suitability on your task.

Keep the sources of loss separate: fingerprint hashing collisions come from the
fingerprint, block pooling from compression, float rounding from the dtype, and
accuracy changes from the model. Only pooling and rounding are compression
concerns.

## Identity, Manifests, and Caching

### What the parameter hash covers

`fp_params_hash()` returns a 16-character SHA-256 prefix of a sorted JSON
payload. It covers:

- `fp_type`
- the fingerprint parameters in `fp_params`
- `method`, unless the method is `None` or the default `geometric` with no
  method parameters
- `method_params`, when present
- `count` and `count_transform`, when non-default

Payload examples:

```python
# Binary call
{"fp_type": "ecfp", "radius": 2}

# Count
{"fp_type": "ecfp", "radius": 2, "count": 1}

# Count with log1p
{"fp_type": "ecfp", "radius": 2, "count": 1, "count_transform": "log1p"}

# Caller-supplied matrix
{
    "fp_type": "ecfp",
    "radius": 2,
    "method": "random_projection",
    "method_params": {"matrix_id": "3f9c2a7d41b8e650", "seed": 42},
}
```

A generator passes its resolved method. An active count generator therefore
also records `method: "uniform"` unless another allowed method was chosen.

Supplying a parameter with its default value explicitly still adds the key to
the payload, so `fp_params={"max_path": 15}` for `erg` produces a different
cache key from omitting it.

The hash does not cover `fp_size`, the compression factor, the output dtype or
encoding, preprocessing, or dependency versions. Changing any of those leaves
the hash unchanged. The hash is a cache key within a fixed generator
configuration; use the manifest for persisted identity.

`fp_type`, `method`, `method_params`, `count`, and `count_transform` are
reserved keys inside `fp_params`. Supplying any of them raises `ValueError`.
NumPy scalar values are normalized to Python scalars, and booleans are
normalized to integers, so `True` and `1` hash identically.

### Matrix and center digests

Before hashing, FPembed replaces a caller-supplied projection matrix with
`matrix_id` and `matrix_shape`. `matrix_id` is a 16-character digest of the
array shape and canonical float64 bytes, so the same values supplied as float32
or float64 share an identity. A different shape or a changed value gives a
different digest.

A precomputed compressor's centering vector is replaced by `center_id`, computed
the same way. The precomputed hash covers `source_id`, `input_dim`, `normalize`,
`center_id`, `method`, and `method_params`. It does not cover the compression
factor or the dtype.

### Manifest schema 2

`representation_manifest()` returns this structure for a generator:

```json
{
  "schema_version": 2,
  "fingerprint": {
    "type": "pubchem",
    "size": 1024,
    "native_size": 881,
    "padding": "tail_zeros",
    "effective_size": 881,
    "encoding": "binary",
    "count_transform": "none",
    "params": {},
    "fixed_params": {}
  },
  "compression": {
    "active": true,
    "factor": 16,
    "method": "geometric",
    "method_params": {"interleave": false},
    "output_dim": 64
  },
  "output": {
    "dim": 64,
    "dtype": "float64",
    "encoding": "normalized_block_value"
  },
  "preprocessing": {
    "canonicalization": {
      "policy": "rdkit_canonical_reparse",
      "applies_to": "smiles_input"
    },
    "external": null
  },
  "versions": {
    "fpembed": "0.3.0",
    "numpy": "2.5.4",
    "rdkit": "2026.3.5",
    "scikit-fingerprints": "2.1.0",
    "selfies": "2.2.0"
  }
}
```

The example is a binary PubChem fingerprint with the default `geometric` method
and `C=16`.

Field meanings:

- `fingerprint.type`: the `fp_type`.
- `fingerprint.size`: the padded `fp_size`.
- `fingerprint.native_size`: the native width before padding.
- `fingerprint.padding`: `"none"` or `"tail_zeros"`.
- `fingerprint.effective_size`: the number of independent bit positions. Each
  count-simulated feature occupies four physical bits, so a simulated
  fingerprint reports a quarter of its native width. The hash slot count equals
  `effective_size`, while the physical array width remains `native_size`.
- `fingerprint.encoding`: `"binary"`, `"count"`, or `"external"`.
- `fingerprint.count_transform`: `"none"` or `"log1p"`.
- `fingerprint.params`: resolved fingerprint parameters. Omitted and explicitly
  default values give the same mapping.
- `fingerprint.fixed_params`: backend values the factory pins that affect bit
  positions and cannot be set through `fp_params`. It is read-only. `rdkit`
  records `count_simulation: false`, folded `pharmacophore` records
  `variant: "folded"` and `use_3D: false`, `erg` records `variant: "bit"`, and
  every other type records `{}`. Execution-only settings (`n_jobs`,
  `batch_size`) and values already recorded by `fingerprint.encoding` are
  excluded. The keys are always disjoint from `fingerprint.params`.
- `compression.active`: whether compression runs.
- `compression.factor`: the factor, or `null` when inactive.
- `compression.method`: the method, or `null` when inactive.
- `compression.method_params`: effective parameters, including `seed`,
  `sparse`, `interleave`, and matrix identity keys when present.
- `compression.output_dim`: the embedding width.
- `output.dim`: the returned array width.
- `output.dtype`: the returned dtype name.
- `output.encoding`: the physical encoding label.
- `preprocessing.canonicalization`: the built-in SMILES policy, or `null` for
  precomputed embeddings.
- `preprocessing.external`: a caller mapping, recorded verbatim and never run.
- `versions`: installed versions of the packages that produced the values. A
  missing distribution can be `null`.

The mapping is rebuilt on every call. Mutating the returned object does not
change the generator.

A precomputed manifest uses `fingerprint.type: "precomputed"`, sets `size`,
`native_size`, and `effective_size` to `input_dim`, uses `padding: "none"`,
sets `fixed_params` to `{}`, and puts `source_id`, `normalize`, and `center_id`
in `fingerprint.params`.

### Output encoding labels

| Case | `output.encoding` |
|---|---|
| Uncompressed binary | `raw_binary` |
| Uncompressed count | `raw_count` |
| Uncompressed count with `log1p` | `raw_count_log1p` |
| Compressed geometric float | `normalized_block_value` |
| Compressed uniform float | `mean_block_value` |
| Compressed linear or log float | `weighted_block_value` |
| Compressed geometric uint16 | `integer_code` |
| Compressed uniform uint16 | `integer_count` |
| Compressed hadamard or random projection, binary | `projection_value` |
| Compressed count with `uniform` | `count_mean_block_value` |
| Compressed count with a projection method | `count_projection_value` |
| Uncompressed precomputed | `external_raw` |
| Compressed precomputed with a projection | `external_projection_value` |

A `uint16` geometric array and its `float64` counterpart are different formats.
The recovery formula relates them, but only the integer form is exact.

### What to store

| Artifact | Store with the embeddings | Why |
|---|---|---|
| Generated embeddings | `representation_manifest()` | Records fingerprint, method, dtype, and versions |
| Caller-supplied matrix | The matrix plus the manifest | The manifest records `matrix_id` and shape, not values |
| Center vector | The center plus the manifest | The manifest records `center_id`, not values |
| External model embeddings | `source_id`, checkpoint identifier, manifest | Identifies the model that produced the input |
| Cache entries | `params_hash` | Stable cache key, not a full identity |

### Caching

Set `cache_size` to a positive integer to enable the in-memory cache. `None`
disables it.

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=16,
    fp_params={"radius": 2},
    cache_size=1024,
)

gen.GetFingerprintFromSmiles("CCO")
gen.GetFingerprintFromSmiles("CCO")
print(gen.cache_info())
gen.clear_cache()
```

With canonicalization on, the cache keys on the canonical SMILES and maps that
form to the embedding. Cached arrays are stored read-only and returned as
writable copies, so mutating a result cannot poison a cache entry. With
`canonicalize=False`, the cache keys on the raw SMILES string instead.

The cache belongs to one generator instance and lives in memory. For
persistence, store `params_hash` alongside an external cache.

## Precomputed Embeddings

### When to use

Use `PrecomputedEmbeddingCompressor` to shrink and normalize continuous
embeddings from another model. For raw molecular fingerprints, use
`EmbeddedFingerprintGenerator` or `compress_fingerprint()`.

The compressor holds only configuration and projection artifacts. It does not
fit, train, or persist anything.

### Compressor configuration

```python
PrecomputedEmbeddingCompressor(
    input_dim: int,
    compression: int | None,
    source_id: str,
    method: str | None = None,
    method_params: dict | None = None,
    dtype: Any = np.float64,
    normalize: str = "none",
    center: npt.ArrayLike | None = None,
)
```

| Parameter | Meaning |
|---|---|
| `input_dim` | Feature length of the external embeddings; positive integer |
| `compression` | Power-of-two factor that divides `input_dim` and is at most half of it; `None` or `0` for no compression |
| `source_id` | Non-empty string identifying the model or checkpoint |
| `method` | `random_projection` or `hadamard`; `None` resolves to `random_projection` when active |
| `method_params` | Method parameters, including an optional matrix |
| `dtype` | `float64` or `float32`; `uint16` and integer types are rejected |
| `normalize` | `"none"` or `"l2"` row normalization |
| `center` | Optional vector of length `input_dim` |

Methods:

| Method | Input | Returns |
|---|---|---|
| `compress(embeddings)` | `(input_dim,)` or `(N, input_dim)`, including `(0, input_dim)` | `(D,)`, `(N, D)`, or `(0, D)` |
| `compress_one(embedding)` | 1-D `(input_dim,)` | `(D,)` |
| `representation_manifest()` | None | Fresh schema 2 mapping |
| `params_hash` | Property | 16-character identity hash |

### Compression and normalization

The compressor subtracts the center before row L2 normalization. An all-zero row
stays zero under L2 normalization. FPembed never fits the center; you supply it.

When compression is active, only projection methods are allowed, because
block-wise outputs are tied to binary block positions and continuous external
embeddings have no such structure. When compression is inactive, no method
runs, and a supplied method is ignored.

### Identity and manifest

The precomputed params hash covers `source_id`, `input_dim`, `normalize`,
`center_id`, `method`, and `method_params`. The compression factor and the dtype
are not covered; record them in the manifest.

The manifest records `center_id` and `matrix_id` in place of the center and
matrix values. Store the arrays themselves next to the manifest, as listed in
[What to store](#what-to-store).

### CLAMP recipe

CLAMP is an external fingerprint model. It is not a core `fp_type`. FPembed
ships no CLAMP weights and does not depend on torch. This recipe treats CLAMP as
a precomputed embedding source.

1. Generate 768-dimensional float32 embeddings with the CLAMP code and
   checkpoint outside FPembed. Torch and the 174 MB weights are external
   dependencies of that step.
2. Record a `source_id` that identifies the exact checkpoint, for example
   `"clamp:<checkpoint-hash>"`.
3. Compress with `random_projection`, because 768 is not a power of two and
   Hadamard requires power-of-two input and output dimensions.
4. Fix the batch size you use for generation. CLAMP measured a maximum drift of
   about 1.9e-6 between batch sizes, so changing the batch size can produce
   slightly different embeddings and break reproducible caching.

```python
import numpy as np
from fpembed import PrecomputedEmbeddingCompressor

clamp_vectors = np.load("clamp_embeddings_768.npy").astype(np.float32)

compressor = PrecomputedEmbeddingCompressor(
    input_dim=768,
    compression=4,
    source_id="clamp:checkpoint-identifier",
    method="random_projection",
    normalize="l2",
)

embeddings = compressor.compress(clamp_vectors)
print(embeddings.shape)  # (N, 192)
print(compressor.params_hash)
```

Hadamard is possible only if you zero-pad each 768-d vector to 1024 before
compression and set `input_dim=1024` with `method="hadamard"`. The padded
positions are zero, and the padded result differs from the 768-d random
projection. Choose one path and record it in the manifest.

Use `compress_one` for a single vector. Use `representation_manifest()` to
persist the source identity, normalization, compression configuration, and
versions alongside the array.

### Persisting embeddings

```python
import json
import numpy as np

np.save("embeddings.npy", embeddings)
with open("embeddings.manifest.json", "w", encoding="utf-8") as handle:
    json.dump(compressor.representation_manifest(), handle, indent=2)
```

Store the manifest with the array. The parameter hash alone omits the
compression factor, the dtype, and dependency versions.

## Recipes

### scikit-learn pipeline

```python
from sklearn.ensemble import RandomForestRegressor
from fpembed import EmbeddedFingerprintGenerator

gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}
)

X, invalid = gen.GetFingerprintsFromSmiles(train_smiles)
y = [label for i, label in enumerate(train_labels) if i not in set(invalid)]

model = RandomForestRegressor(n_estimators=300)
model.fit(X, y)
```

### Count with a projection

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=16,
    count=True,
    method="random_projection",
)
```

Hadamard is also allowed, but `fp_size` and `fp_size // compression` must both
be powers of two.

### Fixed-width PubChem

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="pubchem",
    fp_size=1024,
    compression=16,
)

embedding = gen.GetFingerprintFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
print(gen.native_size)  # 881
print(gen.padding)      # tail_zeros
print(embedding.shape)  # (64,)
```

Laggner uses `fp_size=512` with native 307, and ERG uses `fp_size=512` with
native 315.

### PCA and whitening fitted outside FPembed

Build the raw matrix with the batch API, then fit PCA on training data only:

```python
import numpy as np
from sklearn.decomposition import PCA
from fpembed import EmbeddedFingerprintGenerator, compress_fingerprint

raw_gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=None, fp_params={"radius": 2}
)
X_raw, invalid = raw_gen.GetFingerprintsFromSmiles(train_smiles)
# Filter any labels with `invalid` before training a model.

pca = PCA(n_components=128, random_state=0).fit(X_raw)

X = compress_fingerprint(
    X_raw - pca.mean_,
    size=16,
    method="random_projection",
    method_params={"matrix": pca.components_},
)
```

The generator applies `M @ fingerprint` without subtracting a mean, so the
generator path with `pca.components_` performs a linear projection. For the full
PCA transform, subtract `pca.mean_` and call `compress_fingerprint()`, as above.

For whitening, scale the components by the standard deviations of the fitted
components and keep the same center:

```python
pca = PCA(n_components=128, whiten=True, random_state=0).fit(X_raw)
matrix = pca.components_ / np.sqrt(pca.explained_variance_)[:, None]

X = compress_fingerprint(
    X_raw - pca.mean_,
    size=16,
    method="random_projection",
    method_params={"matrix": matrix},
)
```

`PCA(whiten=True)` applies the scaling inside `transform`; `components_` itself
is not scaled in current scikit-learn. FPembed only validates and applies the
matrix it is given.

### Reuse one projection at several widths

```python
from rdkit import Chem
from fpembed import EmbeddedFingerprintGenerator, truncate_embedding

mol = Chem.MolFromSmiles("CCO")

gen8 = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=1024, compression=8, method="hadamard"
)
gen16 = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=1024, compression=16, method="hadamard"
)

source = gen8.GetFingerprintAsNumPy(mol)
prefix = truncate_embedding(source, 64, method="hadamard")
direct = gen16.GetFingerprintAsNumPy(mol)
```

`prefix` equals `direct` exactly for Hadamard. For random projection, use
`method="random_projection"` and compare with `allclose`, because the scaled
prefix matches within round-off. Block-wise methods are rejected because their
block boundaries change when the output width changes.

## Performance and Memory

### Complexity

| Method | Time | Retained memory |
|---|---|---|
| `geometric`, `linear`, `log`, `uniform` | `O(L)` per row | A weight vector of length C |
| `hadamard` | `O(L log L)` full transform | Sign vector of length L |
| `random_projection` | `O(L D)` dense | One `(L, D)` matrix per parameter set |

Block-wise methods remain the fastest overall.

### Peak memory

`dtype` leaves peak transient memory unchanged during compression. The float
paths compute in float64 and cast on return. Batch methods stack all valid rows
before returning, so the peak includes the output matrix plus the rows being
collected. Iterators avoid the accumulated matrix, but each row is still
computed and returned.

A 100K by 128 float64 matrix is about 100 MB. A 100K by 2048 float64 raw matrix
is about 1.6 GB. Subtract the expected output size from available memory before
choosing a batch API.

### Projection memory

The generator retains one `(L, D)` matrix per parameter set, about 2 MB at
L=2048 and D=128, for its lifetime. Matrices may be reused across generators up
to a fixed memory budget; this is not a process-wide cap.

### Parallelism

Fingerprint generation runs single-threaded inside FPembed, and FPembed adds no
thread pool. Parallelize across molecules or chunks at the application layer,
and size the worker count to the available cores and memory.

### Pharmacophore cost

In our measurements, folded pharmacophore took about 1.781 ms per molecule on
a 50-molecule sample at 2048 bits, against 0.038 ms for ECFP. At that rate, one
million molecules need about 30 minutes of fingerprint generation before
compression. Measure on your target data before making it a default.

## Limitations

- FPembed leaves molecules as written, including salts, tautomers,
  protonation states, and stereochemistry.
- You supply any projection matrix and center; FPembed only validates and
  applies them.
- FPembed trains and persists nothing. Save the arrays, manifest, matrix, and
  center yourself.
- Some fingerprint types reject count mode.
- Fixed-width types require padding, which leaves dead output dimensions.
- Compressed Euclidean distance differs from binary Tanimoto. Keep raw
  fingerprints for exact binary similarity.

## API Reference

### EmbeddedFingerprintGenerator

```python
EmbeddedFingerprintGenerator(
    fp_type: str,
    fp_size: int,
    compression: int | None,
    fp_params: dict | None = None,
    cache_size: int | None = None,
    method: str | None = None,
    method_params: dict | None = None,
    dtype: Any = np.float64,
    *,
    count: bool = False,
    count_transform: str = "none",
)
```

| Parameter | Meaning |
|---|---|
| `fp_type` | One of the fifteen supported identifiers |
| `fp_size` | Fingerprint length; positive integer. For fixed-width types, the padded width |
| `compression` | Power-of-two factor that divides the width and is at most half of it; `None` or `0` for raw output |
| `fp_params` | Fingerprint parameters for the chosen type |
| `cache_size` | Positive integer, or `None` to disable |
| `method` | Compression method, or `None` for mode-based resolution |
| `method_params` | Method parameters, including an optional matrix |
| `dtype` | `float64`, `float32`, or `uint16` |
| `count` | Produce count output instead of binary output |
| `count_transform` | `"none"` or `"log1p"`; requires `count=True` |

`compression` must divide `fp_size`, must be a power of two, and must not
exceed `fp_size / 2`. Unknown `fp_params` keys raise `ValueError`. `count` must
be a boolean.

Read-only properties:

| Property | Meaning |
|---|---|
| `fp_type` | Fingerprint type identifier |
| `fp_size` | Padded width |
| `compression` | Effective factor, `0` when inactive |
| `fp_params` | Copy of the fingerprint parameters |
| `method` | Resolved method name |
| `method_params` | Copy of the effective method parameters |
| `params_hash` | 16-character cache key |
| `dtype` | Normalized output dtype |
| `count` | Count mode flag |
| `count_transform` | Count transform name |
| `native_size` | Native width before padding |
| `padding` | `"none"` or `"tail_zeros"` |

Manifest fingerprint fields:

| Field | Meaning |
|---|---|
| `fingerprint.effective_size` | Independent bit positions; a quarter of `native_size` under count simulation, otherwise equal to it |
| `fingerprint.fixed_params` | Read-only backend values the factory pins; per-type contents are listed in [Manifest schema 2](#manifest-schema-2) |

Methods:

| Method | Returns |
|---|---|
| `GetFingerprintFromSmiles(smiles, validate=True, canonicalize=True)` | `(D,)` array or `None` for invalid SMILES |
| `GetFingerprintFromSelfies(selfies, validate=True, canonicalize=True)` | `(D,)` array or `None` for undecodable SELFIES |
| `GetFingerprintAsNumPy(mol)` | `(D,)` array for an RDKit `Mol` |
| `GetRawFingerprintAsNumPy(mol)` | Uncompressed `(fp_size,)` array |
| `GetFingerprintsFromSmiles(smiles_list, ...)` | `(N, D)` array and invalid indices |
| `GetFingerprintsFromSelfies(selfies_list, ...)` | `(N, D)` array and invalid indices |
| `GetFingerprintsAsNumPy(mols, chunk_size=None)` | `(N, D)` array and invalid indices |
| `IterFingerprints(mols)` | Iterator of `(index, array_or_none)` |
| `IterFingerprintsFromSmiles(smiles_iterable, ...)` | Iterator of `(index, array_or_none)` |
| `IterFingerprintsFromSelfies(selfies_iterable, ...)` | Iterator of `(index, array_or_none)` |
| `representation_manifest(preprocessing=None)` | Fresh schema 2 mapping |
| `cache_info()` | Named tuple with hits, misses, maxsize, currsize |
| `clear_cache()` | Clear the cache |

Batch methods return an empty `(0, D)` array when every input is invalid.

FPembed accepts `validate` for compatibility; parsing always validates.
`chunk_size` is deprecated, has no effect, and emits a `DeprecationWarning`.

### compress_fingerprint

```python
compress_fingerprint(
    vector,
    size: int,
    *,
    method: str = "geometric",
    method_params: dict | None = None,
    dtype: Any = np.float64,
)
```

`vector` is a 1-D `(L,)` or 2-D `(N, L)` array of real numeric values. Boolean,
signed integer, unsigned integer, and floating dtypes are accepted. Complex,
object, string, and datetime dtypes raise `TypeError`. Floating input with NaN
or infinity raises `ValueError`.

`size` is the compression factor. It only has to divide the feature length,
with no power-of-two requirement. The feature axis must be non-empty and evenly
divisible by `size`. Output is `(D,)` or `(N, D)`.

The function accepts counts for all six methods. Geometric reversibility and
integer codes are binary-only. `uniform` on counts is the block mean count.
`random_projection` uses the dense product for non-binary input. `dtype="uint16"`
requires binary input.

For `random_projection`, `method_params["matrix"]` may hold a canonical
`(D, L)` matrix. Its feature axis must match the input length. Validation and
compression do not mutate the input array.

### truncate_embedding

```python
truncate_embedding(
    embedding,
    target_dim: int,
    *,
    method: str,
    source_dim: int | None = None,
)
```

`embedding` may be 1-D or 2-D; leading dimensions beyond 2 are preserved. The
last axis is the feature axis. `target_dim` must be a positive integer no
greater than the feature length and no greater than `source_dim`. `source_dim`
defaults to the feature length.

`method` must be `"hadamard"` or `"random_projection"`. Hadamard returns a copy
of the prefix without scaling. Random projection returns the prefix multiplied
by `sqrt(source_dim / target_dim)`. Block-wise methods and unknown methods raise
`ValueError`. Non-numeric input raises `TypeError`.

The helper is meant for one projection embedding reused at a higher compression
factor. The precondition is the same fingerprint, method parameters, and seed
as the source embedding.

### PrecomputedEmbeddingCompressor

```python
PrecomputedEmbeddingCompressor(
    input_dim: int,
    compression: int | None,
    source_id: str,
    method: str | None = None,
    method_params: dict | None = None,
    dtype: Any = np.float64,
    normalize: str = "none",
    center: npt.ArrayLike | None = None,
)
```

The compressor applies an optional caller center, an optional row L2
normalization, and a projection. It never fits or persists. The parameter table,
the `compress` and `compress_one` behavior, and the center ordering are in
[Precomputed Embeddings](#precomputed-embeddings).

### fp_params_hash

```python
fp_params_hash(
    fp_type: str,
    fp_params: dict,
    *,
    method: str | None = None,
    method_params: dict | None = None,
    count: bool = False,
    count_transform: str = "none",
)
```

Returns a 16-character hexadecimal string. `count` must be a boolean,
`count_transform` must be `"none"` or `"log1p"`, and a non-default transform
requires `count=True`. Reserved keys in `fp_params` raise `ValueError`.

### SMILES helpers

```python
parse_smiles(smiles: str) -> Chem.Mol | None
canonicalize_smiles(smiles: str) -> str | None
canonicalize_to_mol(smiles: str) -> tuple[str, Chem.Mol] | tuple[None, None]
```

`parse_smiles` returns an RDKit `Mol`, or `None` for an unparseable string.
`canonicalize_smiles` returns the canonical SMILES form, or `None`.
`canonicalize_to_mol` parses once and returns `(canonical_smiles, mol)`, or
`(None, None)`. Each raises `TypeError` for non-string input.

## Troubleshooting and FAQ

### Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: fpembed` | Wrong interpreter | Select the interpreter where FPembed is installed, or install with `pip install -e .` |
| `Unsupported fp_type` | Misspelled or unsupported identifier | Use one of the fifteen identifiers in the Fingerprint Types section |
| `Count mode is not supported for fp_type` | Count requested for SECFP, Layered, Pattern, or a fixed-width type | Use a count-capable type or set `count=False` |
| `fp_size must be positive` | Zero or negative width | Pass a positive integer |
| `fp_size is smaller than the backend native_size` | Fixed-width type with a padded width below its native width: 881 for PubChem, 307 for Laggner, or ERG's configured width (`max_path` times the backend unit) | Raise `fp_size` to a power of two at or above the native width |
| `fp_size must be a power of two for fixed-width` | Exact native width or another non-power-of-two value | Use 1024 for PubChem, 512 for Laggner or ERG |
| `Hadamard method requires fp_size to be a power of 2` | Hadamard with a non-power-of-two width | Pad the width or use another method |
| `compression must be a power of 2` | Factor such as 3, 6, or 12 | Use 2, 4, 8, 16, and so on |
| `compression must be <= fp_size / 2` | Factor leaves fewer than two output values | Lower the factor |
| `fp_size must be evenly divisible by compression` | Factor does not divide the width | Choose a divisor |
| `cache_size must be positive` | Zero or negative cache size | Use a positive integer, or `None` to disable the cache |
| `dtype='uint16' is not supported for count fingerprints` | Count output with uint16 | Use float64 or float32 |
| `dtype='uint16' requires binary` | Raw or compressed non-binary values | Use a float dtype |
| `dtype='uint16' has no exact integer representation` | uint16 with linear, log, hadamard, or random projection | Use geometric or uniform, or a float dtype |
| `count_transform requires count=True` | Transform set without count mode | Set `count=True` or remove the transform |
| `count_simulation is a binary-mode option` | Count mode with `count_simulation` in `fp_params` | Remove the option |
| `geometric compression exceeds the maximum` | C > 32 with geometric | Lower C or use another method |
| `input_dim must be evenly divisible by compression` | Factor does not divide the embedding width | Choose a divisor |
| `hadamard requires input_dim and output_dim to be powers of two` | Hadamard with non-power-of-two dimensions | Pad the input or use `random_projection` |
| `center must be a vector of length input_dim` | Center shape mismatch | Pass a 1-D vector with one entry per input feature |
| `Input feature axis must equal input_dim` | Precomputed array width differs from `input_dim` | Reshape or configure the correct width |
| `source_id must be a non-empty string` | Missing model identity | Pass the checkpoint identifier |
| `preprocessing must be a JSON-serializable mapping` | Manifest metadata contains a non-JSON object | Convert it to plain types |

### FAQ

**Why does changing `fp_size` not change `params_hash`?**
The hash is a cache key. It covers the fingerprint type, parameters, method,
method parameters, and count representation. `fp_size`, compression factor,
dtype, preprocessing, and versions are recorded in the manifest instead.

**Can I use count fingerprints with every compression method?**
The generator allows `uniform`, `hadamard`, and `random_projection` when
compression is active. The standalone `compress_fingerprint()` accepts all six,
but geometric reversibility is binary-only.

**Why is my compressed output shape `(64,)` for PubChem at C=16?**
PubChem's native width is 881, and the padded width is 1024. Compression uses
the padded width, so `1024 / 16 = 64`. Eight of those output blocks are dead
padding.

**Why do I get `None` for an invalid SMILES?**
Single-molecule SMILES and SELFIES methods return `None` for unparseable input.
Batch methods omit those rows and return their indices in the second element.
Computation errors propagate.

**Can FPembed fit PCA or whitening for me?**
No. Fit the linear map outside FPembed and pass the result. See
[PCA and whitening fitted outside FPembed](#pca-and-whitening-fitted-outside-fpembed).

## Compatibility

### Python and dependencies

- Python 3.10 to 3.13.
- NumPy 1.26.4 or newer, below 3.
- RDKit 2023.9.6 or newer.
- scikit-fingerprints 2.1 or newer, below 3.
- SELFIES 2.2 or newer.

### API stability

The public API is available from the top-level `fpembed` package:

```python
EmbeddedFingerprintGenerator
compress_fingerprint
parse_smiles
canonicalize_smiles
canonicalize_to_mol
fp_params_hash
truncate_embedding
PrecomputedEmbeddingCompressor
__version__
```

Optional arguments do not change the output, hash, or manifest meaning of
configurations that do not use them. `fpembed.__version__` is the installed
version string.

### Manifest and hash compatibility

The manifest schema version moves independently of the package version. The
current schema is 2. Manifest layout may change between FPembed versions.
Re-derive a manifest rather than comparing serialized bytes across releases.

Hashes for a given configuration are stable across releases. The count and
transform keys appear only when non-default. Treat `params_hash` as one part of
a persisted array's identity. Store the manifest, the package versions, and the
input provenance.

## Glossary

| Term | Meaning |
|---|---|
| Fingerprint | A numeric vector derived from a molecule by a fingerprint algorithm |
| Binary fingerprint | A vector of 0 and 1 values |
| Count fingerprint | A vector of non-negative occurrence counts |
| `fp_size` | The fingerprint length, or the padded width for fixed-width types |
| Native width | The width a fixed-width fingerprint type produces before padding |
| Compression factor | `C`, the number of positions reduced to one output value |
| Output dimension | `D`, the embedding width, `L / C` when compression is active |
| Block-wise method | A method that reduces each block of `C` positions to one value |
| Projection method | A method that combines all positions into each output value |
| `params_hash` | A 16-character cache key derived from the fingerprint and method configuration |
| Manifest | A JSON mapping that records the full representation, including fields the hash omits |
| Precomputed embedding | A continuous embedding produced by another model and compressed by FPembed |

## Original Concept (eMFP)

The original concept of compressing molecular fingerprints with weighted binary
masking comes from this work on Morgan fingerprints:

> Nuñez-Andrade, E. A., Vidal-Daza, I., Gomez-Bombarelli, R., Ryan, J. W., and
> Martin-Martinez, F. J. (2025). Embedded Morgan Fingerprints for more efficient
> molecular property predictions with machine learning. ChemRxiv preprint.
> https://doi.org/10.26434/chemrxiv-2025-6hfp8

Original concept repository: https://github.com/MMLabCodes/eMFP

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

### Software and methods

- RDKit: https://www.rdkit.org/
- scikit-fingerprints: https://github.com/scikit-fingerprints/scikit-fingerprints
- SELFIES: https://github.com/aspuru-guzik-group/selfies
- Random projection and the Johnson-Lindenstrauss lemma: W. B. Johnson and
  J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space",
  Contemporary Mathematics 26, 1984.
- Fast Walsh-Hadamard transform: a standard divide-and-conquer transform over
  Sylvester-ordered Hadamard matrices.

### Citing FPembed

The archived software record has DOI
[10.5281/zenodo.21447996](https://doi.org/10.5281/zenodo.21447996).

## License and Support

FPembed is released under the MIT license. The authoritative license text is in
[LICENSE](LICENSE).

Legal, trademark, commercial, and support details are maintained in the README:
[License](README.md#license),
[Legal & Trademark Notice](README.md#legal--trademark-notice),
[Commercial & Research Services](README.md#commercial--research-services),
[AI Disclosure](README.md#ai-disclosure), and [Support](README.md#support).

The project is provided as-is and may be updated over time. Questions and bug
reports belong in the project issue tracker.
