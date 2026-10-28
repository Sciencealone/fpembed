(p-precomputed-precomputed-embeddings)=
# Precomputed Embeddings

Use this page to compress and normalize continuous embeddings that another model produced. You need it when the vectors come from outside FPembed and you want a smaller output with a recorded identity.

## When to use

Use `PrecomputedEmbeddingCompressor` to shrink and normalize continuous embeddings from another model. For raw molecular fingerprints, use `EmbeddedFingerprintGenerator` or `compress_fingerprint()`.

The compressor holds configuration and projection artifacts. It does not fit, train, or persist anything.

(precomputed-constructor)=
## Compressor configuration

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

The constructor validates every argument, resolves the method, and computes `params_hash`. It reads no data and fits nothing.

Methods:

| Method | Input | Returns |
|---|---|---|
| [`compress(embeddings)`](#precomputed-compress) | `(input_dim,)` or `(N, input_dim)`, including `(0, input_dim)` | `(D,)`, `(N, D)`, or `(0, D)` |
| [`compress_one(embedding)`](#precomputed-compress-one) | 1-D `(input_dim,)` | `(D,)` |
| `representation_manifest()` | None | Fresh schema 2 mapping; the definition is on [Identity, Manifests, and Caching](identity.md#representation_manifest) |
| `params_hash` | Property | 16-character identity hash; the identity rules are on [Identity, Manifests, and Caching](identity.md#what-the-parameter-hash-covers) |

### Errors

(precomputed-argument-type)=
#### An argument has the wrong type

**Message:** `input_dim must be an int, got ...`; `compression must be an int or None, got ...`

**Cause:** The value has a different type than the signature accepts. Booleans do not count as integers for these arguments.

**Fix:** Pass the documented type. Use `None` where the argument is optional.

**Example:** `PrecomputedEmbeddingCompressor("8", 2, "model")` raises the error.

(precomputed-argument-value)=
#### An argument is outside its range

**Message:** `input_dim must be positive, got ...`; `compression must be >= 0, got ...`

**Cause:** An integer argument is outside the accepted range.

**Fix:** Pass a value in the stated range.

**Example:** `PrecomputedEmbeddingCompressor(0, 2, "model")` raises the error.

(precomputed-source-id-empty)=
#### `source_id` must name the source model

**Message:** `source_id must be a non-empty string`

**Cause:** Missing or blank model identity for precomputed embeddings.

**Fix:** Pass the checkpoint or model identifier.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=None, source_id="")` raises the error.

(precomputed-compression-power-of-two)=
#### The compression factor must be a power of two

**Message:** `compression must be a power of 2, got ...`

**Cause:** A factor such as 3, 6, or 12.

**Fix:** Use 2, 4, 8, 16, or another power of two.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=3, source_id="model")` raises the error.

(precomputed-compression-half)=
#### The compression factor exceeds half the width

**Message:** `compression (...) must be <= input_dim / 2 (...)`

**Cause:** The factor leaves fewer than two output values.

**Fix:** Lower the factor so the width holds at least two output values.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=8, source_id="model")` raises the error.

(precomputed-input-dim-divisible)=
#### `input_dim` is not divisible by the compression factor

**Message:** `input_dim (...) must be evenly divisible by compression (...)`

**Cause:** The factor does not divide the precomputed embedding width.

**Fix:** Choose a divisor of `input_dim`.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=100, compression=8, source_id="model")` raises the error.

(precomputed-normalize)=
#### The normalization mode is invalid

**Message:** `normalize must be one of ('none', 'l2'), got ...`

**Cause:** An unknown normalization name.

**Fix:** Pass `"none"` or `"l2"`.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=None, source_id="model", normalize="l1")` raises the error.

(precomputed-dtype)=
#### The dtype is unsupported for precomputed embeddings

**Message:** `dtype must be float64 or float32 for precomputed embeddings`; `dtype='uint16' is not supported for precomputed embeddings`

**Cause:** A dtype the compressor cannot store, such as `uint16`, an integer type, or `float16`.

**Fix:** Use `float64` or `float32`. External values are continuous, so the integer encodings do not apply.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=None, source_id="model", dtype="uint16")` raises the error.

(precomputed-center-length)=
#### The center vector must match `input_dim`

**Message:** `center must be a vector of length input_dim (...)`

**Cause:** The center is not 1-D, or its length differs from `input_dim`.

**Fix:** Pass a 1-D vector with one entry per input feature.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=None, source_id="model", center=np.zeros(4))` raises the error.

(precomputed-method)=
#### The method is not allowed for precomputed compression

**Message:** `Precomputed compression supports only ...`

**Cause:** An active compression factor with a method other than `hadamard` or `random_projection`.

**Fix:** Use `hadamard` or `random_projection`, because block-wise outputs are tied to binary block positions.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=8, compression=2, source_id="model", method="geometric")` raises the error.

(precomputed-hadamard-dimensions)=
#### Hadamard dimensions must be powers of two

**Message:** `hadamard requires input_dim (...) and input_dim // compression (...) to be powers of two.`

**Cause:** Hadamard with non-power-of-two dimensions.

**Fix:** Pad the input or use `random_projection`.

**Example:** `PrecomputedEmbeddingCompressor(input_dim=768, compression=4, source_id="model", method="hadamard")` raises the error because 768 is not a power of two.

The [method_params entry on the compression page](compression.md#compress_fingerprint) covers an unknown key or an invalid seed in `method_params`.

## Compression and normalization

The compressor subtracts the center before row L2 normalization. An all-zero row stays zero under L2 normalization. FPembed never fits the center; you supply it.

When compression is active, FPembed allows only projection methods, because block-wise outputs are tied to binary block positions and continuous external embeddings have no such structure. When compression is inactive, no method runs and FPembed ignores a supplied method.

## Identity and manifest

The precomputed params hash covers `source_id`, `input_dim`, `normalize`, `center_id`, `method`, and `method_params`. The compression factor and the dtype stay out; record them in the manifest.

The manifest records `center_id` and `matrix_id` in place of the center and matrix values. Store the arrays themselves next to the manifest, as listed in [What to store](identity.md#what-to-store).

## CLAMP recipe

CLAMP is an external fingerprint model. It is not a core `fp_type`. FPembed ships no CLAMP weights and does not depend on torch. This recipe treats CLAMP as a precomputed embedding source.

1. Generate 768-dimensional float32 embeddings with the CLAMP code and checkpoint outside FPembed. Torch and the 174 MB weights are external dependencies of that step.
2. Record a `source_id` that identifies the exact checkpoint, for example `"clamp:<checkpoint-hash>"`.
3. Compress with `random_projection`, because 768 is not a power of two and Hadamard requires power-of-two input and output dimensions.
4. Fix the batch size you use for generation. CLAMP measured a maximum drift of about 1.9e-6 between batch sizes, so a change of batch size can produce slightly different embeddings and break reproducible caching.

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

Hadamard is possible only if you zero-pad each 768-d vector to 1024 before compression and set `input_dim=1024` with `method="hadamard"`. The padded positions are zero, and the padded result differs from the 768-d random projection. Choose one path and record it in the manifest.

Use `compress_one` for a single vector. Use `representation_manifest()` to persist the source identity, normalization, compression configuration, and versions alongside the array.

## Persisting embeddings

```python
import json
import numpy as np

np.save("embeddings.npy", embeddings)
with open("embeddings.manifest.json", "w", encoding="utf-8") as handle:
    json.dump(compressor.representation_manifest(), handle, indent=2)
```

Store the manifest with the array. The parameter hash alone omits the compression factor, the dtype, and dependency versions.

(precomputed-compress)=
## `compress`

```python
compress(self, embeddings: npt.ArrayLike) -> npt.NDArray[Any]
```

| Parameter | Meaning |
|---|---|
| `embeddings` | 1-D `(input_dim,)` or 2-D `(N, input_dim)` array, including `(0, input_dim)` |

Returns `(D,)` for a single vector, `(N, D)` for a batch, or `(0, D)` for an empty batch. The method validates the shape and the values, subtracts the center, applies row L2 normalization when configured, and then compresses. An all-zero row stays zero under L2 normalization. FPembed never fits the center; you supply it.

### Errors

(precomputed-compression-input)=
#### The compression input has the wrong shape or values

**Message:** `Input must be 1-D (input_dim,) or 2-D (N, input_dim) with input_dim=...`; `Input contains non-finite values (NaN or infinity)`; `Input dtype '...' is not a real numeric type`

**Cause:** An array with the wrong number of dimensions, a complex or object dtype, or NaN or infinity.

**Fix:** Pass a finite real numeric array of shape `(input_dim,)` or `(N, input_dim)`.

**Example:** `compressor.compress(np.zeros((2, 2, 2)))` raises the shape error.

(precomputed-input-feature-axis)=
#### The input feature axis must match `input_dim`

**Message:** `Input feature axis must equal input_dim (...)`

**Cause:** The width of the array differs from `input_dim`.

**Fix:** Reshape the array or configure the matching width.

**Example:** `compress(np.zeros((10, 4)))` on a compressor with `input_dim=8` raises the error.

(precomputed-compress-one)=
## `compress_one`

```python
compress_one(self, embedding: npt.ArrayLike) -> npt.NDArray[Any]
```

| Parameter | Meaning |
|---|---|
| `embedding` | 1-D `(input_dim,)` array |

Returns the `(D,)` embedding for one vector. The method rejects an array that is not 1-D and then runs the same validation and transform as `compress`.

### Errors

The [`compression input`](#precomputed-compression-input) and [`input feature axis`](#precomputed-input-feature-axis) entries apply here. `compress_one` reports a non-1-D array as `compress_one expects a 1-D vector of length input_dim (...)`.
