(p-compression-compression-methods)=
# Compression Methods

Use this page to choose a compression `method` and set its `method_params`. It shows what each of the six methods does, which methods accept count input, and how the generator resolves a default.

## How compression works

Block-wise methods split the fingerprint into blocks of `C` positions and reduce each block to one number. With `geometric`, each block is read as a `C`-bit integer: for `C=4`, the block `1011` is 11, and the stored value is `11/15`. The mapping is reversible for binary input because each block integer is distinct. `uniform` stores the block mean, which is the fraction of set bits for binary input and the mean count for count input. `linear` and `log` use increasing and logarithmic weights instead of powers of two. Projection methods combine all positions, cost more, and can preserve distances better at high compression.

Six methods are available: four block-wise schemes and two projections.

## Method reference

| Method | Category | `method_params` | Effect |
|---|---|---|---|
| `geometric` | Block-wise | `interleave` | Positional integer code per block; reversible for binary input |
| `linear` | Block-wise | `interleave` | Weighted sum per block |
| `log` | Block-wise | `interleave` | Logarithmically weighted sum per block |
| `uniform` | Block-wise | `interleave` | Mean of the block; block mean count for counts |
| `hadamard` | Global | `seed` | Seeded, truncated Hadamard projection |
| `random_projection` | Global | `seed`, `sparse`, optional `matrix` | Dense or sparse random projection; Johnson-Lindenstrauss style |

All methods produce `D = L / C` output values. Complexity and retained memory are summarized in [Performance and Memory](performance.md#performance-and-memory).

## Method parameters

- Block-wise methods accept `interleave`, default `False`. When it is true, block membership follows a stride rather than contiguous ranges. Interleaving changes which positions share a block, and the output shape stays the same.
- `hadamard` accepts `seed`, default 42. The seed controls the random sign flips applied before the transform.
- `random_projection` accepts `seed` (default 42) and `sparse` (default `False`). `sparse=True` draws the Achlioptas distribution and changes the projection values. The matrix is still stored densely, so `sparse` does not reduce memory.
- `random_projection` also accepts a caller-supplied `matrix`. See [Caller-supplied projection matrix](#caller-supplied-projection-matrix).

Unknown `method_params` keys raise `ValueError`. Seeds must be non-negative integers, and `interleave` and `sparse` must be booleans.

## Method resolution

`method=None` resolves by mode:

| State | Resolved method |
|---|---|
| Compression inactive | `geometric` (no method runs) |
| Binary, compression active | `geometric` |
| Count, compression active | `uniform` |

An explicit forbidden method still raises.

## Count compression rules

With active count compression, only `uniform`, `hadamard`, and `random_projection` are allowed. `geometric`, `linear`, and `log` raise `ValueError`, because their block value is defined on bits or on weighted sums and neither has a count interpretation.

With `uniform`, each output is the mean count in its block, or the mean `log1p` count when the transform is active. With `hadamard` and `random_projection`, each output is a projected count value. Random projection uses the dense product for counts because the sparse path assumes binary non-zeros.

The standalone `compress_fingerprint()` accepts counts for all six methods. The restricted method set applies to the generator.

When count compression is inactive, the count-specific method rejection does not apply, because no compression runs. Checks such as the Hadamard power-of-two rule still apply.

## Geometric limit

`geometric` compression is capped at `C <= 32`. Above that cap, the block integer starts to lose low bits in the float64 accumulation. The cap applies to `geometric` alone and only when compression is active. The other methods are not capped.

(p-compression-caller-supplied-projection-matrix)=
## Caller-supplied projection matrix

`random_projection` accepts a canonical matrix in `method_params={"matrix": M}` with shape `(D, L)`, where `L` is the fingerprint length and `D = L // compression`. FPembed validates the shape, requires a real finite array, converts it to float64, and digests it. It applies the matrix as given rather than fitting it.

The digest, `matrix_id`, and `matrix_shape` enter the parameter hash and the manifest. The matrix contents stay out of the cache key. See [Identity, Manifests, and Caching](identity.md#identity-manifests-and-caching) and the [Reuse one projection at several widths](recipes.md#reuse-one-projection-at-several-widths) recipe.

## Choosing a method

Block-wise methods are fast and simple. `geometric` keeps the widest code range for binary input. `uniform` treats each position equally and is the count default. Projection methods mix information across positions and can retain more information at high compression ratios. `hadamard` requires power-of-two dimensions. `random_projection` carries a distance-preservation statement for the projection itself and costs more.

Method rankings depend on the downstream model. A configuration that works for a tree model may not suit a linear model. Treat any single recommendation as a starting point.

| Use case | Starting point | Check before relying on it |
|---|---|---|
| Tree-based property prediction | `geometric`, C=16 | Compare against raw fingerprints and other methods |
| Linear prediction | `random_projection` | Retune regularization |
| Similarity or retrieval | `random_projection` | Measure recall at k and neighborhood preservation |
| Exact binary similarity | Bit-packed raw fingerprints | Compressed Euclidean distance differs from Tanimoto |
| Dense external embeddings | Projection methods | The binary code argument does not apply |

PCA is a learned linear map fitted on data. FPembed's methods are fixed and do not look at the data. To use PCA, fit it outside FPembed and pass the matrix; the generator applies the matrix without centering. The corrected recipe is in [PCA and whitening fitted outside FPembed](recipes.md#pca-and-whitening-fitted-outside-fpembed).

(compression-compress-fingerprint)=
## `compress_fingerprint`

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

| Parameter | Meaning |
|---|---|
| `vector` | 1-D `(L,)` or 2-D `(N, L)` array of real numeric values |
| `size` | Compression factor; positive integer that divides the feature length |
| `method` | One of the six methods; default `"geometric"` |
| `method_params` | Method parameters, including an optional matrix |
| `dtype` | `float64`, `float32`, or `uint16` |

`vector` accepts boolean, signed integer, unsigned integer, and floating dtypes. Complex, object, string, and datetime dtypes raise `TypeError`. Floating input that contains NaN or infinity raises `ValueError`.

`size` is the compression factor. It has to divide the feature length and needs no power-of-two factor. The feature axis must be non-empty and evenly divisible by `size`. Output is `(D,)` or `(N, D)`.

The function accepts counts for all six methods. Geometric reversibility and integer codes work only with binary input. `uniform` on counts returns the block mean count. `random_projection` uses the dense product for non-binary input. `dtype="uint16"` requires binary input.

For `random_projection`, `method_params["matrix"]` may hold a canonical `(D, L)` matrix. Its feature axis must match the input length. Validation and compression do not mutate the input array.

### Errors

(compression-compression-input)=
#### The compression input has the wrong shape, dtype, or values

**Message:** `Input must be 1-D (L,) or 2-D (N, L); got ndim ...`; `Input feature axis (last axis) must be non-empty`; `Input contains non-finite values (NaN or infinity)`; `Input dtype '...' is not a real numeric type`

**Cause:** An array that is not 1-D or 2-D, has an empty feature axis, holds complex or object values, or holds NaN or infinity.

**Fix:** Pass a finite real numeric array of shape `(L,)` or `(N, L)`.

**Example:** `compress_fingerprint(np.array([np.nan, 0.0]), size=1)` raises the non-finite error.

(compression-compression-divisible)=
#### The input length is not divisible by the compression factor

**Message:** `Input length (...) is not evenly divisible by size (...)`

**Cause:** The factor does not divide the feature length. The generator reports the same shape problem against `fp_size` in [the generator's compression divisibility entry](generator.md#embeddedfingerprintgenerator).

**Fix:** Choose a factor that divides the feature length.

**Example:** `compress_fingerprint(np.ones(100), size=8)` raises the error.

(compression-geometric-max)=
#### Geometric compression exceeds its limit

**Message:** `geometric compression (...) exceeds the maximum of 32`; `geometric block size (...) exceeds the maximum of 32`

**Cause:** A geometric factor above 32. The block integer no longer fits the float64 mantissa, so its low bits are lost.

**Fix:** Use a factor of 32 or lower, or choose another method.

**Example:** `compress_fingerprint(np.ones(66), size=33, method="geometric")` raises the error.

(compression-hadamard-dimensions)=
#### Hadamard dimensions must be powers of two

**Message:** `hadamard requires L (...) and D = L // size (...) to be powers of two.`

**Cause:** Hadamard with a feature length or an output dimension that is not a power of two. [The precomputed compressor](precomputed.md#compressor-configuration) reports the same rule against `input_dim`.

**Fix:** Pad the input or use `random_projection`.

**Example:** `compress_fingerprint(np.ones(1000), size=4, method="hadamard")` raises the error.

(compression-method)=
#### The method is unsupported

**Message:** `Unsupported method '...'`

**Cause:** A method name outside the six supported methods.

**Fix:** Use one of `geometric`, `linear`, `log`, `uniform`, `hadamard`, or `random_projection`.

**Example:** `compress_fingerprint(np.ones(8), size=2, method="pca")` raises the error.

(compression-method-params)=
#### `method_params` has an unknown or invalid value

**Message:** `Unrecognized method_params keys ...`; `seed must be a non-negative integer`; `sparse must be a boolean`; `interleave must be a boolean`; `interleave is not applicable to method '...'`

**Cause:** A key the method does not use, or a value of the wrong type or range. The generator and the precomputed compressor run the same check.

**Fix:** Use only the keys the method documents, with the stated value type.

**Example:** `compress_fingerprint(np.ones(8), size=2, method="random_projection", method_params={"seed": -1})` raises the error.

(compression-matrix)=
#### The projection matrix has the wrong shape, dtype, or values

**Message:** `Projection matrix must be 2-D`; `Projection matrix must have shape (D, L)`; `Projection matrix dtype '...' is not a real numeric type`; `Projection matrix must contain only finite values.`

**Cause:** A supplied matrix that is not 2-D, not `(D, L)`, not real numeric, or not finite.

**Fix:** Pass a real, finite array with shape `(D, L)`, where `L` is the feature length and `D = L // size`.

**Example:** `compress_fingerprint(np.ones(8), size=2, method="random_projection", method_params={"matrix": np.zeros((3, 8))})` raises the shape error.

(compression-uint16-binary)=
#### `uint16` needs binary input

**Message:** `dtype='uint16' requires binary input (every element 0 or 1)`

**Cause:** Raw or compressed values that are not 0 or 1.

**Fix:** Use a float dtype for count or transformed values, or request `uint16` only for a binary fingerprint.

**Example:** `compress_fingerprint(np.array([0.0, 2.0, 0.0, 0.0]), size=2, dtype="uint16")` raises the error.

(compression-dtype)=
#### The dtype cannot hold the integer code

**Message:** `dtype='uint16' with method '...' requires compression <= ...`

**Cause:** `uint16` with `geometric` or `uniform` at a compression factor above the uint16 range for that method.

**Fix:** Lower the factor to the stated limit, or use a float dtype.

**Example:** `compress_fingerprint(np.zeros(64), size=32, method="geometric", dtype="uint16")` raises the error.

(compression-argument-value)=
#### An argument is outside its range

**Message:** `Size must be a positive integer`; `Compression factor must be a positive integer, got ...`; `Block size must be a positive integer, got ...`

**Cause:** A size argument that is zero, negative, or a boolean.

**Fix:** Pass a positive integer size.

**Example:** `compress_fingerprint(np.ones(8), size=0)` raises the error.

(compression-truncate-embedding)=
## `truncate_embedding`

```python
truncate_embedding(
    embedding,
    target_dim: int,
    *,
    method: str,
    source_dim: int | None = None,
)
```

| Parameter | Meaning |
|---|---|
| `embedding` | 1-D or 2-D array; leading dimensions beyond 2 are preserved |
| `target_dim` | Positive integer no greater than the feature length and no greater than `source_dim` |
| `method` | `"hadamard"` or `"random_projection"` |
| `source_dim` | Feature length of the source embedding; defaults to the feature length |

The last axis is the feature axis. `target_dim` must be a positive integer no greater than the feature length and no greater than `source_dim`.

`method` must be `"hadamard"` or `"random_projection"`. Hadamard returns a copy of the prefix without scaling. Random projection returns the prefix multiplied by `sqrt(source_dim / target_dim)`. Block-wise methods and unknown methods raise `ValueError`. Non-numeric input raises `TypeError`.

Use the helper for one projection embedding reused at a higher compression factor. The precondition is the same fingerprint, method parameters, and seed as the source embedding.

### Errors

(compression-truncation)=
#### The truncation target or method is invalid

**Message:** `target_dim (...) exceeds the available feature length (...)`; `target_dim (...) must not exceed source_dim (...)`; `target_dim must be positive, got ...`; `source_dim must be positive, got ...`; `method '...' is block-wise`; `Unsupported method '...' for truncation`; `embedding must have at least one dimension`

**Cause:** A target above the available width, a non-positive dimension, a block-wise method, or an unknown method name.

**Fix:** Use `hadamard` or `random_projection`, and keep `target_dim` at or below the feature length.

**Example:** `truncate_embedding(np.zeros(8), target_dim=16, method="hadamard")` raises the target error.

(compression-argument-type)=
#### An argument has the wrong type

**Message:** `target_dim must be an integer, got ...`; `source_dim must be an integer, got ...`; `embedding must have a numeric dtype, got ...`

**Cause:** The value has a different type than the signature accepts.

**Fix:** Pass the documented type.

**Example:** `truncate_embedding(np.zeros(8), target_dim="4", method="hadamard")` raises the target type error.
