# Working with the Generator

Use this page when you call `EmbeddedFingerprintGenerator` on one molecule, on a stream, or on a batch that may hold invalid input. The examples show the call and the shape or dtype it returns, and one section gives a starting point for each configuration axis. The constructor and method reference, with each parameter table and error entry, follows the usage sections.

## SELFIES and RDKit Mol input

The generator accepts a SELFIES string or an RDKit `Mol` object and returns one embedding per molecule. [GetFingerprintFromSelfies](#generator-getfingerprintfromselfies) and [GetFingerprintAsNumPy](#generator-getfingerprintasnumpy) carry the full definitions.

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

## Raw fingerprint

`compression=None` leaves the fingerprint at its full width, so the output keeps `fp_size` values.

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

## Streaming input

Batch methods collect the valid embeddings before stacking them. For inputs that do not fit in memory, iterate:

```python
for index, embedding in gen.IterFingerprintsFromSmiles(smiles_iterable):
    if embedding is None:
        continue
    store(index, embedding)
```

`IterFingerprints`, `IterFingerprintsFromSmiles`, and `IterFingerprintsFromSelfies` pull one input at a time and yield `(index, embedding_or_none)`. Indices match input order. The iterator does not materialize the input or accumulate rows.

## Invalid input and label alignment

Single-molecule methods return `None` for invalid input. Batch methods return the valid rows plus the indices of invalid inputs, as shown in [Quick Start](quickstart.md#batch-input-and-label-alignment). An all-invalid batch returns an empty `(0, D)` array.

Filter labels with the returned indices before training. Computation errors raise instead of appearing as invalid input.

## Standalone compression

`compress_fingerprint()` compresses one fingerprint without a generator. [Compression Methods](compression.md#compress_fingerprint) carries its definition.

```python
import numpy as np
from fpembed import compress_fingerprint

fp = np.random.randint(0, 2, size=2048)
embedding = compress_fingerprint(fp, size=16)
print(embedding.shape)  # (1, 128)
```

## Output dtype

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

`dtype` shrinks the returned and stored array. Peak transient memory stays the same, because the float paths compute in float64 and cast on return. `dtype` is not part of `params_hash`. [Output Dtype and Exactness](dtype.md#output-dtype-and-exactness) covers the exactness rules.

## Choosing a configuration

| Axis | Starting point | Change it when |
|---|---|---|
| Fingerprint type | `ecfp` with `radius=2` | Your task needs longer-range or substructure relationships |
| `fp_size` | 2048 | You need a wider fingerprint or fewer hashing collisions |
| Compression factor | 16 | Lower it if the model underfits with 128 features; raise it if storage or speed matters more |
| Method | Default for the mode | You need distance preservation at high compression |
| Dtype | `float64` | You need smaller arrays and accept the precision limits |

Change one axis at a time and compare on a validation split. The default method is `geometric` for binary input and `uniform` for count input.

(generator-constructor)=
## `EmbeddedFingerprintGenerator`

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

`compression` must divide `fp_size`, must be a power of two, and must not exceed `fp_size / 2`. Unknown `fp_params` keys raise `ValueError`. `count` must be a boolean.

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
| `fingerprint.fixed_params` | Read-only backend values the factory pins; per-type contents are listed in [Manifest schema 2](identity.md#manifest-schema-2) |

Methods:

| Method | Returns |
|---|---|
| [`GetFingerprintFromSmiles`](#generator-getfingerprintfromsmiles) | `(D,)` array or `None` for invalid SMILES |
| [`GetFingerprintFromSelfies`](#generator-getfingerprintfromselfies) | `(D,)` array or `None` for undecodable SELFIES |
| [`GetFingerprintAsNumPy`](#generator-getfingerprintasnumpy) | `(D,)` array for an RDKit `Mol` |
| [`GetRawFingerprintAsNumPy`](#generator-getrawfingerprintasnumpy) | Uncompressed `(fp_size,)` array |
| [`GetFingerprintsFromSmiles`](#generator-getfingerprintsfromsmiles) | `(N, D)` array and invalid indices |
| [`GetFingerprintsFromSelfies`](#generator-getfingerprintsfromselfies) | `(N, D)` array and invalid indices |
| [`GetFingerprintsAsNumPy`](#generator-getfingerprintsasnumpy) | `(N, D)` array and invalid indices |
| [`IterFingerprints`](#generator-iterfingerprints) | Iterator of `(index, array_or_none)` |
| [`IterFingerprintsFromSmiles`](#generator-iterfingerprintsfromsmiles) | Iterator of `(index, array_or_none)` |
| [`IterFingerprintsFromSelfies`](#generator-iterfingerprintsfromselfies) | Iterator of `(index, array_or_none)` |
| `representation_manifest(preprocessing=None)` | Fresh schema 2 mapping; the definition is on [Identity, Manifests, and Caching](identity.md#representation_manifest) |
| `cache_info()` | Named tuple with hits, misses, maxsize, currsize; the definition is on [Identity, Manifests, and Caching](identity.md#cache_info) |
| `clear_cache()` | Clear the cache; the definition is on [Identity, Manifests, and Caching](identity.md#clear_cache) |

The constructor resolves the method, validates the dtype, builds the backend, and computes `params_hash`. It reads no input data.

### Errors

<details class="fpembed-errors">
<summary>Show 23 entries</summary>

(generator-fp-size-positive)=
#### `fp_size` must be positive

**Message:** `fp_size must be positive, got 0`

**Cause:** The width is zero or negative.

**Fix:** Pass a positive integer.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 0, None)` raises the error.

(generator-fp-size-native)=
#### `fp_size` is below the native width

**Message:** `fp_size (...) is smaller than the backend native_size (...)`

**Cause:** A fixed-width type whose padded width is below its native width: 881 for PubChem, 307 for Laggner, or the ERG width from `max_path`.

**Fix:** Raise `fp_size` to a power of two at or above the native width.

**Example:** `EmbeddedFingerprintGenerator("pubchem", 512, None)` raises the error, and 1024 works.

(generator-fp-size-power-of-two)=
#### A fixed-width type needs a power-of-two `fp_size`

**Message:** `fp_size (...) must be a power of two for fixed-width fp_type '...'`

**Cause:** A fixed-width type with the exact native width or another non-power-of-two value.

**Fix:** Use 1024 for PubChem, or 512 for Laggner and ERG.

**Example:** `EmbeddedFingerprintGenerator("pubchem", 881, None)` raises the error.

(generator-unsupported-fp-type)=
#### Unsupported fingerprint type

**Message:** `Unsupported fp_type '...'`

**Cause:** The `fp_type` value is not one of the fifteen identifiers the generator supports. The message lists the supported set.

**Fix:** Pass a supported identifier, such as `ecfp`, `maccs`, `pubchem`, or `erg`.

**Example:** `EmbeddedFingerprintGenerator("ecfpp", 1024, None)` raises the error, and `EmbeddedFingerprintGenerator("ecfp", 1024, None)` builds.

(generator-count-mode-not-supported)=
#### Count mode is not supported for this type

**Message:** `Count mode is not supported for fp_type '...'`

**Cause:** `count=True` with a type that stores bits rather than counts, such as SECFP, Layered, Pattern, or a fixed-width type.

**Fix:** Use a count-capable type or pass `count=False`.

**Example:** `EmbeddedFingerprintGenerator("maccs", 1024, None, count=True)` raises the error.

(generator-count-simulation-binary)=
#### `count_simulation` is a binary-mode option

**Message:** `fp_params['count_simulation'] is a binary-mode option and cannot be combined with count=True.`

**Cause:** `count_simulation` in `fp_params` while count mode is active.

**Fix:** Remove the option in count mode, or use binary mode.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, fp_params={"count_simulation": True}, count=True)` raises the error.

(generator-cache-size-positive)=
#### `cache_size` must be positive

**Message:** `cache_size must be positive, got ...`

**Cause:** Zero or a negative cache size.

**Fix:** Pass a positive integer, or `None` to disable the cache.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, cache_size=0)` raises the error.

(generator-hadamard-fp-size)=
#### Hadamard needs a power-of-two `fp_size`

**Message:** `Hadamard method requires fp_size to be a power of 2`

**Cause:** The `hadamard` method with a width that is not a power of two.

**Fix:** Pad the width to the next power of two, or choose another method.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1000, None, method="hadamard")` raises the error.

(generator-compression-power-of-two)=
#### The compression factor must be a power of two

**Message:** `compression must be a power of 2, got ...`

**Cause:** A factor such as 3, 6, or 12.

**Fix:** Use 2, 4, 8, 16, or another power of two. [The precomputed compressor](precomputed.md#compressor-configuration) applies the same rule to `input_dim`.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, 3)` raises the error.

(generator-compression-half)=
#### The compression factor exceeds half the width

**Message:** `compression (...) must be <= fp_size / 2 (...)`

**Cause:** The factor leaves fewer than two output values. [The precomputed compressor](precomputed.md#compressor-configuration) compares the factor with `input_dim` instead of `fp_size`.

**Fix:** Lower the factor so the width holds at least two output values.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, 1024)` raises the error.

(generator-compression-divisible)=
#### The width is not divisible by the compression factor

**Message:** `fp_size (...) must be evenly divisible by compression (...)`

**Cause:** The factor does not divide the padded width. The standalone `compress_fingerprint()` reports the same shape problem as [the input-length entry](compression.md#compress_fingerprint).

**Fix:** Choose a factor that divides the width. A power of two divides every power-of-two width.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 100, 8)` raises the error.

(generator-method)=
#### The method is unsupported or not allowed here

**Message:** `Unsupported method '...'`; `method '...' cannot be used with count fingerprints`

**Cause:** An unknown method name, or a binary-only method with count compression. Count compression accepts `uniform`, `hadamard`, and `random_projection`.

**Fix:** Use a listed method.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, 4, method="geometric", count=True)` raises the count error.

(generator-method-params)=
#### `method_params` has an unknown or invalid value

**Message:** `Unrecognized method_params keys ...`; `seed must be a non-negative integer`; `sparse must be a boolean`; `interleave must be a boolean`; `interleave is not applicable to method '...'`; `method_params['matrix'] requires active compression; compression is None or 0.`

**Cause:** A matrix passed without an active compression factor. The unknown-key, seed, sparse, and interleave checks use the same messages and run in the shared validator; [the compression method_params entry](compression.md#compress_fingerprint) covers them.

**Fix:** Use only the keys the method documents, with the stated value type, and pass a matrix only with an active factor.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, method="random_projection", method_params={"matrix": np.zeros((1024, 1024))})` raises the error.

(generator-fp-params)=
#### `fp_params` has an unknown, reserved, or fixed key

**Message:** `Unknown fp_params key(s) for fp_type '...'`; `'variant' is not accepted for fp_type '...'`

**Cause:** A key outside the documented set for the type, or the fixed `variant` key. The reserved keys `fp_type`, `method`, `method_params`, `count`, and `count_transform` raise the same error from the parameter hash; [the fp_params entry on the identity page](identity.md#fp_params_hash) covers that path.

**Fix:** Remove the key. Pass fingerprint settings as generator arguments, and pass the remaining backend parameters in `fp_params`.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, fp_params={"raduis": 2})` raises the unknown-key error.

(generator-dtype)=
#### The dtype is unsupported or incompatible

**Message:** `Unsupported dtype ...`

**Cause:** A dtype outside `float64`, `float32`, and `uint16`, or a name NumPy cannot build.

**Fix:** Use `float64`, `float32`, or `uint16`. Precomputed embeddings accept `float64` and `float32` only; [their dtype entry](precomputed.md#compressor-configuration) covers that rule.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, dtype="float16")` raises the error.

(generator-uint16-count)=
#### `uint16` cannot store counts or non-binary values

**Message:** `dtype='uint16' is not supported for count fingerprints`; `dtype='uint16' requires binary values; count or transformed representations cannot be stored as uint16.`; `dtype='uint16' requires binary input (every value 0 or 1); the exact integer code is undefined otherwise.`

**Cause:** Count output with `dtype="uint16"`, because count values exceed the uint16 range and would wrap. The single-molecule methods repeat the check before the cast and report the raw and compressed forms of the binary rule.

**Fix:** Use `float64` or `float32`, or request `uint16` only for a binary fingerprint.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, count=True, dtype="uint16")` raises the error.

(generator-uint16-no-integer)=
#### This method has no exact `uint16` code

**Message:** `dtype='uint16' has no exact integer representation for method '...'`; `dtype='uint16' with method '...' requires compression <= ...`

**Cause:** `uint16` with `linear`, `log`, `hadamard`, or `random_projection`, or with `geometric` at a compression above 16.

**Fix:** Use `geometric` or `uniform`, lower the compression factor to the stated limit, or use a float dtype.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, 4, method="hadamard", dtype="uint16")` raises the error.

(generator-count-transform-requires-count)=
#### A count transform needs count mode

**Message:** `count_transform requires count=True`

**Cause:** A transform such as `"log1p"` with `count=False`.

**Fix:** Set `count=True` or remove the transform.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, count_transform="log1p")` raises the error.

(generator-count-transform-value)=
#### The count transform is not `none` or `log1p`

**Message:** `count_transform must be one of ('none', 'log1p'), got ...`

**Cause:** An unknown transform name.

**Fix:** Pass `"none"` or `"log1p"`.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, None, count_transform="sqrt")` raises the error.

(generator-argument-type)=
#### An argument has the wrong type

**Message:** `fp_size must be an int, got ...`; `compression must be an int or None, got ...`; `cache_size must be an int or None, got ...`; `count must be a bool, got ...`

**Cause:** The value has a different type than the signature accepts. Booleans do not count as integers for these arguments.

**Fix:** Pass the documented type. Use `None` where the argument is optional.

**Example:** `EmbeddedFingerprintGenerator("ecfp", "1024", None)` raises the error.

(generator-argument-value)=
#### An argument is outside its range

**Message:** `compression must be >= 0, got ...`; `chunk_size must be a positive integer or None`

**Cause:** An integer argument is outside the accepted range. `chunk_size` is deprecated and has no effect, but its value is still checked.

**Fix:** Pass a value in the stated range.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, -1)` raises the error.

(generator-mol-none)=
#### The molecule object is `None`

**Message:** `mol must not be None`

**Cause:** `None` passed where a molecule object is required.

**Fix:** Parse the input first and pass the resulting molecule, or use a SMILES method.

**Example:** `generator.GetFingerprintAsNumPy(None)` raises the error.

(generator-matrix)=
#### The projection matrix has the wrong shape, dtype, or values

**Message:** `Projection matrix must be 2-D`; `Projection matrix must have shape (D, L)`; `Projection matrix dtype '...' is not a real numeric type`; `Projection matrix must contain only finite values.`

**Cause:** A supplied matrix that is not 2-D, not `(output_dim, fp_size)`, not real numeric, or not finite.

**Fix:** Pass a real, finite array with shape `(output_dim, fp_size)`.

**Example:** `EmbeddedFingerprintGenerator("ecfp", 1024, 16, method="random_projection", method_params={"matrix": np.zeros(64)})` raises the 2-D error.

</details>

(generator-getfingerprintfromsmiles)=
## `GetFingerprintFromSmiles`

```python
GetFingerprintFromSmiles(
    smiles: str,
    validate: bool = True,
    canonicalize: bool = True,
) -> npt.NDArray[Any] | None
```

| Parameter | Meaning |
|---|---|
| `smiles` | SMILES string |
| `validate` | Accepted for compatibility; parsing always validates |
| `canonicalize` | Canonicalize the string before keying and parsing; default `True` |

Returns one `(D,)` embedding, or `None` for an unparseable string. With `canonicalize=True`, the method canonicalizes the string, caches on the canonical form, and parses that form for the embedding, so differently spelled inputs for the same molecule embed identically. With `canonicalize=False`, the raw string is the cache key and the method embeds the molecule parsed from it.

### Errors

No error entry is attached to this method. Invalid SMILES returns `None`, and computation errors propagate. A non-string argument raises [the SMILES type entry](concepts.md#canonicalize_smiles).

(generator-getfingerprintfromselfies)=
## `GetFingerprintFromSelfies`

```python
GetFingerprintFromSelfies(
    selfies_str: str,
    validate: bool = True,
    canonicalize: bool = True,
) -> npt.NDArray[Any] | None
```

| Parameter | Meaning |
|---|---|
| `selfies_str` | SELFIES string |
| `validate` | Accepted for compatibility; parsing always validates |
| `canonicalize` | Canonicalize the decoded SMILES before keying and parsing; default `True` |

Returns one `(D,)` embedding, or `None` for a string the SELFIES decoder rejects. The method decodes the SELFIES to SMILES and delegates to `GetFingerprintFromSmiles`. A decoder failure returns `None`; any other exception propagates.

### Errors

No error entry is attached to this method. Undecodable SELFIES returns `None`.

(generator-getfingerprintasnumpy)=
## `GetFingerprintAsNumPy`

```python
GetFingerprintAsNumPy(mol: Chem.Mol) -> npt.NDArray[Any]
```

| Parameter | Meaning |
|---|---|
| `mol` | RDKit `Mol` object |

Returns the compressed, padded `(D,)` embedding for one molecule. The method computes the raw fingerprint, applies the dtype check, and then either returns the raw array when compression is inactive or compresses with the resolved method.

### Errors

The [`mol must not be None`](#generator-mol-none) and [`uint16` cannot store counts or non-binary values](#generator-uint16-count) entries apply to this method.

(generator-getrawfingerprintasnumpy)=
## `GetRawFingerprintAsNumPy`

```python
GetRawFingerprintAsNumPy(mol: Chem.Mol) -> npt.NDArray[Any]
```

| Parameter | Meaning |
|---|---|
| `mol` | RDKit `Mol` object |

Returns the uncompressed `(fp_size,)` fingerprint, padded to the requested width and count-transformed when count mode is active. Count mode returns counts or their `log1p` form in the requested float dtype. `float32` is exact for integer counts below `2**24`.

### Errors

The [`mol must not be None`](#generator-mol-none) and [`uint16` cannot store counts or non-binary values](#generator-uint16-count) entries apply to this method. A count-simulated generator with `dtype="uint16"` reaches the raw form of the binary check.

(generator-getfingerprintsfromsmiles)=
## `GetFingerprintsFromSmiles`

```python
GetFingerprintsFromSmiles(
    smiles_list: Sequence[str],
    validate: bool = True,
    canonicalize: bool = True,
) -> tuple[npt.NDArray[Any], list[int]]
```

Returns a `(N, D)` array of the valid embeddings and a list of the indices whose SMILES did not parse. An all-invalid batch returns an empty `(0, D)` array. Invalid rows are omitted rather than filled.

### Errors

No error entry is attached to this method. A computation error propagates instead of turning into an invalid index.

(generator-getfingerprintsfromselfies)=
## `GetFingerprintsFromSelfies`

```python
GetFingerprintsFromSelfies(
    selfies_list: Sequence[str],
    validate: bool = True,
    canonicalize: bool = True,
) -> tuple[npt.NDArray[Any], list[int]]
```

Returns a `(N, D)` array of the valid embeddings and a list of the indices whose SELFIES did not decode. An all-invalid batch returns an empty `(0, D)` array.

### Errors

No error entry is attached to this method. A computation error propagates instead of turning into an invalid index.

(generator-getfingerprintsasnumpy)=
## `GetFingerprintsAsNumPy`

```python
GetFingerprintsAsNumPy(
    mols: Sequence[Chem.Mol | None],
    chunk_size: int | None = None,
) -> tuple[npt.NDArray[Any], list[int]]
```

| Parameter | Meaning |
|---|---|
| `mols` | Sequence of RDKit `Mol` objects or `None` |
| `chunk_size` | Deprecated; has no effect and emits a `DeprecationWarning` |

Returns a `(N, D)` array of the valid embeddings and a list of the `None` indices. An all-invalid batch returns an empty `(0, D)` array. The method validates `chunk_size` before it warns and then delegates each molecule to `GetFingerprintAsNumPy`.

### Errors

The `chunk_size` range check uses the [argument value entry](#generator-argument-value). The [`mol must not be None`](#generator-mol-none) entry covers the single-molecule method; this batch method reports a `None` row as an invalid index instead.

(generator-iterfingerprints)=
## `IterFingerprints`

```python
IterFingerprints(
    mols: Iterable[Chem.Mol | None],
) -> Iterator[tuple[int, npt.NDArray[Any] | None]]
```

Yields one `(index, embedding_or_none)` pair per molecule and reads one input at a time. A `None` entry yields `None` at its index.

### Errors

No error entry is attached to this method. A computation error propagates from `GetFingerprintAsNumPy`.

(generator-iterfingerprintsfromsmiles)=
## `IterFingerprintsFromSmiles`

```python
IterFingerprintsFromSmiles(
    smiles_list: Iterable[str],
    validate: bool = True,
    canonicalize: bool = True,
) -> Iterator[tuple[int, npt.NDArray[Any] | None]]
```

Yields one `(index, embedding_or_none)` pair per SMILES without materializing the input. An unparseable string yields `None` at its index.

### Errors

No error entry is attached to this method.

(generator-iterfingerprintsfromselfies)=
## `IterFingerprintsFromSelfies`

```python
IterFingerprintsFromSelfies(
    selfies_list: Iterable[str],
    validate: bool = True,
    canonicalize: bool = True,
) -> Iterator[tuple[int, npt.NDArray[Any] | None]]
```

Yields one `(index, embedding_or_none)` pair per SELFIES without materializing the input. A string the decoder rejects yields `None` at its index.

### Errors

No error entry is attached to this method.
