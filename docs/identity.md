(p-identity-identity-manifests-and-caching)=
# Identity, Manifests, and Caching

Use this page to identify an embedding by its parameters, record the full configuration in a manifest, and run the in-memory cache. You need it when you persist embeddings, compare two generator configurations, or trace a cache key back to its source.

## What the parameter hash covers

`fp_params_hash()` returns a 16-character SHA-256 prefix of a sorted JSON payload. The payload covers:

- `fp_type`
- the fingerprint parameters in `fp_params`
- `method`, unless the method is `None` or the default `geometric` with no method parameters
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

A generator passes its resolved method to the hash. An active count generator therefore also records `method: "uniform"` unless you choose another allowed method.

Supplying a parameter with its default value explicitly still adds the key to the payload. `fp_params={"max_path": 15}` for `erg` produces a different cache key from omitting the key.

The hash leaves out `fp_size`, the compression factor, the output dtype or encoding, preprocessing, and dependency versions. Changing any of those leaves the hash unchanged. The hash is a cache key within a fixed generator configuration; use the manifest for persisted identity.

`fp_type`, `method`, `method_params`, `count`, and `count_transform` are reserved keys inside `fp_params`. Supplying any of them raises `ValueError`. NumPy scalar values become Python scalars, and booleans become integers, so `True` and `1` hash identically.

## Matrix and center digests

Before hashing, FPembed replaces a caller-supplied projection matrix with `matrix_id` and `matrix_shape`. `matrix_id` is a 16-character digest of the array shape and canonical float64 bytes, so float32 and float64 copies of the same values share one identity. A different shape or a changed value gives a different digest.

A precomputed compressor's centering vector is replaced by `center_id`, computed the same way. The precomputed hash covers `source_id`, `input_dim`, `normalize`, `center_id`, `method`, and `method_params`. It leaves out the compression factor and the dtype.

(p-identity-manifest-schema-2)=
## Manifest schema 2

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

The example shows a binary PubChem fingerprint with the default `geometric` method and `C=16`.

Field meanings:

- `fingerprint.type`: the `fp_type`.
- `fingerprint.size`: the padded `fp_size`.
- `fingerprint.native_size`: the native width before padding.
- `fingerprint.padding`: `"none"` or `"tail_zeros"`.
- `fingerprint.effective_size`: the number of independent bit positions. Each count-simulated feature occupies four physical bits, so a simulated fingerprint reports a quarter of its native width. The hash slot count equals `effective_size`, while the physical array width remains `native_size`.
- `fingerprint.encoding`: `"binary"`, `"count"`, or `"external"`.
- `fingerprint.count_transform`: `"none"` or `"log1p"`.
- `fingerprint.params`: the resolved fingerprint parameters. Omitting a value and passing its default give the same mapping.
- `fingerprint.fixed_params`: backend values the factory pins that affect bit positions and cannot be set through `fp_params`. The mapping is read-only. `rdkit` records `count_simulation: false`, folded `pharmacophore` records `variant: "folded"` and `use_3D: false`, `erg` records `variant: "bit"`, and every other type records `{}`. Execution-only settings (`n_jobs`, `batch_size`) and values already recorded by `fingerprint.encoding` stay out. The keys are always disjoint from `fingerprint.params`.
- `compression.active`: whether compression runs.
- `compression.factor`: the factor, or `null` when inactive.
- `compression.method`: the method, or `null` when inactive.
- `compression.method_params`: the effective parameters, including `seed`, `sparse`, `interleave`, and matrix identity keys when present.
- `compression.output_dim`: the embedding width.
- `output.dim`: the returned array width.
- `output.dtype`: the returned dtype name.
- `output.encoding`: the physical encoding label.
- `preprocessing.canonicalization`: the built-in SMILES policy, or `null` for precomputed embeddings.
- `preprocessing.external`: a caller mapping, recorded verbatim and never run.
- `versions`: the installed versions of the packages that produced the values. A missing distribution can be `null`.

FPembed rebuilds the mapping on every call. Mutating the returned object does not change the generator.

A precomputed manifest uses `fingerprint.type: "precomputed"`, sets `size`, `native_size`, and `effective_size` to `input_dim`, uses `padding: "none"`, sets `fixed_params` to `{}`, and puts `source_id`, `normalize`, and `center_id` in `fingerprint.params`.

## Output encoding labels

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

A `uint16` geometric array and its `float64` counterpart are different formats. The recovery formula relates them, and only the integer form is exact.

(p-identity-what-to-store)=
## What to store

| Artifact | Store with the embeddings | Why |
|---|---|---|
| Generated embeddings | `representation_manifest()` | Records fingerprint, method, dtype, and versions |
| Caller-supplied matrix | The matrix plus the manifest | The manifest records `matrix_id` and shape, not values |
| Center vector | The center plus the manifest | The manifest records `center_id`, not values |
| External model embeddings | `source_id`, checkpoint identifier, manifest | Identifies the model that produced the input |
| Cache entries | `params_hash` | Stable cache key, not a full identity |

## Caching

Set `cache_size` to a positive integer to enable the in-memory cache. `None` disables it.

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

With canonicalization on, the cache keys on the canonical SMILES and maps that form to the embedding. The cache stores arrays read-only and returns writable copies, so mutating a result cannot poison a cache entry. With `canonicalize=False`, the cache keys on the raw SMILES string instead.

The cache belongs to one generator instance and lives in memory. For persistence, store `params_hash` alongside an external cache.

(identity-fp-params-hash)=
## `fp_params_hash`

```python
fp_params_hash(
    fp_type: str,
    fp_params: dict,
    *,
    method: str | None = None,
    method_params: dict | None = None,
    count: bool = False,
    count_transform: str = "none",
) -> str
```

| Parameter | Meaning |
|---|---|
| `fp_type` | Fingerprint type identifier |
| `fp_params` | Fingerprint parameters for the type |
| `method` | Resolved method name, or `None` |
| `method_params` | Effective method parameters, when present |
| `count` | Count mode flag |
| `count_transform` | `"none"` or `"log1p"`; requires `count=True` |

Returns a 16-character hexadecimal string. The hash covers the sorted JSON payload described in [What the parameter hash covers](#what-the-parameter-hash-covers), and it leaves out `fp_size`, the compression factor, the dtype, preprocessing, and dependency versions.

### Errors

(identity-fp-params)=
#### `fp_params` has a reserved key

**Message:** `[...] reserved key(s) cannot appear in fp_params`

**Cause:** One of the reserved keys `fp_type`, `method`, `method_params`, `count`, and `count_transform` inside `fp_params`.

**Fix:** Remove the key. Pass the setting as a function argument instead.

**Example:** `fp_params_hash("ecfp", {"method": "ecfp"})` raises the error. The generator constructor reports the same message when `fp_params` holds a reserved key; [the fp_params entry on the generator page](generator.md#embeddedfingerprintgenerator) covers the unknown-key and `variant` cases.

(identity-count-transform-value)=
#### The count transform is not `none` or `log1p`

**Message:** `count_transform must be one of ('none', 'log1p'), got ...`

**Cause:** An unknown transform name.

**Fix:** Pass `"none"` or `"log1p"`.

**Example:** `fp_params_hash("ecfp", {}, count_transform="sqrt")` raises the error.

(identity-count-transform-requires-count)=
#### A count transform needs count mode

**Message:** `count_transform requires count=True`

**Cause:** A transform such as `"log1p"` with `count=False`.

**Fix:** Set `count=True` or remove the transform.

**Example:** `fp_params_hash("ecfp", {}, count_transform="log1p")` raises the error.

(identity-argument-type)=
#### An argument has the wrong type

**Message:** `count must be a bool, got ...`

**Cause:** A `count` value that is not a boolean.

**Fix:** Pass `True` or `False`.

**Example:** `fp_params_hash("ecfp", {}, count=1)` raises the error.

(identity-representation-manifest)=
## `representation_manifest`

```python
representation_manifest(self, preprocessing: dict | None = None) -> dict
```

| Parameter | Meaning |
|---|---|
| `preprocessing` | Declarative caller metadata; recorded, never run |

Returns a fresh schema 2 mapping on every call. Mutating the returned object does not change the generator. [Manifest schema 2](#manifest-schema-2) lists the fields and their meanings.

### Errors

(identity-preprocessing-json)=
#### Preprocessing metadata must be JSON-serializable

**Message:** `preprocessing must be a JSON-serializable mapping`

**Cause:** The manifest preprocessing mapping holds a value that `json.dumps` cannot encode.

**Fix:** Convert the values to plain types such as `str`, `int`, `float`, `bool`, `None`, lists, and dicts.

**Example:** A manifest built with `preprocessing={"scaler": np.float64}` raises the error.

(identity-cache-info)=
## `cache_info`

```python
cache_info(self) -> CacheInfo
```

Returns a named tuple with `hits`, `misses`, `maxsize`, and `currsize`. A generator built without `cache_size` returns all zeros.

### Errors

No error entry is attached to this method.

(identity-clear-cache)=
## `clear_cache`

```python
clear_cache(self) -> None
```

Clears both cache levels. The method is a no-op when caching is disabled.

### Errors

No error entry is attached to this method.
