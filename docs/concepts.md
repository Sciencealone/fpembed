# Core Concepts

A fingerprint turns a molecule into a numeric vector, and an embedding shortens that vector to a chosen output dimension. The terms below recur through the manual: fingerprint length, native width, compression factor, output dimension, and identity. Read them before you choose `fp_size` or `compression`.

## Molecular fingerprints

A molecular fingerprint maps a molecule to a numeric vector. A binary fingerprint holds 0 and 1. A count fingerprint holds non-negative occurrence counts.

Set `count=True` to select count mode. Count mode is a representation setting on the generator, and `fp_type` stays the same. Counts are non-negative integers, and FPembed returns `float64` or `float32`. Binary and count configurations have separate identities, so they produce different parameter hashes and manifest encodings.

## Fingerprint length and native width

- `L` is the fingerprint length, named `fp_size` in the API.
- A variable-width fingerprint type produces exactly `L` values.
- A fixed-width fingerprint type has its own native width: PubChem 881, Laggner 307, and ERG 315. For those types, `fp_size` is the padded width. It must be a power of two and at least the native width, and the generator writes zeros beyond the native width.

## Compression factor and output dimension

- `C` is the compression factor, named `compression`.
- `D` is the output dimension. It is `L / C` when compression is active and `L` when it is not.

The generator and `PrecomputedEmbeddingCompressor` require a power-of-two factor that divides the input width and is at most half of it. `compress_fingerprint()` only requires the factor to divide the feature length. With `compression=None` or `0`, no compression runs.

A block-wise method splits the fingerprint into blocks of `C` positions and reduces each block to one value. A projection method combines all positions. [Compression Methods](compression.md#compression-methods) describes each method.

A smaller `fp_size` narrows the fingerprint itself. Hashed fingerprints then have more collisions, because more substructures map to the same positions. Compression leaves the fingerprint at its original width and reduces the output afterward. The two choices differ, so compare them on your validation split.

## Input handling and canonicalization

The generator accepts SMILES, SELFIES, and RDKit `Mol` objects. By default, FPembed canonicalizes SMILES input before keying and parsing. The canonical form makes the cache spelling-invariant and makes order-sensitive fingerprint types return the same embedding for different spellings of the same molecule.

Canonicalization normalizes SMILES spelling, atom ordering, and RDKit's aromatic form. It stops short of chemical standardization: FPembed leaves salts, tautomers, protonation state, and stereochemistry as written. Those choices change scientific meaning and belong to the caller. If a workflow needs standardization, apply it before calling FPembed and describe it in `representation_manifest(preprocessing=...)`.

The `canonicalize` flag selects the canonical branch. FPembed accepts `validate` for compatibility, but parsing always validates, so the flag does not change which branch runs.

(concepts-parse-smiles)=
## `parse_smiles`

```python
parse_smiles(smiles: str) -> Chem.Mol | None
```

| Parameter | Meaning |
|---|---|
| `smiles` | SMILES string |

Returns an RDKit `Mol`, or `None` for a string RDKit cannot parse.

### Errors

(concepts-parse-smiles-smiles-type)=
#### A SMILES argument is not a string

**Message:** `smiles must be a str, got ...`

**Cause:** A non-string value passed to `parse_smiles`.

**Fix:** Pass a Python string.

**Example:** `parse_smiles(123)` raises the error.

(concepts-canonicalize-smiles)=
## `canonicalize_smiles`

```python
canonicalize_smiles(smiles: str) -> str | None
```

| Parameter | Meaning |
|---|---|
| `smiles` | SMILES string |

Returns the canonical SMILES form, or `None` for an unparseable string.

### Errors

(concepts-canonicalize-smiles-smiles-type)=
#### A SMILES argument is not a string

**Message:** `smiles must be a str, got ...`

**Cause:** A non-string value passed to `canonicalize_smiles`.

**Fix:** Pass a Python string.

**Example:** `canonicalize_smiles(123)` raises the error.

(concepts-canonicalize-to-mol)=
## `canonicalize_to_mol`

```python
canonicalize_to_mol(smiles: str) -> tuple[str, Chem.Mol] | tuple[None, None]
```

| Parameter | Meaning |
|---|---|
| `smiles` | SMILES string |

Parses once and returns `(canonical_smiles, mol)`, or `(None, None)` for an unparseable string.

### Errors

(concepts-canonicalize-to-mol-smiles-type)=
#### A SMILES argument is not a string

**Message:** `smiles must be a str, got ...`

**Cause:** A non-string value passed to `canonicalize_to_mol`.

**Fix:** Pass a Python string.

**Example:** `canonicalize_to_mol(123)` raises the error.

## Identity and reproducibility

`params_hash` is a 16-character cache key. It covers the fingerprint type, fingerprint parameters, method, method parameters, and the non-default count representation. It excludes `fp_size`, the compression factor, the output dtype, preprocessing, and dependency versions.

`representation_manifest()` returns schema 2, which records those excluded fields. Persist the manifest alongside an embedding matrix or a deployed model; the hash is only a cache key.

Seed-based projection methods are deterministic given the same seed and NumPy version. The default seed is 42.
