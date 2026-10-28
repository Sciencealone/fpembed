# Compatibility

Use this page to check whether your environment matches the versions FPembed supports and to understand how the public API, manifests, and hashes behave across releases. The glossary defines the terms the rest of the manual uses.

## Python and dependencies

- Python 3.10 to 3.13.
- NumPy 1.26.4 or newer, below 3.
- RDKit 2023.9.6 or newer.
- scikit-fingerprints 2.1 or newer, below 3.
- SELFIES 2.2 or newer.

## API stability

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

Optional arguments do not change the output, hash, or manifest meaning of configurations that leave them unset. `fpembed.__version__` is the installed version string.

## Manifest and hash compatibility

The manifest schema version moves independently of the package version. The current schema is 2. Manifest layout may change between FPembed versions. Re-derive a manifest instead of comparing serialized bytes across releases.

Hashes for a given configuration are stable across releases. The count and transform keys appear only when non-default. Treat `params_hash` as one part of a persisted array's identity, and store the manifest, the package versions, and the input provenance.

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
