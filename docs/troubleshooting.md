# Troubleshooting and FAQ

Use this page when FPembed fails to import or when a dependency is missing. The error entries for the API live on the method pages: each message ends with a link that opens its entry. Check the environment table first, then the FAQ for questions about hashes, compression, output shapes, invalid input, and preprocessing.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'fpembed'` | Wrong interpreter | Select the interpreter where FPembed is installed, or install with `pip install -e .` |
| `ImportError` that names `rdkit` | RDKit is missing or older than the supported range | Install `rdkit>=2023.9.6` in the same environment as FPembed |
| `ImportError` that names `skfp` | scikit-fingerprints is missing or outside the supported range | Install `scikit-fingerprints>=2.1,<3` in the same environment |
| `ImportError` that names `selfies` | SELFIES is missing | Install `selfies>=2.2` to decode SELFIES input |
| An install error or a syntax error on Python 3.9 or 3.14 | The interpreter is outside the supported range | Use Python 3.10, 3.11, 3.12, or 3.13 |

A computation error that carries an FPembed message is documented on the page for the method that raised it. Start at the [API Reference](api.md) and open that method's Errors section.

## FAQ

**Why does changing `fp_size` not change `params_hash`?**
The hash is a cache key for the fingerprint type, parameters, method, method parameters, and count representation. The manifest records `fp_size`, the compression factor, `dtype`, preprocessing, and versions instead.

**Can I use count fingerprints with every compression method?**
The generator allows `uniform`, `hadamard`, and `random_projection` when compression is active. The standalone `compress_fingerprint()` accepts all six methods, but geometric reversibility is binary-only.

**Why is my compressed output shape `(64,)` for PubChem at C=16?**
PubChem's native width is 881 and the padded width is 1024. Compression uses the padded width, so `1024 / 16 = 64`. Eight of those output blocks are dead padding.

**Why do I get `None` for an invalid SMILES?**
Single-molecule SMILES and SELFIES methods return `None` for unparseable input. Batch methods omit those rows and return their indices in the second element. Computation errors propagate.

**Can FPembed fit PCA or whitening for me?**
No. Fit the linear map outside FPembed and pass the result. See [PCA and whitening fitted outside FPembed](recipes.md#pca-and-whitening-fitted-outside-fpembed).
