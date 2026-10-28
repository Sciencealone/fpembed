# FPembed documentation

FPembed turns molecular fingerprints into dense vectors. It generates fingerprints, compresses them, validates input, tracks identity, and caches results, then returns plain NumPy arrays. Use it when you need a standard fingerprint in a smaller, model-ready form and the fingerprint definition has to stay explicit. The table below names the three entry points and the input each one accepts; the toctrees below it lead to the rest of the manual.

| Entry point | Input | Use when |
|---|---|---|
| `EmbeddedFingerprintGenerator` | SMILES, SELFIES, RDKit Mol | You start from molecules |
| `PrecomputedEmbeddingCompressor` | 1-D or 2-D external embeddings | You already have embeddings from another model |
| `compress_fingerprint()` | 1-D or 2-D fingerprint array | You already have fingerprints and want only compression |

```{toctree}
:maxdepth: 2
:caption: Start here

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 2
:caption: Concepts

why-embeddings
```

```{toctree}
:maxdepth: 2
:caption: Reference

generator
fingerprints
compression
dtype
identity
precomputed
recipes
performance
api
```

```{toctree}
:maxdepth: 2
:caption: Help

troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Project

compatibility
about
```

```{toctree}
:maxdepth: 2
:caption: Support

commercial
privacy
```
