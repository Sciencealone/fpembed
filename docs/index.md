# FPembed documentation

FPembed compresses molecular fingerprints into short dense vectors for machine learning. A fingerprint is wide and mostly zeros, and a model pays for every column, so the map shortens what the model reads while the fingerprint definition stays explicit and the configuration is recorded for later reproduction. Nothing is fitted or trained. Input is SMILES, SELFIES, or RDKit Mol, or embeddings from another model, and the output is a plain NumPy array. The table below names the three entry points and the input each one accepts; the toctrees below it lead to the rest of the manual.

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
agent
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
