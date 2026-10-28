# Quick Start

Generate a binary fingerprint, a count fingerprint, and a batch embedding from SMILES, then read the representation manifest. Start here after installation and use the examples as templates for your own pipeline.

## Binary fingerprint from SMILES

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

Start with `ecfp`, `fp_size=2048`, `compression=16`, the default method, and `dtype=float64`, which gives 128 features. From there, change one axis at a time.

## Count fingerprint

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

With `compression=16`, `count=True`, and no explicit method, the method resolves to `uniform`. Each output value is the mean `log1p` count over a block of 16 positions.

(p-quickstart-batch-input-and-label-alignment)=
## Batch input and label alignment

```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "not_a_molecule"]
embeddings, invalid = gen.GetFingerprintsFromSmiles(smiles_list)
print(embeddings.shape)  # (3, 128)
print(invalid)           # [3]

y = [label for i, label in enumerate(labels) if i not in set(invalid)]
```

`embeddings` is a 2-D float array. A batch call drops the invalid SMILES, so filter the labels with the same indices.

## Representation manifest

```python
import json

manifest = gen.representation_manifest()
print(json.dumps(manifest, indent=2))
```

The manifest is a fresh JSON-compatible mapping; the library does not store it.
