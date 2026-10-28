# Recipes

Use these recipes to embed fingerprints for a scikit-learn model, combine count output with a projection, work with fixed-width fingerprints, fit PCA and whitening outside FPembed, and reuse one projection at several output widths.

## scikit-learn pipeline

```python
from sklearn.ensemble import RandomForestRegressor
from fpembed import EmbeddedFingerprintGenerator

gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=16, fp_params={"radius": 2}
)

X, invalid = gen.GetFingerprintsFromSmiles(train_smiles)
y = [label for i, label in enumerate(train_labels) if i not in set(invalid)]

model = RandomForestRegressor(n_estimators=300)
model.fit(X, y)
```

## Count with a projection

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp",
    fp_size=2048,
    compression=16,
    count=True,
    method="random_projection",
)
```

The generator also allows Hadamard, but `fp_size` and `fp_size // compression` must both be powers of two.

## Fixed-width PubChem

```python
gen = EmbeddedFingerprintGenerator(
    fp_type="pubchem",
    fp_size=1024,
    compression=16,
)

embedding = gen.GetFingerprintFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
print(gen.native_size)  # 881
print(gen.padding)      # tail_zeros
print(embedding.shape)  # (64,)
```

Laggner uses `fp_size=512` with native 307, and ERG uses `fp_size=512` with native 315.

(p-recipes-pca-and-whitening-fitted-outside-fpembed)=
## PCA and whitening fitted outside FPembed

Build the raw matrix with the batch API, then fit PCA on training data only:

```python
import numpy as np
from sklearn.decomposition import PCA
from fpembed import EmbeddedFingerprintGenerator, compress_fingerprint

raw_gen = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=2048, compression=None, fp_params={"radius": 2}
)
X_raw, invalid = raw_gen.GetFingerprintsFromSmiles(train_smiles)
# Filter any labels with `invalid` before training a model.

pca = PCA(n_components=128, random_state=0).fit(X_raw)

X = compress_fingerprint(
    X_raw - pca.mean_,
    size=16,
    method="random_projection",
    method_params={"matrix": pca.components_},
)
```

The generator applies `M @ fingerprint` without subtracting a mean, so the generator path with `pca.components_` performs a linear projection. For the full PCA transform, subtract `pca.mean_` and call `compress_fingerprint()`, as above.

For whitening, scale the components by the standard deviations of the fitted components and keep the same center:

```python
pca = PCA(n_components=128, whiten=True, random_state=0).fit(X_raw)
matrix = pca.components_ / np.sqrt(pca.explained_variance_)[:, None]

X = compress_fingerprint(
    X_raw - pca.mean_,
    size=16,
    method="random_projection",
    method_params={"matrix": matrix},
)
```

`PCA(whiten=True)` applies the scaling inside `transform`; `components_` itself is not scaled in current scikit-learn. FPembed validates the matrix it is given and applies it.

(p-recipes-reuse-one-projection-at-several-widths)=
## Reuse one projection at several widths

```python
from rdkit import Chem
from fpembed import EmbeddedFingerprintGenerator, truncate_embedding

mol = Chem.MolFromSmiles("CCO")

gen8 = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=1024, compression=8, method="hadamard"
)
gen16 = EmbeddedFingerprintGenerator(
    fp_type="ecfp", fp_size=1024, compression=16, method="hadamard"
)

source = gen8.GetFingerprintAsNumPy(mol)
prefix = truncate_embedding(source, 64, method="hadamard")
direct = gen16.GetFingerprintAsNumPy(mol)
```

`prefix` equals `direct` exactly for Hadamard. For random projection, use `method="random_projection"` and compare with `allclose`, because the scaled prefix matches within round-off. FPembed rejects block-wise methods because their block boundaries change when the output width changes.
