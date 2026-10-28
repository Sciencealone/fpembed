# Why Embedded Fingerprints

An embedding replaces a wide fingerprint with a shorter dense vector, which cuts feature columns, stored bytes, and the work in the first model layer. Read this page when you weigh a smaller `fp_size` against a compression factor, or when you need the memory and speed numbers behind either choice.

A fingerprint encodes a molecule as a vector of fixed length. A 2048-bit ECFP hashes each substructure into a position and leaves the rest at zero, so few of the 2048 values are nonzero. Training and inference pay for the full width: a tree scans a threshold per position, a dense layer holds a weight per position, and a distance adds a term per position.

An embedding is a shorter, dense vector that carries the fingerprint information in the form the model consumes. The compression factor `C` sets the ratio `D = L / C`. At `C=16`, a 2048-bit fingerprint becomes 128 values. A block-wise method makes each output summarize 16 positions: `geometric` turns that block into one integer scaled to `[0, 1]`, and `uniform` turns it into the fraction of set bits. A projection method mixes all 2048 positions into each of the 128 outputs. The model reads 128 numbers in place of 2048.

You can shrink the input at two stages. A smaller `fp_size` changes the fingerprint itself: the hashing step maps more substructures onto the same positions, so collisions rise and resolution falls before a model sees it. You get a narrower fingerprint without a separate compression step, and you accept more collisions. Compression keeps the fingerprint at full width and reduces the vector after generation, so the fingerprint keeps its resolution and you pay one extra step.

Compare the two on a validation split. A narrower fingerprint and a compressed one can reach the same width and differ in accuracy, because they lose different information. The better choice depends on your substructures and your model.

Some quantities follow from the configuration. Width is `D = L / C`, so `C=16` turns 2048 positions into 128 features, about 16 times fewer inputs. Stored bytes are `D * itemsize`. Accuracy, training time, memory savings, and sample efficiency depend on the model, the data, and the hardware: a gradient-boosted tree and a linear model read the same columns and respond in different ways.

FPembed's maps fall into two families. Block-wise methods reduce each block of `C` positions to one value. With the default `interleave=False`, output `i` depends on contiguous inputs `i*C` through `i*C + C - 1`, so nearby bits stay together and local patterns survive. Projection methods mix all `L` positions into every output, which can retain more information at high `C`. They cost more than a block-wise method, from `O(L log L)` for Hadamard to `O(L * D)` for random projection.

The maps are fixed and data-independent. A molecule gives the same embedding alone or in a batch of a million, during training or at inference. Seeded projections reproduce the map from the same seed and NumPy version, so a model trained on one run reads the same features in the next.

Principal component analysis (PCA) is a fitted alternative. It centers the data and finds the orthogonal directions that carry the most variance: directions at right angles, so each component carries variance the others do not repeat. It orders those directions by variance and keeps the leading ones, which carry most of the variance. When your data sits near a low-dimensional subspace, a fit can keep more structure than a fixed map because it chooses directions from your sample. If two fingerprint positions move together across the sample, PCA can fold them into one component and keep the same information in fewer columns. Dropping low-variance directions can remove noise from distance calculations, though it can also remove signal.

PCA depends on the fit sample. Fit it on training data, and the map follows that sample. PCA works on centered data, scaling changes the components, and a few outliers can rotate the leading directions. A large component count can overfit the training sample. If you refit on new data, the components can rotate, and a model trained on the old map reads different features.

FPembed does not fit PCA. Fit it outside on training data and pass the components as a `(D, L)` matrix through `method_params={"matrix": pca.components_}` with `method="random_projection"`. The generator applies `M @ x` without centering, so that path is a linear projection. For the full transform, subtract `pca.mean_` and call `compress_fingerprint()`. The [PCA recipe](recipes.md#pca-and-whitening-fitted-outside-fpembed) shows both calls.

Choose PCA for linear models and for distances on dense, correlated features, when you have enough rows for a stable fit and preprocessing you can reproduce at inference. Skip it when the dataset is small, when a tree model does the work, or when you need one portable map with no fitted state.

## Storage size

Output dimension is deterministic. `D = L / C` gives 128 features at `C=16`, and `bytes = D * itemsize` gives the stored size, with `itemsize` at 8 bytes for float64, 4 for float32, and 2 for uint16.

| Representation | Per molecule (L=2048) | Per molecule (L=4096) | 100K molecules (L=2048) |
|---|---:|---:|---:|
| Raw binary, float64 | 16 KB | 32 KB | about 1.6 GB |
| Raw binary, bit-packed | 256 B | 512 B | about 25 MB |
| Embedded, C=16, float64 | 1 KB | 2 KB | about 100 MB |
| Embedded, C=16, uint16 | 256 B | 512 B | about 25 MB |
| Embedded, C=32, float64 | 512 B | 1 KB | about 50 MB |

Read the first two rows together. Raw float64 costs 16 KB per molecule because the array holds 2048 values of 8 bytes, and most of them are zero. Bit-packing the same bits costs 256 B, 64 times less, because one byte carries eight positions.

A `geometric` embedding at C=16 in float64 costs 1 KB, which is larger than the bit-packed raw fingerprint. The embedding packs 16 bits into one 8-byte value: 128 values of 8 bytes carry the same 2048 bits that fit in 256 packed bytes. Storing the embedding as uint16 returns it to 256 B.

The feature count is the reliable gain. The model reads 128 inputs instead of 2048 at every step. Choose bit-packed raw when disk or network dominates and the model can consume sparse binary input. Choose the embedding when model input width dominates, and use uint16 when you want byte parity with bit-packed raw.

## Training and inference speed

- A tree model searches for a split by scanning candidate thresholds for each feature. With `D` features instead of `L`, that scan touches about `C` times fewer candidates. Feature and row subsampling, fixed costs per tree, and the histogram build for each node limit the gain, so measure the wall clock.
- A neural network's first dense layer holds a weight matrix of shape `(L, hidden)`. At `D` inputs the matrix becomes `(D, hidden)`, and its parameter count and multiply-adds fall by about `C`. The later layers keep their size, so the saving applies to one layer and its incoming activations.
- A distance method computes a distance by summing over dimensions. At `D` dimensions the inner loop runs about `C` times shorter. Building and maintaining an index, moving data through memory, and reranking candidates after the search keep their full-width cost.

You can bound the realized gain with a budget. If split search takes 40 percent of training time and you cut its work by 16 times, that 40 percent becomes 2.5 percent, so total time falls by 1.6 times. The remaining 60 percent sets the floor; measure your own split.

## Memory during training

A 100K by 2048 float64 feature matrix needs about 1.6 GB. The same 100K rows at 128 float32 features need about 51 MB. Two factors compound: 16 times fewer columns and half the bytes per value, so the example uses about 32 times less memory.

The training matrix scales with row count, width, and dtype. Tree models keep histograms and internal copies that follow the feature count, so the saving reaches those buffers. Smaller input tensors can allow a larger mini-batch on a GPU, and the gain compounds when device memory runs out before the dataset does.

`dtype` changes stored bytes and leaves peak transient memory where it was. The float paths compute in float64 and cast on return, so during compression the raw fingerprint and the float64 working buffers exist at full width. Plan peak memory from the input width and the batch size; plan disk from the output dtype.

## Compression overhead

Warm-cache block-wise compression added about 1 ms per 1000 molecules in our measurements, or about 1 second per million. Treat the figure as an order of magnitude; the cost varies with the method, the width, and the hardware.

Cold-cache projection calls build a sign vector or draw a projection matrix before the first row, so the first batch costs more than later ones. A caller-supplied matrix skips the draw and pays validation on the first call.

Generation cost varies by fingerprint type and can exceed compression. Pharmacophore costs more per molecule than the topological types, while compression of its output costs the same per row. You pay compression once per molecule, and the narrower matrix saves work at training and at every inference call. Time both steps on your hardware before you weigh them.

## Sample efficiency

Distance concentration describes what happens to pairwise distances as dimension grows. In a high-dimensional space the nearest and farthest of a set of random points land at similar distances, because each added dimension adds noise to every pair. A model that separates molecules by distance then needs more rows to find a boundary that holds.

A shorter dense vector can act as a regularizer. The model works from 128 summaries instead of 2048 positions. Each summary mixes a block of bits, or all positions for a projection, so a split on one column draws on many substructures at once. Distances between rows can spread out when the representation is shorter and smoother, which gives a distance-based model more to work with.

Both effects can reverse. A map that discards bits your labels depend on loses accuracy, and a block that sums unrelated substructures can hide the one that matters. With enough rows the wider input fits the same structure and the gain shrinks. Test this benefit with cross-validation; it is the least certain.

## Summary

| Metric | Raw fingerprint (L=2048) | Embedded (D=128) | Caveat |
|---|---|---|---|
| Feature matrix, 100K molecules | about 1.6 GB float64 | about 100 MB float64 | Exact from shape and dtype |
| Per molecule versus bit-packed raw | 256 B | 256 B uint16, 1 KB float64 | Byte parity |
| Tree split candidates | Baseline | About 16x fewer | Realized speedup is workload-dependent |
| First dense layer | `2048 x H` | `128 x H` | Exact parameter count |
| Pairwise distance work | `O(N^2 x 2048)` | `O(N^2 x 128)` | Index overhead remains |
| Small-dataset accuracy | Baseline | Can improve | Model-dependent |
| Large-dataset accuracy | Slightly higher ceiling | Comparable | Empirical |

Feature count, parameter count, and byte count follow from shape and dtype, so you can compute them before you run a benchmark. The speed and accuracy columns are directional: they depend on the model, the hardware, and the dataset.

Start from `ecfp` at 2048 bits with `compression=16` and `dtype="float64"`. That gives 128 features. Change one axis at a time: method, compression factor, dtype, fingerprint type. Measure accuracy, training time, and peak memory on a validation split after each change; the three move in different directions.

For a storage constraint, bit-pack the raw fingerprint or store the embedding as uint16, which reaches byte parity at C=16. For a linear model or similarity search, try `random_projection`: it mixes positions and keeps distances closer to the full-width ones. For a tree model, try `geometric`, which keeps block structure and the widest code range for binary input. Pass a fitted PCA matrix if you trained it on your data and need a learned map.
