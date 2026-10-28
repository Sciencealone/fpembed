(p-performance-performance-and-memory)=
# Performance and Memory

Use this page to estimate the time and memory a fingerprint configuration needs before you start a large batch. It gives the complexity of each compression method, measured peak memory, projection memory, the parallelism rules, and the pharmacophore cost.

## Complexity

| Method | Time | Retained memory |
|---|---|---|
| `geometric`, `linear`, `log`, `uniform` | `O(L)` per row | A weight vector of length C |
| `hadamard` | `O(L log L)` full transform | Sign vector of length L |
| `random_projection` | `O(L D)` dense | One `(L, D)` matrix per parameter set |

Block-wise methods remain the fastest overall.

## Peak memory

`dtype` leaves peak transient memory unchanged during compression. The float paths compute in float64 and cast on return. Batch methods stack all valid rows before returning, so the peak includes the output matrix plus the rows being collected. Iterators avoid the accumulated matrix, and each row is still computed and returned.

A 100K by 128 float64 matrix is about 100 MB. A 100K by 2048 float64 raw matrix is about 1.6 GB. Subtract the expected output size from available memory before choosing a batch API.

## Projection memory

The generator retains one `(L, D)` matrix per parameter set, about 2 MB at L=2048 and D=128, for its lifetime. FPembed may reuse matrices across generators up to a fixed memory budget; this is not a process-wide cap.

## Parallelism

Fingerprint generation runs single-threaded inside FPembed, and FPembed adds no thread pool. Parallelize across molecules or chunks at the application layer, and size the worker count to the available cores and memory.

(p-performance-pharmacophore-cost)=
## Pharmacophore cost

In our measurements, folded pharmacophore took about 1.781 ms per molecule on a 50-molecule sample at 2048 bits, against 0.038 ms for ECFP. At that rate, one million molecules need about 30 minutes of fingerprint generation before compression. Measure on your target data before making it a default.

## Limitations

- FPembed leaves molecules as written, including salts, tautomers, protonation states, and stereochemistry.
- You supply any projection matrix and center; FPembed validates them and applies them.
- FPembed trains and persists nothing. Save the arrays, manifest, matrix, and center yourself.
- Some fingerprint types reject count mode.
- Fixed-width types require padding, which leaves dead output dimensions.
- Compressed Euclidean distance differs from binary Tanimoto. Keep raw fingerprints for exact binary similarity.
