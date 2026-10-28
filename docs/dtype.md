(p-dtype-output-dtype-and-exactness)=
# Output Dtype and Exactness

Use this page when you choose a `dtype` and need to know how the stored value relates to the fingerprint value, when that value stays exact, and how to recover it. The tables give the byte size, the integer codes, and the recovery formulas.

## Accepted dtypes

The generator and `compress_fingerprint()` accept `float64`, `float32`, and `uint16`. The precomputed compressor accepts `float64` and `float32` only. Strings such as `"float32"` and NumPy types such as `np.float32` are accepted.

| `dtype` | Bytes per value | Semantics |
|---|---:|---|
| `float64` | 8 | Default. Geometric value is `n / (2**C - 1)`, up to accumulation round-off. |
| `float32` | 4 | Information-preserving for geometric at C <= 16. About 1e-7 relative error for projections. |
| `uint16` | 2 | Exact integer code for geometric or uniform, binary input only. |

`dtype` shrinks the returned and stored array. Peak transient memory stays the same, because the float paths compute in float64 and cast on return.

`dtype` changes the storage type and keeps the information. It stays out of `params_hash` and does not change cache keys.

With compression inactive, `uint16` stores the raw 0/1 fingerprint. Method restrictions apply only when compression is active.

## float32 exactness

For `geometric` at C <= 16, a float32 value differs from the float64 value by about 0.002 of one code step at C=16. The error is far below the 0.5 needed to misround, so you can recover the block integer exactly with:

```text
n = rint(value * (2**C - 1))
```

The dtype therefore preserves the information for geometric, even when the bits differ from the float64 form.

For count output, float32 is exact for integer counts below `2**24`. Above that bound, float32 cannot represent every integer, and FPembed makes no exactness claim.

## uint16 exact integer codes

`uint16` stores the integer code directly.

| Method | Stored value | Range | Recover the float64 value with |
|---|---|---|---|
| `geometric` | Block integer `sum(b_i * 2**i)`, first bit least significant | `0 .. 2**C - 1` | `value / (2**C - 1)` |
| `uniform` | Block popcount | `0 .. C` | `value / C` |

`interleave=True` changes which positions form a block; the recovery formula stays the same.

`uint16` raises `ValueError` in these cases:

- `linear`, `log`, `hadamard`, or `random_projection`, which have no exact integer form.
- `geometric` with C > 16, where the block integer exceeds the uint16 range.
- `uniform` with C > 65535, which is unreachable in practice but still checked.
- Non-binary input, because the integer identity requires each value to be 0 or 1.
- Count mode, at construction. Counts cannot be stored in uint16.
- Any dtype outside the accepted set, such as `int8`, `uint8`, or `float16`.

## Count output dtype

Count output accepts `float64` and `float32` and rejects `uint16`, because counts cannot be represented exactly.

Uncompressed count output has shape `(fp_size,)`. Compressed count output has shape `(fp_size // compression,)`. With no transform, the float64 values are exact integers. With `log1p`, they are exact `numpy.log1p` values.

## Reversibility is not distance preservation

Geometric recovers the original bits at supported precision, but the decoded values do not preserve Hamming geometry. With C=2, so `2**C - 1 = 3`:

- Codes 1 and 2 differ in both bits, and their decoded values differ by 1/3.
- Codes 0 and 2 differ in one bit, and their decoded values differ by 2/3.

Ranking pairs by Euclidean distance in the decoded space therefore differs from ranking by Hamming distance. Exact binary similarity stays a separate operation. Pooling methods also collide distinct blocks, so measure their suitability on your task.

Keep the sources of loss separate: fingerprint hashing collisions come from the fingerprint, block pooling from compression, float rounding from the dtype, and accuracy changes from the model. Only pooling and rounding are compression concerns.
