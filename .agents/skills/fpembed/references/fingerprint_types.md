# Fingerprint types and compression methods

FPembed supports 15 fingerprint types and six compression methods. The fingerprints come from the scikit-fingerprints backends; compression maps a long bit or count vector to a shorter dense embedding.

## The 15 fingerprint types

| Type | Notes |
|---|---|
| `ecfp` | circular Morgan-style fingerprint; `radius` and related parameters go in `fp_params` |
| `atom_pair` | atom-pair fingerprint |
| `topological_torsion` | topological torsion fingerprint |
| `rdkit` | RDKit fingerprint |
| `layered` | layered substructure fingerprint; binary only |
| `pattern` | substructure pattern fingerprint; binary only |
| `avalon` | Avalon fingerprint |
| `secfp` | SMILES extended connectivity fingerprint; binary only |
| `mhfp` | MinHash fingerprint |
| `map` | MAP4-style fingerprint |
| `lingo` | LINGO fingerprint |
| `pharmacophore` | pharmacophore fingerprint |
| `pubchem` | fixed-width PubChem fingerprint; binary only |
| `laggner` | fixed-width Laggner fingerprint; binary only |
| `erg` | extended reduced graph fingerprint; binary only |

## Count mode

Nine types support `--count`:

`ecfp`, `atom_pair`, `topological_torsion`, `rdkit`, `avalon`, `mhfp`, `map`, `lingo`, `pharmacophore`.

Six do not:

`layered`, `pattern`, `secfp`, `pubchem`, `laggner`, `erg`.

Requesting count mode for a type outside the nine fails with a library error that lists the supported set. Count fingerprints cannot use `uint16`, and the block-wise `geometric`, `linear`, and `log` methods have no count meaning. Count compression therefore uses `uniform`, `hadamard`, or `random_projection`. `--count-transform log1p` applies a `log1p` transform to the counts.

## Fixed-width types

Three types have a backend-native width. The caller's `--fp-size` is the padded width and must be a power of two at or above the native size. The exact native size is rejected because it is not a power of two.

| Type | Native width | Smallest accepted `fp_size` |
|---|---|---|
| `pubchem` | 881 | 1024 |
| `laggner` | 307 | 512 |
| `erg` | 315 | 512 |

Every other type is variable-width, so `--fp-size` is the requested width and needs no padding. When compression is active, `--fp-size` must still be divisible by the compression factor.

## The six compression methods

| Method | Kind | Notes |
|---|---|---|
| `geometric` | block-wise | default for binary fingerprints; reversible within its compression cap; `uint16` allowed |
| `linear` | block-wise | linear block reduction |
| `log` | block-wise | logarithmic block reduction |
| `uniform` | block-wise | block mean; default for count mode; `uint16` allowed |
| `hadamard` | global projection | deterministic, prefix-consistent, supports `--truncate-to` |
| `random_projection` | global projection | accepts `--seed` and supports `--truncate-to` |

Block-wise methods and Hadamard are deterministic. `random_projection` draws its matrix from a seed; pass `--seed` to make it reproducible. Most readthedocs pages describe the FPembed library in more detail; the agent landing page is at https://fpembed.readthedocs.io/en/latest/agent/?utm_source=agent&utm_medium=skill&utm_campaign=featurize.
