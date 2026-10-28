# Fingerprint Types

Use this page to choose a `fp_type`, check whether it supports count mode, and set its parameters. The catalog gives the native width and the default parameters of all fifteen types, and the sections after it state the count, Lingo, pharmacophore, and fixed-width rules.

## Choosing a fingerprint type

Start with `ecfp` for general-purpose property prediction and similarity. `atom_pair` and `topological_torsion` capture longer-range relationships. `rdkit`, `layered`, `pattern`, and `avalon` are path- or substructure-based. `secfp`, `mhfp`, and `map` are hashed circular variants. `lingo` works on SMILES substrings. `pharmacophore` uses pharmacophore points and is much slower than the topological types. `pubchem`, `laggner`, and `erg` are fixed-width legacy fingerprints.

## Catalog

| Type | `fp_type` | Count mode | Native width | Main parameters and defaults |
|---|---|:---:|---:|---|
| Extended Connectivity (Morgan) | `ecfp` | Yes | `fp_size` | `radius=2`, `include_chirality=False`, `use_pharmacophoric_invariants=False`, `use_bond_types=True`, `count_simulation=False` |
| Atom Pair | `atom_pair` | Yes | `fp_size` | `min_distance=1`, `max_distance=30`, `use_pharmacophoric_invariants=False`, `count_simulation=True` |
| Topological Torsion | `topological_torsion` | Yes | `fp_size` | `torsion_atom_count=4`, `count_simulation=True` |
| RDKit | `rdkit` | Yes | `fp_size` | `min_path=1`, `max_path=7`, `use_bond_order=True`, `num_bits_per_feature=2`, `linear_paths_only=False`, `use_pharmacophoric_invariants=False` |
| Layered | `layered` | No | `fp_size` | `min_path=1`, `max_path=7`, `linear_paths_only=False` |
| Pattern | `pattern` | No | `fp_size` | None |
| Avalon | `avalon` | Yes | `fp_size` | None |
| SECFP | `secfp` | No | `fp_size` | `radius=3`, `min_radius=1`, `isomeric_smiles=False`, `kekulize=True` |
| MHFP | `mhfp` | Yes | `fp_size` | `radius=3`, `min_radius=1`, `isomeric_smiles=False`, `sssr_rings=True` |
| MAP4 | `map` | Yes | `fp_size` | `radius=2`, `include_chirality=False`, `include_duplicated_shingles=False` |
| Lingo | `lingo` | Yes | `fp_size` | `substring_length=4` |
| Pharmacophore | `pharmacophore` | Yes | `fp_size` | `min_points=2`, `max_points=3`; folded 2D only |
| PubChem | `pubchem` | No | 881 | No type parameters; use padded `fp_size=1024` |
| Laggner | `laggner` | No | 307 | No type parameters; use padded `fp_size=512` |
| ERG | `erg` | No | Follows `max_path` (315 at 15) | `max_path=15`; fixed bit variant, so the native width follows `max_path`; use a padded `fp_size` at or above it |

Pass all parameter values through `fp_params`. Omitted and explicitly default values produce the same fingerprint configuration and manifest parameters. `fp_params` rejects `random_state` and `variant`.

## Count support

`secfp`, `layered`, `pattern`, `pubchem`, `laggner`, and `erg` reject `count=True` with a `ValueError` that names the supported count types and the requested type. The other nine types support count mode.

`ecfp`, `atom_pair`, and `topological_torsion` also accept `count_simulation`, a binary-mode option: combining it with `count=True` raises `ValueError`. The catalog lists the per-type defaults, `False` for `ecfp` and `True` for `atom_pair` and `topological_torsion`. Changing the option changes bit positions, so fingerprints and models built with the old setting are not interchangeable. `rdkit` does not accept it.

## Lingo

`lingo` produces a substring fingerprint over canonical SMILES. Its only parameter is `substring_length`, default 4. Lingo supports count mode, and the canonical-SMILES branch keeps different spellings of the same molecule identical.

## Folded pharmacophore

`pharmacophore` selects the folded 2D variant with `variant="folded"` and `use_3D=False`. Set the point range with `min_points` (default 2) and `max_points` (default 3). You cannot select `raw_bits`, a 3D mode, or a different variant. Pharmacophore supports count mode.

Pharmacophore is much slower than the topological types. See [Pharmacophore cost](performance.md#pharmacophore-cost) for numbers.

## Fixed-width types

PubChem and Laggner have a fixed native width. ERG has a native width that follows `max_path`: 315 at the default 15. `fp_size` is the padded width you request. The rules are:

- `fp_size` must be a power of two and at least the native width, including when it equals the native width. The exact native widths 881 and 307 are not powers of two and are rejected; ERG's configured width is treated the same way.
- Values are placed at their native indices and the tail is filled with zeros.
- The raw and compressed shapes follow the padded `fp_size`.
- A variable-width type accepts any positive `fp_size`, and its native width equals `fp_size`.

The compression behavior at C=16 follows from the padded width:

| Type | Native | Padded `fp_size` | Live output blocks | Dead blocks | Real positions in tail block |
|---|---:|---:|---:|---:|---:|
| `pubchem` | 881 | 1024 | 56 of 64 | 8 of 64 (12.5%) | 1 of 16 |
| `laggner` | 307 | 512 | 20 of 32 | 12 of 32 (37.5%) | 3 of 16 |
| `erg` | 315 | 512 | 20 of 32 | 12 of 32 (37.5%) | 11 of 16 |

A dead block is entirely padding and reduces to zero. The tail block is only partly populated. These costs follow from the fixed native width and the power-of-two padded width.
