"""Batch and streaming featurization with an invalid-row audit, optional
deduplication, raw fingerprint export, and a representation manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from _common import (
    embed_structures,
    emit,
    fail,
    iter_structures,
    load_structures,
    projection_seed,
    resolve_input_format,
    run_doctor,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Turn SMILES or SELFIES into FPembed fingerprint embeddings.",
    )
    parser.add_argument("--input", help="CSV, .smi, .sdf, or .mol input path")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES or SELFIES column name")
    parser.add_argument("--id-col", help="optional ID column name in a CSV")
    parser.add_argument("--id-prop", default="_Name", help="SDF property holding IDs")
    parser.add_argument(
        "--input-format", choices=("smiles", "selfies", "mol"), default="smiles",
        help="input entity format; .sdf and .mol suffixes imply mol",
    )
    parser.add_argument("--fp-type", default="ecfp", help="fingerprint type")
    parser.add_argument("--fp-size", type=int, default=2048, help="fingerprint length")
    parser.add_argument("--compression", type=int, default=16, help="compression factor")
    parser.add_argument("--count", action="store_true", help="count mode")
    parser.add_argument(
        "--count-transform", choices=("none", "log1p"), default="none",
        help="count transform",
    )
    parser.add_argument(
        "--method", default=None, help="compression method (library default if omitted)"
    )
    parser.add_argument(
        "--dtype", choices=("float64", "float32", "uint16"), default="float64",
        help="output dtype",
    )
    parser.add_argument("--cache-size", type=int, default=None, help="embedding cache size")
    parser.add_argument("--stream", action="store_true", help="iterate and write chunks")
    parser.add_argument("--raw", action="store_true", help="export the raw fingerprint")
    parser.add_argument("--dedupe", action="store_true", help="canonicalize and dedupe")
    parser.add_argument(
        "--manifest", action="store_true",
        help="include the representation manifest in the JSON",
    )
    parser.add_argument("--output", default="embeddings.npy", help="output path")
    parser.add_argument(
        "--output-format", choices=("npy", "csv"), default="npy", help="output format"
    )
    parser.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    parser.add_argument("--seed", type=int, default=None, help="random_projection seed")
    return parser


def _canonical(value: Any, input_format: str) -> str | None:
    """Return a canonical string for dedupe, or None when the entry is invalid."""
    if input_format == "mol":
        from rdkit import Chem

        return None if value is None else Chem.MolToSmiles(value)

    import fpembed

    return fpembed.canonicalize_smiles(value)


def _prepare(strings: list, dedupe: bool,
             input_format: str) -> tuple[list, list[int], list[int], int]:
    """Return structures to embed, their source indices, pre-known invalid rows, duplicates."""
    if not dedupe:
        return list(strings), list(range(len(strings))), [], 0

    seen: set[str] = set()
    kept: list = []
    kept_indices: list[int] = []
    invalid: list[int] = []
    duplicates = 0
    for index, value in enumerate(strings):
        canonical = _canonical(value, input_format)
        if canonical is None:
            invalid.append(index)
        elif canonical in seen:
            duplicates += 1
        else:
            seen.add(canonical)
            kept.append(value)
            kept_indices.append(index)
    return kept, kept_indices, invalid, duplicates


def _build_generator(args: argparse.Namespace) -> Any:
    """Construct the generator from the parsed flags."""
    from fpembed import EmbeddedFingerprintGenerator

    compression = None if args.raw else args.compression
    method_params = {} if args.raw else projection_seed(args.method, args.seed)
    return EmbeddedFingerprintGenerator(
        fp_type=args.fp_type, fp_size=args.fp_size, compression=compression,
        fp_params={}, cache_size=args.cache_size,
        method=None if args.raw else args.method, method_params=method_params,
        dtype=args.dtype, count=args.count, count_transform=args.count_transform,
    )


def _output_dim(args: argparse.Namespace) -> int:
    """Return the embedding width that the generator will produce."""
    if args.raw or args.compression in (None, 0):
        return args.fp_size
    return args.fp_size // args.compression


def _write_npy(path: str | Path, matrix: Any) -> None:
    """Write the matrix only to a .npy file."""
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, matrix)


def _csv_header(dim: int, has_ids: bool) -> list[str]:
    """Return the CSV header with the optional ID column first."""
    header = ["id"] if has_ids else []
    return header + [f"dim_{i}" for i in range(dim)]


def _write_csv(path: str | Path, matrix: Any, ids: list[str] | None,
               source_indices: list[int]) -> None:
    """Write the matrix and optional IDs to a CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_csv_header(matrix.shape[1], ids is not None))
        for row, source in enumerate(source_indices):
            values = [matrix[row, column].item() for column in range(matrix.shape[1])]
            writer.writerow(([ids[source]] if ids is not None else []) + values)


def _run_batch(gen: Any, prepared: list, source_indices: list[int],
               pre_invalid: list[int], input_format: str,
               ) -> tuple[Any, list[int], list[int]]:
    """Featurize a materialized input; return matrix, invalid and valid source indices."""
    matrix, invalid = embed_structures(gen, prepared, input_format)

    invalid_set = set(invalid)
    mapped_invalid = {source_indices[position] for position in invalid_set}
    invalid_indices = sorted(set(pre_invalid) | mapped_invalid)
    valid_sources = [
        source_indices[position]
        for position in range(len(prepared))
        if position not in invalid_set
    ]
    return matrix, invalid_indices, valid_sources


def _run_stream(gen: Any, prepared: list, source_indices: list[int],
                pre_invalid: list[int], ids: list[str] | None,
                args: argparse.Namespace, input_format: str) -> tuple[int, list[int]]:
    """Featurize lazily, writing CSV rows as they arrive; return count and invalid indices."""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter_structures(gen, prepared, input_format)

    invalid = set(pre_invalid)
    valid = 0
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_csv_header(_output_dim(args), ids is not None))
        for position, embedding in iterator:
            if embedding is None:
                invalid.add(source_indices[position])
                continue
            values = [embedding[column].item() for column in range(len(embedding))]
            source = source_indices[position]
            writer.writerow(([ids[source]] if ids is not None else []) + values)
            valid += 1
    return valid, sorted(invalid)


def _run(args: argparse.Namespace) -> int:
    """Execute one featurization run and emit the JSON envelope."""
    import fpembed

    input_format = resolve_input_format(args.input, args.input_format)
    structures, ids, _ = load_structures(
        args.input, input_format, smiles_col=args.smiles_col,
        id_col=args.id_col, id_prop=args.id_prop,
    )
    prepared, source_indices, pre_invalid, duplicates = _prepare(
        structures, args.dedupe, input_format
    )
    gen = _build_generator(args)

    if args.stream:
        n_valid, invalid_indices = _run_stream(
            gen, prepared, source_indices, pre_invalid, ids, args, input_format
        )
    else:
        matrix, invalid_indices, valid_sources = _run_batch(
            gen, prepared, source_indices, pre_invalid, input_format
        )
        n_valid = matrix.shape[0]
        if args.output_format == "npy":
            _write_npy(args.output, matrix)
        else:
            _write_csv(args.output, matrix, ids, valid_sources)

    result: dict[str, Any] = {
        "status": "ok",
        "command": "featurize",
        "n_input": len(structures),
        "n_valid": n_valid,
        "n_invalid": len(invalid_indices),
        "invalid_indices": invalid_indices,
        "output": str(args.output),
        "output_format": args.output_format,
        "input_format": input_format,
        "fp_type": args.fp_type,
        "fp_size": args.fp_size,
        "compression": None if args.raw else args.compression,
        "count": args.count,
        "dtype": args.dtype,
        "method": gen.method,
        "raw": args.raw,
        "streamed": args.stream,
        "fpembed_version": fpembed.__version__,
    }
    if ids is not None:
        result["invalid_ids"] = [ids[index] for index in invalid_indices]
    if duplicates:
        result["n_duplicates"] = duplicates
    if args.manifest:
        result["manifest"] = gen.representation_manifest()
    emit(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the featurize CLI and return the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        doctor = run_doctor()
        installed = bool(doctor["fpembed"]["installed"])
        emit(
            {
                "status": "ok" if installed else "error",
                "command": "featurize",
                **doctor,
            }
        )
        return 0 if installed else 1

    if args.input is None:
        return fail("--input is required unless --doctor is given", "featurize")
    if args.stream and args.output_format == "npy":
        return fail(
            "--stream requires --output-format csv; .npy needs the full matrix",
            "featurize",
        )

    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary preserves the library message
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
