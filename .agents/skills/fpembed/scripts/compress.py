"""Compress external embeddings or raw fingerprints with FPembed.

External matrices go through ``PrecomputedEmbeddingCompressor``, raw
fingerprint arrays through ``compress_fingerprint``, and ``--truncate-to``
applies ``truncate_embedding`` for the prefix-consistent projection methods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _common import emit, fail, load_matrix, projection_seed, run_doctor

_PROJECTION_METHODS = ("hadamard", "random_projection")


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compress external embeddings or raw fingerprints with FPembed.",
    )
    parser.add_argument("--input", help=".npy or .csv matrix (or a single vector)")
    parser.add_argument(
        "--mode", choices=("external", "fingerprint"), default="external",
        help="external embeddings or raw fingerprint arrays",
    )
    parser.add_argument("--compression", type=int, help="compression factor")
    parser.add_argument(
        "--method", default=None,
        help="external: random_projection/hadamard; fingerprint: any of the six methods",
    )
    parser.add_argument(
        "--dtype", choices=("float64", "float32", "uint16"), default="float64",
        help="output dtype (uint16 is fingerprint-only)",
    )
    parser.add_argument("--source-id", help="model or checkpoint name (external only)")
    parser.add_argument(
        "--normalize", choices=("none", "l2"), default="none",
        help="row normalization (external only)",
    )
    parser.add_argument("--center", help="path to a .npy center vector (external only)")
    parser.add_argument(
        "--truncate-to", type=int, default=None,
        help="keep this many leading dimensions via truncate_embedding",
    )
    parser.add_argument("--output", default="compressed.npy", help="output path")
    parser.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    parser.add_argument("--seed", type=int, default=None, help="random_projection seed")
    return parser


def _resolve_method(args: argparse.Namespace) -> str:
    """Return the explicit method or the mode-specific default."""
    if args.method is not None:
        return args.method
    return "geometric" if args.mode == "fingerprint" else "random_projection"


def _load_center(path: str | None) -> Any:
    """Load the optional center vector from a .npy file."""
    if path is None:
        return None
    import numpy as np

    return np.load(path)


def _compress_external(array: Any, args: argparse.Namespace, method: str) -> Any:
    """Compress an external embedding matrix or vector."""
    from fpembed import PrecomputedEmbeddingCompressor

    compressor = PrecomputedEmbeddingCompressor(
        array.shape[-1],
        args.compression,
        source_id=args.source_id,
        method=method,
        method_params=projection_seed(method, args.seed),
        dtype=args.dtype,
        normalize=args.normalize,
        center=_load_center(args.center),
    )
    return compressor.compress(array)


def _compress_fingerprint(array: Any, args: argparse.Namespace, method: str) -> Any:
    """Compress a raw fingerprint array with any supported method."""
    from fpembed import compress_fingerprint

    return compress_fingerprint(
        array,
        args.compression,
        method=method,
        method_params=projection_seed(method, args.seed),
        dtype=args.dtype,
    )


def _n_input(array: Any) -> int:
    """Return the number of input rows, counting a single vector as one."""
    return 1 if array.ndim == 1 else int(array.shape[0])


def _save(path: str | Path, array: Any) -> None:
    """Write the result to a .npy file, creating parent directories."""
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)


def _run(args: argparse.Namespace, method: str) -> int:
    """Execute one compression run and emit the JSON envelope."""
    import fpembed

    if args.truncate_to is not None and method not in _PROJECTION_METHODS:
        return fail(
            f"--truncate-to requires method hadamard or random_projection, got {method!r}",
            "compress",
        )

    array = load_matrix(args.input)
    if args.mode == "external":
        if not args.source_id:
            return fail("--source-id is required for external mode", "compress")
        result = _compress_external(array, args, method)
    else:
        result = _compress_fingerprint(array, args, method)

    if args.truncate_to is not None:
        from fpembed import truncate_embedding

        result = truncate_embedding(result, args.truncate_to, method=method)

    _save(args.output, result)

    payload: dict[str, Any] = {
        "status": "ok",
        "command": "compress",
        "mode": args.mode,
        "n_input": _n_input(array),
        "input_dim": int(array.shape[-1]),
        "output_dim": int(result.shape[-1]),
        "compression": args.compression,
        "method": method,
        "dtype": args.dtype,
        "output": str(args.output),
        "fpembed_version": fpembed.__version__,
    }
    if args.truncate_to is not None:
        payload["truncated_to"] = args.truncate_to
    if args.mode == "external":
        payload["source_id"] = args.source_id
    emit(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the compress CLI and return the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        doctor = run_doctor()
        installed = bool(doctor["fpembed"]["installed"])
        emit(
            {
                "status": "ok" if installed else "error",
                "command": "compress",
                **doctor,
            }
        )
        return 0 if installed else 1

    if args.input is None:
        return fail("--input is required unless --doctor is given", "compress")
    if args.compression is None:
        return fail("--compression is required unless --doctor is given", "compress")
    if args.mode == "external" and not args.source_id:
        return fail("--source-id is required for external mode", "compress")

    try:
        return _run(args, _resolve_method(args))
    except Exception as exc:  # noqa: BLE001 - CLI boundary preserves the library message
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
