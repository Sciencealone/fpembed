"""Documentation URLs for FPembed error messages.

The module performs no I/O. It builds a string from a fixed site root and the
page, anchor, and key stored in ``ERROR_TARGETS``. The package appends the
result to a documented validation error, so a reader can open the matching
entry on the method page.
"""

_DOCS_BASE = "https://fpembed.readthedocs.io/en/latest/"
_UTM = "utm_source=fpembed&utm_medium=exception&utm_campaign=errors"

ERROR_TARGETS = {
    "generator-fp-size-positive": (
        "generator", "generator-fp-size-positive", "`fp_size` must be positive",
    ),
    "generator-fp-size-native": (
        "generator", "generator-fp-size-native",
        "`fp_size` is below the native width",
    ),
    "generator-fp-size-power-of-two": (
        "generator", "generator-fp-size-power-of-two",
        "A fixed-width type needs a power-of-two `fp_size`",
    ),
    "generator-unsupported-fp-type": (
        "generator", "generator-unsupported-fp-type", "Unsupported fingerprint type",
    ),
    "generator-count-mode-not-supported": (
        "generator", "generator-count-mode-not-supported",
        "Count mode is not supported for this type",
    ),
    "generator-count-simulation-binary": (
        "generator", "generator-count-simulation-binary",
        "`count_simulation` is a binary-mode option",
    ),
    "generator-cache-size-positive": (
        "generator", "generator-cache-size-positive", "`cache_size` must be positive",
    ),
    "generator-hadamard-fp-size": (
        "generator", "generator-hadamard-fp-size",
        "Hadamard needs a power-of-two `fp_size`",
    ),
    "generator-compression-power-of-two": (
        "generator", "generator-compression-power-of-two",
        "The compression factor must be a power of two",
    ),
    "generator-compression-half": (
        "generator", "generator-compression-half",
        "The compression factor exceeds half the width",
    ),
    "generator-compression-divisible": (
        "generator", "generator-compression-divisible",
        "The width is not divisible by the compression factor",
    ),
    "generator-method": (
        "generator", "generator-method",
        "The method is unsupported or not allowed here",
    ),
    "generator-method-params": (
        "generator", "generator-method-params",
        "`method_params` has an unknown or invalid value",
    ),
    "generator-fp-params": (
        "generator", "generator-fp-params",
        "`fp_params` has an unknown, reserved, or fixed key",
    ),
    "generator-dtype": (
        "generator", "generator-dtype", "The dtype is unsupported or incompatible",
    ),
    "generator-uint16-count": (
        "generator", "generator-uint16-count",
        "`uint16` cannot store counts or non-binary values",
    ),
    "generator-uint16-no-integer": (
        "generator", "generator-uint16-no-integer",
        "This method has no exact `uint16` code",
    ),
    "generator-count-transform-requires-count": (
        "generator", "generator-count-transform-requires-count",
        "A count transform needs count mode",
    ),
    "generator-count-transform-value": (
        "generator", "generator-count-transform-value",
        "The count transform is not `none` or `log1p`",
    ),
    "generator-argument-type": (
        "generator", "generator-argument-type", "An argument has the wrong type",
    ),
    "generator-argument-value": (
        "generator", "generator-argument-value", "An argument is outside its range",
    ),
    "generator-mol-none": (
        "generator", "generator-mol-none", "The molecule object is `None`",
    ),
    "generator-matrix": (
        "generator", "generator-matrix",
        "The projection matrix has the wrong shape, dtype, or values",
    ),
    "compression-compression-input": (
        "compression", "compression-compression-input",
        "The compression input has the wrong shape, dtype, or values",
    ),
    "compression-compression-divisible": (
        "compression", "compression-compression-divisible",
        "The input length is not divisible by the compression factor",
    ),
    "compression-geometric-max": (
        "compression", "compression-geometric-max",
        "Geometric compression exceeds its limit",
    ),
    "compression-hadamard-dimensions": (
        "compression", "compression-hadamard-dimensions",
        "Hadamard dimensions must be powers of two",
    ),
    "compression-method": (
        "compression", "compression-method", "The method is unsupported",
    ),
    "compression-method-params": (
        "compression", "compression-method-params",
        "`method_params` has an unknown or invalid value",
    ),
    "compression-matrix": (
        "compression", "compression-matrix",
        "The projection matrix has the wrong shape, dtype, or values",
    ),
    "compression-uint16-binary": (
        "compression", "compression-uint16-binary", "`uint16` needs binary input",
    ),
    "compression-dtype": (
        "compression", "compression-dtype",
        "The dtype cannot hold the integer code",
    ),
    "compression-argument-value": (
        "compression", "compression-argument-value",
        "An argument is outside its range",
    ),
    "compression-truncation": (
        "compression", "compression-truncation",
        "The truncation target or method is invalid",
    ),
    "compression-argument-type": (
        "compression", "compression-argument-type", "An argument has the wrong type",
    ),
    "precomputed-argument-type": (
        "precomputed", "precomputed-argument-type", "An argument has the wrong type",
    ),
    "precomputed-argument-value": (
        "precomputed", "precomputed-argument-value",
        "An argument is outside its range",
    ),
    "precomputed-source-id-empty": (
        "precomputed", "precomputed-source-id-empty",
        "`source_id` must name the source model",
    ),
    "precomputed-compression-power-of-two": (
        "precomputed", "precomputed-compression-power-of-two",
        "The compression factor must be a power of two",
    ),
    "precomputed-compression-half": (
        "precomputed", "precomputed-compression-half",
        "The compression factor exceeds half the width",
    ),
    "precomputed-input-dim-divisible": (
        "precomputed", "precomputed-input-dim-divisible",
        "`input_dim` is not divisible by the compression factor",
    ),
    "precomputed-normalize": (
        "precomputed", "precomputed-normalize", "The normalization mode is invalid",
    ),
    "precomputed-dtype": (
        "precomputed", "precomputed-dtype",
        "The dtype is unsupported for precomputed embeddings",
    ),
    "precomputed-center-length": (
        "precomputed", "precomputed-center-length",
        "The center vector must match `input_dim`",
    ),
    "precomputed-method": (
        "precomputed", "precomputed-method",
        "The method is not allowed for precomputed compression",
    ),
    "precomputed-hadamard-dimensions": (
        "precomputed", "precomputed-hadamard-dimensions",
        "Hadamard dimensions must be powers of two",
    ),
    "precomputed-compression-input": (
        "precomputed", "precomputed-compression-input",
        "The compression input has the wrong shape or values",
    ),
    "precomputed-input-feature-axis": (
        "precomputed", "precomputed-input-feature-axis",
        "The input feature axis must match `input_dim`",
    ),
    "identity-fp-params": (
        "identity", "identity-fp-params", "`fp_params` has a reserved key",
    ),
    "identity-count-transform-value": (
        "identity", "identity-count-transform-value",
        "The count transform is not `none` or `log1p`",
    ),
    "identity-count-transform-requires-count": (
        "identity", "identity-count-transform-requires-count",
        "A count transform needs count mode",
    ),
    "identity-argument-type": (
        "identity", "identity-argument-type", "An argument has the wrong type",
    ),
    "identity-preprocessing-json": (
        "identity", "identity-preprocessing-json",
        "Preprocessing metadata must be JSON-serializable",
    ),
    "concepts-parse-smiles-smiles-type": (
        "concepts", "concepts-parse-smiles-smiles-type",
        "A SMILES argument is not a string",
    ),
    "concepts-canonicalize-smiles-smiles-type": (
        "concepts", "concepts-canonicalize-smiles-smiles-type",
        "A SMILES argument is not a string",
    ),
    "concepts-canonicalize-to-mol-smiles-type": (
        "concepts", "concepts-canonicalize-to-mol-smiles-type",
        "A SMILES argument is not a string",
    ),
}


def docs_url(key: str) -> str:
    """Return the documentation URL for one method-plus-error key."""
    page, anchor, _ = ERROR_TARGETS[key]
    return f"{_DOCS_BASE}{page}/?{_UTM}&utm_content={key}#{anchor}"


def with_docs(message: str, key: str) -> str:
    """Return the error message with its documentation link appended."""
    return f"{message} See {docs_url(key)}"
