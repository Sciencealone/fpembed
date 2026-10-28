# Installation

Install FPembed from source, from PyPI, or into a fresh Conda environment, then confirm the installed version. Follow this page after you check the requirements, and return to it when you rebuild an environment or pin a version.

## Requirements

FPembed requires Python 3.10 or newer, below 3.14. Runtime dependencies are NumPy 1.26.4 or newer below 3, RDKit 2023.9.6 or newer, scikit-fingerprints 2.1 or newer below 3, and SELFIES 2.2 or newer.

Count fingerprints, fixed-width fingerprint types, and precomputed embeddings require FPembed 0.3.0 or newer.

## Install from source

Clone the repository and install the package in editable mode.

```bash
git clone https://github.com/Sciencealone/fpembed.git
cd fpembed
python -m pip install -e .
```

## Install from PyPI

```bash
python -m pip install fpembed
```

Use the source install when the PyPI release is older than 0.3.0.

## Conda

Create the environment, activate it, and install the released package into it.

```bash
conda create -n fpembed python=3.12
conda activate fpembed
python -m pip install fpembed
```

## Verification

Print the version that the import reports.

```python
import fpembed

print(fpembed.__version__)
```

The distribution metadata and `fpembed.__version__` should agree. `pip show fpembed` prints the installed metadata.
