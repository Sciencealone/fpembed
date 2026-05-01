# FPembed NiceGUI Optimization App

An interactive web application for optimizing molecular fingerprint embeddings using Optuna hyperparameter search and Random Forest models. Built with [NiceGUI](https://nicegui.io/), this app provides an event-driven UI with live progress updates, Altair-based visualizations, and JSON export.

## Features

- **Dataset selection** — Load RedDB, NFA, or QM9 molecular datasets with filtering and stratified subset sampling
- **Fingerprint configuration** — Choose from six FP types (ECFP, Atom Pair, Topological Torsion, RDKit, Layered, Pattern) with per-type parameter ranges and size bounds
- **Molecular descriptor toggle** — Optionally include 73 RDKit molecular descriptors in the optimization search space. Enabled by default: Optuna decides per-trial whether descriptors improve model performance. Disabling the checkbox excludes descriptors from all trials, using fingerprints only
- **Random Forest tuning** — Configure hyperparameter search bounds for n_estimators, max_depth, and min_samples_split
- **Optuna optimization** — Run background optimization with configurable trial count and time limits
- **Live progress** — Real-time trial counter, elapsed time, and intermediate results via timer-based polling
- **Results visualization** — Top-10 results table, R²/MAPE/MSE bar charts, and predicted-vs-actual scatter plots with molecule image tooltips
- **Export** — Download results as JSON with full reproducibility metadata

## Prerequisites

- **Conda environment**: `fpembed` (see `environment.yml` at the project root)

```bash
conda activate fpembed
```

## Installation

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

Or install the fpembed package with app extras:

```bash
pip install fpembed[app]
```

## Usage

**Warning**: the demo app uses a cache to speed up the calculations. Please provide at least 100 GB of free disk space before the evaluation. The obsolete cache file `examples/nicegui_app/cache.db` can be deleted manually afterward.

From the `examples/nicegui_app/` directory:

```bash
python app.py
```

The app starts on [http://localhost:8080](http://localhost:8080) using NiceGUI's built-in Uvicorn server. No external web server configuration is needed.

## Datasets

Datasets are stored in `examples/datasets/` at the project root. The `config.yaml` file references this directory via a relative path:

```yaml
paths:
  datasets_dir: "../datasets"
```

The three compressed CSV datasets (RedDB, NFA, QM9) live in `examples/datasets/`.
