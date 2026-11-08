# FPembed agent skill

FPembed ships an agent skill that maps a small set of command-line workflows to the library's public API. The skill lives in the repository at `.agents/skills/fpembed/`, with `SKILL.md` for discovery, four scripts under `scripts/`, four documents under `references/`, and a demo fixture at `resources/demo.csv`. An agent loads `SKILL.md`, runs a script, and reads one JSON object from stdout.

## Dependency tiers

| Tier | Packages | Scripts |
|---|---|---|
| 1 | FPembed and NumPy | `featurize.py`, `compress.py`, `diversity.py` |
| 2 | tier 1 plus scikit-learn | `qsar_engine.py` |

Tier 1 covers featurization, compression, and chemical-space exploration. Tier 2 adds the scikit-learn estimators behind QSAR. Run the doctor first to report which tier the environment provides, the installed FPembed version, and whether that version meets the minimum of 0.3.0.

## Scripts

| Script | Purpose | Tier |
|---|---|---|
| `featurize.py` | Turn SMILES, SELFIES, or RDKit Mols into dense fingerprint embeddings, batch or streaming, with an invalid-row audit. | 1 |
| `compress.py` | Compress an external embedding matrix or a raw fingerprint array. | 1 |
| `qsar_engine.py` | Fit and predict regression or classification models with scikit-learn. | 2 |
| `diversity.py` | Pick max-min exemplars, score outliers, or query nearest neighbors. | 1 |

Each script prints one JSON object to stdout, sends diagnostics to stderr, and exits non-zero on failure. `--help` lists every flag, and `--doctor` runs the self-check.

## Input entities

`featurize.py` and `qsar_engine.py fit` accept three kinds of input:

- SMILES, from a CSV column named `smiles` by default, or from a `.smi` file.
- SELFIES, from a CSV column or a `.smi` file selected with `--input-format selfies`.
- RDKit Mol, from a `.sdf` or `.mol` file. The file suffix implies `--input-format mol`.

Use `--id-col` for a CSV identifier column and `--id-prop` for an SDF property, with `_Name` as the default. Invalid rows are reported by index and, when identifiers are known, by ID, so labels never shift silently. For SDF training, `qsar_engine.py fit` reads the target from the property named by `--target-prop`.

## Quick start

Run these commands from the skill folder. The demo fixture has `smiles`, `id`, `target`, and `activity` columns.

Check the environment:

```bash
python scripts/featurize.py --doctor
```

Featurize the fixture:

```bash
python scripts/featurize.py --input resources/demo.csv --output embeddings.npy
```

Fit a regression model on the `target` column:

```bash
python scripts/qsar_engine.py fit --smiles-csv resources/demo.csv --target-col target --model-out surrogate.joblib
```

Predict with the saved bundle:

```bash
python scripts/qsar_engine.py predict --model-in surrogate.joblib --smiles-csv resources/demo.csv --output-csv predictions.csv
```

The default settings are `ecfp`, `fp_size` 2048, and compression 16, which gives 128 features per molecule. The QSAR fit reports `r2`, `rmse`, and `mae` for regression, or `roc_auc`, `f1`, and `accuracy` for classification. The predict step rebuilds the embedding generator from the saved configuration, so the generator is never pickled. For the underlying library options see [Working with the Generator](generator.md), [Compression Methods](compression.md), and [Precomputed Embeddings](precomputed.md). The full command-line flag set is in `references/flags.md`.

## Auto-discovery

| Platform | Skills directory read automatically |
|---|---|
| OpenCode | `.opencode/skills/` and `.agents/skills/` |
| Claude Code | `.claude/skills/` |
| Cursor | `.agents/skills/` and `.cursor/skills/` |
| GitHub Copilot | `.github/skills/`, `.claude/skills/`, and `.agents/skills/` |
| Antigravity | `.agents/skills/` |
| Codex CLI | `.agents/skills/` |
| Gemini CLI | `.gemini/skills/` and `.agents/skills/` |

Install the skill with the skills CLI, which copies the published `fpembed` folder into a directory the platform scans:

```bash
npx skills add Sciencealone/fpembed --skill fpembed
```

Add `-a <agent>` to target one tool and `-g` for a global install. The command requires Node 22.20 or newer. Two alternatives install the same folder:

```bash
gh skill install Sciencealone/fpembed fpembed --agent <agent> --scope user --allow-hidden-dirs
gemini skills install https://github.com/Sciencealone/fpembed --scope workspace
```

`gh skill install` requires GitHub CLI 2.90 or newer. The skill folder must be published to GitHub before any installer can find it.

If you already have a repository clone, copy the folder manually as a fallback. OpenCode, Cursor, GitHub Copilot, Antigravity, Codex CLI, and Gemini CLI read `.agents/skills/` directly; Claude Code needs a copy under its own directory. Both commands below do the same copy.

On macOS and Linux:

```bash
mkdir -p .claude/skills && cp -r .agents/skills/fpembed .claude/skills/fpembed
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .claude/skills | Out-Null; Copy-Item -Recurse -Force .agents/skills/fpembed .claude/skills/fpembed
```

The installers write into the matching directory above. Without one, opening the repository root as the workspace loads the skill for that project session in the platforms that read `.agents/skills/`; Claude Code still needs the manual copy. The skill is project-scoped unless you install globally with `-g` or copy it to a user-level directory such as `~/.agents/skills/` or `~/.claude/skills/`.

`pip install fpembed` installs the library into `site-packages` and does not install the skill. To use the skill, install it with the skills CLI or copy the skill folder into a discovered skills directory.

## References

The `references/` folder holds four documents. `install.md` covers the per-platform paths, the Claude Code copy, the pip-install caveat, and the dependency tiers. `flags.md` holds the full flag tables. `fingerprint_types.md` describes the supported fingerprint types, count mode, and fixed widths. `quickstart.md` walks through the same flow as the quick start above with more detail. These files belong to the skill folder and sit outside this documentation site. For library background, see [Installation](installation.md) and [Quick Start](quickstart.md).

## Link attribution

Links generated by the skill use the query string `utm_source=agent&utm_medium=skill&utm_campaign=<context>`, where the context is `quickstart`, `doctor`, `featurize`, `compress`, `qsar`, or `diversity`. This page is the destination for all of them.
