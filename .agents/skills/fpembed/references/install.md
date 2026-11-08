# Installing the FPembed skill

The FPembed skill is a folder of instructions and scripts that agents load from a skills directory. It ships in the FPembed repository under `.agents/skills/fpembed/` and runs the public FPembed API. Nothing in the scripts reaches the network.

## Dependency tiers

The skill has two dependency tiers.

| Tier | Packages | Covers |
|---|---|---|
| 1 | FPembed and `numpy` | `featurize.py`, `compress.py`, `diversity.py` |
| 2 | tier 1 plus `scikit-learn` | `qsar_engine.py` |

Tier 2 is additive. A machine that only needs featurization, compression, or diversity works with tier 1 alone. Running `qsar_engine.py` without scikit-learn fails with a clear install message and a non-zero exit code. The scripts check the installed FPembed version at run time; the minimum is 0.3.0.

Install FPembed into the environment your agent uses:

```
pip install fpembed numpy
pip install scikit-learn
```

The second command is optional and only needed for QSAR.

## Auto-discovery paths

Each platform scans one or more skills directories. The repository places the skill at `.agents/skills/fpembed/`, which several platforms read directly.

| Platform | Auto-discovered skills directories |
|---|---|
| OpenCode | `.opencode/skills/` and `.agents/skills/` |
| Claude Code | `.claude/skills/` |
| Cursor | `.agents/skills/` and `.cursor/skills/` |
| GitHub Copilot | `.github/skills/`, `.claude/skills/`, and `.agents/skills/` |
| Antigravity | `.agents/skills/` |
| Codex CLI | `.agents/skills/` |
| Gemini CLI | `.gemini/skills/` and `.agents/skills/` |

OpenCode, Cursor, GitHub Copilot, Antigravity, Codex CLI, and Gemini CLI read `.agents/skills/` directly. Claude Code needs a copy under `.claude/skills/`.

## Installing the skill

Install the skill with the skills CLI, which resolves the published FPembed repository and copies the `fpembed` folder into a skills directory your agent scans:

```
npx skills add Sciencealone/fpembed --skill fpembed
```

Add `-a <agent>` to target one tool, for example `-a claude-code` or `-a codex`, and `-g` to install globally instead of into the current project. The command requires Node 22.20 or newer.

Two alternatives install the same folder:

```
gh skill install Sciencealone/fpembed fpembed --agent <agent> --scope user --allow-hidden-dirs
gemini skills install https://github.com/Sciencealone/fpembed --scope workspace
```

`gh skill install` requires GitHub CLI 2.90 or newer. All three commands read the repository, so the skill folder must be published to GitHub before any installer can find it. The auto-discovery table above shows the directory each platform scans; an installer writes into the matching one.

## Manual copy fallback

If you already have a repository clone, copy the skill folder yourself. This is also the path Claude Code needs when no installer targets it, because Claude Code reads `.claude/skills/` rather than `.agents/skills/`.

Open the repository root, not a parent folder, because these tools walk upward from the working directory and stop at the workspace boundary. Run the command for your platform once from the repository root. Both commands do the same copy.

On macOS and Linux:

```
mkdir -p .claude/skills && cp -r .agents/skills/fpembed .claude/skills/fpembed
```

On Windows PowerShell:

```
New-Item -ItemType Directory -Force .claude/skills | Out-Null; Copy-Item -Recurse -Force .agents/skills/fpembed .claude/skills/fpembed
```

## The pip-install caveat

`pip install fpembed` installs the library into `site-packages`. It does not install the skill folder, and tools do not scan site-packages for skills. To get the skill, install it with the skills CLI or copy `.agents/skills/fpembed/` into your platform's skills directory.

## Documentation

The install and quickstart guide is at https://fpembed.readthedocs.io/en/latest/agent.html?utm_source=agent&utm_medium=skill&utm_campaign=quickstart. The bundled `references/quickstart.md` shows the same flow with local commands.
