"""Sphinx configuration for the FPembed documentation site."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INIT = ROOT / "src" / "fpembed" / "__init__.py"
VERSION = re.search(
    r'__version__ = "([^"]+)"',
    PACKAGE_INIT.read_text(encoding="utf-8"),
).group(1)

project = "FPembed"
author = "Mikhail Markovskii"
copyright = "2026, Mikhail Markovskii"
author_email = "contact@fpembed.com"
version = VERSION
release = VERSION
language = "en"

extensions = ["myst_parser"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "sphinx_rtd_theme"
html_title = "FPembed documentation"
html_baseurl = "https://fpembed.readthedocs.io/en/latest/"

html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css", "consent.css"]
html_js_files = [("consent.js", {"defer": "defer"})]

html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

html_context = {
    "display_github": True,
    "github_user": "Sciencealone",
    "github_repo": "fpembed",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
linkcheck_anchors = True
linkcheck_timeout = 15
