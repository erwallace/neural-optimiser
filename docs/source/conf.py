# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# Make the src/ directory importable so autodoc can find neural_optimiser
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "neural-optimiser"
author = "Ewan Wallace"
copyright = f"{date.today().year}, {author}"

# Dynamic versioning
release = "0.0.0"
try:
    from importlib.metadata import PackageNotFoundError, version  # py3.8+

    try:
        release = version("neural-optimiser")
    except PackageNotFoundError:
        try:
            from neural_optimiser import __version__  # noqa: F401

            release = __version__
        except Exception:
            pass
except Exception:
    pass

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "myst_parser",
]

extlinks = {
    "pyg": ("https://pytorch-geometric.readthedocs.io/en/latest/generated/%s.html", "%s"),
}

# Support .rst and .md sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Autodoc/autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Napoleon (Google/Numpy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# Intersphinx links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://pytorch.org/docs/stable/", {}),
}

# MyST config
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "tasklist",
]
