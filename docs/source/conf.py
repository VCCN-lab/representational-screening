# repscreen documentation build configuration file.

import os
import sys
from importlib.metadata import version as _pkg_version

import matplotlib
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath("../.."))

# Save figures with a tight bounding box so edge content is not
# clipped.  sphinx-gallery resets these per example, so they are also
# re-applied via the ``reset_modules`` hook below.
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.2


# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxarg.ext",
]

# Configuration for sphinx-gallery
# The generated gallery under ``auto_examples`` is committed, so the
# documentation builds from it without running the examples.  Set
# REPSCREEN_BUILD_GALLERY=1 to execute them and regenerate that
# output.  ``run_stale_examples`` follows the same switch, so a build
# without it reuses the committed output rather than re-running an
# example whose source has changed.
_build_gallery = (
    os.environ.get("REPSCREEN_BUILD_GALLERY", "0") == "1"
)


def _tight_savefig(gallery_conf, fname):
    """Re-apply a tight bounding box after each example is reset.

    sphinx-gallery's default ``reset_modules`` calls
    ``plt.rcdefaults()`` before every example, which drops the
    tight-bbox settings and would otherwise clip figures (y-labels,
    outside legends) at the edges.
    """
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.2


class ExampleOrderKey(FileNameSortKey):
    """Order gallery examples from basic to advanced use cases."""

    _order = (
        "plot_screening_run.py",
        "plot_compactness_measures.py",
        "plot_curated_stimuli.py",
        "plot_null_distribution.py",
        "plot_category_signatures.py",
    )

    def __call__(self, filename):
        name = os.path.basename(filename)
        if name in self._order:
            return self._order.index(name)
        return len(self._order)


sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"(__init__|_synthetic)\.py",
    "within_subsection_order": ExampleOrderKey,
    "reset_modules": ("matplotlib", "seaborn", _tight_savefig),
    "plot_gallery": _build_gallery,
    "run_stale_examples": _build_gallery,
}

# Configuration for sphinx-copybutton
copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Add any paths that contain templates here
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"

# General information about the project
project = "repscreen"
copyright = "2026, the repscreen developers"
author = "the repscreen developers"

# Get version from installed package.  repscreen must be installed
# (via ``uv pip install -e ".[docs]"``) for the build to succeed.
_version = _pkg_version("repscreen")
version = _version
release = _version

language = "en"
exclude_patterns = []
pygments_style = "default"
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

extensions.append("sphinx_rtd_theme")
html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
    ]
}

htmlhelp_basename = "repscreen"

# LaTeX output
latex_elements = {}
latex_documents = [
    (
        master_doc,
        "repscreen.tex",
        "repscreen Documentation",
        "Contributors",
        "manual",
    ),
]

# Manual page output
man_pages = [
    (
        master_doc,
        "repscreen",
        "repscreen Documentation",
        [author],
        1,
    ),
]

# Texinfo output
texinfo_documents = [
    (
        master_doc,
        "repscreen",
        "repscreen Documentation",
        author,
        "repscreen",
        "Representational screening for stimulus curation.",
        "Miscellaneous",
    ),
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": (
        "https://scikit-learn.org/stable/",
        None,
    ),
}

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_show_sourcelink = True
html_show_sphinx = True

# Custom CSS for RTD theme
html_css_files = [
    "custom_rtd.css",
]

html_context = {
    "display_github": True,
    "github_user": "PeerHerholz",
    "github_repo": "representational-screening",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

suppress_warnings = ["config.cache", "duplicate.object"]
