# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import date
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/psimpy/'))


# -- Project information -----------------------------------------------------

project = 'psimpy'
copyright = f'{date.today().year}, Hu Zhao'
author = 'Hu Zhao'

import psimpy
version = psimpy.__version__
release = version


# -- General configuration ---------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'myst_nb',
    # 'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
]

# autoapi_dirs = ['../src']
autoclass_content = 'init'
autodoc_member_order = 'bysource'
add_module_names = False

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# configure Sphinx-Gallery
sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/emulator",
        "../examples/inference",
        "../examples/sampler",
        "../examples/sensitivity",
        "../examples/simulator",
    ],  # path to your example scripts,
    "gallery_dirs": [
        "auto_examples/emulator",
        "auto_examples/inference",
        "auto_examples/sampler",
        "auto_examples/sensitivity",
        "auto_examples/simulator",
    ],  # path to where to save gallery generated output
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

html_title = "PSimPy's documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]
