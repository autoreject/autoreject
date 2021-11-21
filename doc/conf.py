"""Configure autoreject docs."""
import os
import sys
from datetime import date

import sphinx_bootstrap_theme

import autoreject


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, '..', 'autoreject')))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'sphinx_copybutton',
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autosummary_generate = True  # generate autosummary even if no references

# configure numpydoc
numpydoc_xref_param_type = True
numpydoc_show_class_members = False  # noqa:E501  https://stackoverflow.com/a/34604043/5201771

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
author = 'Mainak Jas'
project = 'autoreject'
td = date.today()
copyright = f'2016-{td.year}, {author}. Last updated on {td.isoformat()}'

# The short X.Y version.
version = autoreject.__version__
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------------------------------------

# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False

html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'navbar_title': 'autoreject',
    'bootswatch_theme': "united",
    'navbar_sidebarrel': False,  # no "previous / next" navigation
    'navbar_pagenav': False,  # no "Page" navigation in sidebar
    'bootstrap_version': '3',
    'navbar_links': [
        ("Examples", "auto_examples/index"),
        ("Explanation", "explanation"),
        ("FAQ", "faq"),
        ("API", "api"),
        ("What's new", "whats_new"),
        ("GitHub", "https://github.com/autoreject/autoreject", True)
    ],

}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'mne': ('https://mne.tools/dev', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
}
intersphinx_timeout = 5

sphinx_gallery_conf = {
    'doc_module': 'autoreject',
    'reference_url': {
        'autoreject': None,
    },
    'backreferences_dir': 'generated',
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
}
