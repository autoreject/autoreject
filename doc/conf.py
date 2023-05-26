"""Configure autoreject docs."""
import os
import sys
from datetime import date

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
    'sphinx_github_role',
]

# configure sphinx-github-role
github_default_org_project = ("autoreject", "autoreject")

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autosummary_generate = True  # generate autosummary even if no references

# configure numpydoc
numpydoc_xref_param_type = True
numpydoc_show_class_members = False  # noqa:E501  https://stackoverflow.com/a/34604043/5201771
numpydoc_xref_ignore = {
    # words
    "instance", "of", "shape", "object",
    # shapes
    "n_interpolate", "consensus", "n_epochs", "n_channels", "n_data_channels",
}

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
author = 'autoreject developers'
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
html_show_sourcelink = False
html_copy_source = False

html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'icon_links': [
        dict(name='GitHub',
             url='https://github.com/autoreject/autoreject',
             icon='fab fa-github-square'),
    ],
    'icon_links_label': 'Quick Links',  # for screen reader
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_toc_level': 1,
}

html_context = {
    'versions_dropdown': {
        'dev': 'v0.5 (devel)',
        'stable': 'v0.4 (stable)',
        'v0.3': 'v0.3',
        'v0.2': 'v0.2',
    },
}

html_sidebars = {}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'mne': ('https://mne.tools/dev', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
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
