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

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.githubpages',
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

# configure numpydoc
numpydoc_xref_param_type = True
numpydoc_show_class_members = False  # noqa:E501  https://stackoverflow.com/a/34604043/5201771
numpydoc_xref_ignore = {
    # words
    "instance", "of", "shape", "object",
    # shapes
    "n_interpolate", "consensus", "n_epochs", "n_channels", "n_data_channels",
}

# generate autosummary even if no references
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['style.css']

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
switcher_version_match = "dev" if "dev" in release else version

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
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "switcher": {
        # "json_url": "https://mne.tools/dev/_static/versions.json",
        "json_url": "https://raw.githubusercontent.com/sappelhoff/autoreject/pydata/sphinx/doc/_static/versions.json",  # noqa: E501
        "version_match": switcher_version_match,
    },
}

html_context = {
    "default_mode": "auto",
    # next 3 are for the "edit this page" button
    "github_user": "autoreject",
    "github_repo": "autoreject",
    "github_version": "master",
    "doc_path": "doc",
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
