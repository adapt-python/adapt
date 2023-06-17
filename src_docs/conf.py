# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

user = "antoinedemathelin"
project = 'adapt'
copyright = '2020, ADAPT team, Michelin and Centre Borelli, ENS Paris-Saclay'
author = 'ADAPT team'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc',
 'sphinx.ext.autosummary',
 'numpydoc',
 'sphinx.ext.linkcode',
 'nbsphinx',
 'sphinx_gallery.load_style',
 'IPython.sphinxext.ipython_console_highlighting'
 ]


numpydoc_show_inherited_class_members = False
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# napoleon_use_rtype = False

# generate autosummary even if no references
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

#root_doc = "contents"
#root_doc = "toctree_generation"
#master_doc = "toctree_generation"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_additional_pages = {'index': 'index.html'}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
#html_sidebars = {'**': ['localtoc.html', 'sourcelink.html', 'searchbox.html']}

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/custom.js',
]

html_show_sourcelink = False

nbsphinx_thumbnails = {
"examples/Sample_bias_example": "_static/images/sample_bias_corr_img.png",
"examples/Office_example": "_static/images/office_item.png",
"examples/Flowers_example": "_static/images/finetuned.png",
}

#sphinx_gallery_conf = {
#    'reference_url': {"adapt": None},
#    'examples_dirs': '../../examples',
#    'gallery_dirs': 'auto_examples',
#    "doc_module": "adapt",
#    "backreferences_dir": os.path.join("generated")}
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    #'doc_module'          : ('sphinx_gallery', 'adapt.feature_based')}


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['fullname']:
        return None
    if not info['module']:
        return None
    
    object_name = info['fullname'].lower()
    
    if "transfertree" in object_name:
        object_name = "transfer_tree"
    if "tradaboost" in object_name:
        object_name = "tradaboost"
    if "regular" in object_name:
        object_name = "regular"
    
    object_name = "_" + object_name
    
    if "utils" in info['module'] or "metrics" in info['module']:
        filename = info['module'].replace('.', '/')
    else:
        filename = os.path.join(info['module'].replace('.', '/'), object_name)
    
    return 'https://github.com/%s/%s/tree/master/%s.py' % (user, project, filename)