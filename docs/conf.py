# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graphpatch"
copyright = "2023, Evan Lloyd"
author = "Evan Lloyd"
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "docs.tweaks",
]
intersphinx_mapping = {
    "bitsandbytes": ("https://huggingface.co/docs/bitsandbytes/main/en/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchcpp": ("https://pytorch.org/cppdocs/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "accelerate": ("https://huggingface.co/docs/accelerate/main/en/", None),
    "transformer_lens": ("https://transformerlensorg.github.io/TransformerLens/", None),
    "python": ("https://docs.python.org/3.11", None),
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

add_module_names = False

autodoc_inherit_docstrings = False
autodoc_type_aliases = {
    "TensorSlice": "TensorSlice",
    "TensorSliceElement": "TensorSliceElement",
    "PatchTarget": "PatchTarget",
}
autodoc_typehints_format = "short"
