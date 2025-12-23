project = "resfo-utilities"  # noqa: INP001
copyright = "2022, Equinor"  # noqa: A001
author = "Equinor"
release = "1.0.0"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
}
nitpick_ignore = [
    ("py:class", "ArrayLike"),
    ("py:class", "annotated_types.Gt"),
    ("py:class", "annotated_types.Interval"),
    ("py:class", "collections.abc.Buffer"),
    ("py:class", "hypothesis.strategies.SearchStrategy"),
    ("py:class", "np.bool_"),
    ("py:class", "np.float32"),
    ("py:class", "npt.ArrayLike"),
    ("py:class", "npt.NDArray"),
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int32"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "resfo.format.Format"),
]
language = "python"
html_theme = "sphinx_rtd_theme"
autodoc_type_aliases = {"npt.ArrayLike": "ArrayLike"}
