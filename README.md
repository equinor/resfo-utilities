resfo-utilities
===============


resfo-utilities is a library for working with output from
 several reservoir simulators such as [opm
flow](https://github.com/OPM/opm-simulators).

Installation
============

`pip install resfo-utilities`


How to contribute
=================

We use uv to have one synchronized development environment for all packages.
See [installing uv](https://docs.astral.sh/uv/getting-started/installation/). We
recommend either installing uv using your systems package manager, or creating
a small virtual environment you install base packages into (such as `uv` and `pre-commit`).

Once uv is installed, you can get a development environment by running:

```sh
git clone https://github.com/equinor/resfo-utilities
cd resfo-utilities
uv sync --all-extras
```


You should set up `pre-commit` to ensure style checks are done as you commit:

```bash
uv run pre-commit install --hook-type pre-push
```

Documentation
==============================

The documentation is hosted and can be found at [readthedocs.io](https://resfo-utilities.readthedocs.io/en/latest/).

To build the documentation locally, run:
```shell
uv run --all-extras sphinx-build -W -n -E docs/source docs/output
```
To open the documentation, open the `index.html` file in the output folder in
a web browser.
