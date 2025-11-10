User Guide
==========
Resfo-utilities is a toolkit for working with parsed reservoir simulator output parsed from
Resfo. Resfo is an abbreviation for "REservoir Simulation Fortran Output".

Purpose
-------
OPM FLOW outputs files on various formats. Resfo is parsing these files and read them
into Python objects. Resfo-utilities is a toolkit to work with the output from Resfo.

Getting started
---------------

Details regarding the output format of reservoir simulators can be found `here <https://resfo.readthedocs.io/en/latest/the_file_format.html>`_.
The resfo-utilities package can be downloaded from pypi using

.. code-block:: shell

    uv pip install resfo-utilities


A typical starting point when working with result files is to read it into your python
application. Resfo contains reading utility for this purpose.

.. code-block:: python

    import resfo
    filename = "output.egrid"
    stream = open(filename)

    for entry in resfo.lazyread(stream):
        ...


´´´
