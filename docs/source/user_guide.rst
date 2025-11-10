User Guide
==========
Resfo is an abbreviation for "REservoir Simulation Fortran Output"

Purpose
-------
OPM FLOW outputs files on various formats. Resfo is parsing these files and read them
into Python objects. Resfo-utilities is a toolkit to work with the output from Resfo.

OPM FLOW output format
----------------------
The output files of OPM flow follows a pattern where the file is divided into entries.
Each entry is typically multiple lines long, where the first line contains metadata and
the following lines contains measurement data. The first line starts with a "header" - a
string describing the measurement type followed by a number representing the time of
measurement. The line is concluded with another string describing the type of the data.
The type of data can be string, integer, float, boolean and missing value.

A generic example of this pattern looks like:

.. code-block:: text

    'MEASUREMENT' 1234 'INTE'
     1111 2222 3333
     4444 5555 6666
     7777 8888 9999

Getting started
---------------
The resfo package can be downloaded from pypi using

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
