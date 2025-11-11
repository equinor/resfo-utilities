======================================================================
resfo-utilities: A library for working with reservoir simulator output
======================================================================

resfo-utilities is a library for working with output from several reservoir
simulators such as `opm flow <https://github.com/OPM/opm-simulators>`__.

Quick Start Guide
-----------------

.. code-block:: console

   pip install resfo-utilities

We assume that you have some reservoir simulator input (a ``.DATA`` file), but if
you want to start testing you can pick up one `here
<https://github.com/OPM/opm-tests>`__.

By giving that input to a reservoir simulator (flow in this case):


.. code-block:: console

   flow MY_INPUT.DATA

the simulator will output a number of files (depending on what options
are set in the input file). resfo-utilities is used to read and analyze
such output. The summary files (.SMSPEC, .UNSMRY, etc.) can be read using
:class:`resfo_utilities.SummaryReader` while the egrid file (.EGRID and .FEGRID) can be
analyzed using the :class:`resfo_utilities.CornerpointGrid`.



.. toctree::
   :maxdepth: 2

   self
   user_guide
   api_reference
   glossary
