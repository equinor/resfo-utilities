Glossary
========

.. glossary::

    corner-point grid
        A corner-point grid is a tessellation of a 3D volume where
        each cell is a hexahedron. See the `corner-point grid wikipedia page`_  or
        `OPM Flow manual`_ section 6.2.2 for more information.

    pillar
        In a :term:`corner-point grid` the pillars are a set of straight lines used
        to define each cell. The 8 corner vertices of each cell is determined by
        giving the height along four adjacent pillars.

    UTM-coordinates
        The Universal Transverse Mercator (UTM) is a coordinate system based on
        the transverse Mercator map projection of the Earth spheroid. See
        the `UTM wikipedia page`_.

    formatted files
        The output files come in two flavours: formatted and unformatted. If the ``.DATA`` file contains
        ``FMTOUT`` then the output will be formatted. Generally, formatted file
        extension start with an "F", so ``.FEGRID`` instead of ``.EGRID``.

    well connection
        In reservoir simulation, wells consists of connections. These are the
        i,j,k coordinates in the :term:`corner-point grid` which the `well trajectory`_
        go through.

.. _OPM Flow manual: https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf
.. _corner-point grid wikipedia page: https://en.wikipedia.org/wiki/Corner-point_grid
.. _UTM wikipedia page: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
.. _well trajectory: https://wiki.aapg.org/Wellbore_trajectory
