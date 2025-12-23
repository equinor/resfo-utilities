"""
The testing module implements hypothesis generators for data commonly
found in reservoir simulator output.

The basic usage is to use either the ``egrids`` generator::

    from hypothesis import given
    from resfo_utilities import egrids, EGrid

    @given(egrids)
    def test_egrid(egrid: EGrid):
        print(egrid.shape) # tuple ni,nj,nk
        egrid.to_file("MY_CASE.EGRID")

or the ``summaries`` generator::

    from resfo_utilities import SummaryReader
    from resfo_utilities.testing import summaries
    from io import BytesIO
    from hypothesis import given

    @given(summary=summaries())
    def test_that_the_read_values_matches_those_in_the_input(summary):
        smspec, unsmry = summary
        smspec_buf = BytesIO()
        unsmry_buf = BytesIO()
        smspec.to_file(smspec_buf)
        unsmry.to_file(unsmry_buf)
        smspec_buf.seek(0)
        unsmry_buf.seek(0)

        summary = SummaryReader(
            smspec=lambda: smspec_buf,
            summaries=[lambda: unsmry_buf]
            )
"""

from ._egrid_generator import (
    CoordinateType,
    EGrid,
    Filehead,
    GlobalGrid,
    GridFormat,
    GridHead,
    GridRelative,
    GridUnit,
    RockModel,
    TypeOfGrid,
    Units,
    egrids,
)
from ._summary_generator import (
    Date,
    Simulator,
    Smspec,
    SmspecIntehead,
    SummaryMiniStep,
    SummaryStep,
    UnitSystem,
    Unsmry,
    smspecs,
    summaries,
    summary_variables,
)

__all__ = [
    "CoordinateType",
    "Date",
    "EGrid",
    "Filehead",
    "GlobalGrid",
    "GridFormat",
    "GridHead",
    "GridRelative",
    "GridUnit",
    "RockModel",
    "Simulator",
    "Smspec",
    "SmspecIntehead",
    "SummaryMiniStep",
    "SummaryStep",
    "TypeOfGrid",
    "UnitSystem",
    "Units",
    "Unsmry",
    "egrids",
    "smspecs",
    "summaries",
    "summary_variables",
]
