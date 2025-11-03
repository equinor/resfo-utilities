"""
The testing module implements hypothesis generators for data commonly
found in reservoir simulator output.
"""

from ._egrid_generator import (
    Units,
    GridRelative,
    GridUnit,
    CoordinateType,
    TypeOfGrid,
    RockModel,
    GridFormat,
    Filehead,
    GridHead,
    GlobalGrid,
    EGrid,
    egrids,
)

from ._summary_generator import (
    summary_variables,
    UnitSystem,
    Simulator,
    SmspecIntehead,
    Date,
    Smspec,
    smspecs,
    SummaryMiniStep,
    SummaryStep,
    Unsmry,
    summaries,
)

__all__ = [
    "Units",
    "GridRelative",
    "GridUnit",
    "CoordinateType",
    "TypeOfGrid",
    "RockModel",
    "GridFormat",
    "Filehead",
    "GridHead",
    "GlobalGrid",
    "EGrid",
    "egrids",
    "summary_variables",
    "UnitSystem",
    "Simulator",
    "SmspecIntehead",
    "Date",
    "Smspec",
    "smspecs",
    "SummaryMiniStep",
    "SummaryStep",
    "Unsmry",
    "summaries",
]
