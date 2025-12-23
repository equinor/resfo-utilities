"""
The egrid fileformat is a file used by reservoir simulators such as opm
flow containing the grid geometry.

For details about the data format see https://resfo.readthedocs.io/en/latest/the_file_format.html


The basic usage is to use the ``egrids`` generator::

    from hypothesis import given
    from resfo_utilities import egrids, EGrid

    @given(egrids)
    def test_egrid(egrid: EGrid):
        print(egrid.shape) # tuple ni,nj,nk
        egrid.to_file("MY_CASE.EGRID")


egrid files contain tuples of keywords and list of data values
of one type (An array with a name). The enums in this file generally describe
a range of values for a position in one of these lists, the dataclasses describe
the values of one keyword or a collection of those, named a file section.

The following egrid file contents (as keyword/array pairs)::

  ("FILEHEAD", [2001,3,0,3,0,0,0])
  ("GRIDUNIT", "METRES   ")

is represented by::

    EGrid(
        Filehead(2001,3,3,TypeOfGrid.CORNER_POINT,RockModel(0),GridFormat(0)),
        GridUnit("METRES   ")
    )

Where ``EGrid`` is a section of the file, ``Filehead`` and ``GridUnit`` are
keywords.

Generally, the data layout of these objects map 1-to-1 with some section of an
valid egrid file.

keywords implement the `to_egrid` that convert from the object representation
to the in file representation.
"""

from dataclasses import astuple, dataclass
from enum import Enum, auto, unique
from os import PathLike
from typing import IO, Any, assert_never

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import resfo
from hypothesis.extra.numpy import arrays


@unique
class Units(Enum):
    """The Grids distance units."""

    METRES = auto()
    CM = auto()
    FEET = auto()

    def to_egrid(self) -> str:
        return self.name.ljust(8)


@unique
class GridRelative(Enum):
    """GridRelative is the second value given GRIDUNIT keyword.

    MAP means map relative units, while
    leaving it blank means relative to the origin given by the
    MAPAXES keyword.
    """

    MAP = auto()
    ORIGIN = auto()

    def to_egrid(self) -> str:
        if self == GridRelative.MAP:
            return "MAP".ljust(8)
        return "".ljust(8)


@dataclass
class GrdeclKeyword:
    """An abstract grdecl keyword.

    Gives a general implementation of to/from grdecl which recurses on
    fields. Ie. a dataclass such as
    >>> class A(GrdeclKeyword):
    ...     ...
    >>> class B(GrdeclKeyword):
    ...     ...

    >>> @dataclass
    ... class MyKeyword(GrdeclKeyword):
    ...     field1: A
    ...     field2: B

    will have a to_egrid method that will be similar to

    >>> def to_egrid(self):
    ...     return [self.field1.to_egrid(), self.field2.to_egrid]
    """

    def to_egrid(self) -> list[Any]:
        return [value.to_egrid() for value in astuple(self)]


@dataclass
class GridUnit(GrdeclKeyword):
    """Defines the units used for grid dimensions.

    The first value is a string describing the units used, defaults to METRES,
    known accepted other units are FIELD and LAB. The last value describes
    whether the measurements are relative to the map or to the origin of
    MAPAXES.
    """

    unit: Units = Units.METRES
    grid_relative: GridRelative = GridRelative.ORIGIN


@unique
class CoordinateType(Enum):
    """The coordinate system type given in the SPECGRID keyword.

    This is given by either T or F in the last value of SPECGRID, meaning
    either cylindrical or cartesian coordinates respectively.
    """

    CARTESIAN = auto()
    CYLINDRICAL = auto()

    def to_egrid(self) -> int:
        if self == CoordinateType.CARTESIAN:
            return 0
        return 1


@unique
class TypeOfGrid(Enum):
    """
    A Grid has three possible data layout formats, UNSTRUCTURED, CORNER_POINT,
    BLOCK_CENTER and COMPOSITE (meaning combination of the two former).
    """

    COMPOSITE = 0
    CORNER_POINT = 1
    UNSTRUCTURED = 2
    BLOCK_CENTER = 3

    @property
    def alternate_value(self) -> int:
        """Inverse of alternate_code."""
        alternate_value = 0
        match self:
            case TypeOfGrid.CORNER_POINT:
                alternate_value = 0
            case TypeOfGrid.UNSTRUCTURED:
                alternate_value = 1
            case TypeOfGrid.COMPOSITE:
                alternate_value = 2
            case TypeOfGrid.BLOCK_CENTER:
                alternate_value = 3

            case default:
                assert_never(default)
        return alternate_value


@unique
class RockModel(Enum):
    """Type of rock model."""

    SINGLE_PERMEABILITY_POROSITY = 0
    DUAL_POROSITY = 1
    DUAL_PERMEABILITY = 2


@unique
class GridFormat(Enum):
    """
    The format of the "original grid", ie., what
    method was used to construct the values in the file.
    """

    UNKNOWN = 0
    IRREGULAR_CORNER_POINT = 1
    REGULAR_CARTESIAN = 2


@dataclass
class Filehead:
    """
    The first keyword in an egrid file is the FILEHEAD
    keyword, containing metadata about the file and its
    content.
    """

    version_number: np.int32
    year: np.int32
    version_bound: np.int32
    type_of_grid: TypeOfGrid
    rock_model: RockModel
    grid_format: GridFormat

    def to_egrid(self) -> np.ndarray:
        """
        Returns:
            List of values, as layed out after the FILEHEAD keyword for
            the given filehead.
        """
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.version_number
        result[1] = self.year
        result[3] = self.version_bound
        result[4] = self.type_of_grid.alternate_value
        result[5] = self.rock_model.value
        result[6] = self.grid_format.value
        return result


@dataclass
class GridHead:
    """
    Both for lgr (see LGRSection) and the global grid (see GlobalGrid)
    the GRIDHEAD keyword indicates the start of the grid layout for that
    section.
    """

    type_of_grid: TypeOfGrid
    num_x: np.int32
    num_y: np.int32
    num_z: np.int32
    grid_reference_number: np.int32
    numres: np.int32
    nseg: np.int32
    coordinate_type: CoordinateType
    lgr_start: tuple[np.int32, np.int32, np.int32]
    lgr_end: tuple[np.int32, np.int32, np.int32]

    def to_egrid(self) -> np.ndarray:
        # The data is expected to consist of
        # 100 integers, but only a subset is used.
        result = np.zeros((100,), dtype=np.int32)
        result[0] = self.type_of_grid.value
        result[1] = self.num_x
        result[2] = self.num_y
        result[3] = self.num_z
        result[4] = self.grid_reference_number
        result[24] = self.numres
        result[25] = self.nseg
        result[26] = self.coordinate_type.to_egrid()
        result[[27, 28, 29]] = np.array(self.lgr_start)
        result[[30, 31, 32]] = np.array(self.lgr_end)
        return result


@dataclass
class GlobalGrid:  # noqa: PLW1641
    """
    The global grid contains the layout of the grid before
    refinements, and the sectioning into grid coarsening
    through the optional corsnum keyword.
    """

    grid_head: GridHead
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalGrid):
            return False
        if self.actnum is None:
            return other.actnum is None
        if other.actnum is None:
            return self.actnum is None
        return (
            self.grid_head == other.grid_head
            and np.array_equal(self.actnum, other.actnum)
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    def to_egrid(self) -> list[tuple[str, Any]]:
        result = [
            ("GRIDHEAD", self.grid_head.to_egrid()),
            ("COORD   ", self.coord.astype(np.float32)),
            ("ZCORN   ", self.zcorn.astype(np.float32)),
        ]
        if self.actnum is not None:
            result.append(("ACTNUM  ", self.actnum.astype(np.int32)))
        result.append(("ENDGRID ", np.array([], dtype=np.int32)))
        return result


@dataclass
class EGrid:
    """Contains the data of an EGRID file.

    Args:
        file_head:
            The file header starting with the FILEHEAD keyword
        global_grid:
            The global grid
    """

    file_head: Filehead
    grid_unit: GridUnit
    global_grid: GlobalGrid

    @property
    def shape(self) -> tuple[np.int32, np.int32, np.int32]:
        grid_head = self.global_grid.grid_head
        return (grid_head.num_x, grid_head.num_y, grid_head.num_z)

    def to_file(self, filelike: str | PathLike[str] | IO[Any]) -> None:
        """write the EGrid to file.

        Args:
            filelike: The egrid file to write to.
        """
        contents = []
        contents.append(("FILEHEAD", self.file_head.to_egrid()))
        contents.append(("GRIDUNIT", self.grid_unit.to_egrid()))  # type: ignore
        contents += self.global_grid.to_egrid()
        resfo.write(filelike, contents)


_finites = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

_indices = st.integers(min_value=1, max_value=4)


def _zcorns(dims: tuple[int, int, int]) -> st.SearchStrategy[npt.NDArray[Any]]:
    return arrays(
        shape=8 * dims[0] * dims[1] * dims[2],
        dtype=np.float32,
        elements=_finites,
    )


_types_of_grid = st.just(TypeOfGrid.CORNER_POINT)
_file_heads = st.builds(
    Filehead,
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=2000, max_value=2022),
    st.integers(min_value=0, max_value=5),
    _types_of_grid,
    grid_format=st.just(GridFormat.IRREGULAR_CORNER_POINT),
)

_grid_heads = st.builds(
    GridHead,
    _types_of_grid,
    _indices,
    _indices,
    _indices,
    _indices,
    st.just(1),
    st.just(1),
    coordinate_type=st.just(CoordinateType.CARTESIAN),
    lgr_start=st.tuples(_indices, _indices, _indices),
    lgr_end=st.tuples(_indices, _indices, _indices),
)


@st.composite
def _global_grids(draw: st.DrawFn) -> GlobalGrid:
    grid_head = draw(_grid_heads)
    dims = (int(grid_head.num_x), int(grid_head.num_y), int(grid_head.num_z))
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = arrays(
        shape=corner_size,
        dtype=np.float32,
        elements=_finites,
    )
    actnum = st.one_of(
        st.just(None),
        arrays(
            shape=dims[0] * dims[1] * dims[2],
            dtype=np.int32,
            elements=st.integers(min_value=0, max_value=3),
        ),
    )
    return GlobalGrid(
        coord=draw(coord),
        zcorn=draw(_zcorns(dims)),
        actnum=draw(actnum),
        grid_head=grid_head,
    )


egrids = st.builds(EGrid, _file_heads, global_grid=_global_grids())

__all__ = [
    "CoordinateType",
    "EGrid",
    "Filehead",
    "GlobalGrid",
    "GrdeclKeyword",
    "GridFormat",
    "GridHead",
    "GridRelative",
    "GridUnit",
    "RockModel",
    "TypeOfGrid",
    "Units",
    "egrids",
]
