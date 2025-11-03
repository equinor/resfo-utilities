from resfo_utilities import (
    CornerpointGrid,
    InvalidEgridFileError,
    MapAxes,
    InvalidGridError,
)
from resfo_utilities.testing import egrids
import resfo
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, assume, example
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays, from_dtype
from itertools import product
from contextlib import suppress


def write_to_buffer(file_contents):
    buffer = BytesIO()
    resfo.write(buffer, file_contents)
    buffer.seek(0)
    return buffer


def pad_to(lst: list[int], target_len: int):
    return np.pad(lst, (0, target_len - len(lst)), mode="constant")


@given(egrids)
def test_that_cornerpoint_grid_reads_all_generated_grids(egrid):
    buf = BytesIO()
    egrid.to_file(buf)
    buf.seek(0)
    with suppress(InvalidEgridFileError):  # for radial grids
        _ = CornerpointGrid.read_egrid(buf)


def test_that_read_egrid_raises_invalid_egrid_file_when_gridhead_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(write_to_buffer([("GRIDHEAD", resfo.MESS)]))


def test_that_read_egrid_raises_invalid_egrid_file_when_gridhead_is_too_short():
    with pytest.raises(InvalidEgridFileError, match="contained too few elements"):
        CornerpointGrid.read_egrid(write_to_buffer([("GRIDHEAD", [1.0])]))


def test_that_read_egrid_raises_invalid_egrid_file_when_coord_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(
            write_to_buffer([("GRIDHEAD", [1, 1, 1, 1]), ("COORD   ", resfo.MESS)])
        )


def test_that_read_egrid_raises_invalid_egrid_file_the_coordinate_system_is_radial():
    grid_head_array = pad_to([1, 1, 1, 1], 100)
    grid_head_array[26] = 1  # set to 1 to indicate radial grid
    with pytest.raises(InvalidEgridFileError, match="contains a radial grid"):
        CornerpointGrid.read_egrid(write_to_buffer([("GRIDHEAD", grid_head_array)]))


def test_that_read_egrid_warns_when_the_global_grid_does_not_have_reference_number_zero():
    with pytest.warns(UserWarning, match="reference number 1, expected 0"):
        CornerpointGrid.read_egrid(
            write_to_buffer(
                [
                    ("GRIDHEAD", [1, 1, 1, 1, 1]),
                    ("COORD   ", [1.0] * (8 * 3)),
                    ("ZCORN   ", [1.0] * 8),
                ],
            )
        )


def test_that_read_egrid_raises_invalid_egrid_file_when_coord_has_too_many_values():
    with pytest.raises(InvalidEgridFileError, match="did not match grid dimensions"):
        CornerpointGrid.read_egrid(
            write_to_buffer(
                [
                    ("GRIDHEAD", [1, 1, 1, 1]),
                    ("COORD   ", [1.0]),
                    ("ZCORN   ", [1.0] * 8),
                ],
            )
        )


def test_that_read_egrid_raises_invalid_egrid_file_when_mapaxes_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(write_to_buffer([("MAPAXES ", resfo.MESS)]))


def test_that_read_egrid_raises_invalid_egrid_file_when_mapaxes_has_too_many_values():
    with pytest.raises(InvalidEgridFileError, match="contained too few elements"):
        CornerpointGrid.read_egrid(write_to_buffer([("MAPAXES ", [1.0])]))


def test_that_grid_with_invalid_coord_shape_raises():
    with pytest.raises(InvalidGridError, match="coord had invalid dimensions"):
        CornerpointGrid(
            coord=np.array([], dtype=np.float32),
            zcorn=np.array([[[[0, 0, 0, 0, 1, 1, 1, 1]]]], dtype=np.float32),
        )


def test_that_grid_with_invalid_zcorn_shape_raises():
    with pytest.raises(InvalidGridError, match="zcorn had invalid dimensions"):
        CornerpointGrid(
            coord=np.array(
                [
                    [[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                    [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
                ],
                dtype=np.float32,
            ),
            zcorn=np.array([], dtype=np.float32),
        )


def test_that_grid_with_zcorn_and_coord_shape_mismatch_raises():
    with pytest.raises(
        InvalidGridError, match="zcorn and coord dimensions do not match"
    ):
        CornerpointGrid(
            coord=np.array(
                [
                    [[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                    [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
                ],
                dtype=np.float32,
            ),
            zcorn=np.array(
                [[[[0, 0, 0, 0, 1, 1, 1, 1]]], [[[0, 0, 0, 0, 1, 1, 1, 1]]]],
                dtype=np.float32,
            ),
        )


@pytest.mark.parametrize(
    "contents_after_global_grid",
    [
        [],
        [
            ("LGR     ", np.array([b"LGR1    "], dtype="|S8")),
            ("LGRPARNT", np.array([b"        "], dtype="|S8")),
            (
                "GRIDHEAD",
                pad_to([1, 2, 2, 2, 1] + [0] * 19 + [1, 1, 0, 2, 2, 2, 2, 2, 2], 100),
            ),
            ("COORD   ", np.arange(200, 254)),
            ("ZCORN   ", np.arange(300, 364)),
            ("ACTNUM  ", np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=">i4")),
            ("HOSTNUM ", np.array([8, 8, 8, 8, 8, 8, 8, 8], dtype=">i4")),
            ("ENDGRID ", np.array([], dtype=">i4")),
            ("ENDLGR  ", np.array([], dtype=">i4")),
            ("NNCHEAD ", pad_to([0, 1], 10)),
            ("NNC1    ", np.array([], dtype=">i4")),
            ("NNC2    ", np.array([], dtype=">i4")),
            ("NNCL    ", np.array([], dtype=">i4")),
            ("NNCG    ", np.array([], dtype=">i4")),
        ],
    ],
)
def test_that_read_egrid_fetches_the_geometry_from_the_global_grid_in_the_file(
    contents_after_global_grid,
):
    grid = CornerpointGrid.read_egrid(
        write_to_buffer(
            [
                ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
                ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
                ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
                ("GRIDHEAD", pad_to([1, 2, 2, 2], 100)),
                ("COORD   ", np.arange(0, 54, dtype=">f4")),
                ("ZCORN   ", np.arange(100, 164, dtype=">f4")),
                ("ACTNUM  ", np.ones((8,), dtype=">i4")),
                ("ENDGRID ", np.array([], dtype=">i4")),
                *contents_after_global_grid,
            ]
        )
    )
    assert grid.map_axes.origin == (0.0, 0.0)
    assert grid.map_axes.y_axis == (0.0, 1.0)
    assert grid.map_axes.x_axis == (1.0, 0.0)
    # coord indecies are (ni, nj, nk, 3)
    # where each triplet is the x,y,z coordinate
    # of the pillar at (i,j,k)

    # The opm manual describes the order in the input as follows:
    # "COORD defines a set of coordinate lines or pillars for a reservoir grid via
    # an array. A total of 6 x (NX+1) x (NY+1) lines must be specified for each
    # coordinate data set (or reservoir). For multiple reservoirs, where
    # NUMRES is greater than one, there must be 6 x (NX+1) x (NY+1) x NUMRES values.
    # For Cartesian geometry, each line is defined by the (x, y, z) coordinates of
    # two distinct points on the line. The lines are entered with I cycling fastest then J."
    assert grid.coord.tolist() == [
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],  # Top and base points for i,j=0,0
            [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
            [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
        ],
        [
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[24.0, 25.0, 26.0], [27.0, 28.0, 29.0]],
            [[42.0, 43.0, 44.0], [45.0, 46.0, 47.0]],
        ],
        [
            [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]],
            [[48.0, 49.0, 50.0], [51.0, 52.0, 53.0]],
        ],
    ]
    # coord indecies are (ni, nj, nk, 8)
    # where the 8 values describe depth of the corners
    # for cell (i,j,k)
    # Order of heights for each corner is
    # (N(orth) means higher y, E(east) means higer x, T(op) means lower z (depth))
    # TSW TSE TNW TNE BSW BSE BNW BNE

    # The opm manual describes the order of the input as follows:
    # "ZCORN defines the depth of each corner point of a grid block
    # on the pillars defining the reservoir grid. A total of 8 x NX x NY x NZ values
    # are needed to fully define all the depths in the model. The depths
    # specifying the top of the first layer are entered first with one point
    # for each pillar for each grid block. The points are entered
    # with the X axis cycling fastest. Next come the depths of the bottom
    # of the first layer. The top of layer two follows etc."
    assert grid.zcorn.tolist() == [
        [
            [
                [100.0, 101.0, 104.0, 105.0, 116.0, 117.0, 120.0, 121.0],
                [132.0, 133.0, 136.0, 137.0, 148.0, 149.0, 152.0, 153.0],
            ],
            [
                [108.0, 109.0, 112.0, 113.0, 124.0, 125.0, 128.0, 129.0],
                [140.0, 141.0, 144.0, 145.0, 156.0, 157.0, 160.0, 161.0],
            ],
        ],
        [
            [
                [102.0, 103.0, 106.0, 107.0, 118.0, 119.0, 122.0, 123.0],
                [134.0, 135.0, 138.0, 139.0, 150.0, 151.0, 154.0, 155.0],
            ],
            [
                [110.0, 111.0, 114.0, 115.0, 126.0, 127.0, 130.0, 131.0],
                [142.0, 143.0, 146.0, 147.0, 158.0, 159.0, 162.0, 163.0],
            ],
        ],
    ]


@pytest.fixture
def unit_cell_grid():
    """A Corner point grid which just contains the unit cube as a cell"""
    return CornerpointGrid(
        coord=np.array(
            [
                [[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array([[[[0, 0, 0, 0, 1, 1, 1, 1]]]], dtype=np.float32),
    )


def test_that_interior_points_are_in_the_cell(unit_cell_grid):
    assert unit_cell_grid.point_in_cell((0.5, 0.5, 0.5), 0, 0, 0)
    assert unit_cell_grid.point_in_cell(
        [(0.5, 0.5, 0.5), (0.25, 0.25, 0.25)], 0, 0, 0
    ).tolist() == [True, True]


def test_that_points_on_corners_are_in_the_cell(unit_cell_grid):
    assert unit_cell_grid.point_in_cell(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
        0,
        0,
        0,
    ).all()


def test_that_points_on_faces_are_in_the_cell(unit_cell_grid):
    assert unit_cell_grid.point_in_cell(
        [
            (0.0, 0.5, 0.5),
            (1.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 1.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 1.0),
        ],
        0,
        0,
        0,
    ).all()


def test_that_transform_points_does_not_scale_by_map_axes():
    assert MapAxes((0.0, 10.0), (0.0, 0.0), (1.0, 0.0)).transform_map_points(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    ).tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]

    assert MapAxes((0.0, 1.0), (0.0, 0.0), (10.0, 0.0)).transform_map_points(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    ).tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]


def test_that_transform_points_translates_by_origin():
    assert_allclose(
        MapAxes((100.0, 51.0), (100.0, 50.0), (101.0, 50.0)).transform_map_points(
            np.array([[101.0, 50.0, 0.0], [100.0, 51.0, 0.0], [101.0, 51.0, 1.0]])
        ),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
    )


coordinates = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1.0e9, max_value=1.0e9, width=32
)


@st.composite
def regular_grids(draw):
    ni, nj, nk = draw(st.tuples(*([st.integers(min_value=1, max_value=10)] * 3)))
    height = draw(
        st.floats(
            min_value=16.0,
            max_value=2**32,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        )
    )
    top_depth = draw(
        st.floats(
            min_value=0.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        )
    )
    bot_depth = top_depth + height
    coord = np.zeros((ni + 1, nj + 1, 2, 3), dtype=np.float32)
    zcorn = np.zeros((ni, nj, nk, 8), dtype=np.float32)
    for i, j in product(range(ni + 1), range(nj + 1)):
        coord[i, j, 0] = [i, j, top_depth]
        coord[i, j, 1] = [i, j, bot_depth]
    for i, j, k in product(range(ni), range(nj), range(nk)):
        zcorn[i, j, k] = [height * (k / nk) + top_depth] * 4 + [
            height * ((k + 1) / nk) + top_depth
        ] * 4
    return CornerpointGrid(coord, zcorn)


points = st.tuples(coordinates, coordinates, coordinates)


@given(
    grid=regular_grids(),
    point=points,
    data=st.data(),
)
def test_that_found_cell_contains_point(grid, point, data):
    (cell,) = grid.find_cell_containing_point(np.array([point], dtype=np.float32))
    if cell is None:
        # select one point to check, the point should be
        # in none of the cells
        i, j, k = data.draw(
            st.tuples(
                st.integers(min_value=0, max_value=grid.zcorn.shape[0] - 1),
                st.integers(min_value=0, max_value=grid.zcorn.shape[1] - 1),
                st.integers(min_value=0, max_value=grid.zcorn.shape[2] - 1),
            )
        )
        assert not grid.point_in_cell(point, i, j, k, tolerance=1e-14)
    else:
        assert grid.point_in_cell(point, *cell)


@given(
    grid=regular_grids(),
    point=points,
    data=st.data(),
)
def test_that_on_regular_grids_point_in_cell_is_the_same_as_in_bounding_box(
    grid, point, data
):
    cell = data.draw(
        st.tuples(
            st.integers(min_value=0, max_value=grid.zcorn.shape[0] - 1),
            st.integers(min_value=0, max_value=grid.zcorn.shape[1] - 1),
            st.integers(min_value=0, max_value=grid.zcorn.shape[2] - 1),
        )
    )
    cell_corners = grid.cell_corners(*cell)
    tolerance = 1e-6
    min_point = cell_corners.min(axis=0) - tolerance
    max_point = cell_corners.max(axis=0) + tolerance

    # avoid points close to the boundary
    assume(np.all(np.abs(point - min_point) >= 0.01))
    assume(np.all(np.abs(max_point - point) >= 0.01))

    assert grid.point_in_cell(point, *cell, tolerance) == (
        np.all(min_point <= point) and np.all(point <= max_point)
    )


def test_that_map_coordinates_parameter_sets_the_coordinate_system_for_points():
    # unit cell grid with map axes
    # translating origin to 100, 100
    grid = CornerpointGrid(
        coord=np.array(
            [
                [[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array([[[[0, 0, 0, 0, 1, 1, 1, 1]]]], dtype=np.float32),
        map_axes=MapAxes((100.0, 101.0), (100.0, 100.0), (101.0, 100.0)),
    )

    # A map contained in cell 0,0,0
    point = np.array([[100.5, 100.5, 0.5]], dtype=np.float32)
    # By default points are in the map coordinate system
    assert grid.find_cell_containing_point(point) == [(0, 0, 0)]

    assert grid.find_cell_containing_point(point, map_coordinates=False) == [None]


def test_that_point_in_cell_correctly_orders_zcorn_against_coord():
    """This is a regression test for a bug where point_in_cell
    did not order coord points correctly against zcorn"""
    grid = CornerpointGrid(
        coord=np.array(
            [
                [
                    [[581, 800, 675], [581, 800, 86]],
                    [[581, 700, 683], [581, 700, 110]],
                ],
                [
                    [[681, 800, 657], [681, 800, 48]],
                    [[681, 700, 670], [681, 700, 68]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [
                [
                    [
                        [729, 707, 733, 718, 735, 712, 738, 723],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )

    # A point contained in cell 0,0,0
    point = np.array([[662, 752, 718]], dtype=np.float64)
    # By default points are in the map coordinate system
    assert grid.point_in_cell(point, 0, 0, 0)
    assert grid.find_cell_containing_point(point) == [(0, 0, 0)]


def test_that_point_in_cell_correctly_assign_vertices_to_faces():
    """This is a regression test for a bug where point_in_cell
    assigned vertices to faces in a way that created a hole in the cell surface"""
    grid = CornerpointGrid(
        coord=np.array(
            [
                [
                    [[0, 0, 0], [0, 0, 1]],
                    [[1, 0, 0], [1, 0, 1]],
                ],
                [
                    [[0, 2, 0], [0, 1, 1]],
                    [[1, 1, 0], [1, 2, 1]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 1, 1, 1, 1],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )

    # A point contained in cell 0,0,0
    point = np.array([[0.9, 1.0, 0.75]], dtype=np.float64)

    assert grid.point_in_cell(point, 0, 0, 0)
    assert grid.find_cell_containing_point(point) == [(0, 0, 0)]


@given(*([st.floats(min_value=-1.0, max_value=2.0)] * 3))
@pytest.mark.parametrize(
    "bottom_heights",
    [
        pytest.param(
            [0.05, 1.0, 1.0, 0.05],
            id="the bottom face has two elevated diagonally opposed corners",
        ),
        pytest.param(
            [0.0, 1.0, 1.0, 1.0],
            id="one bottom corner is collapsed onto the corresponding top corner",
        ),
        pytest.param(
            [0.0, 0.0, 1.0, 1.0],
            id="one bottom edge is collapsed onto the corresponding top edge",
        ),
    ],
)
def test_point_in_cell_considers_cells_as_trilinear_shapes(bottom_heights, x, y, z):
    # All grids are unit grids except for the bottom face
    grid = CornerpointGrid(
        coord=np.array(
            [
                [[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]],
                [[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array([[[[0, 0, 0, 0, *bottom_heights]]]], dtype=np.float32),
    )

    def bottom_face_depth(x, y):
        """The depth of the bottom face at x,y by bilinear interpolation"""
        xy_1 = (1 - x) * grid.zcorn[0, 0, 0, 4] + x * grid.zcorn[0, 0, 0, 5]
        xy_2 = (1 - x) * grid.zcorn[0, 0, 0, 6] + x * grid.zcorn[0, 0, 0, 7]
        return (1 - y) * xy_1 + y * xy_2

    tolerance = 1e-6

    def in_bounding_box(point):
        return all(-1 * tolerance <= c <= 1 + tolerance for c in point)

    # avoid points that is very close to the boundary to avoid
    # numerical issues
    assume(np.abs(z) >= 0.01)
    assume(np.abs(bottom_face_depth(x, y) - z) >= 0.01)

    # only the bottom face is different from the bounding box
    # so containment is the same as the conjunction of the two conditions
    assert grid.point_in_cell([x, y, z], 0, 0, 0, tolerance) == (
        z <= bottom_face_depth(x, y) and in_bounding_box((x, y, z))
    )


def test_that_zero_height_pillar_is_invalid():
    grid = CornerpointGrid(
        coord=np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [[[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]]], dtype=np.float32
        ),
        map_axes=None,
    )
    with pytest.raises(InvalidGridError, match="Grid contains zero height pillars"):
        grid.cell_corners(0, 0, 0)


def test_that_cells_with_infinite_pillars_are_invalid():
    grid = CornerpointGrid(
        coord=np.array(
            [
                [
                    [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
                    [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
                ],
                [
                    [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
                    [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
        ),
        map_axes=None,
    )

    with pytest.raises(InvalidGridError, match="The corners of the cell"):
        grid.cell_corners(0, 0, 0)


@st.composite
def single_cell_grids(draw):
    nice_elements = dict(
        allow_nan=False, allow_infinity=False, max_value=2**12, min_value=-(2**12)
    )
    coord = draw(
        arrays(
            np.float32,
            (2, 2, 2, 3),
            elements=nice_elements,
        )
    )

    for i in range(coord.shape[0]):
        for j in range(coord.shape[1]):
            bot_pillar_z = np.float32(coord[i, j, 0, 2] + 0.1)
            coord[i, j, 1] = draw(
                from_dtype(
                    np.dtype(np.float32),
                    **{
                        **nice_elements,
                        **dict(min_value=bot_pillar_z, max_value=2**14),
                    },
                )
            )

    grid = CornerpointGrid(
        coord=coord,
        zcorn=draw(arrays(np.float32, (1, 1, 1, 8), elements=nice_elements)),
    )
    try:
        grid.cell_corners(0, 0, 0)
    except InvalidGridError:
        assume(False)
    return grid


@given(single_cell_grids(), points)
@example(
    grid=CornerpointGrid(
        coord=np.array(
            [
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
        ),
        map_axes=None,
    ),
    point=(0.0, 0.0, 0.0),
).via("discovered failure")
@example(
    grid=CornerpointGrid(
        coord=np.array(
            [
                [
                    [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 1.0, 0.0], [2.0, 1.0, 2.0]],
                ],
                [
                    [[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]],
                    [[0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
        ),
        map_axes=None,
    ),
    point=(0.0, 0.0, 0.0),
).via("constructed grid with bulging face")
@example(
    grid=CornerpointGrid(
        coord=np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zcorn=np.array(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32
        ),
        map_axes=None,
    ),
    point=(0.0, 0.0, 9.999999747378752e-06),
).via("discovered failure")
def test_that_in_single_cell_grids_found_and_contains_are_the_same(
    grid: CornerpointGrid, point: tuple[float, float, float]
):
    assert bool(grid.find_cell_containing_point([point])[0]) == grid.point_in_cell(
        point, 0, 0, 0
    )
