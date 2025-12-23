import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from resfo_utilities.testing import (
    EGrid,
    Filehead,
    GlobalGrid,
    GridHead,
    TypeOfGrid,
    egrids,
)


@given(st.builds(GlobalGrid))
def test_that_global_grid_eq_is_reflexive(global_grid):
    assert global_grid == global_grid


@given(*([st.builds(GlobalGrid)] * 3))
def test_that_global_grid_eq_is_transitive(a, b, c):
    if a == b and b == c:
        assert a == c


@given(st.builds(Filehead))
def test_that_filehead_to_egrid_writes_expected_positions_and_size(fh: Filehead):
    arr = fh.to_egrid()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (100,)
    assert arr.dtype == np.int32
    assert arr[0] == fh.version_number
    assert arr[1] == fh.year
    assert arr[3] == fh.version_bound
    assert arr[4] == fh.type_of_grid.alternate_value
    assert arr[5] == fh.rock_model.value
    assert arr[6] == fh.grid_format.value


@given(st.builds(GridHead))
def test_that_gridhead_to_egrid_populates_dimension_and_coordinate_type_indices(
    gh: GridHead,
):
    arr = gh.to_egrid()
    assert arr[0] == gh.type_of_grid.value
    assert (arr[1], arr[2], arr[3]) == (gh.num_x, gh.num_y, gh.num_z)
    assert arr[4] == gh.grid_reference_number
    assert arr[24] == gh.numres
    assert arr[25] == gh.nseg
    assert arr[26] == gh.coordinate_type.to_egrid()
    assert tuple(arr[[27, 28, 29]]) == gh.lgr_start
    assert tuple(arr[[30, 31, 32]]) == gh.lgr_end


@given(egrids)
def test_that_egrid_shape_matches_global_grid_head_dimensions(grid: EGrid):
    nx, ny, nz = grid.shape
    gh = grid.global_grid.grid_head
    assert (nx, ny, nz) == (gh.num_x, gh.num_y, gh.num_z)


@given(egrids)
def test_that_egrids_strategy_produces_corner_point_grids_only(grid: EGrid):
    assert grid.file_head.type_of_grid == TypeOfGrid.CORNER_POINT
    assert grid.global_grid.grid_head.type_of_grid == TypeOfGrid.CORNER_POINT


@given(st.builds(Filehead))
def test_that_filehead_to_egrid_alternate_value_is_valid(fh: Filehead):
    arr = fh.to_egrid()
    alternate_values = {tg.alternate_value for tg in TypeOfGrid}
    assert arr[4] in alternate_values


@given(st.builds(GlobalGrid))
def test_that_globalgrid_to_egrid_contains_expected_keywords(gg: GlobalGrid):
    items = gg.to_egrid()
    keywords = [k for k, _ in items]
    assert "COORD   " in keywords
    assert "ZCORN   " in keywords
    if gg.actnum is not None:
        assert "ACTNUM  " in keywords
    assert keywords[-1] == "ENDGRID "
