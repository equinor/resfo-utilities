import subprocess
from pathlib import Path

import pytest
from pytest import approx

from resfo_utilities import CornerpointGrid, MapAxes


@pytest.mark.usefixtures("eightcells")
def test_that_we_can_read_the_eightcells_grid_from_the_simulator(
    tmp_path: Path,
    simulator_cmd: list[str],
) -> None:
    subprocess.run([*simulator_cmd, str(tmp_path / "EIGHTCELLS")], check=False)

    grid = CornerpointGrid.read_egrid(str(tmp_path / "EIGHTCELLS.EGRID"))

    assert grid.coord.shape == (3, 3, 2, 3)
    assert grid.coord[0, 0].tolist() == [
        approx([0.1, 0.2, 0.3]),
        approx([0.4, 0.5, 100.6]),
    ]
    assert grid.coord[1, 0].tolist() == [
        approx([50.7, 0.8, 0.9]),
        approx([51.1, 1.2, 101.3]),
    ]
    assert grid.coord[2, 0].tolist() == [
        approx([101.4, 1.5, 1.6]),
        approx([101.7, 1.8, 101.9]),
    ]
    assert grid.coord[0, 1].tolist() == [
        approx([2.0, 52.1, 2.3]),
        approx([2.4, 52.5, 102.6]),
    ]
    assert grid.coord[1, 1].tolist() == [
        approx([52.7, 52.8, 2.9]),
        approx([53.0, 53.1, 103.2]),
    ]
    assert grid.coord[2, 1].tolist() == [
        approx([103.3, 53.4, 3.5]),
        approx([103.6, 53.7, 103.8]),
    ]
    assert grid.coord[0, 2].tolist() == [
        approx([3.9, 104.0, 4.1]),
        approx([4.2, 104.3, 104.4]),
    ]
    assert grid.coord[1, 2].tolist() == [
        approx([54.5, 104.6, 4.7]),
        approx([54.8, 104.9, 105.0]),
    ]
    assert grid.coord[2, 2].tolist() == [
        approx([105.1, 105.2, 5.3]),
        approx([105.4, 105.5, 105.6]),
    ]

    assert grid.zcorn.shape == (2, 2, 2, 8)
    # Order of heights for each corner is
    # (N(orth) means higher y, E(east) means higer x, T(op) means lower z (depth))
    # TSW TSE TNW TNE BSW BSE BNW BNE
    assert grid.zcorn[0, 0, 0, :].tolist() == approx(
        [0.0, 0.1, 0.4, 0.5, 50.0, 50.1, 50.4, 50.5],
    )
    assert grid.zcorn[1, 0, 0, :].tolist() == approx(
        [0.2, 0.3, 0.6, 0.7, 50.2, 50.3, 50.6, 50.7],
    )
    assert grid.zcorn[0, 1, 0, :].tolist() == approx(
        [0.8, 0.9, 1.2, 1.3, 50.8, 50.9, 51.2, 51.3],
    )
    assert grid.zcorn[1, 1, 0, :].tolist() == approx(
        [1.0, 1.1, 1.4, 1.5, 51.0, 51.1, 51.4, 51.5],
    )
    assert grid.zcorn[0, 0, 1, :].tolist() == approx(
        [51.6, 51.7, 52.0, 52.1, 100.0, 100.1, 100.4, 100.5],
    )
    assert grid.zcorn[1, 0, 1, :].tolist() == approx(
        [51.8, 51.9, 52.2, 52.3, 100.2, 100.3, 100.6, 100.7],
    )
    assert grid.zcorn[0, 1, 1, :].tolist() == approx(
        [52.4, 52.5, 52.8, 52.9, 100.8, 100.9, 101.2, 101.3],
    )
    assert grid.zcorn[1, 1, 1, :].tolist() == approx(
        [52.6, 52.7, 53.0, 53.1, 101.0, 101.1, 101.4, 101.5],
    )
    assert grid.map_axes == MapAxes(
        y_axis=approx((0.01, 1.01)),
        origin=approx((0.01, 0.01)),
        x_axis=approx((1.01, 0.01)),
    )

    assert grid.point_in_cell((25, 25, 25), 0, 0, 0)
    assert grid.find_cell_containing_point([(25, 25, 25)]) == [(0, 0, 0)]
    assert grid.find_cell_containing_point([(200, 200, 200)]) == [None]
    assert not grid.point_in_cell((225, 225, 225), 0, 0, 0)
