from itertools import product

import numpy as np
import pytest

from resfo_utilities import CornerpointGrid

TILT_FACTOR = 0.5
FAULT_THROW = 5.0

# Grid factories produce unit-cell grids (each cell is 1x1x1) so that test
# point coordinates are easy to reason about. Total grid depth equals nk.


def _make_regular_grid(ni, nj, nk):
    coord = np.zeros((ni + 1, nj + 1, 2, 3), dtype=np.float32)
    zcorn = np.zeros((ni, nj, nk, 8), dtype=np.float32)
    for i, j in product(range(ni + 1), range(nj + 1)):
        coord[i, j, 0] = [i, j, 0.0]
        coord[i, j, 1] = [i, j, float(nk)]
    for i, j, k in product(range(ni), range(nj), range(nk)):
        zcorn[i, j, k] = [k] * 4 + [k + 1] * 4
    return coord, zcorn


def _make_tilted_grid(ni, nj, nk):
    """Pillars lean TILT_FACTOR units in +x per unit of depth."""
    coord = np.zeros((ni + 1, nj + 1, 2, 3), dtype=np.float32)
    zcorn = np.zeros((ni, nj, nk, 8), dtype=np.float32)
    for i, j in product(range(ni + 1), range(nj + 1)):
        coord[i, j, 0] = [i, j, 0.0]
        coord[i, j, 1] = [i + TILT_FACTOR * nk, j, float(nk)]
    for i, j, k in product(range(ni), range(nj), range(nk)):
        zcorn[i, j, k] = [k] * 4 + [k + 1] * 4
    return coord, zcorn


def _make_faulted_grid(ni, nj, nk):
    """Cells with j >= nj//2 are thrown down by FAULT_THROW in z."""
    coord = np.zeros((ni + 1, nj + 1, 2, 3), dtype=np.float32)
    zcorn = np.zeros((ni, nj, nk, 8), dtype=np.float32)
    for i, j in product(range(ni + 1), range(nj + 1)):
        coord[i, j, 0] = [i, j, 0.0]
        coord[i, j, 1] = [i, j, float(nk) + FAULT_THROW]
    for i, j, k in product(range(ni), range(nj), range(nk)):
        offset = FAULT_THROW if j >= nj // 2 else 0.0
        zcorn[i, j, k] = [k + offset] * 4 + [k + 1 + offset] * 4
    return coord, zcorn


@pytest.fixture
def large_regular_grid():
    ni, nj, nk = 50, 50, 10
    coord, zcorn = _make_regular_grid(ni, nj, nk)
    return CornerpointGrid(coord, zcorn)


@pytest.fixture
def large_tilted_grid():
    ni, nj, nk = 50, 50, 10
    coord, zcorn = _make_tilted_grid(ni, nj, nk)
    return CornerpointGrid(coord, zcorn)


@pytest.fixture
def large_faulted_grid():
    ni, nj, nk = 50, 50, 10
    coord, zcorn = _make_faulted_grid(ni, nj, nk)
    return CornerpointGrid(coord, zcorn)


def test_benchmark_find_cell_regular_points_inside(large_regular_grid, benchmark):
    def run():
        assert large_regular_grid.find_cell_containing_point(
            [(i + 25.5, j + 25.5, 2.5) for i, j in product(range(10), range(10))],
        ) == [(i + 25, j + 25, 2) for i, j in product(range(10), range(10))]

    benchmark(run)


def test_benchmark_find_cell_regular_points_outside(large_regular_grid, benchmark):
    def run():
        assert large_regular_grid.find_cell_containing_point(
            [(i + 125.5, j + 125.5, 20.5) for i, j in product(range(10), range(10))],
        ) == [None for i, j in product(range(10), range(10))]

    benchmark(run)


def test_benchmark_find_cell_tilted_points_inside(large_tilted_grid, benchmark):
    def run():
        pts = [
            (i + 25.5 + TILT_FACTOR * 2.5, j + 25.5, 2.5)
            for i, j in product(range(10), range(10))
        ]
        results = large_tilted_grid.find_cell_containing_point(pts)
        assert all(r is not None for r in results)

    benchmark(run)


def test_benchmark_find_cell_tilted_points_outside(large_tilted_grid, benchmark):
    x_shift = (50 + TILT_FACTOR * 10) * 10
    pts = [(x_shift + i, j + 25.5, 2.5) for i, j in product(range(10), range(10))]

    def run():
        assert large_tilted_grid.find_cell_containing_point(
            pts,
        ) == [None for _ in pts]

    benchmark(run)


def test_benchmark_find_cell_faulted_points_inside(large_faulted_grid, benchmark):
    pts = [(i + 25.5, j + 20.5, 2.5) for i, j in product(range(5), range(5))] + [
        (i + 25.5, j + 25.5, 2.5 + FAULT_THROW) for i, j in product(range(5), range(5))
    ]

    def run():
        results = large_faulted_grid.find_cell_containing_point(
            pts,
        )
        assert all(r is not None for r in results)

    benchmark(run)


def test_benchmark_find_cell_faulted_points_outside(large_faulted_grid, benchmark):
    pts = [(i + 125.5, j + 125.5, 20.5) for i, j in product(range(10), range(10))]

    def run():
        assert large_faulted_grid.find_cell_containing_point(
            pts,
        ) == [None for _ in pts]

    benchmark(run)
