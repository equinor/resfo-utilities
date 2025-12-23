from itertools import product

import numpy as np
import pytest

from resfo_utilities import CornerpointGrid


@pytest.fixture
def large_regular_grid():
    ni, nj, nk = 50, 50, 10
    height = 100
    top_depth = 0.0
    bot_depth = top_depth + height
    coord = np.zeros((ni + 1, nj + 1, 2, 3), dtype=np.float32)
    zcorn = np.zeros((ni, nj, nk, 8), dtype=np.float32)
    for i, j in product(range(ni + 1), range(nj + 1)):
        coord[i, j, 0] = [i, j, top_depth]
        coord[i, j, 1] = [i, j, bot_depth]
    for i, j, k in product(range(ni), range(nj), range(nk)):
        zcorn[i, j, k] = [height * (k / nk)] * 4 + [height * ((k + 1) / nk)] * 4
    return CornerpointGrid(coord, zcorn)


def test_benchmark_find_cell(large_regular_grid, benchmark):
    def run():
        assert large_regular_grid.find_cell_containing_point(
            [(i + 25.5, j + 25.5, 20.5) for i, j in product(range(10), range(10))],
        ) == [(i + 25, j + 25, 2) for i, j in product(range(10), range(10))]

    benchmark(run)
