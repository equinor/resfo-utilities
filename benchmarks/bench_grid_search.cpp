#include <benchmark/benchmark.h>
#include <vector>

#include "grid_search.hpp"
#include "column_interval_tree.hpp"
#include "bench_helpers.hpp"


using PointSet = std::vector<Eigen::Vector3d>(*)(int, int, int, int, int);

using GridFactory = void(*)(int, int, int, std::vector<float>&, std::vector<float>&);

template<GridFactory MakeGrid, PointSet MakePoints>
static void BM_GridSearch(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);
    const int n_pts = state.range(2);
    const int n_pts_outside = state.range(3);

    std::vector<float> coord, zcorn;
    MakeGrid(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    auto column_bboxes = resfo::create_column_bounding_boxes(coord.data(), dims, {0.0f, static_cast<float>(nk)});
    resfo::ColumnIntervalTree tree(std::move(column_bboxes));

    auto points = MakePoints(ni, nj, nk, n_pts, n_pts_outside);

    for (auto _ : state) {
        for (const auto& p : points) {
            auto r = resfo::grid_search(p, coord.data(), zcorn.data(), dims, tree, 1e-6f);
            benchmark::DoNotOptimize(r);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pts);
}

// Register with given name and two grid depths (10 and 50 cells). All grids
// have 50 cells in x- and y-direction.
#define REGISTER_GRID_SEARCH_BENCHMARK(name, grid_fn, pts_fn, n_pts, n_pts_outside) \
    BENCHMARK_TEMPLATE(BM_GridSearch, grid_fn, pts_fn)                              \
        ->Name(name)                                                                \
        ->Args({50, 10, n_pts, n_pts_outside})                                      \
        ->Args({50, 50, n_pts, n_pts_outside})

// All points inside
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/regular_grid/all_points_inside",
    make_grid_regular, make_points_regular, 100, 0
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/regular_grid/all_points_outside",
    make_grid_regular, make_points_regular, 100, 100
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/regular_grid/5_points_outside",
    make_grid_regular, make_points_regular, 100, 5
);

REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/tilted_grid/all_points_inside",
    make_grid_tilted, make_points_tilted, 100, 0
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/tilted_grid/all_points_outside",
    make_grid_tilted, make_points_tilted, 100, 100
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/tilted_grid/5_points_outside",
    make_grid_tilted, make_points_tilted, 100, 5
);

REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/faulted_grid/all_points_inside",
    make_grid_faulted, make_points_faulted, 100, 0
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/faulted_grid/all_points_outside",
    make_grid_faulted, make_points_faulted, 100, 100
);
REGISTER_GRID_SEARCH_BENCHMARK(
    "BM_GridSearch/faulted_grid/5_points_outside",
    make_grid_faulted, make_points_faulted, 100, 5
);

BENCHMARK_MAIN();
