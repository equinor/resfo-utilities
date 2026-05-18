#include <benchmark/benchmark.h>
#include <utility>
#include <vector>

#include "grid_search.hpp"
#include "bench_helpers.hpp"

using CellResult = std::optional<std::tuple<int, int, int>>;

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

    auto points = MakePoints(ni, nj, nk, n_pts, n_pts_outside);

    auto [z_min, z_max] = std::minmax_element(zcorn.begin(), zcorn.end());
    auto top = resfo::pillar_z_intersection(coord.data(), dims, *z_min);
    auto bot = resfo::pillar_z_intersection(coord.data(), dims, *z_max);

    std::vector<CellResult> results;
    results.reserve(points.size());
    std::optional<std::pair<int, int>> prev_ij;

    for (auto _ : state) {
        for (const auto& p : points) {
            auto r = resfo::grid_search(p, coord.data(), zcorn.data(), dims, top, bot, 1e-6f, prev_ij);
            if (r.has_value())
                results.push_back(std::make_tuple(r->i, r->j, r->k));
            else
                results.push_back(std::nullopt);

            prev_ij = r ? std::make_optional(std::make_pair(r->i, r->j)) : std::nullopt;
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
