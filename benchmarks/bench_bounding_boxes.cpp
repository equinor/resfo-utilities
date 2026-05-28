#include <benchmark/benchmark.h>
#include <vector>

#include "column_interval_tree.hpp"
#include "grid.hpp"
#include "bench_helpers.hpp"

static void BM_CreateColumnBoundingBoxes(benchmark::State& state)
{
    const int ni = state.range(0);
    const int nj = state.range(0);
    const int nk = state.range(1);

    std::vector<float> coord, zcorn;
    make_grid_regular(ni, nj, nk, coord, zcorn);
    resfo::GridDimensions dims{ni, nj, nk};

    for (auto _ : state) {
        auto boxes = resfo::create_column_bounding_boxes(coord.data(), dims, {0.0f, static_cast<float>(nk)});
        benchmark::DoNotOptimize(boxes);
    }

    state.SetItemsProcessed(state.iterations() * ni * nj);
}

BENCHMARK(BM_CreateColumnBoundingBoxes)
    ->Args({50, 10})
    ->Args({50, 50})
    ->Args({100, 10})
    ->Args({100, 50});

BENCHMARK_MAIN();
