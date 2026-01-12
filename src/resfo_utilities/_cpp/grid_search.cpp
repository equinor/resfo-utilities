#include "grid_search.hpp"

#include <unordered_set>
#include <queue>
#include <vector>
#include <functional>

#include "point_in_cell.hpp"

namespace resfo {

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const std::vector<float>& top, const std::vector<float>& bot, float tolerance,
    std::optional<std::pair<int, int>> prev_ij) {

    float bound_tol = 20.0*tolerance;

    if (dims.ni <= 0 || dims.nj <= 0) {
        return std::nullopt;
    }

    auto intersection = pillar_z_intersection(coord, dims, p[2]);

    std::priority_queue<QuadNode, std::vector<QuadNode>, std::greater<QuadNode>> queue;
    std::unordered_set<std::pair<int, int>, PairHash> visited;

    if (prev_ij.has_value()) {
        queue.emplace(prev_ij->first, prev_ij->second, p, 1, 1, intersection, dims);
    } else {
        queue.emplace(dims.ni / 2, dims.nj / 2, p, dims.ni / 2, dims.nj / 2, intersection,
                      dims);
    }

    visited.insert({queue.top().i, queue.top().j});

    while (!queue.empty()) {
        QuadNode node = queue.top();
        queue.pop();

        int i = node.i;
        int j = node.j;

        float dist_from_bounds = distance_from_bounds(p, top, bot, i, j, dims);
        if (dist_from_bounds <= 2 * bound_tol) {
            for (int k = 0; k < dims.nk; ++k) {
                int zcorn_idx = (i * dims.nj * dims.nk + j * dims.nk + k) * NUM_CORNERS;
                auto [z_min, z_max] = std::minmax_element(zcorn + zcorn_idx, zcorn + zcorn_idx + NUM_CORNERS);

                if (p[2] >= *z_min - 2 * bound_tol && p[2] <= *z_max + 2 * bound_tol) {
                    if (resfo::point_in_cell(p, i, j, k, coord, zcorn, dims, tolerance)) {
                        return CellIndex{i, j, k};
                    }
                }
            }
        }

        int size_i = node.i_neighbourhood;
        for (int di : {-1 * size_i, -1, 0, 1, size_i}) {
            int ni = i + di;
            if (ni < 0 || ni >= dims.ni) continue;

            int size_j = node.j_neighbourhood;
            for (int dj : {-1 * size_j, -1, 0, 1, size_j}) {
                int nj = j + dj;
                if (nj < 0 || nj >= dims.nj) continue;

                if (visited.find({ni, nj}) == visited.end()) {
                    queue.emplace(ni, nj, p, std::max(size_i / 2, 1), std::max(size_j / 2, 1),
                                  intersection, dims);
                    visited.insert({ni, nj});
                }
            }
        }
    }

    return std::nullopt;
}

}  // namespace resfo
