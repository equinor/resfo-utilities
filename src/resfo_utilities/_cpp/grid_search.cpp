#include "grid_search.hpp"

#include <array>
#include <limits>
#include <vector>

#include "point_in_cell.hpp"

namespace resfo {

struct Candidate {
    int i, j, k;
    float z_dist;
};

// Gather candidate cells from the given columns, filtering by z-range.
static std::vector<Candidate> gather_z_candidates(
    const std::vector<std::pair<int, int>>& columns,
    const float* zcorn, const GridDimensions& dims,
    float pz, float bound_tol)
{
    std::vector<Candidate> candidates;
    candidates.reserve(columns.size() * 2);

    for (const auto& [ci, cj] : columns) {
        for (int k = 0; k < dims.nk; ++k) {
            int zcorn_idx = (ci * dims.nj * dims.nk + cj * dims.nk + k) * NUM_CORNERS;
            auto [z_min_it, z_max_it] = std::minmax_element(
                zcorn + zcorn_idx, zcorn + zcorn_idx + NUM_CORNERS);
            if (pz >= *z_min_it - 2 * bound_tol && pz <= *z_max_it + 2 * bound_tol) {
                float z_center = (*z_min_it + *z_max_it) * 0.5f;
                candidates.emplace_back(Candidate{ci, cj, k, std::abs(z_center - pz)});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.z_dist < b.z_dist; });
    return candidates;
}

// Test candidates in z-distance order, returning the first match.
static std::optional<CellIndex> test_candidates(
    const std::vector<Candidate>& candidates,
    const Eigen::Vector3d& p, const float* coord, const float* zcorn,
    const GridDimensions& dims, float tolerance)
{
    for (const auto& c : candidates) {
        if (resfo::point_in_cell(p, c.i, c.j, c.k, coord, zcorn, dims, tolerance)) {
            return CellIndex{c.i, c.j, c.k};
        }
    }
    return std::nullopt;
}

// Order candidate columns by distance from the (x,y) bounding box at query
// depth. Each column is defined by 4 pillars; we interpolate their (x,y) at
// the query z and compute how far the point lies from the resulting bbox.
// Columns are sorted by this distance so that the most likely hits are tested
// first. No column is discarded — the trilinear cell interior can extend
// beyond the pillar-corner bounding box at a given depth.
static void order_columns_by_depth_distance(
    std::vector<std::pair<int, int>>& columns,
    const float* coord, const GridDimensions& dims,
    float px, float py, float pz, float tol)
{
    std::vector<std::pair<float, int>> distances;
    distances.reserve(columns.size());

    for (int idx = 0; idx < static_cast<int>(columns.size()); ++idx) {
        const auto& [ci, cj] = columns[idx];
        const std::array<int, 4> pillar_idx = {
            (ci * (dims.nj + 1) + cj) * 6,
            (ci * (dims.nj + 1) + (cj + 1)) * 6,
            ((ci + 1) * (dims.nj + 1) + cj) * 6,
            ((ci + 1) * (dims.nj + 1) + (cj + 1)) * 6
        };

        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        for (int v = 0; v < 4; ++v) {
            int base = pillar_idx[v];
            float z_top = coord[base + 2];
            float z_bot = coord[base + 5];

            float dz = z_bot - z_top;
            float x_at_z, y_at_z;
            if (std::abs(dz) <= std::numeric_limits<float>::epsilon() * std::max(std::abs(z_top), std::abs(z_bot))) {
                x_at_z = coord[base];
                y_at_z = coord[base + 1];
            } else {
                float t = (pz - z_top) / dz;
                x_at_z = coord[base]     + t * (coord[base + 3] - coord[base]);
                y_at_z = coord[base + 1] + t * (coord[base + 4] - coord[base + 1]);
            }

            min_x = std::min(min_x, x_at_z);
            max_x = std::max(max_x, x_at_z);
            min_y = std::min(min_y, y_at_z);
            max_y = std::max(max_y, y_at_z);
        }

        float dx = std::max(0.f, min_x - tol - px) + std::max(0.f, px - max_x - tol);
        float dy = std::max(0.f, min_y - tol - py) + std::max(0.f, py - max_y - tol);
        distances.push_back({dx + dy, idx});
    }

    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<int, int>> sorted;
    sorted.reserve(columns.size());
    for (const auto& [_, idx] : distances) {
        sorted.push_back(columns[idx]);
    }
    columns = std::move(sorted);
}

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const ColumnIntervalTree& tree, float tolerance) {

    float bound_tol = 20.0f * tolerance;

    if (dims.ni <= 0 || dims.nj <= 0 || dims.nk <= 0) {
        return std::nullopt;
    }

    auto columns = tree.query(static_cast<float>(p[0]), static_cast<float>(p[1]), bound_tol);
    order_columns_by_depth_distance(
        columns, coord, dims,
        static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
        bound_tol);

    const float pz = static_cast<float>(p[2]);
    auto candidates = gather_z_candidates(columns, zcorn, dims, pz, bound_tol);
    return test_candidates(candidates, p, coord, zcorn, dims, tolerance);
}

}  // namespace resfo
