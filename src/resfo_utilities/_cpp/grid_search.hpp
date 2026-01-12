#pragma once

#include <cmath>
#include <optional>
#include <vector>
#include <algorithm>
#include <array>

#include <Eigen/Dense>

namespace resfo {

const int NUM_CORNERS = 8;

struct GridDimensions {
    int ni;
    int nj;
    int nk;
};

struct CellIndex {
    int i;
    int j;
    int k;
};

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

/* The GridSearch algorithm is a search of QuadNodes. Each QuadNode describes a
 * pillar at (i,j) in the grid (see the docstring CornerpointGrid). The QuadNodes
 * are visited in order of a heuristic. This heuristic is calculated by first
 * finding the intersection of the plane z=p[2] and the pillars. This gives
 * a quad Q(i,j). The heuristic is the manhattan distance from p to center(Q(i,j)).
 */
class QuadNode {
public:
    int i;
    int j;
    Eigen::Vector3d p;
    int i_neighbourhood;
    int j_neighbourhood;
    float distance;

    QuadNode(int i_, int j_, const Eigen::Vector3d& p_, int i_neigh, int j_neigh,
             const std::vector<float>& intersection, const GridDimensions& dims)
        : i(i_), j(j_), p(p_), i_neighbourhood(i_neigh), j_neighbourhood(j_neigh) {
        distance = distance_intersection_center(intersection, dims);
    }

    bool operator>(const QuadNode& other) const {
        return distance > other.distance;
    }

private:
    /* The distance from p to Q(i,j). */
    float distance_intersection_center(const std::vector<float>& intersection,
                                       const GridDimensions& dims) const {
        int idx_00 = (i * (dims.nj + 1) + j) * 2;
        int idx_10 = ((i + 1) * (dims.nj + 1) + j) * 2;
        int idx_11 = ((i + 1) * (dims.nj + 1) + (j + 1)) * 2;
        int idx_01 = (i * (dims.nj + 1) + (j + 1)) * 2;

        float center_x = (intersection[idx_00] + intersection[idx_10] +
                          intersection[idx_11] + intersection[idx_01]) * 0.25f;
        float center_y = (intersection[idx_00 + 1] + intersection[idx_10 + 1] +
                          intersection[idx_11 + 1] + intersection[idx_01 + 1]) * 0.25f;

        return std::abs(center_x - p[0]) + std::abs(center_y - p[1]);
    }
};

/* The intersection of the pillars defined by coord and the plane z*/
inline std::vector<float> pillar_z_intersection(
    const float* coord, const GridDimensions& dims, float z) {
    int num_pillars = (dims.ni + 1) * (dims.nj + 1);
    std::vector<float> result(2*num_pillars);

    for (int idx = 0; idx < num_pillars; ++idx) {
        int base = idx * 6;
        float x1 = coord[base];
        float y1 = coord[base + 1];
        float z1 = coord[base + 2];
        float x2 = coord[base + 3];
        float y2 = coord[base + 4];
        float z2 = coord[base + 5];

        float t = (z - z1) / (z2 - z1);
        result[idx*2] = x1 + t * (x2 - x1);
        result[idx*2+1] = y1 + t * (y2 - y1);
    }

    return result;
}

inline float distance_from_bounds(const Eigen::Vector3d& p, const std::vector<float>& top,
                                  const std::vector<float>& bot, int i, int j,
                                  const GridDimensions& dims) {
    int idx_00 = (i * (dims.nj + 1) + j) * 2;
    int idx_10 = ((i + 1) * (dims.nj + 1) + j) * 2;
    int idx_11 = ((i + 1) * (dims.nj + 1) + (j + 1)) * 2;
    int idx_01 = (i * (dims.nj + 1) + (j + 1)) * 2;


    std::array<float, NUM_CORNERS> vertices_x{
        top[idx_00],     top[idx_10],     top[idx_11],     top[idx_01],
        bot[idx_00],     bot[idx_10],     bot[idx_11],     bot[idx_01]
    };
    std::array<float, NUM_CORNERS> vertices_y{
        top[idx_00 + 1], top[idx_10 + 1], top[idx_11 + 1], top[idx_01 + 1],
        bot[idx_00 + 1], bot[idx_10 + 1], bot[idx_11 + 1], bot[idx_01 + 1]
    };

    auto [min_x, max_x] = std::minmax_element(vertices_x.begin(), vertices_x.end());
    auto [min_y, max_y] = std::minmax_element(vertices_y.begin(), vertices_y.end());

    float x_dist = std::max({*min_x - p[0], p[0] - *max_x, 0.0});
    float y_dist = std::max({*min_y - p[1], p[1] - *max_y, 0.0});

    return x_dist + y_dist;
}

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const std::vector<float>& top, const std::vector<float>& bot, float tolerance,
    std::optional<std::pair<int, int>> prev_ij);

}  // namespace resfo
