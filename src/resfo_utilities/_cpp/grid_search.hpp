#pragma once

#include <optional>

#include <Eigen/Dense>

#include "column_interval_tree.hpp"
#include "grid.hpp"

namespace resfo {

std::optional<CellIndex> grid_search(
    const Eigen::Vector3d& p, const float* coord, const float* zcorn, const GridDimensions& dims,
    const ColumnIntervalTree& tree, float tolerance,
    // Optional (ni, nj, nk) mask; only selected cells are searched.
    const bool* cell_mask = nullptr);

}  // namespace resfo
