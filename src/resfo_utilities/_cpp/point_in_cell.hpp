#pragma once
#include "grid_search.hpp"

#include <Eigen/Dense>

namespace resfo {

bool point_in_cell(const Eigen::Vector3d& point, int i, int j, int k, const float* coord,
                   const float* zcorn, const GridDimensions& dims, float tolerance);

}  // namespace resfo
