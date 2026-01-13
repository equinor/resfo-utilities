#include "point_in_cell.hpp"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include "ceres/ceres.h"

namespace resfo {

/*
 * To determine whether the point is a given cell,
 * the inverse of a trilinear hexahedral (Q1) finite-element mapping
 * is solved. If it has a solution then the point is in the cell.
 * see "The Finite Element Method: A Practical Course" G. R. Liu & S. S. Quek chapter 9.3.
 */

struct HexInverseCost : ceres::CostFunction {
    private:
    const std::vector<double> corners;  // The corners of the cell, size 24 (8 × 3)
    Eigen::Vector3d point;


    static constexpr std::array<std::array<int,3>,NUM_CORNERS> reference_cube_corners = {{
        {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {-1, 1, 1},  {1, 1, 1}
    }};

    public:
    HexInverseCost(std::vector<double>&& v, const Eigen::Vector3d& p)
        : corners(std::move(v)), point(p) {
        set_num_residuals(3);
        *mutable_parameter_block_sizes() = {3};
        }

    bool Evaluate(
        const double* const* parameters,
        double* residuals,
        double** jacobians
    ) const override {
        const double xi   = parameters[0][0];
        const double eta  = parameters[0][1];
        const double zeta = parameters[0][2];

        Eigen::Vector3d mapped(0.0, 0.0, 0.0);

        for (int v = 0; v < NUM_CORNERS; ++v) {
            double shape =
                0.125 * (1 + xi   * reference_cube_corners[v][0]) *
                        (1 + eta  * reference_cube_corners[v][1]) *
                        (1 + zeta * reference_cube_corners[v][2]);

            mapped[0] += shape * corners[v * 3 + 0];
            mapped[1] += shape * corners[v * 3 + 1];
            mapped[2] += shape * corners[v * 3 + 2];
        }

        Eigen::Vector3d r = mapped - point;
        residuals[0] = r[0];
        residuals[1] = r[1];
        residuals[2] = r[2];

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();

            for (int v = 0; v < NUM_CORNERS; ++v) {
                // N is here the shape function
                double dN_dxi =
                    0.125 * reference_cube_corners[v][0] *
                    (1 + eta  * reference_cube_corners[v][1]) *
                    (1 + zeta * reference_cube_corners[v][2]);

                double dN_deta =
                    0.125 * reference_cube_corners[v][1] *
                    (1 + xi   * reference_cube_corners[v][0]) *
                    (1 + zeta * reference_cube_corners[v][2]);

                double dN_dzeta =
                    0.125 * reference_cube_corners[v][2] *
                    (1 + xi   * reference_cube_corners[v][0]) *
                    (1 + eta  * reference_cube_corners[v][1]);

                for (int dim = 0; dim < 3; ++dim) {
                    J(dim, 0) += dN_dxi   * corners[v * 3 + dim];
                    J(dim, 1) += dN_deta  * corners[v * 3 + dim];
                    J(dim, 2) += dN_dzeta * corners[v * 3 + dim];
                }
            }
        }

        return true;
    }
};

std::vector<double> cell_corners(int i, int j, int k, const float* coord, const float* zcorn,
                                 const GridDimensions& dims) {
    std::vector<double> vertices(24, 0.0);

    // Pillar indices for the four corners of the cell (i,j)
    // (i,j), (i,j+1), (i+1,j), (i+1,j+1)
    int pillar_idx[4] = {
        (i * (dims.nj + 1) + j) * 6,
        (i * (dims.nj + 1) + (j + 1)) * 6,
        ((i + 1) * (dims.nj + 1) + j) * 6,
        ((i + 1) * (dims.nj + 1) + (j + 1)) * 6
    };

    std::array<float, 3> top[4], bot[4];
    for (int p = 0; p < 4; ++p) {
        top[p] = {coord[pillar_idx[p]], coord[pillar_idx[p] + 1], coord[pillar_idx[p] + 2]};
        bot[p] = {coord[pillar_idx[p] + 3], coord[pillar_idx[p] + 4], coord[pillar_idx[p] + 5]};
    }

    int zcorn_idx = (i * dims.nj * dims.nk + j * dims.nk + k) * NUM_CORNERS;
    std::array<float, NUM_CORNERS> z_values;
    std::copy_n(zcorn + zcorn_idx, NUM_CORNERS, z_values.begin());

    // The zcorn ordering is: TSW, TSE, TNW, TNE, BSW, BSE, BNW, BNE
    // where SW = (i,j), SE = (i+1,j), NW = (i,j+1), NE = (i+1,j+1)
    // Map zcorn index to pillar index
    std::array<int, NUM_CORNERS> pillar_order = {0, 2, 1, 3, 0, 2, 1, 3};

    for (int v = 0; v < NUM_CORNERS; ++v) {
        int p_idx = pillar_order[v];
        const auto& p_top = top[p_idx];
        const auto& p_bot = bot[p_idx];

        float height_diff = p_bot[2] - p_top[2];
        float t = (z_values[v] - p_top[2]) / height_diff;

        vertices[v * 3] = p_top[0] + t * (p_bot[0] - p_top[0]);
        vertices[v * 3 + 1] = p_top[1] + t * (p_bot[1] - p_top[1]);
        vertices[v * 3 + 2] = z_values[v];
    }

    return vertices;
}

bool point_in_cell(const Eigen::Vector3d& point, int i, int j, int k, const float* coord,
                   const float* zcorn, const GridDimensions& dims, float tolerance) {

    auto corners = cell_corners(i, j, k, coord, zcorn, dims);
    double xi_eta_zeta[3] = {0.0, 0.0, 0.0}; // initial guess

    ceres::Problem problem;

    problem.AddParameterBlock(xi_eta_zeta, 3);

    // Enforce [-1,1] bounds
    for (int i = 0; i < 3; ++i) {
        problem.SetParameterLowerBound(xi_eta_zeta, i, -1.0);
        problem.SetParameterUpperBound(xi_eta_zeta, i,  1.0);
    }


    HexInverseCost* cost_function = new HexInverseCost(std::move(corners), point);

    problem.AddResidualBlock(
        cost_function,
        nullptr, // no robust loss needed
        xi_eta_zeta
    );

    double tolerance_squared = static_cast<double>(tolerance) * static_cast<double>(tolerance);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    options.function_tolerance = tolerance_squared;
    options.gradient_tolerance = tolerance_squared;
    options.parameter_tolerance = tolerance_squared;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return
        summary.termination_type == ceres::CONVERGENCE &&
        summary.final_cost < tolerance;
}

}  // namespace resfo
