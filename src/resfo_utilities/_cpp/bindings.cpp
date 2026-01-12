#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "grid_search.hpp"
#include "point_in_cell.hpp"
#include <Eigen/Dense>
namespace py = pybind11;

std::vector<std::optional<std::tuple<int, int, int>>> find_cells_containing_points(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> zcorn_array,
    float tolerance) {

    auto points_buf = points_array.request();
    auto coord_buf = coord_array.request();
    auto zcorn_buf = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("Points array must have shape (n, 3)");
    }

    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3) {
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    }

    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8) {
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");
    }

    const float* points = static_cast<const float*>(points_buf.ptr);
    const float* coord = static_cast<const float*>(coord_buf.ptr);
    const float* zcorn = static_cast<const float*>(zcorn_buf.ptr);

    auto zcorn_shape = zcorn_buf.shape;
    resfo::GridDimensions dims{
        static_cast<int>(zcorn_shape[0]),
        static_cast<int>(zcorn_shape[1]),
        static_cast<int>(zcorn_shape[2])
    };

    auto [z_min, z_max] = std::minmax_element(zcorn, zcorn + zcorn_buf.size);

    auto top_intersection = resfo::pillar_z_intersection(coord, dims, *z_min);
    auto bot_intersection = resfo::pillar_z_intersection(coord, dims, *z_max);

    size_t num_points = points_buf.shape[0];
    std::vector<std::optional<std::tuple<int, int, int>>> results;
    results.reserve(num_points);
    std::optional<std::pair<int, int>> prev_ij;

    for (size_t p_idx = 0; p_idx < num_points; ++p_idx) {
        Eigen::Vector3d p{
            points[p_idx * 3],
            points[p_idx * 3 + 1],
            points[p_idx * 3 + 2]
        };

        auto result = resfo::grid_search(
            p, coord, zcorn, dims, top_intersection, bot_intersection, tolerance,
            prev_ij);

        if (result.has_value()) {
            results.push_back(std::make_tuple(result->i, result->j, result->k));
            prev_ij = std::make_pair(result->i, result->j);
        } else {
            results.push_back(std::nullopt);
            prev_ij = std::nullopt;
        }
    }
    return results;
}

py::array_t<bool> point_in_cell_wrapper(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_array,
    int i, int j, int k,
    py::array_t<float, py::array::c_style | py::array::forcecast> coord_array,
    py::array_t<float, py::array::c_style | py::array::forcecast> zcorn_array,
    float tolerance) {

    auto points_buf = points_array.request();
    auto coord_buf = coord_array.request();
    auto zcorn_buf = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3) {
        throw std::runtime_error("Points array must have shape (n, 3)");
    }

    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3) {
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    }

    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8) {
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");
    }

    const float* points = static_cast<const float*>(points_buf.ptr);
    const float* coord = static_cast<const float*>(coord_buf.ptr);
    const float* zcorn = static_cast<const float*>(zcorn_buf.ptr);

    auto zcorn_shape = zcorn_buf.shape;
    resfo::GridDimensions dims{
        static_cast<int>(zcorn_shape[0]),
        static_cast<int>(zcorn_shape[1]),
        static_cast<int>(zcorn_shape[2])
    };

    size_t num_points = points_buf.shape[0];
    auto result = py::array_t<bool>(num_points);
    auto result_buf = result.request();
    bool* result_ptr = static_cast<bool*>(result_buf.ptr);

    for (size_t p_idx = 0; p_idx < num_points; ++p_idx) {
        Eigen::Vector3d p{
            points[p_idx * 3],
            points[p_idx * 3 + 1],
            points[p_idx * 3 + 2]
        };

        result_ptr[p_idx] = resfo::point_in_cell(p, i, j, k, coord, zcorn, dims, tolerance);
    }

    return result;
}

PYBIND11_MODULE(_grid_cpp, m) {
    m.doc() = "Fast C++ implementation of grid search algorithms";

    m.def("find_cells_containing_points", &find_cells_containing_points,
          py::arg("points"),
          py::arg("coord"),
          py::arg("zcorn"),
          py::arg("tolerance") = 1e-6f,
          "Find cells containing given points");

    m.def("point_in_cell", &point_in_cell_wrapper,
          py::arg("points"),
          py::arg("i"),
          py::arg("j"),
          py::arg("k"),
          py::arg("coord"),
          py::arg("zcorn"),
          py::arg("tolerance") = 1e-6f,
          "Check if points are in a specific cell");
}
