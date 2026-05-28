#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "column_interval_tree.hpp"
#include "grid_search.hpp"
#include "point_in_cell.hpp"

#include <Eigen/Dense>
namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Validated grid arrays extracted from numpy buffers.
struct GridArrays {
    const float* points;
    const float* coord;
    const float* zcorn;
    resfo::GridDimensions dims;
    size_t num_points;
    size_t zcorn_size;
};

static GridArrays validate_and_extract(
    FloatArray& points_array, FloatArray& coord_array, FloatArray& zcorn_array)
{
    auto points_buf = points_array.request();
    auto coord_buf  = coord_array.request();
    auto zcorn_buf  = zcorn_array.request();

    if (points_buf.ndim != 2 || points_buf.shape[1] != 3)
        throw std::runtime_error("Points array must have shape (n, 3)");
    if (coord_buf.ndim != 4 || coord_buf.shape[2] != 2 || coord_buf.shape[3] != 3)
        throw std::runtime_error("Coord array must have shape (ni+1, nj+1, 2, 3)");
    if (zcorn_buf.ndim != 4 || zcorn_buf.shape[3] != 8)
        throw std::runtime_error("Zcorn array must have shape (ni, nj, nk, 8)");

    return GridArrays{
        static_cast<const float*>(points_buf.ptr),
        static_cast<const float*>(coord_buf.ptr),
        static_cast<const float*>(zcorn_buf.ptr),
        resfo::GridDimensions{
            static_cast<int>(zcorn_buf.shape[0]),
            static_cast<int>(zcorn_buf.shape[1]),
            static_cast<int>(zcorn_buf.shape[2])
        },
        static_cast<size_t>(points_buf.shape[0]),
        static_cast<size_t>(zcorn_buf.size)
    };
}

using CellResult = std::optional<std::tuple<int, int, int>>;

static CellResult to_result(const std::optional<resfo::CellIndex>& r) {
    if (r.has_value())
        return std::make_tuple(r->i, r->j, r->k);
    return std::nullopt;
}

std::vector<CellResult> find_cells_containing_points(
    FloatArray points_array, FloatArray coord_array, FloatArray zcorn_array,
    float tolerance)
{
    auto g = validate_and_extract(points_array, coord_array, zcorn_array);

    if (g.num_points == 0) return std::vector<CellResult>();

    auto build_interval_tree = [&g]() {
        auto [z_min, z_max] = std::minmax_element(g.zcorn, g.zcorn + g.zcorn_size);
        auto bboxes = resfo::create_column_bounding_boxes(g.coord, g.dims, {*z_min, *z_max});
        return resfo::ColumnIntervalTree(std::move(bboxes));
    };

    std::vector<CellResult> results;
    results.reserve(g.num_points);

    auto tree = build_interval_tree();
    Eigen::Map<const Eigen::Matrix3Xf> point_map(g.points, 3, g.num_points);
    for (size_t i = 0; i < g.num_points; ++i) {
        auto r = resfo::grid_search(
            point_map.col(i).cast<double>(), g.coord, g.zcorn, g.dims, tree, tolerance);
        results.push_back(to_result(r));
    }
    return results;
}

py::array_t<bool> point_in_cell_wrapper(
    FloatArray points_array, int i, int j, int k,
    FloatArray coord_array, FloatArray zcorn_array,
    float tolerance)
{
    auto g = validate_and_extract(points_array, coord_array, zcorn_array);

    auto result = py::array_t<bool>(g.num_points);
    auto result_buf = result.request();
    bool* result_ptr = static_cast<bool*>(result_buf.ptr);

    Eigen::Map<const Eigen::Matrix3Xf> point_map(g.points, 3, g.num_points);
    for (size_t idx = 0; idx < g.num_points; ++idx) {
        result_ptr[idx] = resfo::point_in_cell(
            point_map.col(idx).cast<double>(), i, j, k, g.coord, g.zcorn, g.dims, tolerance);
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
