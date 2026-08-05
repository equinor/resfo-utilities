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
using BoolArray = py::array_t<bool, py::array::c_style | py::array::forcecast>;

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
    float tolerance, std::optional<BoolArray> cell_mask_array)
{
    auto g = validate_and_extract(points_array, coord_array, zcorn_array);

    const bool* cell_mask = nullptr;
    if (cell_mask_array.has_value()) {
        auto mask_buf = cell_mask_array->request();
        if (mask_buf.ndim != 3 || mask_buf.shape[0] != g.dims.ni
            || mask_buf.shape[1] != g.dims.nj || mask_buf.shape[2] != g.dims.nk)
            throw std::runtime_error("Cell mask array must have shape (ni, nj, nk)");
        cell_mask = static_cast<const bool*>(mask_buf.ptr);
    }

    if (g.num_points == 0) return std::vector<CellResult>();

    // The z-range the column bounding boxes must cover. With a cell mask,
    // only the selected cells contribute, giving tighter boxes.
    auto z_range = [&g, cell_mask]() -> std::optional<std::pair<float, float>> {
        if (g.zcorn_size == 0) return std::nullopt;
        if (cell_mask == nullptr) {
            auto [z_min, z_max] = std::minmax_element(g.zcorn, g.zcorn + g.zcorn_size);
            return std::make_pair(*z_min, *z_max);
        }
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();
        bool any_selected = false;
        const size_t num_cells = g.zcorn_size / resfo::NUM_CORNERS;
        for (size_t c = 0; c < num_cells; ++c) {
            if (!cell_mask[c]) continue;
            any_selected = true;
            const float* cell = g.zcorn + c * resfo::NUM_CORNERS;
            auto [lo, hi] = std::minmax_element(cell, cell + resfo::NUM_CORNERS);
            z_min = std::min(z_min, *lo);
            z_max = std::max(z_max, *hi);
        }
        if (!any_selected) return std::nullopt;
        return std::make_pair(z_min, z_max);
    }();

    if (!z_range.has_value()) return std::vector<CellResult>(g.num_points);

    std::vector<CellResult> results;
    results.reserve(g.num_points);

    resfo::ColumnIntervalTree tree(
        resfo::create_column_bounding_boxes(g.coord, g.dims, *z_range, cell_mask));
    Eigen::Map<const Eigen::Matrix3Xf> point_map(g.points, 3, g.num_points);
    for (size_t i = 0; i < g.num_points; ++i) {
        auto r = resfo::grid_search(
            point_map.col(i).cast<double>(), g.coord, g.zcorn, g.dims, tree, tolerance,
            cell_mask);
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
          py::arg("cell_mask") = py::none(),
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
