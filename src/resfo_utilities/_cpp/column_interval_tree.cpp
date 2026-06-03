#include "column_interval_tree.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace resfo {

// Compute (x, y) bounding boxes for each column by interpolating the four
// corner pillars at the zcorn depth range. Pillars are lines in 3D space, so
// the (x, y) extent of a column varies with depth. Using the full zcorn
// z-range ensures the bounding boxes enclose the pillar positions across the
// entire grid depth.
std::vector<ColumnBoundingBox> create_column_bounding_boxes(const float* coord,
                                                            const resfo::GridDimensions& dims,
                                                            const std::pair<float, float>& z_minmax) {
    std::vector<ColumnBoundingBox> boxes;
    boxes.reserve(static_cast<size_t>(dims.ni) * dims.nj);
    const auto& [z_min, z_max] = z_minmax;

    for (int i = 0; i < dims.ni; ++i) {
        for (int j = 0; j < dims.nj; ++j) {
            const std::array<int, 4> pillar_idx{
                (i * (dims.nj + 1) + j) *  6,
                (i * (dims.nj + 1) + (j + 1)) * 6,
                ((i + 1) * (dims.nj + 1) + j) * 6,
                ((i + 1) * (dims.nj + 1) + (j + 1)) * 6
            };

            ColumnBoundingBox box;
            box.cell_index = {i, j, 0};

            auto update_bounds = [&box](const std::pair<float, float>& xy) {
                const auto& [x, y] = xy;
                box.min_x = std::min(box.min_x, x);
                box.max_x = std::max(box.max_x, x);
                box.min_y = std::min(box.min_y, y);
                box.max_y = std::max(box.max_y, y);
            };

            for (int v = 0; v < 4; ++v) {
                int idx = pillar_idx[v];

                float x1 = coord[idx];
                float y1 = coord[idx + 1];
                float z1 = coord[idx + 2];
                float x2 = coord[idx + 3];
                float y2 = coord[idx + 4];
                float z2 = coord[idx + 5];

                // Check difference to avoid division by close-to-zero numbers
                float dz = z2 - z1;
                if (std::abs(dz) <= std::numeric_limits<float>::epsilon() * std::max(std::abs(z1), std::abs(z2))) {
                    update_bounds({x1, y1});
                    continue;
                }

                auto pillar_x_y_at = [&](float z) {
                    float t = (z - z1) / dz;
                    float x = x1 + t * (x2 - x1);
                    float y = y1 + t * (y2 - y1);
                    return std::pair<float, float>{x, y};
                };

                update_bounds(pillar_x_y_at(z_min));
                update_bounds(pillar_x_y_at(z_max));
            }

            boxes.push_back(std::move(box));
        }
    }

    return boxes;
}


// When transposed_ is true, the x and y fields in each ColumnBoundingBox
// have been swapped so that the primary tree axis is always "x".
int ColumnIntervalTree::build(std::vector<ColumnBoundingBox> boxes) {
    if (boxes.empty()) return -1;

    // Midpoint = median of all interval midpoints to keep the tree balanced.
    std::vector<float> mids;
    mids.reserve(boxes.size());
    for (const auto& b : boxes)
        mids.push_back((b.min_x + b.max_x) * 0.5f);
    std::nth_element(mids.begin(), mids.begin() + mids.size() / 2, mids.end());
    float mid = mids[mids.size() / 2];

    std::vector<ColumnBoundingBox> left_boxes, right_boxes, spanning;
    for (auto& b : boxes) {
        if (b.max_x < mid)      left_boxes.push_back(std::move(b));
        else if (b.min_x > mid) right_boxes.push_back(std::move(b));
        else                    spanning.push_back(std::move(b));
    }
    // All content has been moved out of boxes by now, but clear to avoid
    // accidental access to moved content.
    boxes.clear();

    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    nodes_[idx].mid    = mid;
    // Make one copy of the spanning set because we need two sets sorted by
    // different keys for efficient querying.
    nodes_[idx].by_min = spanning;
    nodes_[idx].by_max = std::move(spanning);
    std::sort(nodes_[idx].by_min.begin(), nodes_[idx].by_min.end(),
              [](const auto& a, const auto& b) { return a.min_x < b.min_x; });
    std::sort(nodes_[idx].by_max.begin(), nodes_[idx].by_max.end(),
              [](const auto& a, const auto& b) { return a.max_x > b.max_x; });

    nodes_[idx].left  = build(std::move(left_boxes));
    nodes_[idx].right = build(std::move(right_boxes));
    return idx;
}

// x and y here refer to the (possibly transposed) coordinates stored in each
// ColumnBoundingBox — see build() and the constructor for details.
void ColumnIntervalTree::query_node(int idx, float x, float y, float tol,
                                    std::vector<std::pair<int,int>>& results) const {
    if (idx == -1) return;
    const Node& node = nodes_[idx];

    if (x <= node.mid) {
        // All spanning intervals have max_x >= mid >= x, so containment in X reduces
        // to min_x <= x + tol. The list is sorted by min_x ascending: stop at first miss.
        for (const auto& b : node.by_min) {
            if (b.min_x > x + tol) break;
            if (b.overlaps_y(y, tol))
                 results.emplace_back(b.cell_index.i, b.cell_index.j);
        }
        query_node(node.left, x, y, tol, results);
    } else {
        // All spanning intervals have min_x <= mid < x, so containment in X reduces
        // to max_x >= x - tol. The list is sorted by max_x descending: stop at first miss.
        for (const auto& b : node.by_max) {
            if (b.max_x < x - tol) break;
            if (b.overlaps_y(y, tol))
                results.emplace_back(b.cell_index.i, b.cell_index.j);
        }
        query_node(node.right, x, y, tol, results);
    }
}

ColumnIntervalTree::ColumnIntervalTree(std::vector<ColumnBoundingBox> boxes) {
    if (boxes.empty()) return;

    // Choose the primary tree axis as whichever has the smaller total bbox
    // extent — this minimises the size of spanning sets and therefore the
    // number of candidates that reach the secondary-axis linear scan.
    // For a tilted grid (lean in x), x-extents grow with tilt*depth while
    // y-extents stay unit-width, so building on y is much more efficient.
    float sum_x = 0.f, sum_y = 0.f;
    for (const auto& b : boxes) {
        sum_x += b.max_x - b.min_x;
        sum_y += b.max_y - b.min_y;
    }
    transposed_ = (sum_y < sum_x);
    if (transposed_) {
        for (auto& b : boxes) {
            std::swap(b.min_x, b.min_y);
            std::swap(b.max_x, b.max_y);
        }
    }

    nodes_.reserve(boxes.size());
    build(std::move(boxes));
}

std::vector<std::pair<int,int>> ColumnIntervalTree::query(float x, float y, float tolerance) const {
    std::vector<std::pair<int,int>> results;
    if (nodes_.empty()) return results;
    // Swap coordinates to match the axis the tree was built on.
    if (transposed_) std::swap(x, y);
    query_node(0, x, y, tolerance, results);
    return results;
}

}  // namespace resfo
