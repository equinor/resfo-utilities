#pragma once

#include <limits>
#include <utility>
#include <vector>

#include "grid.hpp"

namespace resfo {

// Bounding box of a column of cells defined by four pillars of the corner-point
// grid.
struct ColumnBoundingBox {
    resfo::CellIndex cell_index;

    float min_y = std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();

    float max_y = std::numeric_limits<float>::lowest();
    float max_x = std::numeric_limits<float>::lowest();

    bool overlaps_y(float y, float tol = 1e-6f) const {
        return (min_y - tol <= y && y <= max_y + tol);
    }
};

std::vector<ColumnBoundingBox> create_column_bounding_boxes(
    const float* coord,
    const resfo::GridDimensions& dims,
    const std::pair<float, float>& z_minmax
);

// 1D interval tree over ColumnBoundingBox with secondary-axis filtering.
//
// The X dimension is indexed with a classic interval tree (O(log n + k')
// stabbing query); each candidate is then filtered by Y containment. This
// works best under the assumption that the overlap of the pillar bounding
// boxes is small. If the bounding boxes' overlap is large, e.g. if the pillars
// are heavily tilted, the query of the ColumnIntervalTree degrades to a
// brute-force approach.
class ColumnIntervalTree {
private:
    struct Node {
        float mid;
        // Spanning intervals sorted by primary-axis min ascending  (used when
        // query <= mid).
        std::vector<ColumnBoundingBox> by_min;
        // Spanning intervals sorted by primary-axis max descending (used when
        // query >  mid).
        std::vector<ColumnBoundingBox> by_max;

        // Indices of left and right child nodes in the nodes_ vector, or -1
        // for no child.
        int left  = -1;
        int right = -1;
    };

    // Stores the tree structure in a vector for cache efficiency. The root
    // node is at index 0.
    std::vector<Node> nodes_;
    // When true the tree is built on the Y axis (x and y coords are swapped in
    // the stored bboxes).  The query() method swaps its x,y arguments before
    // descending the tree and the results are returned with (i,j) unmodified.
    bool transposed_ = false;

    // Returns index of the root node, or -1 for an empty tree.
    int build(std::vector<ColumnBoundingBox> boxes);
    void query_node(int idx, float x, float y, float tol,
                    std::vector<std::pair<int,int>>& results) const;

public:
    ColumnIntervalTree() = default;
    explicit ColumnIntervalTree(std::vector<ColumnBoundingBox> boxes);

    // Returns (i,j) indices of all pillar columns whose bounding box contains
    // (x,y).
    std::vector<std::pair<int,int>> query(float x, float y,
                                          float tolerance = 1.e-6f) const;
};

}  // namespace resfo
