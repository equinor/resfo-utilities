#pragma once

namespace resfo {

constexpr int NUM_CORNERS = 8;

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

} /* namespace resfo */
