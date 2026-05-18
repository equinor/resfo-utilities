#pragma once

#include <algorithm>
#include <vector>
#include <Eigen/Dense>

// Tilt applied to every pillar in make_grid_tilted (units of x per unit of depth).
static constexpr float TILT_FACTOR = 0.5f;

// Fault throw applied to the downthrown block (j >= nj/2) in make_grid_faulted.
static constexpr float FAULT_THROW = 5.0f;

// ---------------------------------------------------------------------------
// Grid factories
// ---------------------------------------------------------------------------

// Regular grid: vertical pillars, unit cell spacing, nk uniform layers.
// coord layout: [(ni+1)*(nj+1)] pillars, each 6 floats [x,y,z_top, x,y,z_bot]
// zcorn layout: [ni*nj*nk*8] corner depths
inline void make_grid_regular(int ni, int nj, int nk,
                      std::vector<float>& coord,
                      std::vector<float>& zcorn)
{
    coord.resize((ni + 1) * (nj + 1) * 6);
    for (int i = 0; i <= ni; ++i) {
        for (int j = 0; j <= nj; ++j) {
            int p = (i * (nj + 1) + j) * 6;
            coord[p + 0] = static_cast<float>(i);
            coord[p + 1] = static_cast<float>(j);
            coord[p + 2] = 0.0f;
            coord[p + 3] = static_cast<float>(i);
            coord[p + 4] = static_cast<float>(j);
            coord[p + 5] = static_cast<float>(nk);
        }
    }

    zcorn.resize(ni * nj * nk * 8);
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            for (int k = 0; k < nk; ++k) {
                int base = (i * nj * nk + j * nk + k) * 8;
                float z_top = static_cast<float>(k);
                float z_bot = static_cast<float>(k + 1);
                for (int c = 0; c < 4; ++c) zcorn[base + c]     = z_top;
                for (int c = 0; c < 4; ++c) zcorn[base + 4 + c] = z_bot;
            }
        }
    }
}

// Tilted grid: pillars lean TILT_FACTOR units in +x per unit of depth.
// At depth z the x position of pillar i is: i + TILT_FACTOR * z.
// This causes pillar bounding boxes to overlap heavily in x, stressing 2D
// spatial indices.
inline void make_grid_tilted(int ni, int nj, int nk,
                              std::vector<float>& coord,
                              std::vector<float>& zcorn)
{
    coord.resize((ni + 1) * (nj + 1) * 6);
    for (int i = 0; i <= ni; ++i) {
        for (int j = 0; j <= nj; ++j) {
            int p = (i * (nj + 1) + j) * 6;
            coord[p + 0] = static_cast<float>(i);
            coord[p + 1] = static_cast<float>(j);
            coord[p + 2] = 0.0f;
            coord[p + 3] = static_cast<float>(i) + TILT_FACTOR * nk;
            coord[p + 4] = static_cast<float>(j);
            coord[p + 5] = static_cast<float>(nk);
        }
    }

    zcorn.resize(ni * nj * nk * 8);
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            for (int k = 0; k < nk; ++k) {
                int base = (i * nj * nk + j * nk + k) * 8;
                for (int c = 0; c < 4; ++c) zcorn[base + c]     = static_cast<float>(k);
                for (int c = 0; c < 4; ++c) zcorn[base + 4 + c] = static_cast<float>(k + 1);
            }
        }
    }
}

// Faulted grid: cells with j >= nj/2 are thrown down by FAULT_THROW in z.
// The upthrown block (j < nj/2) has z in [k, k+1]; the downthrown block
// (j >= nj/2) has z in [k + FAULT_THROW, k + 1 + FAULT_THROW].
// Pillar z_bot spans the full range so no pillar is truncated.
inline void make_grid_faulted(int ni, int nj, int nk,
                               std::vector<float>& coord,
                               std::vector<float>& zcorn)
{
    coord.resize((ni + 1) * (nj + 1) * 6);
    for (int i = 0; i <= ni; ++i) {
        for (int j = 0; j <= nj; ++j) {
            int p = (i * (nj + 1) + j) * 6;
            coord[p + 0] = static_cast<float>(i);
            coord[p + 1] = static_cast<float>(j);
            coord[p + 2] = 0.0f;
            coord[p + 3] = static_cast<float>(i);
            coord[p + 4] = static_cast<float>(j);
            coord[p + 5] = static_cast<float>(nk) + FAULT_THROW; // spans both blocks
        }
    }

    zcorn.resize(ni * nj * nk * 8);
    for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            float offset = (j >= nj / 2) ? FAULT_THROW : 0.0f;
            for (int k = 0; k < nk; ++k) {
                int base = (i * nj * nk + j * nk + k) * 8;
                float z_top = static_cast<float>(k) + offset;
                float z_bot = static_cast<float>(k + 1) + offset;
                for (int c = 0; c < 4; ++c) zcorn[base + c]     = z_top;
                for (int c = 0; c < 4; ++c) zcorn[base + 4 + c] = z_bot;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Point generators
// ---------------------------------------------------------------------------

// A set of query points scattered across a regular grid interior.
inline std::vector<Eigen::Vector3d> make_points_regular(int ni, int nj, int nk, int n_pts, int n_pts_outside)
{
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n_pts);
    for (int idx = 0; idx < n_pts; ++idx) {
        double fi = static_cast<double>(idx % ni) + 0.5;
        double fj = static_cast<double>((idx / ni) % nj) + 0.5;
        double fk = static_cast<double>((idx / (ni * nj)) % nk) + 0.5;
        pts.emplace_back(fi, fj, fk);
    }

    // Shift points outside of the grid
    const double x_shift = ni * 10.0;
    const double y_shift = nj * 10.0;
    const int n_pts_to_shift = std::clamp(n_pts_outside, 0, n_pts);
    for (int i = 0; i < n_pts_to_shift; ++i) {
        pts[i][0] += x_shift;
        pts[i][1] += y_shift;
    }

    return pts;
}


// Points at the cell centres of the tilted grid.
// At depth z = k + 0.5 the cell centre x is: ci + 0.5 + TILT_FACTOR * (k + 0.5).
inline std::vector<Eigen::Vector3d> make_points_tilted(int ni, int nj, int nk, int n_pts, int n_pts_outside)
{
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n_pts);
    for (int idx = 0; idx < n_pts; ++idx) {
        int ci = idx % ni;
        int cj = (idx / ni) % nj;
        int ck = (idx / (ni * nj)) % nk;
        double fk = ck + 0.5;
        double fx = ci + 0.5 + TILT_FACTOR * fk;
        pts.emplace_back(fx, cj + 0.5, fk);
    }

    const double x_shift = (ni + TILT_FACTOR * nk) * 10.0;
    const double y_shift = nj * 10.0;
    const int n_pts_to_shift = std::clamp(n_pts_outside, 0, n_pts);
    for (int i = 0; i < n_pts_to_shift; ++i) {
        pts[i][0] += x_shift;
        pts[i][1] += y_shift;
    }
    return pts;
}

// Points at the cell centres of the faulted grid (z accounts for fault throw).
inline std::vector<Eigen::Vector3d> make_points_faulted(int ni, int nj, int nk, int n_pts, int n_pts_outside)
{
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n_pts);
    for (int idx = 0; idx < n_pts; ++idx) {
        int ci = idx % ni;
        int cj = (idx / ni) % nj;
        int ck = (idx / (ni * nj)) % nk;
        double offset = (cj >= nj / 2) ? FAULT_THROW : 0.0;
        pts.emplace_back(ci + 0.5, cj + 0.5, ck + 0.5 + offset);
    }

    const double x_shift = ni * 10.0;
    const double y_shift = nj * 10.0;
    const int n_pts_to_shift = std::clamp(n_pts_outside, 0, n_pts);
    for (int i = 0; i < n_pts_to_shift; ++i) {
        pts[i][0] += x_shift;
        pts[i][1] += y_shift;
    }
    return pts;
}
