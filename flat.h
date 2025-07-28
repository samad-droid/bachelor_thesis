#pragma once

// ===== core =====
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>

// ===== polyscope =====
// adjust the include paths to wherever polyscope lives in your tree
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include "external/polyscope/include/polyscope/curve_network.h"
#include <glm/glm.hpp>

// ------------------------------------------------------------
// Flat<T> (moved from your working version, but cleaned a bit)
// ------------------------------------------------------------
template <typename Scalar = double>
class Flat {
public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // ctor
    Flat(const Vector& origin, const Matrix& basis)
        : origin_(origin), basis_(basis) {
        assert(basis_.cols() <= basis_.rows());
    }

    // accessors
    const Vector& origin() const { return origin_; }
    const Matrix& basis()  const { return basis_; }

    // dimensions
    int dimension() const        { return static_cast<int>(basis_.cols()); }
    int ambientDimension() const { return static_cast<int>(basis_.rows()); }

    // membership test
    bool contains(const Vector& point, Scalar eps = Scalar(1e-8)) const {
        Vector diff      = point - origin_;
        Vector proj      = basis_.transpose() * diff;
        Vector flatPoint = origin_ + basis_ * proj;
        return (point - flatPoint).norm() < eps;
    }

    // orthogonal projection
    Vector project(const Vector& point) const {
        Vector diff  = point - origin_;
        Vector proj  = basis_.transpose() * diff;
        return origin_ + basis_ * proj;
    }

    // from local coordinates to ambient point
    Vector pointFromCoords(const Vector& coords) const {
        assert(coords.size() == basis_.cols());
        return origin_ + basis_ * coords;
    }

    void print() const {
        std::cout << "Flat of dimension " << dimension()
                  << " in ambient dimension " << ambientDimension() << "\n";
        std::cout << "Origin:\n" << origin_.transpose() << "\n";
        std::cout << "Basis vectors (columns):\n" << basis_ << "\n";
    }

private:
    Vector origin_;
    Matrix basis_;
};

// ------------------------------------------------------------
// 1) Sample N noisy points from an arbitrary-dimensional flat
// ------------------------------------------------------------
template <typename Scalar = double, class URNG>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
generateNoisyFlatSamples(const Flat<Scalar>& flat,
                         int N,
                         Scalar coordExtent,   // sample local coords u in [-coordExtent, coordExtent]^k
                         Scalar noiseStd,      // Gaussian noise std-dev in the ambient space
                         URNG& rng) {
    using namespace Eigen;

    const int k = flat.dimension();
    const int n = flat.ambientDimension();

    if (N <= 0) throw std::invalid_argument("N must be > 0");
    if (k <= 0) throw std::invalid_argument("Flat dimension must be > 0");

    std::uniform_real_distribution<Scalar> uni(-coordExtent, coordExtent);
    std::normal_distribution<Scalar> gauss(Scalar(0), noiseStd);

    Matrix<Scalar, Dynamic, Dynamic> points(N, n);

    for (int i = 0; i < N; ++i) {
        // sample local coordinates
        Matrix<Scalar, Dynamic, 1> local(k);
        for (int j = 0; j < k; ++j) local(j) = uni(rng);

        // map to ambient
        Matrix<Scalar, Dynamic, 1> p = flat.pointFromCoords(local);

        // add noise in ambient coords
        for (int j = 0; j < n; ++j) p(j) += gauss(rng);

        points.row(i) = p.transpose();
    }

    return points;
}
//_____________________________________________________________
//2) Sample N noisy planes from an arbitrary-dimensional flat
//
template <typename Scalar = double, class URNG>
std::vector<Flat<Scalar>> generateRandomFlats(int numFlats,
                                              int ambientDim,
                                              int flatDim,
                                              Scalar originSpread,
                                              URNG& rng) {
    using namespace Eigen;
    using Vec = Matrix<Scalar, Dynamic, 1>;
    using Mat = Matrix<Scalar, Dynamic, Dynamic>;

    if (numFlats <= 0) throw std::invalid_argument("numFlats must be > 0");
    if (flatDim <= 0 || flatDim >= ambientDim) throw std::invalid_argument("flatDim must be in (0, ambientDim)");

    std::vector<Flat<Scalar>> flats;
    std::uniform_real_distribution<Scalar> uni(-originSpread, originSpread);

    for (int i = 0; i < numFlats; ++i) {
        // Random origin
        Vec origin = Vec::NullaryExpr(ambientDim, [&]() { return uni(rng); });

        // Random basis (ambientDim x flatDim), orthonormalized
        Mat B = Mat::Random(ambientDim, flatDim);
        HouseholderQR<Mat> qr(B);
        Mat B_orth = qr.householderQ() * Mat::Identity(ambientDim, flatDim);

        flats.emplace_back(origin, B_orth);
    }

    return flats;
}
// ------------------------------------------------------------
// Helpers to build mesh/curve for k = 2 / k = 1 in 3D
// ------------------------------------------------------------
template <typename Scalar = double>
void buildPlanePatchMesh(const Flat<Scalar>& plane,
                         Scalar s,
                         int res,
                         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
                         Eigen::MatrixXi& F) {
    using namespace Eigen;
    if (plane.dimension() != 2 || plane.ambientDimension() != 3) {
        throw std::runtime_error("buildPlanePatchMesh: needs k=2, n=3");
    }
    const int nV = res * res;
    const int nF = 2 * (res - 1) * (res - 1);
    V.resize(nV, 3);
    F.resize(nF, 3);

    auto idx = [res](int i, int j) { return i * res + j; };

    for (int i = 0; i < res; ++i) {
        for (int j = 0; j < res; ++j) {
            Scalar u = -s + 2.0 * s * Scalar(i) / Scalar(res - 1);
            Scalar v = -s + 2.0 * s * Scalar(j) / Scalar(res - 1);
            Matrix<Scalar, 2, 1> uv; uv << u, v;
            auto p = plane.pointFromCoords(uv);
            V.row(idx(i, j)) = p.transpose();
        }
    }

    int f = 0;
    for (int i = 0; i < res - 1; ++i) {
        for (int j = 0; j < res - 1; ++j) {
            int v00 = idx(i,     j);
            int v10 = idx(i + 1, j);
            int v01 = idx(i,     j + 1);
            int v11 = idx(i + 1, j + 1);
            F.row(f++) << v00, v10, v11;
            F.row(f++) << v00, v11, v01;
        }
    }
}

template <typename Scalar = double>
void buildLinePatchCurve(const Flat<Scalar>& line,
                         Scalar s,
                         int res,
                         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V,
                         Eigen::MatrixXi& E) {
    using namespace Eigen;
    if (line.dimension() != 1 || line.ambientDimension() != 3) {
        throw std::runtime_error("buildLinePatchCurve: needs k=1, n=3");
    }
    V.resize(res, 3);
    E.resize(res - 1, 2);

    for (int i = 0; i < res; ++i) {
        Scalar u = -s + 2.0 * s * Scalar(i) / Scalar(res - 1);
        Matrix<Scalar, 1, 1> coord; coord << u;
        auto p = line.pointFromCoords(coord);
        V.row(i) = p.transpose();
    }
    for (int i = 0; i < res - 1; ++i) {
        E.row(i) << i, i + 1;
    }
}

// ------------------------------------------------------------
// 2) Visualize in 3D (throws if ambient dimension != 3)
//    - k = 2: show a plane patch + point cloud
//    - k = 1: show a line segment + point cloud
//    - k = 0: just the origin
// ------------------------------------------------------------
template <typename Scalar = double>
void visualizeFlatSamples3D(const Flat<Scalar>& flat,
                            const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& points,
                            const std::string& namePrefix,
                            Scalar patchHalfSize = 1.0,
                            int res = 15) {
    using namespace Eigen;

    const int n = flat.ambientDimension();
    const int k = flat.dimension();

    if (n != 3) {
        throw std::runtime_error("visualizeFlatSamples3D: ambient dimension must be 3, failing as requested.");
    }

    // register the scattered points
    auto pc = polyscope::registerPointCloud(namePrefix + " points", points.template cast<double>());
    pc->setPointRadius(0.005);

    if (k == 2) {
        Matrix<Scalar, Dynamic, Dynamic> V;
        MatrixXi F;
        buildPlanePatchMesh(flat, patchHalfSize, res, V, F);

        auto mesh = polyscope::registerSurfaceMesh(namePrefix + " flat patch",
                                                   V.template cast<double>(),
                                                   F);
        mesh->setTransparency(0.5f);
        mesh->setSurfaceColor(glm::vec3(0.2f, 0.6f, 1.0f));
    } else if (k == 1) {
        Matrix<Scalar, Dynamic, Dynamic> V;
        MatrixXi E;
        buildLinePatchCurve(flat, patchHalfSize, res, V, E);

        auto curve = polyscope::registerCurveNetwork(namePrefix + " flat line",
                                                     V.template cast<double>(),
                                                     E);
        curve->setRadius(0.0025);
        curve->setColor(glm::vec3(0.9f, 0.2f, 0.2f));
    } else if (k == 0) {
        // Degenerate case: just a single point (origin)
        Eigen::Matrix<Scalar, 1, 3> V;
        V = flat.project(flat.origin()).transpose();
        polyscope::registerPointCloud(namePrefix + " flat (0D)", V.template cast<double>());
    } else {
        throw std::runtime_error("visualizeFlatSamples3D: only k = 0,1,2 are visualized.");
    }
}

// ------------------------------------------------------------
// Build a triangulated quad mesh on a Flat plane by sampling a grid in [-s, s]^2
// ------------------------------------------------------------
inline void buildPlaneMesh(const Flat<>& plane,
                           double s,
                           int res,
                           Eigen::MatrixXd& V,
                           Eigen::MatrixXi& F) {
    using namespace Eigen;

    const int nV = res * res;
    const int nF = 2 * (res - 1) * (res - 1);

    V.resize(nV, 3);
    F.resize(nF, 3);

    auto idx = [res](int i, int j) { return i * res + j; };

    // vertices
    for (int i = 0; i < res; ++i) {
        for (int j = 0; j < res; ++j) {
            double u = -s + 2.0 * s * static_cast<double>(i) / static_cast<double>(res - 1);
            double v = -s + 2.0 * s * static_cast<double>(j) / static_cast<double>(res - 1);

            Eigen::Vector2d uv(u, v);
            Eigen::VectorXd p = plane.pointFromCoords(uv); // should be 3D
            V.row(idx(i, j)) = p.transpose();
        }
    }

    // faces (two triangles per quad)
    int f = 0;
    for (int i = 0; i < res - 1; ++i) {
        for (int j = 0; j < res - 1; ++j) {
            int v00 = idx(i,     j);
            int v10 = idx(i + 1, j);
            int v01 = idx(i,     j + 1);
            int v11 = idx(i + 1, j + 1);

            // triangle 1
            F.row(f++) << v00, v10, v11;
            // triangle 2
            F.row(f++) << v00, v11, v01;
        }
    }
}
