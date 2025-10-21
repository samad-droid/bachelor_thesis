#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include "external/polyscope/include/polyscope/curve_network.h"
#include <glm/glm.hpp>

// ------------------------------------------------------------
// Flat<T>
// ------------------------------------------------------------
template <typename Scalar = double>
class Flat {
public:

    Flat(const Eigen::Matrix<double, Eigen::Dynamic, 1>& origin, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& basis)
        : origin_(origin), basis_(basis) {
        assert((basis_.cols() <= basis_.rows()) && "Number of basis vectors needs to be lower than dimension of ambient space");
    }

    // accessors
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& origin() const { return origin_; }
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& basis()  const { return basis_; }

    int flatDimension() const        { return static_cast<int>(basis_.cols()); }
    int ambientDimension() const { return static_cast<int>(basis_.rows()); }

    // orthogonal projection
    Eigen::Matrix<double, Eigen::Dynamic, 1> project(const Eigen::Matrix<double, Eigen::Dynamic, 1>& point) const {
        assert((point.size() == origin_.size()) && "Point must be in the ambient space of the flat");
        Eigen::Matrix<double, Eigen::Dynamic, 1> diff  = point - origin_;
        Eigen::Matrix<double, Eigen::Dynamic, 1> proj  = basis_.transpose() * diff;
        return origin_ + basis_ * proj;
    }

    // from local coordinates to ambient point
    Eigen::Matrix<double, Eigen::Dynamic, 1> pointFromCoords(const Eigen::Matrix<double, Eigen::Dynamic, 1>& coords) const {
        assert(coords.size() == basis_.cols());
        return origin_ + basis_ * coords;
    }

    void print() const {
        std::cout << "Flat of dimension " << flatDimension()
                  << " in ambient dimension " << ambientDimension() << "\n";
        std::cout << "Origin:\n" << origin_.transpose() << "\n";
        std::cout << "Basis vectors (columns):\n" << basis_ << "\n";
    }

private:
    Eigen::Matrix<double, Eigen::Dynamic, 1> origin_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> basis_;
};

// ------------------------------------------------------------
// 1) Sample N noisy points from a k-dimensional flat with local coordinates [-coordExt, coordExt]
// ------------------------------------------------------------
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
generateNoisyFlatSamples(const Flat<double>& flat,
                         int N,
                         double coordExtent,
                         double noiseStd,
                         std::mt19937& rng) {

    const int k = flat.flatDimension();
    const int n = flat.ambientDimension();
    if (N <= 0) throw std::invalid_argument("N must be > 0");
    if (k <= 0) throw std::invalid_argument("Flat dimension must be > 0");

    std::uniform_real_distribution<double> uni(-coordExtent, coordExtent);
    std::normal_distribution<double> gauss(0.0, noiseStd);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> points(N, n);

    // create local coordinates around given flat with uniform distribution uni,
    // transform them into coordinates in ambient space and add noise
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> local(k);
        for (int j = 0; j < k; ++j) local(j) = uni(rng);
        Eigen::Matrix<double, Eigen::Dynamic, 1> p = flat.pointFromCoords(local);
        for (int j = 0; j < n; ++j) p(j) += gauss(rng);

        points.row(i) = p.transpose();
    }

    return points;
}
//-------------------------------------------------------------
//2) Sample N noisy Flats with random origin point (ambientDim x 1) and basis matrix (ambientDim x flatDim)
//-------------------------------------------------------------
std::vector<Flat<double>> generateRandomFlats(int numFlats,
                                              int ambientDim,
                                              int flatDim,
                                              double originSpread,
                                              std::mt19937& rng) {
    if (numFlats <= 0) throw std::invalid_argument("numFlats must be > 0");
    if (flatDim <= 0 || flatDim >= ambientDim) throw std::invalid_argument("flatDim must be in (0, ambientDim)");

    std::vector<Flat<double>> flats;
    std::uniform_real_distribution<double> uni(-originSpread, originSpread);

    for (int i = 0; i < numFlats; ++i) {
        // Random origin
        Eigen::Matrix<double, Eigen::Dynamic, 1> origin = Eigen::Matrix<double, Eigen::Dynamic, 1>::NullaryExpr(ambientDim, [&]() { return uni(rng); });
        // Random basis (ambientDim x flatDim), orthonormalized
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(ambientDim, flatDim);
        Eigen::HouseholderQR<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> qr(B);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B_orth = qr.householderQ() * Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(ambientDim, flatDim);
        origin = origin - B_orth * (B_orth.transpose() * origin);
        flats.emplace_back(origin, B_orth);
    }

    return flats;
}
// ------------------------------------------------------------
// Helpers to build mesh/curve for k = 2 / k = 1 in 3D
// ------------------------------------------------------------
void buildPlanePatchMesh(const Flat<double>& plane,
                         double s,
                         int res,
                         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& V,
                         Eigen::MatrixXi& F) {
    if (plane.flatDimension() != 2 || plane.ambientDimension() != 3) {
        throw std::runtime_error("buildPlanePatchMesh: needs dimension of flat=2, dimension of ambient space=3");
    }
    const int nV = res * res;
    const int nF = 2 * (res - 1) * (res - 1);
    V.resize(nV, 3);
    F.resize(nF, 3);

    auto idx = [res](int i, int j) { return i * res + j; };

    for (int i = 0; i < res; ++i) {
        for (int j = 0; j < res; ++j) {
            double u = -s + 2.0 * s * double(i) / double(res - 1);
            double v = -s + 2.0 * s * double(j) / double(res - 1);
            Eigen::Matrix<double, 2, 1> uv; uv << u, v;
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

void buildLinePatchCurve(const Flat<double>& line,
                         double s,
                         int res,
                         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& V,
                         Eigen::MatrixXi& E) {

    if (line.flatDimension() != 1 || line.ambientDimension() != 3) {
        throw std::runtime_error("buildLinePatchCurve: needs needs dimension of flat=1, dimension of ambient space=3");
    }
    V.resize(res, 3);
    E.resize(res - 1, 2);

    for (int i = 0; i < res; ++i) {
        double u = -s + 2.0 * s * double(i) / double(res - 1);
        Eigen::Matrix<double, 1, 1> coord; coord << u;
        auto p = line.pointFromCoords(coord);
        V.row(i) = p.transpose();
    }
    for (int i = 0; i < res - 1; ++i) {
        E.row(i) << i, i + 1;
    }
}

// ------------------------------------------------------------
// 2) Visualize in 3D (throws error if ambient dimension != 3)
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

    const int n = flat.ambientDimension();
    const int k = flat.flatDimension();

    if (n != 3) {
        throw std::runtime_error("visualizeFlatSamples3D: ambient dimension must be 3, failing as requested.");
    }

    auto pc = polyscope::registerPointCloud(namePrefix + " points", points.template cast<double>());
    pc->setPointRadius(0.005);

    if (k == 2) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V;
        Eigen::MatrixXi F;
        buildPlanePatchMesh(flat, patchHalfSize, res, V, F);

        auto mesh = polyscope::registerSurfaceMesh(namePrefix + " flat patch",
                                                   V.template cast<double>(),
                                                   F);
        mesh->setTransparency(0.5f);
        mesh->setSurfaceColor(glm::vec3(0.2f, 0.6f, 1.0f));
    } else if (k == 1) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V;
        Eigen::MatrixXi E;
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
        throw std::runtime_error("visualizeFlatSamples3D: only dimension = 0,1,2 are visualized.");
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

// Save points and their cluster ID to a CSV file
void savePointsToCSV(const std::string& filename,
                     const std::vector<Eigen::MatrixXd>& allPoints,
                     const std::vector<int>& clusterLabels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Determine ambient dimension from first batch of points
    if (allPoints.empty()) {
        throw std::runtime_error("No points to save.");
    }
    int ambientDim = allPoints[0].cols();

    // Write header: x0,x1,...,x{ambientDim-1},cluster
    for (int d = 0; d < ambientDim; ++d) {
        file << "x" << d << ",";
    }
    file << "flat number\n";

    // Write points
    for (size_t i = 0; i < allPoints.size(); ++i) {
        const auto& mat = allPoints[i];
        int cluster = clusterLabels[i];
        for (int j = 0; j < mat.rows(); ++j) {
            for (int d = 0; d < ambientDim; ++d) {
                file << mat(j, d);
                if (d < ambientDim - 1) file << ",";
            }
            file << "," << cluster << "\n";
        }
    }
    file.close();
    std::cout << "Saved CSV to " << filename << "\n";
}

double computeMeanProjectionError(const Eigen::MatrixXd& noisyPts, const Flat<>& flat) {
    double totalError = 0.0;
    int N = noisyPts.rows();
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXd x = noisyPts.row(i).transpose();
        Eigen::VectorXd proj = flat.project(x);
        double err = (x - proj).norm();
        totalError += err;
    }
    return totalError / static_cast<double>(N);
}
