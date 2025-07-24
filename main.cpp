#include <iostream>
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include <vector>
#include <glm/glm.hpp>
#include "external/eigen/Eigen/Dense"
#include <random>
#include "flat.h"

// Build a triangulated quad mesh on a Flat plane by sampling a grid in [-s, s]^2
void buildPlaneMesh(const Flat<>& plane,
                    double s,          // half-size of the square patch on the plane
                    int res,           // samples per side (>= 2)
                    Eigen::MatrixXd& V,// (res*res) x 3 vertex positions
                    Eigen::MatrixXi& F // 2*(res-1)*(res-1) x 3 triangle indices
) {
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

            Vector2d uv(u, v);
            VectorXd p = plane.pointFromCoords(uv); // should be 3D
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

int main() {

    polyscope::init();

    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    constexpr int ambientDim = 3;  // 3D space
    constexpr int flatDim = 2;     // A plane
    constexpr int N = 200;

    // Create the flat
    Vec origin(ambientDim);
    origin << 1.0, 2.0, 0.5;

    Mat basis(ambientDim, flatDim);
    basis.col(0) = Eigen::Vector3d(1, 0, 1).normalized();
    basis.col(1) = Eigen::Vector3d(0, 1, 1).normalized();

    Flat<> plane(origin, basis);

    // Random generators
    std::default_random_engine rng;
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.2);  // adjustable

    Mat points(N, ambientDim);

    for (int i = 0; i < N; ++i) {
        // Generate random flat coords in 2D
        Eigen::Vector2d localCoord;
        localCoord << uniform(rng), uniform(rng);

        // Map to 3D
        Vec p = plane.pointFromCoords(localCoord);

        // Add Gaussian noise
        for (int j = 0; j < ambientDim; ++j) {
            p(j) += noise(rng);
        }

        points.row(i) = p.transpose();
    }
    Eigen::Vector3d v0 = basis.col(0);
    Eigen::Vector3d v1 = basis.col(1);
    Eigen::Vector3d n = v0.cross(v1).normalized();

    double totalProjNoise = 0.0;

    for (int i = 0; i < points.rows(); ++i) {
        Eigen::VectorXd p = points.row(i).transpose();
        Eigen::VectorXd proj = plane.project(p);
        Eigen::VectorXd diff = p - proj;

        double signedDist = n.dot(diff);  // projection along normal
        totalProjNoise += std::abs(signedDist);
    }

    double avgProjNoise = totalProjNoise / points.rows();
    std::cout << "Average projection onto flat normal direction: " << avgProjNoise << std::endl;

    // ===== second plane =====
    Vec origin2(ambientDim);
    origin2 << -0.5, 1.8, 1.2;            // different place

    // pick a different normal to tilt the plane
    Eigen::Vector3d n2 = Eigen::Vector3d(0.3, -0.6, 0.74).normalized();

    // build an orthonormal basis spanning the plane from that normal
    Eigen::Vector3d t0 = n2.unitOrthogonal().normalized();   // any unit vector orthogonal to n2
    Eigen::Vector3d t1 = n2.cross(t0).normalized();

    Mat basis2(ambientDim, flatDim);
    basis2.col(0) = t0;
    basis2.col(1) = t1;

    Flat<> plane2(origin2, basis2);

    // sample points on the second plane
    Mat points2(N, ambientDim);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d localCoord;
        localCoord << uniform(rng), uniform(rng);

        Vec p = plane2.pointFromCoords(localCoord);

        for (int j = 0; j < ambientDim; ++j) {
            p(j) += noise(rng);
        }
        points2.row(i) = p.transpose();
    }

    // measure average projection noise for plane2 as well
    Eigen::Vector3d v0_2 = basis2.col(0);
    Eigen::Vector3d v1_2 = basis2.col(1);
    Eigen::Vector3d n_2  = v0_2.cross(v1_2).normalized();

    double totalProjNoise2 = 0.0;
    for (int i = 0; i < points2.rows(); ++i) {
        Eigen::VectorXd p = points2.row(i).transpose();
        Eigen::VectorXd proj = plane2.project(p);
        Eigen::VectorXd diff = p - proj;

        double signedDist = n_2.dot(diff);
        totalProjNoise2 += std::abs(signedDist);
    }
    double avgProjNoise2 = totalProjNoise2 / points2.rows();
    std::cout << "Average projection onto second flat normal direction: "
              << avgProjNoise2 << std::endl;

    // ===== Build and register plane meshes =====
    double s   = 1.2;   // half-width of the square on each plane
    int    res = 15;    // grid resolution

    Eigen::MatrixXd V1, V2;
    Eigen::MatrixXi F1, F2;

    buildPlaneMesh(plane,  s, res, V1, F1);
    buildPlaneMesh(plane2, s, res, V2, F2);

    // Visualize
    auto pc1 = polyscope::registerPointCloud("Scattered Plane 1 Points", points);
    auto pc2 = polyscope::registerPointCloud("Scattered Plane 2 Points", points2);
    pc2->setPointColor(glm::vec3(0.9f, 0.2f, 0.2f));

    auto mesh1 = polyscope::registerSurfaceMesh("Plane 1 mesh", V1, F1);
    mesh1->setTransparency(0.5f);
    mesh1->setSurfaceColor(glm::vec3(0.2f, 0.6f, 1.0f));

    auto mesh2 = polyscope::registerSurfaceMesh("Plane 2 mesh", V2, F2);
    mesh2->setTransparency(0.5f);
    mesh2->setSurfaceColor(glm::vec3(1.0f, 0.3f, 0.2f));

    polyscope::show();

    return 0;
}
