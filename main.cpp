#include <iostream>
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include <vector>
#include <glm/glm.hpp>
#include "external/eigen/Eigen/Dense"
#include <random>
#include "flat.h"

// Helper function to compute mean projection error
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

int main() {
    polyscope::init();

    using Scalar = double;
    using Vec    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    std::mt19937 rng(std::random_device{}());
    int    N           = 300;
    double coordExtent = 1.0;
    double noiseStd    = 0.2;

    // ----- Plane A (tilted in XZ, centered at (0, 0, 0))
    Vec originA(3); originA << 0.0, 0.0, 0.0;
    Mat basisA(3, 2);
    basisA.col(0) = Eigen::Vector3d(1, 0, 1).normalized();  // diagonal in XZ
    basisA.col(1) = Eigen::Vector3d(0, 1, 0);               // Y axis

    Flat<Scalar> flatA(originA, basisA);
    Mat pointsA = generateNoisyFlatSamples(flatA, N, coordExtent, noiseStd, rng);

    double errorA = computeMeanProjectionError(pointsA, flatA);
    std::cout << "Plane A: Mean projection error = " << errorA << "\n";

    // ----- Plane B (tilted in YZ, centered at (2, 2, 2))
    Vec originB(3); originB << 2.0, 2.0, 2.0;
    Mat basisB(3, 2);
    basisB.col(0) = Eigen::Vector3d(0, 1, 1).normalized();  // diagonal in YZ
    basisB.col(1) = Eigen::Vector3d(1, 0, -1).normalized(); // diagonal in X(-Z)

    Flat<Scalar> flatB(originB, basisB);
    Mat pointsB = generateNoisyFlatSamples(flatB, N, coordExtent, noiseStd, rng);

    double errorB = computeMeanProjectionError(pointsB, flatB);
    std::cout << "Plane B: Mean projection error = " << errorB << "\n";

    // ----- Visualize both planes + their points
    try {
        visualizeFlatSamples3D(flatA, pointsA, "Plane A ", 1.5, 20);
        visualizeFlatSamples3D(flatB, pointsB, "Plane B ", 1.5, 20);
    } catch (const std::exception& e) {
        std::cerr << "Visualization failed: " << e.what() << "\n";
        return 1;
    }

    // ----- Line A (diagonal in XZ, passes through origin)
    Vec originLA(3); originLA << 2.0, 3.6, 6.5;
    Mat basisLA(3, 1);
    basisLA.col(0) = Eigen::Vector3d(1, 0, 1).normalized(); // XZ diagonal

    Flat<Scalar> lineA(originLA, basisLA);
    Mat pointsLA = generateNoisyFlatSamples(lineA, N, coordExtent, noiseStd, rng);

    double errorLA = computeMeanProjectionError(pointsLA, lineA);
    std::cout << "Line A: Mean projection error = " << errorLA << "\n";

    // ----- Line B (diagonal in YZ, offset in space)
    Vec originLB(3); originLB << 2.0, 2.0, 0.0;
    Mat basisLB(3, 1);
    basisLB.col(0) = Eigen::Vector3d(0, 1, 1).normalized(); // YZ diagonal

    Flat<Scalar> lineB(originLB, basisLB);
    Mat pointsLB = generateNoisyFlatSamples(lineB, N, coordExtent, noiseStd, rng);

    double errorLB = computeMeanProjectionError(pointsLB, lineB);
    std::cout << "Line B: Mean projection error = " << errorLB << "\n";

    // ----- Visualize lines and their noisy point clouds
    try {
        visualizeFlatSamples3D(lineA, pointsLA, "Line A ", /*patchHalfSize*/ coordExtent, /*res*/ 40);
        visualizeFlatSamples3D(lineB, pointsLB, "Line B ", /*patchHalfSize*/ coordExtent, /*res*/ 40);
    } catch (const std::exception& e) {
        std::cerr << "Visualization failed: " << e.what() << "\n";
        return 1;
    }

    std::mt19937 rng2(311);
    int numPlanes = 30;
    int ambientDim = 3;
    int flatDim = 2;
    double originSpread = 3.0;

    std::vector<Flat<>> planes = generateRandomFlats(numPlanes, ambientDim, flatDim, originSpread, rng2);
    for (int i = 0; i < planes.size(); ++i) {
        visualizeFlatSamples3D(planes[i], {}, "Plane_" + std::to_string(i), 1.0, 15);
    }

    polyscope::show();
    return 0;
}
