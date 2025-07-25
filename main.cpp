#include <iostream>
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include <vector>
#include <glm/glm.hpp>
#include "external/eigen/Eigen/Dense"
#include <random>
#include "flat.h"

int main() {

    polyscope::init();

    using Scalar = double;
    using Vec    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Mat    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // ----- Choose ambient / flat dimensions
    constexpr int ambientDim = 3;
    constexpr int flatDim    = 2;   // try 1 for a line, 2 for a plane

    // ----- Define a flat in R^ambientDim
    Vec origin(ambientDim);
    origin << 1.0, 2.0, 0.5;

    Mat basis(ambientDim, flatDim);
    basis.col(0) = Eigen::Vector3d(1, 0, 1).normalized();
    basis.col(1) = Eigen::Vector3d(0, 1, 1).normalized();

    Flat<Scalar> flat(origin, basis);

    // ----- Sample noisy points
    std::default_random_engine rng(1234);
    int    N           = 500;
    double coordExtent = 1.0;
    double noiseStd    = 0.15;

    Mat pts = generateNoisyFlatSamples(flat, N, coordExtent, noiseStd, rng);

    // ----- Visualize (throws if ambientDim != 3)
    try {
        visualizeFlatSamples3D(flat, pts, "MyFlat ", /*patchHalfSize=*/1.2, /*res=*/20);
    } catch (const std::exception& e) {
        std::cerr << "Visualization failed: " << e.what() << "\n";
        return 1;
    }

    polyscope::show();
    return 0;
}
