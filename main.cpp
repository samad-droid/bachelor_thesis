#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <glm/glm.hpp>

#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include "external/eigen/Eigen/Dense"

#include "flat.h"
#include "ransac_multiD.h"  // ← using the generic version now

int main() {
    polyscope::init();

    /*// Parameters
    std::mt19937 rng2(244);
    double coordExtent  = 1.0;
    double noiseStd     = 0.2;
    int ambientDim      = 6;
    double originSpread = 6.0;

    constexpr int MIN_INLIERS       = 50;
    constexpr int RANSAC_ITERATIONS = 1000;
    constexpr double RANSAC_THRESHOLD = 0.38;

    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    int clusterId = 1;

    // --- Planes ---
    {
        std::vector<Flat<>> planes = generateRandomFlats(3, ambientDim, 2, originSpread, rng2);
        for (int i = 0; i < planes.size(); ++i) {
            Eigen::MatrixXd pts = generateNoisyFlatSamples(planes[i], 300, coordExtent, noiseStd, rng2);
            double err = computeMeanProjectionError(pts, planes[i]);
            std::cout << "Random Plane " << i << ": Mean projection error = " << err << "\n";
            visualizeFlatSamples3D(planes[i], pts, "Random Plane " + std::to_string(i), 1.0, 20);
            allPoints.push_back(pts);
            clusterLabels.push_back(clusterId++);
        }
    }

    // --- Lines ---
    {
        std::vector<Flat<>> lines = generateRandomFlats(2, ambientDim, 1, originSpread, rng2);
        for (int i = 0; i < lines.size(); ++i) {
            Eigen::MatrixXd pts = generateNoisyFlatSamples(lines[i], 300, coordExtent, noiseStd, rng2);
            double err = computeMeanProjectionError(pts, lines[i]);
            std::cout << "Random Line " << i << ": Mean projection error = " << err << "\n";
            visualizeFlatSamples3D(lines[i], pts, "Random Line " + std::to_string(i), 1.0, 40);
            allPoints.push_back(pts);
            clusterLabels.push_back(clusterId++);
        }
    }*/

    const int ambientDim = 3;
    const int numFlats = 5;
    const double originSpread = 6.0;
    const double coordExtent = 1.0;
    const double noiseStd = 0.2;
    const int numDataPointsPerFlat = 300;

    constexpr int MIN_INLIERS       = 50;
    constexpr int RANSAC_ITERATIONS = 1000;
    constexpr double RANSAC_THRESHOLD = 0.38;
    constexpr int FIXED_DIMENSION = 1;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> flatDimDist(1, ambientDim - 1);

    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    int clusterId = 0;

    // Generate flats with random flat dimensions between 1 and 5
    for (int i = 0; i < numFlats; ++i) {
        int flatDim = flatDimDist(rng);

        // Generate a single random flat with specified ambient and flat dimension
        std::vector<Flat<>> flats = generateRandomFlats(1, ambientDim, flatDim, originSpread, rng);
        Flat<> flat = flats[0];

        Eigen::MatrixXd pts = generateNoisyFlatSamples(flat, numDataPointsPerFlat, coordExtent, noiseStd, rng);
        double err = computeMeanProjectionError(pts, flat);
        std::cout << "Random Flat " << i << " (dim=" << flatDim << "): Mean projection error = " << err << "\n";

        // Visualization only possible if ambientDim == 3 (your visualize function expects that)
        if (ambientDim == 3) {
            visualizeFlatSamples3D(flat, pts, "Random Flat " + std::to_string(i), 1.0, 20);
        } else {
            std::cout << "Visualization skipped (ambientDim != 3).\n";
        }

        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    savePointsToCSV("../generated_data2.csv", allPoints, clusterLabels);

    try {
        // Flatten all Eigen::MatrixXd clusters into a single vector<Eigen::VectorXd>
        std::vector<Eigen::VectorXd> allVecPoints;
        for (const auto& mat : allPoints) {
            for (int i = 0; i < mat.rows(); ++i) {
                allVecPoints.push_back(mat.row(i).transpose());
            }
        }

        // Run multi-model RANSAC
        auto detectedModels = multiRansacAffine(allVecPoints, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_INLIERS, FIXED_DIMENSION);
        saveSubspacesToCSV(detectedModels, "../detected_subspaces.csv");

        std::cout << "Detected " << detectedModels.size() << " subspaces\n";

        // Visualization
        for (int i = 0; i < detectedModels.size(); ++i) {
            const auto& model = detectedModels[i];
            std::cout << "Model " << i << ": dim=" << model.basis.cols()
                      << ", inliers=" << model.inliers.size() << "\n";

            // Only visualize if 3D data
            if (model.origin.size() == 3) {
                visualizeSubspace3D(model, "RANSAC Subspace " + std::to_string(i));

                // Visualize inlier points
                std::vector<glm::vec3> inlierPoints;
                for (int idx : model.inliers) {
                    const Eigen::VectorXd& p = allVecPoints[idx];
                    inlierPoints.push_back(glm::vec3(
                        static_cast<float>(p(0)),
                        static_cast<float>(p(1)),
                        static_cast<float>(p(2))
                    ));
                }
                polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

    polyscope::show();
    return 0;
}
