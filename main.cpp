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
#include "ransac_multiD.h"
#include "experiment_config.h"

int main() {
    polyscope::init();

    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    int clusterId = 0;

    // Generate flats with random flat dimensions [1, ambientDim - 1]
    for (int i = 0; i < numFlats; ++i) {
        int flatDim = flatDimDist(rng);

        // Generate a single random flat with specified ambient and flat dimension
        std::vector<Flat<>> flats = generateRandomFlats(1, ambientDim, flatDim, originSpread, rng);
        Flat<> flat = flats[0];

        Eigen::MatrixXd pts = generateNoisyFlatSamples(flat, numDataPointsPerFlat, coordExtent, noiseStd, rng);
        double err = computeMeanProjectionError(pts, flat);
        std::cout << "Random Flat " << i << " (dim=" << flatDim << "): Mean projection error = " << err << "\n";

        if (ambientDim == 3) {
            visualizeFlatSamples3D(flat, pts, "Random Flat " + std::to_string(i), 1.0, 20);
        } else {
            std::cout << "Visualization skipped (ambientDim != 3).\n";
        }

        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    savePointsToCSV(pointsCSV, allPoints, clusterLabels);

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
        saveSubspacesToCSV(detectedModels, ransacCSV);

        std::cout << "Detected " << detectedModels.size() << " subspaces\n";

        // Visualization
        for (int i = 0; i < detectedModels.size(); ++i) {
            const auto& model = detectedModels[i];
            std::cout << "Model " << i << ": dim=" << model.basis.cols()
                      << ", inliers=" << model.inliers.size() << "\n";

            if (model.origin.size() == 3) {
                visualizeSubspace3D(model, "RANSAC Subspace " + std::to_string(i));

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
