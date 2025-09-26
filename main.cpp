#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <glm/glm.hpp>
#include <queue>

#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include "external/eigen/Eigen/Dense"

#include "flat.h"
#include "ransac_multiD.h"
#include "experiment_config.h"
//#include "experiment_config_2.h"
//#include "progressive.h"

#include <limits>
#include <algorithm>

#include "experiment_config.h"

int main() {
    polyscope::init();

    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    int clusterId = 0;

    // Generate flats
    for (int i = 0; i < numFlats; ++i) {
        int flatDim = flatDimDist(rng);

        std::vector<Flat<>> flats = generateRandomFlats(1, ambientDim, flatDim, originSpread, rng);
        Flat<> flat = flats[0];

        Eigen::MatrixXd pts = generateNoisyFlatSamples(flat, numDataPointsPerFlat, coordExtent, noiseStd, rng);
        double err = computeMeanProjectionError(pts, flat);
        std::cout << "Random Flat " << i << " (dim=" << flatDim << "): Mean projection error = " << err << "\n";

        // Visualization
        if (ambientDim == 3) {
            visualizeFlatSamples3D(flat, pts, "Random Flat " + std::to_string(i), 1.0, 20);
        } else if (ambientDim == 2) {
            visualizeFlatSamples2D(pts, "Random Flat " + std::to_string(i)); // call header version
        } else {
            std::cout << "Visualization skipped (ambientDim != 3).\n";
        }

        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    savePointsToCSV(pointsCSV, allPoints, clusterLabels);

    try {
        std::vector<Eigen::VectorXd> allVecPoints;
        for (const auto& mat : allPoints) {
            for (int i = 0; i < mat.rows(); ++i) {
                allVecPoints.push_back(mat.row(i).transpose());
            }
        }

        auto detectedModels = multiRansacAffine(allVecPoints, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_INLIERS, FIXED_DIMENSION);
        std::cout << "Greedy multi-RANSAC detected " << detectedModels.size() << " subspaces\n";

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

            } else if (model.origin.size() == 2) {
                visualizeSubspace2D(model, "RANSAC Subspace " + std::to_string(i), allVecPoints);

                std::vector<glm::vec3> inlierPoints;
                for (int idx : model.inliers) {
                    const Eigen::VectorXd& p = allVecPoints[idx];
                    inlierPoints.push_back(glm::vec3(
                        static_cast<float>(p(0)),
                        static_cast<float>(p(1)),
                        0.0f
                    ));
                }
                polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);
            } else {
                std::cout << "Unknown ambient dim for model.origin: " << model.origin.size() << "\n";
            }
        }

        // Intersection / Jaccard / Normalized Matrices
        int n = detectedModels.size();
        Eigen::MatrixXi intersectionMatrix = Eigen::MatrixXi::Zero(n, n);
        Eigen::MatrixXd jaccardMatrix = Eigen::MatrixXd::Zero(n, n);
        Eigen::MatrixXd normalizedMatrix = Eigen::MatrixXd::Zero(n, n);
        int totalPoints = static_cast<int>(allVecPoints.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                int interCount = 0;
                for (int idx : detectedModels[i].inliers)
                    if (detectedModels[j].inliers.find(idx) != detectedModels[j].inliers.end()) ++interCount;

                intersectionMatrix(i, j) = intersectionMatrix(j, i) = interCount;

                int unionCount = static_cast<int>(detectedModels[i].inliers.size()
                                  + detectedModels[j].inliers.size() - interCount);
                double jaccard = (unionCount > 0) ? static_cast<double>(interCount) / unionCount : 0.0;
                jaccardMatrix(i, j) = jaccardMatrix(j, i) = jaccard;

                double normalized = (totalPoints > 0) ? static_cast<double>(interCount) / totalPoints : 0.0;
                normalizedMatrix(i, j) = normalizedMatrix(j, i) = normalized;
            }
        }

        std::cout << "Intersection matrix:\n" << intersectionMatrix << "\n\n";
        std::cout << "Jaccard similarity matrix:\n" << jaccardMatrix << "\n\n";
        std::cout << "Normalized intersection matrix:\n" << normalizedMatrix << "\n\n";

        // Clustering using header function
        std::vector<double> thresholds = {JACCARD_THRESHOLD};
        for (double thr : thresholds) {
            int numClusters = 0;
            auto clusterIdVec = clusterSubspaces(jaccardMatrix, thr, numClusters); // call header version

            std::cout << "\n=== Clustering with threshold " << thr << " ===\n";
            std::cout << "Total clusters: " << numClusters << "\n";

            for (int c = 0; c < numClusters; c++) {
                std::cout << "Cluster " << c << ": ";
                for (int i = 0; i < n; i++)
                    if (clusterIdVec[i] == c) std::cout << i << " ";
                std::cout << "\n";
            }

            for (int i = 0; i < n; i++)
                detectedModels[i].clusterId = clusterIdVec[i];

            saveSubspacesToCSV(detectedModels, ransacCSV);
        }

    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

    polyscope::show();
    return 0;
}
