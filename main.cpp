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
#include "experiment_config_2.h"
#include "progressive.h"

#include <limits>
#include <algorithm>
#include "qdf.h"
#include "qdf_analysis.h"
#include "mean_Qdf.h"
#include "mean_qdf_lines.h"
#include "merge_csv.h"
#include "cluster_points_to subspaces.h"
#include "clustering_accuracy.h"
#include "median_qdf.h"
#include "medianW_qdf.h"

// Helper function to save subspaces to CSV in the format expected by loadSubspacesFromCSV_fixedDim
// We do NOT write a header to prevent 'stoi' errors in the loader.
void saveSubsToCSV(const std::vector<AffineSubspaceModel>& models, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    int count = 0;
    for (const auto& model : models) {
        // Format: clusterId, origin_0, ..., origin_n, basis_dim, basis_00, ...
        file << model.clusterId;

        // Write Origin
        for (int i = 0; i < model.origin.size(); ++i) {
            file << "," << model.origin(i);
        }

        // Write Basis Rank (dimensions)
        file << "," << model.basis.cols();

        // Write Basis entries (flattened column by column)
        for (int j = 0; j < model.basis.cols(); ++j) {
            for (int i = 0; i < model.basis.rows(); ++i) {
                file << "," << model.basis(i, j);
            }
        }
        file << "\n";
        count++;
    }
    file.close();
    std::cout << "Saved " << count << " subspaces to " << filename << std::endl;
}

// Unused QDF saver kept for compilation compatibility if needed
void saveQDFToCSV(const std::string& filename, const std::vector<QDF>& qdfs) {}

int main() {
    polyscope::init();

    // Load Data
    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    loadPointsFromCSV(inputCSV, allPoints, clusterLabels);

    try {
        // Flatten points for RANSAC
        std::vector<Eigen::VectorXd> allVecPoints;
        for (const auto& mat : allPoints) {
            for (int i = 0; i < mat.rows(); ++i) {
                allVecPoints.push_back(mat.row(i).transpose());
            }
        }

        // 1. Run RANSAC to detect subspaces
        auto detectedModels = multiRansacAffine(allVecPoints, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_INLIERS, FIXED_DIMENSION);
        recomputeAllInliers(detectedModels, allVecPoints, RANSAC_THRESHOLD);
        std::cout << "Greedy multi-RANSAC detected " << detectedModels.size() << " subspaces\n";

        // 2. Compute Similarities (Jaccard)
        int n = detectedModels.size();
        Eigen::MatrixXi intersectionMatrix = Eigen::MatrixXi::Zero(n, n);
        Eigen::MatrixXd jaccardMatrix = Eigen::MatrixXd::Zero(n, n);
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
            }
        }

        // 3. Cluster the Subspaces
        std::vector<double> thresholds = {JACCARD_THRESHOLD};
        for (double thr : thresholds) {
            int numClusters = 0;
            auto clusterIdVec = clusterSubspacesAlglib(jaccardMatrix, thr, numClusters);

            // =========================================================
            // CRITICAL FIX: Reassign -1 IDs to unique positive IDs
            // =========================================================
            int nextUniqueId = numClusters; // Start assigning new IDs after the last valid cluster
            if (nextUniqueId == 0) nextUniqueId = 0; // Ensure we start at least at 0

            for (int i = 0; i < n; i++) {
                if (clusterIdVec[i] == -1) {
                    clusterIdVec[i] = nextUniqueId++;
                }
                detectedModels[i].clusterId = clusterIdVec[i];
            }
            numClusters = nextUniqueId; // Update total count
            // =========================================================

            std::cout << "\n=== Clustering with threshold " << thr << " ===\n";
            std::cout << "Total clusters (including singletons): " << numClusters << "\n";

            // Visualization Colors
            std::vector<glm::vec3> clusterColors(numClusters);
            for (int c = 0; c < numClusters; c++) {
                // Simple Golden Angle approximation for distinct colors
                float hue = static_cast<float>(c) * 137.508f;
                hue = fmod(hue, 360.0f);
                float s = 0.8f;
                float v = 0.9f;

                float c_val = v * s;
                float x = c_val * (1 - std::abs(fmod(hue / 60.0f, 2) - 1));
                float m = v - c_val;
                float r=0, g=0, b=0;
                if(hue < 60) { r=c_val; g=x; }
                else if(hue < 120) { r=x; g=c_val; }
                else if(hue < 180) { g=c_val; b=x; }
                else if(hue < 240) { g=x; b=c_val; }
                else if(hue < 300) { r=x; b=c_val; }
                else { r=c_val; b=x; }
                clusterColors[c] = glm::vec3(r+m, g+m, b+m);
            }

            // Visualize RANSAC Models
            for (int i = 0; i < detectedModels.size(); ++i) {
                const auto& model = detectedModels[i];
                glm::vec3 color = clusterColors[model.clusterId];

                if (model.origin.size() == 3) {
                    visualizeSubspace3D(model, "RANSAC Subspace " + std::to_string(i), color);
                } else if (model.origin.size() == 2) {
                    visualizeSubspace2D(model, "RANSAC Subspace " + std::to_string(i), allVecPoints, color);
                }
            }

            // Save the detected models (NOW with valid IDs)
            saveSubsToCSV(detectedModels, ransacCSV);
        }

        // 4. Assign points to the loaded subspaces
        // Now that the CSV contains valid IDs (>=0), this function should load them correctly.
        std::cout << "Assigning points to RANSAC subspaces..." << std::endl;
        assignPointsToSubspaces(inputCSV, ransacCSV, outputCSV, CLUSTERING_THRESHOLD);

        // 5. Compute Metrics
        computeClusteringMetrics(inputCSV, outputCSV);

    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

    polyscope::show(200);
    return 0;
}