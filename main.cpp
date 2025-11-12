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
//#include "experiment_config.h"
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

void saveQDFToCSV(const std::string& filename, const std::vector<QDF>& qdfs) {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

    if (qdfs.empty()) return;

    // Determine dimensions from first QDF
    int n = qdfs[0].A().rows();
    int basis_dim = qdfs[0].basis().cols();

    // Write header
    file << "cluster";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file << ",A" << i << j;
    for (int i = 0; i < n; i++)
        file << ",b" << i;
    file << ",c\n";

    // Write each QDF
    for (const auto& qdf : qdfs) {
        int cluster_id = qdf.clusterId();
        Eigen::MatrixXd A = qdf.A();
        Eigen::VectorXd b = qdf.b();
        double c = qdf.c();

        file << cluster_id;
        for (int i = 0; i < A.rows(); i++)
            for (int j = 0; j < A.cols(); j++)
                file << "," << A(i,j);
        for (int i = 0; i < b.size(); i++)
            file << "," << b(i);
        file << "," << c << "\n";

        std::cout << "Saved QDF for cluster " << cluster_id << "\n"; // sanity check
    }

    file.close();
    std::cout << "CSV file written with header: " << filename << "\n";
}

//namespace fs = std::filesystem;

int main() {
    polyscope::init();

    /*
    std::vector<Eigen::MatrixXd> allPoints1;
    std::vector<int> clusterLabels1;
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

        allPoints1.push_back(pts);
        clusterLabels1.push_back(clusterId++);
    }

    savePointsToCSV(pointsCSV, allPoints1, clusterLabels1);
    */
    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    loadPointsFromCSV(inputCSV, allPoints, clusterLabels);
    

    try {
        std::vector<Eigen::VectorXd> allVecPoints;
        for (const auto& mat : allPoints) {
            for (int i = 0; i < mat.rows(); ++i) {
                allVecPoints.push_back(mat.row(i).transpose());
            }
        }

        auto detectedModels = multiRansacAffine(allVecPoints, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_INLIERS, FIXED_DIMENSION, MAX_MODELS);
        recomputeAllInliers(detectedModels, allVecPoints, RANSAC_THRESHOLD);
        std::cout << "Greedy multi-RANSAC detected " << detectedModels.size() << " subspaces\n";

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

            // assign cluster ids
            for (int i = 0; i < n; i++)
                detectedModels[i].clusterId = clusterIdVec[i];

            // assign distinct colors per cluster using HSV wheel
            std::vector<glm::vec3> clusterColors(numClusters);
            for (int c = 0; c < numClusters; c++) {
                float hue = static_cast<float>(c) / std::max(1, numClusters); // evenly spaced [0,1)
                float s = 0.9f;  // strong saturation
                float v = 0.9f;  // bright

                float r, g, b;
                int i = int(hue * 6.0f);
                float f = hue * 6.0f - i;
                float p = v * (1.0f - s);
                float q = v * (1.0f - f * s);
                float t = v * (1.0f - (1.0f - f) * s);
                switch (i % 6) {
                    case 0: r = v, g = t, b = p; break;
                    case 1: r = q, g = v, b = p; break;
                    case 2: r = p, g = v, b = t; break;
                    case 3: r = p, g = q, b = v; break;
                    case 4: r = t, g = p, b = v; break;
                    case 5: r = v, g = p, b = q; break;
                }
                clusterColors[c] = glm::vec3(r, g, b);
            }

            // ======================================
            // Show all points (neutral gray background)
            // ======================================
            std::vector<glm::vec3> allPoints;
            for (const auto& p : allVecPoints) {
                allPoints.push_back(glm::vec3(
                    static_cast<float>(p(0)),
                    static_cast<float>(p(1)),
                    p.size() > 2 ? static_cast<float>(p(2)) : 0.0f
                ));
            }

            auto pcAll = polyscope::registerPointCloud("All Points", allPoints);
            pcAll->setPointColor(glm::vec3(0.8f, 0.8f, 0.8f)); // light gray
            pcAll->setPointRadius(0.005f);

            // ===== Filter out clusters with only 1 subspace =====
            std::map<int, int> clusterCounts;
            for (const auto& model : detectedModels) {
                clusterCounts[model.clusterId]++;
            }

            // Remove models belonging to clusters with just 1 subspace
            std::vector<AffineSubspaceModel> filteredModels;
            for (const auto& model : detectedModels) {
                if (clusterCounts[model.clusterId] > 1) {
                    filteredModels.push_back(model);
                }
            }
            detectedModels = std::move(filteredModels);

            // Visualization per model with cluster color
            for (int i = 0; i < detectedModels.size(); ++i) {
                const auto& model = detectedModels[i];
                glm::vec3 color = clusterColors[model.clusterId];

                std::cout << "Model " << i << ": dim=" << model.basis.cols()
                          << ", inliers=" << model.inliers.size()
                          << ", cluster=" << model.clusterId << "\n";

                if (model.origin.size() == 3) {
                    glm::vec3 color = colorFromClusterId(model.clusterId);
                    visualizeSubspace3D(model, "RANSAC Subspace " + std::to_string(i), color);

                    std::vector<glm::vec3> inlierPoints;
                    for (int idx : model.inliers) {
                        const Eigen::VectorXd& p = allVecPoints[idx];
                        inlierPoints.push_back(glm::vec3(
                            static_cast<float>(p(0)),
                            static_cast<float>(p(1)),
                            static_cast<float>(p(2))
                        ));
                    }
                    auto pc = polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);
                    pc->setPointColor(color);
                    pc->setEnabled(true);

                } else if (model.origin.size() == 2) {
                    glm::vec3 color = colorFromClusterId(model.clusterId);
                    visualizeSubspace2D(model, "RANSAC Subspace " + std::to_string(i), allVecPoints, color);

                    std::vector<glm::vec3> inlierPoints;
                    for (int idx : model.inliers) {
                        const Eigen::VectorXd& p = allVecPoints[idx];
                        inlierPoints.push_back(glm::vec3(
                            static_cast<float>(p(0)),
                            static_cast<float>(p(1)),
                            0.0f
                        ));
                    }
                    auto pc = polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);
                    pc->setPointColor(color);
                    pc->setEnabled(true);

                } else {
                    std::cout << "Unknown ambient dim for model.origin: " << model.origin.size() << "\n";
                }
            }

            saveSubspacesToCSV(detectedModels, ransacCSV);
        }

    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

 std::ifstream infile(ransacCSV);
if (!infile) {
    std::cerr << "Error: could not open subspaces.csv\n";
    return 1;
}

std::ofstream outfile(qdfCSV);
if (!outfile) {
    std::cerr << "Error: could not open qdfs.csv for writing\n";
    return 1;
}

std::string line;
    std::vector<QDF> qdfList;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);
        if (tokens.size() < 6) continue;

        int cluster_id = std::stoi(tokens[0]);
        int n = ambientDim;
        int basis_dim = std::stoi(tokens[1 + n]);

        Eigen::VectorXd origin(n);
        for (int i = 0; i < n; i++) origin[i] = std::stod(tokens[1 + i]);

        Eigen::MatrixXd basis(n, basis_dim);
        int basis_start = 1 + n + 1;
        for (int j = 0; j < basis_dim; j++)
            for (int i = 0; i < n; i++)
                basis(i, j) = std::stod(tokens[basis_start + j*n + i]);

        QDF qdf(origin, basis, cluster_id);
        qdfList.push_back(qdf);
    }

    // Save all QDFs
    saveQDFToCSV(qdfCSV, qdfList);
    QDFAnalysis::analyzeQDFClusters(qdfCSV);
    MeanQDF::computeMeanQDF(qdfCSV, meanCSV);
    //polyscope::removeAllStructures();
    MeanQDFLines::visualizeMeanQDF(meanCSV);

    //merging of ransacCSV and meanCSV no longer necessary because we throw out ransac lines that are not in a cluster
    //anyway and we only use meanQDF lines for clustering
    //mergeDetectedAndMeanQDF(ransacCSV, meanCSV, mergeCSV);

    assignPointsToSubspaces(inputCSV, meanCSV, outputCSV, CLUSTERING_THRESHOLD);
    computeClusteringMetrics(inputCSV, outputCSV);

    polyscope::show();
    return 0;
}
