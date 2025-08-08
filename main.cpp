#include <iostream>
#include <fstream>
#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include <vector>
#include <glm/glm.hpp>
#include "external/eigen/Eigen/Dense"
#include <random>
#include "flat.h"
#include "ransac_line.h"

// Save points and their cluster ID to a CSV file
void savePointsToCSV(const std::string& filename, const std::vector<Eigen::MatrixXd>& allPoints, const std::vector<int>& clusterLabels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file << "x,y,z,cluster\n";

    for (size_t i = 0; i < allPoints.size(); ++i) {
        const auto& mat = allPoints[i];
        int cluster = clusterLabels[i];

        for (int j = 0; j < mat.rows(); ++j) {
            file << mat(j, 0) << "," << mat(j, 1) << "," << mat(j, 2) << "," << cluster << "\n";
        }
    }

    file.close();
    std::cout << "Saved CSV to " << filename << "\n";
}


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

    // Parameters
    std::mt19937 rng2(244);
    double coordExtent = 1.0;
    double noiseStd    = 0.2;
    int ambientDim     = 3;
    double originSpread = 6.0;

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
    }

    savePointsToCSV("../generated_data2.csv", allPoints, clusterLabels);

    try {
        std::vector<Eigen::Vector3d> allPoints = loadCSV("../generated_data.csv");

        auto detectedLines = multiRansacLines(allPoints, 1000, 0.38);
        saveLinesToCSV(detectedLines, "../detected_lines.csv");

        std::cout << "Detected " << detectedLines.size() << " lines\n";

        for (int i = 0; i < detectedLines.size(); ++i) {
            const auto& line = detectedLines[i];
            std::cout << "Line " << i << " has " << line.inliers.size() << " inliers.\n";

            VisualLine visLine = lineModelToVisualLine(line);
            visualizeRansacLine(visLine, "RANSAC Line " + std::to_string(i));

            // Visualize inliers separately
            std::vector<glm::vec3> inlierPoints;
            for (int idx : line.inliers) {
                const Eigen::Vector3d& p = allPoints[idx];
                inlierPoints.push_back(glm::vec3(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())));
            }
            polyscope::registerPointCloud("Line " + std::to_string(i) + " Inliers", inlierPoints);
        }
    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

    polyscope::show();
    return 0;
}
