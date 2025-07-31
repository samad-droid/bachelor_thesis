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

    // --- Generate 3 random planes and compute their errors
    //rng=299
    std::mt19937 rng2(244);
    double coordExtent = 1.0;
    double noiseStd    = 0.2;
    int ambientDim = 3;
    double originSpread = 6.0;

    std::vector<Flat<>> planes = generateRandomFlats(3, ambientDim, 2, originSpread, rng2);
    for (int i = 0; i < planes.size(); ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pts = generateNoisyFlatSamples(planes[i], 300, coordExtent, noiseStd, rng2);
        double err = computeMeanProjectionError(pts, planes[i]);
        std::cout << "Random Plane " << i << ": Mean projection error = " << err << "\n";
        visualizeFlatSamples3D(planes[i], pts, "Random Plane " + std::to_string(i), 1.0, 20);
    }

    // --- Generate 2 random lines and compute their errors
    std::vector<Flat<>> lines = generateRandomFlats(2, ambientDim, 1, originSpread, rng2);
    for (int i = 0; i < lines.size(); ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pts = generateNoisyFlatSamples(lines[i], 300, coordExtent, noiseStd, rng2);
        double err = computeMeanProjectionError(pts, lines[i]);
        std::cout << "Random Line " << i << ": Mean projection error = " << err << "\n";
        visualizeFlatSamples3D(lines[i], pts, "Random Line " + std::to_string(i), 1.0, 40);
    }
    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;

    int clusterId = 1;

    // Save planes
    for (int i = 0; i < planes.size(); ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pts = generateNoisyFlatSamples(planes[i], 200, coordExtent, noiseStd, rng2);
        double err = computeMeanProjectionError(pts, planes[i]);
        std::cout << "Random Plane " << i << ": Mean projection error = " << err << "\n";
        visualizeFlatSamples3D(planes[i], pts, "Random Plane " + std::to_string(i), 1.0, 20);
        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    // Save lines
    for (int i = 0; i < lines.size(); ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pts = generateNoisyFlatSamples(lines[i], 200, coordExtent, noiseStd, rng2);
        double err = computeMeanProjectionError(pts, lines[i]);
        std::cout << "Random Line " << i << ": Mean projection error = " << err << "\n";
        visualizeFlatSamples3D(lines[i], pts, "Random Line " + std::to_string(i), 1.0, 40);
        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    // Write to CSV
    savePointsToCSV("../generated_data.csv", allPoints, clusterLabels);
    polyscope::show();
    return 0;
}
