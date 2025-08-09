#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <iostream>

#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/curve_network.h"
#include "external/polyscope/include/polyscope/point_cloud.h"

// ===== Model definition =====
struct AffineSubspaceModel {
    Eigen::VectorXd origin;          // point on subspace
    Eigen::MatrixXd basis;           // columns = orthonormal basis vectors
    std::vector<int> inliers;        // indices of inliers in the original dataset
};

// ===== CSV loader for any dimension =====
inline std::vector<Eigen::VectorXd> loadCSV(const std::string& filename) {
    std::vector<Eigen::VectorXd> points;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> coords;
        while (std::getline(ss, cell, ',')) coords.push_back(std::stod(cell));
        Eigen::VectorXd p(coords.size());
        for (size_t i = 0; i < coords.size(); ++i) p(i) = coords[i];
        points.push_back(p);
    }
    return points;
}

// ===== Distance from point to affine subspace =====
inline double pointSubspaceDistance(const Eigen::VectorXd& p,
                                    const Eigen::VectorXd& origin,
                                    const Eigen::MatrixXd& basis) {
    Eigen::VectorXd diff = p - origin;
    Eigen::VectorXd proj = basis * (basis.transpose() * diff);
    return (diff - proj).norm();
}

// ===== Fit affine subspace from a set of points =====
inline AffineSubspaceModel fitAffineSubspace(const std::vector<Eigen::VectorXd>& sample) {
    const int dim = sample[0].size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    for (auto& p : sample) mean += p;
    mean /= sample.size();

    Eigen::MatrixXd centered(dim, sample.size());
    for (size_t i = 0; i < sample.size(); ++i)
        centered.col(i) = sample[i] - mean;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU);
    Eigen::MatrixXd basis = svd.matrixU().leftCols(sample.size()-1); // span of points

    return {mean, basis, {}};
}

// ===== RANSAC for variable-dimension affine subspaces =====
inline AffineSubspaceModel ransacAffine(const std::vector<Eigen::VectorXd>& points,
                                        int iterations, double threshold, int min_inliers,
                                        int fixedDim = 0) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    AffineSubspaceModel bestModel;
    int bestInlierCount = 0;

    const int ambientDim = points[0].size();

    for (int iter = 0; iter < iterations; ++iter) {
        int k;
        if (fixedDim > 0) {
            k = fixedDim;
        } else {
            // Random dimension between 1 and ambientDim-1
            k = 1 + (rng() % (ambientDim - 1));
        }

        // Sample k+1 unique points
        std::unordered_set<int> chosen;
        while (chosen.size() < static_cast<size_t>(k+1))
            chosen.insert(dist(rng));
        std::vector<Eigen::VectorXd> sample;
        for (int idx : chosen) sample.push_back(points[idx]);

        auto model = fitAffineSubspace(sample);
        model.basis = model.basis.leftCols(k);

        std::vector<int> inliers;
        for (int i = 0; i < points.size(); ++i) {
            if (pointSubspaceDistance(points[i], model.origin, model.basis) < threshold)
                inliers.push_back(i);
        }

        if ((int)inliers.size() > bestInlierCount && (int)inliers.size() >= min_inliers) {
            bestInlierCount = (int)inliers.size();
            model.inliers = inliers;
            bestModel = model;
        }
    }

    return bestModel;
}

// ===== Remove inliers =====
inline std::vector<Eigen::VectorXd> removeInliers(const std::vector<Eigen::VectorXd>& points,
                                                  const std::vector<int>& inliers) {
    std::unordered_set<int> inlierSet(inliers.begin(), inliers.end());
    std::vector<Eigen::VectorXd> remaining;
    for (int i = 0; i < points.size(); ++i)
        if (inlierSet.find(i) == inlierSet.end()) remaining.push_back(points[i]);
    return remaining;
}

// ===== Multi-model RANSAC =====
inline std::vector<AffineSubspaceModel> multiRansacAffine(std::vector<Eigen::VectorXd> data,
                                                          int iterations, double threshold, int min_inliers,
                                                          int fixedDim = 0) {
    std::vector<AffineSubspaceModel> models;
    while ((int)data.size() >= min_inliers) {
        auto best = ransacAffine(data, iterations, threshold, min_inliers, fixedDim);
        if ((int)best.inliers.size() < min_inliers) break;
        models.push_back(best);
        data = removeInliers(data, best.inliers);
    }
    return models;
}

// ===== Save subspaces to CSV =====
// Format: origin_x, origin_y, ..., basis_dim, basis_vectors(flat)
inline void saveSubspacesToCSV(const std::vector<AffineSubspaceModel>& models,
                               const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    out << "origin...,basis_dim,basis_vectors...\n";
    for (const auto& m : models) {
        for (int i = 0; i < m.origin.size(); ++i) {
            out << m.origin(i) << ",";
        }
        out << m.basis.cols() << ",";
        for (int c = 0; c < m.basis.cols(); ++c) {
            for (int r = 0; r < m.basis.rows(); ++r) {
                out << m.basis(r,c);
                if (!(c == m.basis.cols()-1 && r == m.basis.rows()-1)) out << ",";
            }
        }
        out << "\n";
    }
    std::cout << "Saved " << models.size() << " subspaces to " << filename << "\n";
}

// ===== Optional 3D visualization =====
void visualizeSubspace3D(const AffineSubspaceModel& model, const std::string& name) {
    int dim = model.basis.cols();
    if (dim < 1 || dim > 3) {
        std::cerr << "Error: unsupported subspace dimension: " << dim << "\n";
        return;
    }

    Eigen::Vector3d origin = model.origin;

    if (dim == 1) {
        // Draw line segment along basis.col(0)
        Eigen::Vector3d p1 = origin - model.basis.col(0) * 5.0;
        Eigen::Vector3d p2 = origin + model.basis.col(0) * 5.0;

        std::vector<std::array<double, 3>> points = {
            {p1.x(), p1.y(), p1.z()},
            {p2.x(), p2.y(), p2.z()}
        };
        std::vector<std::array<size_t, 2>> edges = {{0, 1}};
        polyscope::registerCurveNetwork(name, points, edges);

    } else if (dim == 2) {
        // Draw plane patch using basis.col(0) and basis.col(1)
        double s = 5.0;
        Eigen::Vector3d u = model.basis.col(0);
        Eigen::Vector3d v = model.basis.col(1);

        std::vector<std::array<double, 3>> squarePoints = {
            { (origin + (-s) * u + (-s) * v).x(), (origin + (-s) * u + (-s) * v).y(), (origin + (-s) * u + (-s) * v).z() },
            { (origin + (-s) * u + ( s) * v).x(), (origin + (-s) * u + ( s) * v).y(), (origin + (-s) * u + ( s) * v).z() },
            { (origin + ( s) * u + ( s) * v).x(), (origin + ( s) * u + ( s) * v).y(), (origin + ( s) * u + ( s) * v).z() },
            { (origin + ( s) * u + (-s) * v).x(), (origin + ( s) * u + (-s) * v).y(), (origin + ( s) * u + (-s) * v).z() }
        };

        std::vector<std::array<size_t, 2>> squareEdges = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}
        };

        polyscope::registerCurveNetwork(name + "_plane", squarePoints, squareEdges);
    } else {
        // 3D subspace visualization - optional, probably not needed for your case
        std::cerr << "3D subspace visualization not implemented\n";
    }
}


