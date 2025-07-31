#pragma once

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include "external/polyscope/include/polyscope/polyscope.h"
#include <glm/glm.hpp>

using Vec3 = Eigen::Vector3d;
using PointList = std::vector<Vec3>;

// Structure to hold a line model and its inliers
struct LineModel {
    Vec3 a, b; // points defining the line segment
    std::vector<int> inliers;
};

// Load 3D points (x, y, z) from CSV file
inline PointList loadCSV(const std::string& filename) {
    PointList points;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        double x, y, z;

        std::getline(ss, cell, ','); x = std::stod(cell);
        std::getline(ss, cell, ','); y = std::stod(cell);
        std::getline(ss, cell, ','); z = std::stod(cell);
        // Ignore cluster column if present
        points.emplace_back(x, y, z);
    }

    return points;
}

// Compute perpendicular distance from point p to line through (a, b)
inline double pointLineDistance(const Vec3& p, const Vec3& a, const Vec3& b) {
    Vec3 ab = b - a;
    Vec3 ap = p - a;
    return (ab.cross(ap)).norm() / ab.norm();
}

// Run RANSAC to fit a line to the point cloud
inline LineModel ransacLine(const PointList& points, int iterations, double threshold) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    LineModel bestModel;
    int bestInlierCount = 0;

    for (int i = 0; i < iterations; ++i) {
        int idx1 = dist(rng), idx2 = dist(rng);
        if (idx1 == idx2) continue;

        Vec3 a = points[idx1];
        Vec3 b = points[idx2];

        std::vector<int> inliers;
        for (int j = 0; j < points.size(); ++j) {
            double d = pointLineDistance(points[j], a, b);
            if (d < threshold) inliers.push_back(j);
        }

        if (inliers.size() > bestInlierCount) {
            bestInlierCount = static_cast<int>(inliers.size());
            bestModel = { a, b, inliers };
        }
    }

    return bestModel;
}

// Minimum inliers to accept a line during multi-line RANSAC
constexpr int MIN_INLIERS = 50;

// Remove inliers from point set
inline PointList removeInliers(const PointList& points, const std::vector<int>& inliers) {
    PointList remaining;
    std::unordered_set<int> inlierSet(inliers.begin(), inliers.end());
    for (int i = 0; i < points.size(); ++i) {
        if (inlierSet.find(i) == inlierSet.end()) {
            remaining.push_back(points[i]);
        }
    }
    return remaining;
}

// Multi-line RANSAC: iteratively find lines and remove inliers
inline std::vector<LineModel> multiRansacLines(PointList data, int iterations, double threshold) {
    std::vector<LineModel> detectedLines;

    while (true) {
        if (data.size() < MIN_INLIERS) break;  // Not enough points to fit a line

        LineModel best = ransacLine(data, iterations, threshold);

        if (best.inliers.size() < MIN_INLIERS) break;  // No meaningful line found

        detectedLines.push_back(best);

        // Remove inliers from data
        data = removeInliers(data, best.inliers);
    }

    return detectedLines;
}

// Save inliers of the best model to a CSV
inline void saveLineModel(const LineModel& model, const PointList& points, const std::string& filename) {
    std::ofstream out(filename);
    out << "x,y,z\n";
    for (int idx : model.inliers) {
        const Vec3& p = points[idx];
        out << p.x() << "," << p.y() << "," << p.z() << "\n";
    }
    std::cout << "Saved best line inliers to: " << filename << "\n";
}

// Save multiple line models to CSV with point and direction columns
inline void saveLinesToCSV(const std::vector<LineModel>& lines, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    out << "px,py,pz,dx,dy,dz\n";  // point and direction

    for (const auto& line : lines) {
        Vec3 dir = (line.b - line.a).normalized();
        out << line.a.x() << "," << line.a.y() << "," << line.a.z() << ",";
        out << dir.x() << "," << dir.y() << "," << dir.z() << "\n";
    }
    std::cout << "Saved " << lines.size() << " lines to " << filename << "\n";
}


struct VisualLine {
    Eigen::Vector3d point;
    Eigen::Vector3d direction;
};

// Convert LineModel to VisualLine for visualization
inline VisualLine lineModelToVisualLine(const LineModel& model) {
    VisualLine vis;
    vis.point = model.a;
    vis.direction = (model.b - model.a).normalized();
    return vis;
}

// Visualize line segment with Polyscope
inline void visualizeRansacLine(const VisualLine& line, const std::string& name) {
    std::vector<glm::vec3> linePoints;

    Eigen::Vector3d p1 = line.point;
    Eigen::Vector3d p2 = line.point + line.direction * 5.0;  // Extend line for visualization

    linePoints.push_back(glm::vec3(static_cast<float>(p1(0)), static_cast<float>(p1(1)), static_cast<float>(p1(2))));
    linePoints.push_back(glm::vec3(static_cast<float>(p2(0)), static_cast<float>(p2(1)), static_cast<float>(p2(2))));

    std::vector<std::array<size_t, 2>> edges{{0, 1}};
    polyscope::registerCurveNetwork(name, linePoints, edges);
}
