#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <map> // Added for debug counting

#include "ransac_multiD.h"

// ---------------------------------------------------------
// Simple CSV splitter
// ---------------------------------------------------------
inline std::vector<std::string> splitCSVLineSimple(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

// ---------------------------------------------------------
// Load generated points and infer ambient dimension
// ---------------------------------------------------------
inline std::vector<Eigen::VectorXd> loadGeneratedPointsCSV(
    const std::string &filename,
    std::vector<int> &originalLabels,
    int &ambientDim)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open generated data file: " + filename);

    std::vector<Eigen::VectorXd> points;
    std::string line;

    // read header (if any)
    if (!std::getline(file, line))
        throw std::runtime_error("Empty generated data file: " + filename);

    // attempt to detect number of columns from first data line
    if (!std::getline(file, line))
        throw std::runtime_error("No data rows in generated data file: " + filename);

    {
        auto toks = splitCSVLineSimple(line);
        if (toks.size() < 2)
            throw std::runtime_error("Unexpected format in generated data: " + line);
        // last column is the flat number label -> ambientDim = cols - 1
        ambientDim = static_cast<int>(toks.size()) - 1;

        Eigen::VectorXd p(ambientDim);
        for (int i = 0; i < ambientDim; ++i) p(i) = std::stod(toks[i]);
        points.push_back(p);
        originalLabels.push_back(std::stoi(toks.back()));
    }

    // remaining lines
    while (std::getline(file, line)) {
        auto toks = splitCSVLineSimple(line);
        if (toks.size() < (size_t)ambientDim + 1) continue; // skip malformed
        Eigen::VectorXd p(ambientDim);
        for (int i = 0; i < ambientDim; ++i) p(i) = std::stod(toks[i]);
        points.push_back(p);
        originalLabels.push_back(std::stoi(toks.back()));
    }

    std::cout << "Loaded " << points.size() << " generated points (ambientDim=" << ambientDim << ")\n";
    return points;
}

// ---------------------------------------------------------
// Load subspaces from all_lines.csv using known ambientDim
// ---------------------------------------------------------
inline std::vector<AffineSubspaceModel> loadSubspacesFromCSV_fixedDim(const std::string &filename, int ambientDim) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open subspaces file: " + filename);

    std::vector<AffineSubspaceModel> models;
    std::string line;
    if (!std::getline(file, line)) return models; // skip header

    while (std::getline(file, line)) {
        auto toks = splitCSVLineSimple(line);
        if (toks.empty()) continue;

        size_t idx = 0;
        AffineSubspaceModel m;
        try {
            m.clusterId = std::stoi(toks[idx++]);
        } catch (...) {
            std::cerr << "Skipping row with invalid cluster id: " << line << "\n";
            continue;
        }

        // === Case 1: mean_qdf_lines format ===
        if (toks.size() == (size_t)(1 + 2 * ambientDim)) {
            m.origin = Eigen::VectorXd(ambientDim);
            Eigen::VectorXd b_new(ambientDim);

            // Read origin (x_m)
            for (int i = 0; i < ambientDim; ++i)
                m.origin(i) = std::stod(toks[idx++]);

            // Read b_new (another point on the line)
            for (int i = 0; i < ambientDim; ++i)
                b_new(i) = std::stod(toks[idx++]);

            // Compute direction vector
            Eigen::VectorXd dir = b_new - m.origin;

            // Normalize direction to unit length
            double len = dir.norm();
            if (len < 1e-12) {
                std::cerr << "Warning: degenerate line (clusterId=" << m.clusterId << ")\n";
                continue;
            }
            dir /= len;

            // Store normalized direction as 1-column basis matrix
            m.basis = Eigen::MatrixXd(ambientDim, 1);
            m.basis.col(0) = dir;

            models.push_back(std::move(m));
            continue;
        }

        // === Case 2: detected_subspaces format ===
        if (toks.size() < (size_t)(1 + ambientDim + 1)) {
            std::cerr << "Skipping malformed subspace row: " << line << "\n";
            continue;
        }

        m.origin = Eigen::VectorXd(ambientDim);
        for (int d = 0; d < ambientDim; ++d)
            m.origin(d) = std::stod(toks[idx++]);

        int basisDim = std::stoi(toks[idx++]);

        if ((int)(toks.size() - idx) != ambientDim * basisDim) {
            std::cerr << "Warning: basis size mismatch for cluster "
                      << m.clusterId << ": expected " << (ambientDim * basisDim)
                      << " got " << (toks.size() - idx) << "\n";
            continue;
        }

        m.basis = Eigen::MatrixXd(ambientDim, basisDim);
        for (int c = 0; c < basisDim; ++c)
            for (int r = 0; r < ambientDim; ++r)
                m.basis(r, c) = std::stod(toks[idx++]);

        models.push_back(std::move(m));
    }

    std::cout << "Loaded " << models.size() << " subspace models from " << filename << "\n";
    return models;
}

// ---------------------------------------------------------
// Main assignment function: reads points, reads subspaces,
// assigns each point to closest subspace regardless of distance.
// ---------------------------------------------------------
inline void assignPointsToSubspaces(
    const std::string &generatedDataFile,
    const std::string &allLinesFile,
    const std::string &outputFile,
    double /*threshold*/ ) // Threshold argument kept for compatibility but ignored
{
    std::vector<int> originalLabels;
    int ambientDim = 0;
    auto points = loadGeneratedPointsCSV(generatedDataFile, originalLabels, ambientDim);
    if (points.empty())
        throw std::runtime_error("No points loaded from " + generatedDataFile);

    auto models = loadSubspacesFromCSV_fixedDim(allLinesFile, ambientDim);
    std::cout << "DEBUG: loaded " << models.size() << " models\n";

    if (models.empty()) {
        std::cerr << "Warning: no valid subspace models loaded from " << allLinesFile << "\n";
    }

    std::ofstream out(outputFile);
    if (!out.is_open())
        throw std::runtime_error("Cannot open output file: " + outputFile);

    // header
    for (int i = 0; i < ambientDim; ++i) {
        out << "x" << i << ",";
    }
    out << "cluster_id\n";

    std::map<int, int> assignmentCounts;

    for (size_t i = 0; i < points.size(); ++i) {
        double minDist = std::numeric_limits<double>::infinity();
        int bestCluster = -1;

        // Find closest subspace
        for (const auto &m : models) {
            if (m.origin.size() != ambientDim || m.basis.rows() != ambientDim) {
                continue;
            }

            double d = pointSubspaceDistance(points[i], m.origin, m.basis);
            if (d < minDist) {
                minDist = d;
                bestCluster = m.clusterId;
            }
        }

        // --- CHANGED: Removed threshold check. We force assignment to the closest cluster. ---
        // if (minDist > threshold) bestCluster = -1; // This logic is deleted.

        assignmentCounts[bestCluster]++;

        // write point and assigned cluster
        for (int j = 0; j < ambientDim; ++j) {
            out << points[i](j) << ",";
        }
        out << bestCluster << "\n";
    }

    std::cout << "\n=== Assignment Summary ===\n";
    for(auto const& [cluster, count] : assignmentCounts) {
        std::cout << "Cluster " << cluster << ": " << count << " points assigned\n";
    }
    std::cout << "Saved clustered points to " << outputFile << "\n";
}