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
// Expect line format produced by saveSubspacesToCSV:
// cluster_id, origin[0],...,origin[d-1], basis_dim, basis_flattened (d * basis_dim numbers)
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
        // cluster_id, x_m_0...x_m_d-1, b_new_0...b_new_d-1
        if (toks.size() == (size_t)(1 + 2 * ambientDim)) {
            m.origin = Eigen::VectorXd(ambientDim);
            m.basis = Eigen::MatrixXd::Zero(ambientDim, 1); // single direction = difference

            for (int i = 0; i < ambientDim; ++i)
                m.origin(i) = std::stod(toks[idx++]);
            for (int i = 0; i < ambientDim; ++i)
                m.basis(i, 0) = std::stod(toks[idx++]) - m.origin(i); // direction = b_new - x_m

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
// assigns each point to closest subspace by pointSubspaceDistance,
// writes output CSV replacing flat number with cluster_id
// ---------------------------------------------------------
inline void assignPointsToSubspaces(
    const std::string &generatedDataFile,
    const std::string &allLinesFile,
    const std::string &outputFile)
{
    std::vector<int> originalLabels;
    int ambientDim = 0;
    auto points = loadGeneratedPointsCSV(generatedDataFile, originalLabels, ambientDim);
    if (points.empty())
        throw std::runtime_error("No points loaded from " + generatedDataFile);

    auto models = loadSubspacesFromCSV_fixedDim(allLinesFile, ambientDim);
    if (models.empty()) {
        std::cerr << "Warning: no valid subspace models loaded from " << allLinesFile << "\n";
        // still write a file with cluster_id = -1 for all points
    }

    std::ofstream out(outputFile);
    if (!out.is_open())
        throw std::runtime_error("Cannot open output file: " + outputFile);

    // header x0,x1,...,x{d-1},cluster_id
    for (int i = 0; i < ambientDim; ++i) {
        out << "x" << i << ",";
    }
    out << "cluster_id\n";

    for (size_t i = 0; i < points.size(); ++i) {
        double minDist = std::numeric_limits<double>::infinity();
        int bestCluster = -1;

        for (const auto &m : models) {
            // guard: origin and basis dimension should match ambientDim
            if (m.origin.size() != ambientDim || m.basis.rows() != ambientDim) {
                std::cerr << "Skipping model with mismatched dimension (clusterId=" << m.clusterId << ")\n";
                continue;
            }

            double d = pointSubspaceDistance(points[i], m.origin, m.basis);
            if (d < minDist) {
                minDist = d;
                bestCluster = m.clusterId;
            }
        }

        // write point and assigned cluster (or -1 if none)
        for (int j = 0; j < ambientDim; ++j) {
            out << points[i](j) << ",";
        }
        out << bestCluster << "\n";
    }

    std::cout << "Saved clustered points to " << outputFile << " (assigned using " << models.size() << " subspaces)\n";
}
