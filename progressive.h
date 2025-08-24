#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <random>
#include <iostream>

#include "ransac_multiD.h"

// Progressive-X
inline std::vector<AffineSubspaceModel> progressiveAffine(
    const std::vector<Eigen::VectorXd>& points,
    int iterations,
    double threshold,
    int min_inliers,
    int fixedDim)
{
    if (points.empty()) return {};

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    const int ambientDim = points[0].size();
    std::vector<AffineSubspaceModel> models;

    // Candidate models collected during iterations
    std::vector<AffineSubspaceModel> candidates;

    for (int iter = 0; iter < iterations; ++iter) {
        int k = fixedDim;
        if (k <= 0 || k >= ambientDim) {
            throw std::runtime_error("Invalid FIXED_DIMENSION for progressive RANSAC");
        }

        // Sample k+1 unique points
        std::unordered_set<int> chosen;
        while (chosen.size() < static_cast<size_t>(k + 1))
            chosen.insert(dist(rng));

        std::vector<Eigen::VectorXd> sample;
        for (int idx : chosen) sample.push_back(points[idx]);

        auto model = fitAffineSubspace(sample);
        model.basis = model.basis.leftCols(k);

        // Build consensus set
        std::unordered_set<int> inliers;
        for (int i = 0; i < points.size(); ++i) {
            if (pointSubspaceDistance(points[i], model.origin, model.basis) < threshold)
                inliers.insert(i);
        }

        if ((int)inliers.size() >= min_inliers) {
            model.inliers = inliers;
            candidates.push_back(model);
        }
    }

    // === Progressive-X selection ===
    std::vector<bool> used(points.size(), false);

    for (auto& cand : candidates) {
        int newInliers = 0;
        for (int idx : cand.inliers) {
            if (!used[idx]) newInliers++;
        }

        if (newInliers >= min_inliers) {
            models.push_back(cand);
            for (int idx : cand.inliers) used[idx] = true;
        }
    }

    return models;
}
