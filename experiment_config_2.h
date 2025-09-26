#pragma once
#include <string>
#include <random>

// ===== Synthetic Data Creation =====
inline const int ambientDim = 2;
inline const int numFlats = 7;
inline const double originSpread = 6.0;
inline const double coordExtent = 1.0;
inline const double noiseStd = 0.4;
inline const int numDataPointsPerFlat = 300;
// ===== RANSAC Parameters =====
inline constexpr int MIN_INLIERS = numDataPointsPerFlat * 0.15;
inline constexpr int RANSAC_ITERATIONS = 1000;
inline constexpr double RANSAC_THRESHOLD = 0.45;
inline constexpr int FIXED_DIMENSION = 1;
inline constexpr double JACCARD_THRESHOLD = 0.08;
// ===== RNG =====
inline std::random_device rd;
inline std::mt19937 rng(632);
inline std::uniform_int_distribution<int> flatDimDist(1, ambientDim - 1);
// ===== Output File Names =====
inline const std::string pointsCSV = "../generated_data_experiment2.csv";
inline const std::string ransacCSV = "../detected_subspaces_experiment2.csv";
