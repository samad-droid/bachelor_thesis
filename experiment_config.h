#pragma once
#include <string>
#include <random>

#include "experiment_config.h"

// ===== Synthetic Data Creation =====
inline const int ambientDim = 3;
inline const int numFlats = 7;
inline const double originSpread = 8.0;
inline const double coordExtent = 1.0;
inline const double noiseStd = 0.35;
inline const int numDataPointsPerFlat = 300;
// ===== RANSAC Parameters =====
inline constexpr int MAX_MODELS = 100;
inline constexpr int MIN_INLIERS = numDataPointsPerFlat * 0.10;
// inline constexpr int MIN_INLIERS = numDataPointsPerFlat * (1.0/numFlats);
inline constexpr int RANSAC_ITERATIONS = 15;
inline constexpr double RANSAC_THRESHOLD = 0.2;
inline constexpr int FIXED_DIMENSION = 1;
inline constexpr double JACCARD_THRESHOLD = 0.1;
// ===== RNG =====
inline std::random_device rd;
inline std::mt19937 rng(54);
inline std::uniform_int_distribution<int> flatDimDist(1, ambientDim - 1);
// ===== 2d Synthetic Data =====
inline const std::string stair4CSV = "../stair_4.csv";
inline const std::string star5CSV = "../star5_data.csv";
inline const std::string star11CSV = "../star11_data.csv";
// ===== Output File Names =====
inline const std::string pointsCSV = "../generated_data_experiment1.csv";
inline const std::string ransacCSV = "../detected_subspaces_experiment1.csv";
inline const std::string qdfCSV = "../qdf_representation_experiment1.csv";
inline const std::string meanCSV = "../mean_qdf_experiment1.csv";
inline const std::string outputCSV = "../clustered_data_experiment1.csv";
//inline const std::string mergeCSV = "../all_lines_experiment1.csv";
// ===== Run Parameters =====
inline const int NUM_RUNS = 3;
inline const std::string inputCSV = pointsCSV;