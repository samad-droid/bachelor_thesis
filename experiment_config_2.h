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
inline constexpr int MAX_MODELS = 20;
inline constexpr int MIN_INLIERS = 20;
inline constexpr int RANSAC_ITERATIONS = 200;
inline constexpr double RANSAC_THRESHOLD = 0.012;
inline constexpr int FIXED_DIMENSION = 1;
inline constexpr double JACCARD_THRESHOLD = 0.22;
// ===== Clustering =====
inline constexpr double CLUSTERING_THRESHOLD = 0.02;
// ===== RNG =====
inline std::random_device rd;
inline std::mt19937 rng(632);
inline std::uniform_int_distribution<int> flatDimDist(1, ambientDim - 1);
// ===== 2d Synthetic Data =====
inline const std::string stair4CSV = "../stair_4.csv";
inline const std::string star5CSV = "../star5_data.csv";
inline const std::string star11CSV = "../star11_data.csv";
inline const std::string inputCSV = star11CSV;
// ===== Output File Names =====
inline const std::string pointsCSV = "../generated_data_experiment2.csv";
inline const std::string ransacCSV = "../detected_subspaces_experiment2.csv";
inline const std::string qdfCSV = "../qdf_representation_experiment2.csv";
inline const std::string meanCSV = "../mean_qdf_experiment2.csv";
inline const std::string outputCSV = "../clustered_data_experiment2.csv";
inline const std::string mergeCSV = "../all_lines_experiment2.csv";
// ===== Run Parameters =====
inline const int NUM_RUNS = 10;