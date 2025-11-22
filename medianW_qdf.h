#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>

class MedianWQDF {
public:
    static void computeMedianWQDF(const std::string& input_filename, const std::string& output_filename) {
        std::ifstream in(input_filename);
        if (!in) {
            std::cerr << "Error: cannot open " << input_filename << "\n";
            return;
        }

        std::string header;
        std::getline(in, header);

        std::string firstLine;
        std::getline(in, firstLine);
        if (firstLine.empty()) {
            std::cerr << "Error: CSV appears empty after header.\n";
            return;
        }

        // Infer dimension
        std::vector<std::string> firstTokens = splitCSV(firstLine);
        int total_cols = (int)firstTokens.size();

        // Formula: total_cols = 1 (id) + n*n (A) + n (r) -> n^2 + n + 1 - total = 0
        int n = std::round((std::sqrt(4 * (total_cols - 2) + 1) - 1) / 2);
        if (n <= 0) {
            std::cerr << "Error: cannot infer dimension from CSV, got " << total_cols << " columns.\n";
            return;
        }

        std::cout << "Detected dimension n = " << n << " from " << total_cols << " columns.\n";

        // Reset reading
        in.clear();
        in.seekg(0);
        std::getline(in, header);

        // Store samples as flattened vectors for Weiszfeld
        // Each sample is a vector of size (n*n + n)
        std::map<int, std::vector<Eigen::VectorXd>> clusters;

        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> tokens = splitCSV(line);
            if ((int)tokens.size() < total_cols) continue;

            int cluster_id = std::stoi(tokens[0]);

            // Flatten A and r into a single vector
            // Vector layout: [A_00, A_01, ..., A_nn, r_0, ..., r_n]
            Eigen::VectorXd sample(n * n + n);
            int idx = 1;
            
            // Fill A part
            for (int i = 0; i < n * n; ++i) {
                sample(i) = std::stod(tokens[idx++]);
            }

            // Fill r part (note: input CSV stores r = -2*A*p0, so r = input / -2.0 is not needed if CSV is raw coeffs)
            // BUT your previous code did: r(i) = std::stod(tokens[idx++]) / -2.0;
            // I will keep your logic: reading the raw values, converting, then storing.
            for (int i = 0; i < n; ++i) {
                sample(n * n + i) = std::stod(tokens[idx++]) / -2.0; 
            }

            clusters[cluster_id].push_back(sample);
        }

        // Output file
        std::ofstream out(output_filename);
        out << "cluster_id";
        for (int i = 0; i < n; ++i) out << ",x_m_" << i;
        for (int i = 0; i < n; ++i) out << ",b_new_" << i;
        out << "\n";

        // Process each cluster
        for (auto& [cluster_id, samples] : clusters) {
            int count = samples.size();
            if (count < 1) continue;

            // ---- Compute Geometric Median via Weiszfeld Algorithm ----
            Eigen::VectorXd median_flat = computeGeometricMedian(samples);

            // ---- Unpack Median into A_med and r_med ----
            Eigen::MatrixXd A_med(n, n);
            Eigen::VectorXd r_med(n);

            int idx = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A_med(i, j) = median_flat(idx++);
                }
            }
            for (int i = 0; i < n; ++i) {
                r_med(i) = median_flat(idx++);
            }

            // ---- Original Logic: Eigen decomposition & Reconstruction ----
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_med);
            // Use standard eigenvectors/values (sorted low to high by Eigen usually)
            Eigen::MatrixXd U = es.eigenvectors();
            Eigen::VectorXd lambda = es.eigenvalues();

            // Smallest eigenvector is typically column 0 in Eigen's SelfAdjointEigenSolver
            Eigen::VectorXd A_out = U.col(0); 

            // Pseudoinverse of Lambda
            Eigen::MatrixXd lambda_pinv = Eigen::MatrixXd::Zero(n, n);
            double eps = 1e-10;
            for (int i = 0; i < n; ++i)
                if (std::abs(lambda(i)) > eps)
                    lambda_pinv(i, i) = 1.0 / lambda(i);

            Eigen::MatrixXd A_med_pinv = U * lambda_pinv * U.transpose();

            Eigen::VectorXd x_m = A_med_pinv * r_med;
            Eigen::VectorXd b_new = x_m - A_out * (A_out.transpose() * x_m);

            out << cluster_id;
            for (int i = 0; i < n; ++i) out << "," << x_m(i);
            for (int i = 0; i < n; ++i) out << "," << b_new(i);
            out << "\n";
        }

        std::cout << "✅ Median QDF (Weiszfeld) written to " << output_filename << "\n";
    }

private:
    // --- Weiszfeld Algorithm Implementation ---
    static Eigen::VectorXd computeGeometricMedian(const std::vector<Eigen::VectorXd>& points, int max_iter = 100, double tol = 1e-5) {
        if (points.empty()) return Eigen::VectorXd();
        if (points.size() == 1) return points[0];
        if (points.size() == 2) return 0.5 * (points[0] + points[1]);

        // Initial guess: Arithmetic Mean (Center of Mass)
        Eigen::VectorXd current_median = Eigen::VectorXd::Zero(points[0].size());
        for (const auto& p : points) current_median += p;
        current_median /= (double)points.size();

        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd numerator = Eigen::VectorXd::Zero(current_median.size());
            double denominator = 0.0;
            bool hit_point = false;

            for (const auto& p : points) {
                double dist = (p - current_median).norm();

                // Handle singularity (if current guess lands exactly on a data point)
                if (dist < 1e-12) {
                    hit_point = true;
                    // If we land on a point, the algorithm can get stuck or divide by zero.
                    // A simple fix for production code is to perturb slightly or just skip update.
                    // Here we break and assume we found a "good enough" median or move slightly.
                    // Proper handling: check optimality condition for this point.
                    // For simplicity in this snippet: just perturb.
                    dist = 1e-12; 
                }

                double weight = 1.0 / dist;
                numerator += weight * p;
                denominator += weight;
            }

            Eigen::VectorXd next_median = numerator / denominator;
            
            if ((next_median - current_median).norm() < tol) {
                return next_median;
            }
            current_median = next_median;
        }
        return current_median;
    }

    static std::vector<std::string> splitCSV(const std::string& line) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            tokens.push_back(item);
        }
        return tokens;
    }
};