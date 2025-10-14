
#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>

class MeanQDF {
public:
    static void computeMeanQDF(const std::string& input_filename, const std::string& output_filename) {
        // Read input CSV
        std::ifstream in(input_filename);
        if (!in) {
            std::cerr << "Error: cannot open " << input_filename << "\n";
            return;
        }

        // Skip header
        std::string header;
        std::getline(in, header);

        // Maps to store sums for each cluster
        std::map<int, Eigen::Matrix3d> Q_sums;  // Sum of A matrices
        std::map<int, Eigen::Vector3d> r_sums;  // Sum of b vectors (divided by -2)
        std::map<int, int> cluster_counts;      // Count of entries per cluster

        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> tokens = splitCSV(line);
            if (tokens.size() < 14) continue;  // cluster + 9 for A + 3 for b + 1 for c

            int cluster_id = std::stoi(tokens[0]);

            // Read A matrix (Q in the quadratic form)
            Eigen::Matrix3d A;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    A(i, j) = std::stod(tokens[1 + i*3 + j]);
                }
            }

            // Read b vector and divide by -2 to get r
            Eigen::Vector3d r;
            for (int i = 0; i < 3; i++) {
                r(i) = std::stod(tokens[10 + i]) / -2.0;
            }

            // Accumulate sums
            if (Q_sums.find(cluster_id) == Q_sums.end()) {
                Q_sums[cluster_id] = A;
                r_sums[cluster_id] = r;
                cluster_counts[cluster_id] = 1;
            } else {
                Q_sums[cluster_id] += A;
                r_sums[cluster_id] += r;
                cluster_counts[cluster_id]++;
            }
        }

        // Process each cluster and write results
        std::ofstream out(output_filename);
        out << "cluster_id,x_m_0,x_m_1,x_m_2,b_new_0,b_new_1,b_new_2\n";

        for (const auto& [cluster_id, Q_sum] : Q_sums) {
            // ✅ Skip clusters with less than 2 lines
            if (cluster_counts[cluster_id] < 2) {
                continue;
            }

            // 1. Compute Q* and r*
            Eigen::Matrix3d Q_star = Q_sum;
            Eigen::Vector3d r_star = r_sums[cluster_id];

            // 2. Eigendecomposition of Q*
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Q_star);
            Eigen::Matrix3d U = es.eigenvectors();
            Eigen::Vector3d lambda = es.eigenvalues();

            // Get eigenvector corresponding to smallest eigenvalue (A_out)
            Eigen::Vector3d A_out = U.col(0);  // Assuming eigenvalues are in ascending order

            // 3. Compute pseudoinverse
            Eigen::Matrix3d lambda_pinv = Eigen::Matrix3d::Zero();
            double epsilon = 1e-10;  // Threshold for considering eigenvalue as zero
            for (int i = 0; i < 3; i++) {
                if (std::abs(lambda(i)) > epsilon) {
                    lambda_pinv(i, i) = 1.0 / lambda(i);
                }
            }
            Eigen::Matrix3d Q_star_pinv = U * lambda_pinv * U.transpose();

            // 4. Compute minimizer x_m
            Eigen::Vector3d x_m = Q_star_pinv * r_star;

            // 5. Compute closest point to origin
            Eigen::Vector3d b_new = x_m - A_out * (A_out.transpose() * x_m);

            // Write results to output file
            out << cluster_id << ","
                << x_m(0) << "," << x_m(1) << "," << x_m(2) << ","
                << b_new(0) << "," << b_new(1) << "," << b_new(2) << "\n";
        }
    }

private:
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
