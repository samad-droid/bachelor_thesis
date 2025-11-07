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

class MeanQDF {
public:
    static void computeMeanQDF(const std::string& input_filename, const std::string& output_filename) {
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

        // Peek into first data line to infer dimension
        std::vector<std::string> firstTokens = splitCSV(firstLine);
        int total_cols = (int)firstTokens.size();

        // total = 1 (cluster) + n*n (A) + n (b) + 1 (c)
        int n = std::round((std::sqrt(4 * (total_cols - 2) + 1) - 1) / 2);
        if (n <= 0) {
            std::cerr << "Error: cannot infer dimension from CSV, got " << total_cols << " columns.\n";
            return;
        }

        std::cout << "Detected dimension n = " << n << " from " << total_cols << " columns.\n";

        // Reset stream to process all lines including the one we read
        in.clear();
        in.seekg(0);
        std::getline(in, header); // skip header again

        // Data structures
        std::map<int, Eigen::MatrixXd> Q_sums;
        std::map<int, Eigen::VectorXd> r_sums;
        std::map<int, int> cluster_counts;

        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> tokens = splitCSV(line);
            if ((int)tokens.size() < total_cols) continue;

            int cluster_id = std::stoi(tokens[0]);

            // Read A (n x n)
            Eigen::MatrixXd A(n, n);
            int idx = 1;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    A(i, j) = std::stod(tokens[idx++]);
                }
            }

            // Read b (divided by -2)
            Eigen::VectorXd r(n);
            for (int i = 0; i < n; ++i) {
                r(i) = std::stod(tokens[idx++]) / -2.0;
            }

            // Skip c term (last token)
            // double c = std::stod(tokens[idx]);

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

        // Output
        std::ofstream out(output_filename);
        out << "cluster_id";
        for (int i = 0; i < n; ++i) out << ",x_m_" << i;
        for (int i = 0; i < n; ++i) out << ",b_new_" << i;
        out << "\n";

        for (const auto& [cluster_id, Q_sum] : Q_sums) {
            if (cluster_counts[cluster_id] < 2) continue;

            Eigen::MatrixXd Q_star = Q_sum;
            Eigen::VectorXd r_star = r_sums[cluster_id];

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q_star);
            Eigen::MatrixXd U = es.eigenvectors();
            Eigen::VectorXd lambda = es.eigenvalues();

            Eigen::VectorXd A_out = U.col(0);  // smallest eigenvector

            // Pseudoinverse
            Eigen::MatrixXd lambda_pinv = Eigen::MatrixXd::Zero(n, n);
            double eps = 1e-10;
            for (int i = 0; i < n; ++i)
                if (std::abs(lambda(i)) > eps)
                    lambda_pinv(i, i) = 1.0 / lambda(i);

            Eigen::MatrixXd Q_star_pinv = U * lambda_pinv * U.transpose();

            Eigen::VectorXd x_m = Q_star_pinv * r_star;
            Eigen::VectorXd b_new = x_m - A_out * (A_out.transpose() * x_m);

            out << cluster_id;
            for (int i = 0; i < n; ++i) out << "," << x_m(i);
            for (int i = 0; i < n; ++i) out << "," << b_new(i);
            out << "\n";
        }

        std::cout << "✅ Mean QDF written to " << output_filename << "\n";
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
