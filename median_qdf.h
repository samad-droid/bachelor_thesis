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

class MedianQDF {
public:
    static void computeMedianQDF(const std::string& input_filename, const std::string& output_filename) {
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

        // Store all samples per cluster
        struct ClusterData {
            std::vector<Eigen::MatrixXd> A_list;
            std::vector<Eigen::VectorXd> r_list;
        };

        std::map<int, ClusterData> clusters;

        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> tokens = splitCSV(line);
            if ((int)tokens.size() < total_cols) continue;

            int cluster_id = std::stoi(tokens[0]);

            int idx = 1;
            Eigen::MatrixXd A(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    A(i, j) = std::stod(tokens[idx++]);

            Eigen::VectorXd r(n);
            for (int i = 0; i < n; ++i)
                r(i) = std::stod(tokens[idx++]) / -2.0;

            clusters[cluster_id].A_list.push_back(A);
            clusters[cluster_id].r_list.push_back(r);
        }

        // Output file
        std::ofstream out(output_filename);
        out << "cluster_id";
        for (int i = 0; i < n; ++i) out << ",x_m_" << i;
        for (int i = 0; i < n; ++i) out << ",b_new_" << i;
        out << "\n";

        // Process each cluster
        for (auto& [cluster_id, data] : clusters) {
            int count = data.A_list.size();
            if (count < 2) continue;

            // ---- Compute elementwise median A ----
            Eigen::MatrixXd A_med(n, n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    std::vector<double> vals(count);
                    for (int k = 0; k < count; ++k)
                        vals[k] = data.A_list[k](i, j);
                    std::sort(vals.begin(), vals.end());
                    A_med(i, j) = vals[count / 2];
                }
            }

            // ---- Compute elementwise median r ----
            Eigen::VectorXd r_med(n);
            for (int i = 0; i < n; ++i) {
                std::vector<double> vals(count);
                for (int k = 0; k < count; ++k)
                    vals[k] = data.r_list[k](i);
                std::sort(vals.begin(), vals.end());
                r_med(i) = vals[count / 2];
            }

            // Eigen decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A_med);
            Eigen::MatrixXd U = es.eigenvectors();
            Eigen::VectorXd lambda = es.eigenvalues();

            Eigen::VectorXd A_out = U.col(0); // smallest eigenvector

            // Pseudoinverse
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

        std::cout << "✅ Median QDF written to " << output_filename << "\n";
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
