#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace QDFAnalysis {

using Matrix = Eigen::MatrixXd;

// Helper to split CSV line by comma
inline std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        tokens.push_back(item);
    }
    return tokens;
}

// Read QDF CSV and analyze clusters
inline void analyzeQDFClusters(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return;
    }

    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        std::cerr << "Error: empty file " << filename << "\n";
        return;
    }
    auto header = splitCSV(headerLine);

    // find A.. columns
    std::vector<int> aIndices;
    for (size_t i = 0; i < header.size(); ++i) {
        if (!header[i].empty() && (header[i][0] == 'A' || header[i][0] == 'a')) {
            aIndices.push_back((int)i);
        }
    }
    if (aIndices.empty()) {
        std::cerr << "No Aij columns found in header.\n";
        return;
    }

    int numA = (int)aIndices.size();
    int d = (int)std::round(std::sqrt(numA));
    if (d * d != numA) {
        std::cerr << "Error: found " << numA << " A entries, not a square.\n";
        return;
    }

    // cluster_id -> vector of A matrices
    std::map<int, std::vector<Matrix>> clusters;

    std::string line;
    int rowIdx = 0;
    while (std::getline(in, line)) {
        ++rowIdx;
        if (line.empty()) continue;

        auto tokens = splitCSV(line);
        if (tokens.size() < 1 + numA) continue;

        int clusterId = std::stoi(tokens[0]);
        Matrix A(d, d);
        for (int k = 0; k < numA; ++k) {
            int col = aIndices[k];
            double val = std::stod(tokens[col]);
            int r = k / d;
            int c = k % d;
            A(r, c) = val;
        }
        clusters[clusterId].push_back(A);
    }

    std::cout << std::fixed << std::setprecision(6);

    // Process each cluster
    for (auto& kv : clusters) {
        int cid = kv.first;
        auto& subspaces = kv.second;
        int n = (int)subspaces.size();
        if (n <= 1) continue;

        std::cout << "\n=== Cluster " << cid << " : " << n << " subspaces ===\n";
        std::cout << "Pair , Eigenvalues\n";

        // Precompute Q = I - A A^T
        Matrix I = Matrix::Identity(d, d);
        std::vector<Matrix> Qs;
        for (auto& A : subspaces) {
            Matrix Q = I - A * A.transpose();
            Q = 0.5 * (Q + Q.transpose()); // enforce symmetry
            Qs.push_back(Q);
        }

        // Compute pairwise eigenvalues
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Matrix M = 0.5 * (Qs[i] + Qs[j]);
                M = 0.5 * (M + M.transpose());

                Eigen::SelfAdjointEigenSolver<Matrix> es(M);
                if (es.info() != Eigen::Success) {
                    std::cout << i+1 << "-" << j+1 << " , <eig-fail>\n";
                    continue;
                }

                auto eigs = es.eigenvalues();
                std::cout << i+1 << "-" << j+1 << " , ";
                for (int k = 0; k < eigs.size(); k++) {
                    std::cout << eigs[k];
                    if (k + 1 < eigs.size()) std::cout << ", ";
                }
                std::cout << "\n";
            }
        }
    }
}

} // namespace QDFAnalysis
