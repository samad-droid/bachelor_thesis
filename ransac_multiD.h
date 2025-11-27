#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <iostream>

#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/curve_network.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "color_utils.h"
// ALGLIB INCLUDES (Adjust path if needed based on your CMake setup)
#include "external/alglib/alglib-cpp/src/dataanalysis.h"
#include "external/alglib/alglib-cpp/src/stdafx.h"

// ===== Model definition =====
struct AffineSubspaceModel {
    Eigen::VectorXd origin;                  // point on subspace
    Eigen::MatrixXd basis;                   // columns = orthonormal basis vectors
    std::unordered_set<int> inliers;         // indices of inliers
    int clusterId = -1;
};

// ===== CSV loader for any dimension =====
inline std::vector<Eigen::VectorXd> loadCSV(const std::string& filename) {
    std::vector<Eigen::VectorXd> points;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> coords;
        while (std::getline(ss, cell, ',')) coords.push_back(std::stod(cell));
        Eigen::VectorXd p(coords.size());
        for (size_t i = 0; i < coords.size(); ++i) p(i) = coords[i];
        points.push_back(p);
    }
    return points;
}

// ===== Distance from point to affine subspace =====
inline double pointSubspaceDistance(const Eigen::VectorXd& p,
                                    const Eigen::VectorXd& origin,
                                    const Eigen::MatrixXd& basis) {
    Eigen::VectorXd diff = p - origin;
    Eigen::VectorXd proj = basis * (basis.transpose() * diff);
    return (diff - proj).norm();
}

// ===== Fit affine subspace from a set of points =====
inline AffineSubspaceModel fitAffineSubspace(const std::vector<Eigen::VectorXd>& sample) {
    const int dim = sample[0].size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    for (auto& p : sample) mean += p;
    mean /= sample.size();

    Eigen::MatrixXd centered(dim, sample.size());
    for (size_t i = 0; i < sample.size(); ++i)
        centered.col(i) = sample[i] - mean;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU);
    Eigen::MatrixXd basis = svd.matrixU().leftCols(sample.size()-1); // span of points

    return {mean, basis, {}};
}

// ===== RANSAC for variable-dimension affine subspaces =====
inline AffineSubspaceModel ransacAffine(const std::vector<Eigen::VectorXd>& points,
                                        int iterations, double threshold, int min_inliers,
                                        int fixedDim = 0) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    AffineSubspaceModel bestModel;
    int bestInlierCount = 0;
    const int ambientDim = points[0].size();

    for (int iter = 0; iter < iterations; ++iter) {
        int k;
        if (fixedDim > 0) {
            k = fixedDim;
        } else {
            // Random dimension between 1 and ambientDim-1
            k = 1 + (rng() % (ambientDim - 1));
        }

        // Sample k+1 unique points
        std::unordered_set<int> chosen;
        while (chosen.size() < static_cast<size_t>(k+1))
            chosen.insert(dist(rng));
        std::vector<Eigen::VectorXd> sample;
        for (int idx : chosen) sample.push_back(points[idx]);

        auto model = fitAffineSubspace(sample);
        model.basis = model.basis.leftCols(k);

        std::unordered_set<int> inliers;
        for (int i = 0; i < points.size(); ++i) {
            if (pointSubspaceDistance(points[i], model.origin, model.basis) < threshold)
                inliers.insert(i);
        }

        if ((int)inliers.size() > bestInlierCount && (int)inliers.size() >= min_inliers) {
            bestInlierCount = (int)inliers.size();
            model.inliers = inliers;
            bestModel = model;
        }
    }

    return bestModel;
}

// ===== Remove inliers =====
inline std::vector<Eigen::VectorXd> removeInliers(const std::vector<Eigen::VectorXd>& points,
                                                  const std::unordered_set<int>& inliers) {
    std::vector<Eigen::VectorXd> remaining;
    for (int i = 0; i < points.size(); ++i)
        if (inliers.find(i) == inliers.end()) remaining.push_back(points[i]);
    return remaining;
}

// ===== Multi-model RANSAC =====
/*
inline std::vector<AffineSubspaceModel> multiRansacAffine(std::vector<Eigen::VectorXd> data,
                                                          int iterations, double threshold, int min_inliers,
                                                          int fixedDim = 0) {
    std::vector<AffineSubspaceModel> models;
    std::vector<int> globalIndices(data.size());
    std::iota(globalIndices.begin(), globalIndices.end(), 0); // 0,1,2,...

    while (static_cast<int>(data.size()) >= min_inliers) {
        auto best = ransacAffine(data, iterations, threshold, min_inliers, fixedDim);
        if (static_cast<int>(best.inliers.size()) < min_inliers) break;

        // Convert local inliers → global indices
        std::unordered_set<int> globalInliers;
        for (int idx : best.inliers)
            globalInliers.insert(globalIndices[idx]);
        best.inliers = std::move(globalInliers);

        models.push_back(best);

        // Remove inliers from both data and globalIndices
        std::vector<Eigen::VectorXd> newData;
        std::vector<int> newGlobal;
        for (int i = 0; i < data.size(); ++i) {
            if (best.inliers.find(globalIndices[i]) == best.inliers.end()) {
                newData.push_back(data[i]);
                newGlobal.push_back(globalIndices[i]);
            }
        }
        data = std::move(newData);
        globalIndices = std::move(newGlobal);
    }

    return models;
}
*/


inline std::vector<AffineSubspaceModel> multiRansacAffine(const std::vector<Eigen::VectorXd>& data,
                                                          int iterations, double threshold, int min_inliers,
                                                          int fixedDim = 0, int maxModels = 100) {
    std::vector<AffineSubspaceModel> models;

    // Keep finding models until we hit maxModels or can't find enough inliers
    while (models.size() < maxModels) {
        auto best = ransacAffine(data, iterations, threshold, min_inliers, fixedDim);
        if (static_cast<int>(best.inliers.size()) < min_inliers) break;

        models.push_back(best);
    }

    return models;
}

inline void recomputeAllInliers(std::vector<AffineSubspaceModel>& models,
                                const std::vector<Eigen::VectorXd>& allData,
                                double threshold) {
    for (auto& model : models) {
        model.inliers.clear();
        for (int j = 0; j < allData.size(); ++j) {
            double dist = pointSubspaceDistance(allData[j], model.origin, model.basis);
            if (dist < threshold) {
                model.inliers.insert(j);
            }
        }
    }
}

inline void saveSubspacesToCSV(const std::vector<AffineSubspaceModel>& models,
                               const std::string& filename) {
    // First, count subspaces per cluster
    std::map<int, int> clusterCounts;
    for (const auto& m : models) {
        clusterCounts[m.clusterId]++;
    }

    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    out << "cluster_id,origin...,basis_dim,basis_vectors...\n";
    int savedCount = 0;

    for (const auto& m : models) {
        // Skip clusters with only 1 subspace
        if (clusterCounts[m.clusterId] <= 1) continue;

        out << m.clusterId << ",";

        for (int i = 0; i < m.origin.size(); ++i) {
            out << m.origin(i) << ",";
        }
        out << m.basis.cols() << ",";

        for (int c = 0; c < m.basis.cols(); ++c) {
            for (int r = 0; r < m.basis.rows(); ++r) {
                out << m.basis(r, c);
                if (!(c == m.basis.cols() - 1 && r == m.basis.rows() - 1)) out << ",";
            }
        }
        out << "\n";
        savedCount++;
    }
    std::cout << "Saved " << savedCount << " subspaces to " << filename << "\n";
}



// ===== 3D visualization =====
void visualizeSubspace3D(const AffineSubspaceModel& model, const std::string& name, const glm::vec3& color) {
    int dim = model.basis.cols();
    if (dim < 1 || dim > 3) {
        std::cerr << "Error: unsupported subspace dimension: " << dim << "\n";
        return;
    }

    Eigen::Vector3d origin = model.origin;

    if (dim == 1) {
        Eigen::Vector3d p1 = origin - model.basis.col(0) * 5.0;
        Eigen::Vector3d p2 = origin + model.basis.col(0) * 5.0;

        std::vector<std::array<double, 3>> points = {
            {p1.x(), p1.y(), p1.z()},
            {p2.x(), p2.y(), p2.z()}
        };
        std::vector<std::array<size_t, 2>> edges = {{0, 1}};
        auto curve = polyscope::registerCurveNetwork(name, points, edges);
        curve->setColor(color);

    } else if (dim == 2) {
        double s = 5.0;
        Eigen::Vector3d u = model.basis.col(0);
        Eigen::Vector3d v = model.basis.col(1);

        std::vector<std::array<double, 3>> squarePoints = {
            { (origin + (-s) * u + (-s) * v).x(), (origin + (-s) * u + (-s) * v).y(), (origin + (-s) * u + (-s) * v).z() },
            { (origin + (-s) * u + ( s) * v).x(), (origin + (-s) * u + ( s) * v).y(), (origin + (-s) * u + ( s) * v).z() },
            { (origin + ( s) * u + ( s) * v).x(), (origin + ( s) * u + ( s) * v).y(), (origin + ( s) * u + ( s) * v).z() },
            { (origin + ( s) * u + (-s) * v).x(), (origin + ( s) * u + (-s) * v).y(), (origin + ( s) * u + (-s) * v).z() }
        };

        std::vector<std::array<size_t, 2>> squareEdges = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}
        };

        auto plane = polyscope::registerCurveNetwork(name + "_plane", squarePoints, squareEdges);
        plane->setColor(color);

    } else {
        std::cerr << "3D subspace visualization not implemented\n";
    }
}

// ========================= 2D Visualization Helpers =========================

// visualize points in 2D (embedded in XY plane)
inline void visualizeFlatSamples2D(const Eigen::MatrixXd& pts, const std::string& name, float radius = 0.01f) {
    std::vector<glm::vec3> ps;
    ps.reserve(pts.rows());
    for (int i = 0; i < pts.rows(); ++i) {
        float x = static_cast<float>(pts(i, 0));
        float y = static_cast<float>(pts(i, 1));
        ps.emplace_back(x, y, 0.0f); // embed in XY plane
    }
    auto pc = polyscope::registerPointCloud(name, ps);
    if (pc) pc->setPointRadius(radius);
}

// visualize a 2D affine subspace (dim 0, 1, or 2)
template<typename ModelT>
inline void visualizeSubspace2D(const ModelT& model,
                                const std::string& name,
                                const std::vector<Eigen::VectorXd>& allVecPoints,
                                const glm::vec3& color,          // NEW
                                double coordExtent = 1.0) {
    if (model.origin.size() < 2) return;
    Eigen::Vector2d origin2 = model.origin.head(2).template cast<double>();
    int subDim = model.basis.cols();

    // 0D: just the origin
    if (subDim == 0) {
        std::vector<glm::vec3> pts{{(float)origin2(0), (float)origin2(1), 0.0f}};
        auto pc = polyscope::registerPointCloud(name + " (origin)", pts);
        if (pc) {
            pc->setPointRadius(0.05f);
            pc->setPointColor(color);   // apply color
        }
        return;
    }

    // 1D: line along basis.col(0)
    if (subDim == 1) {
        Eigen::Vector2d dir = model.basis.col(0).head(2).template cast<double>();
        double nrm = dir.norm();
        if (nrm < 1e-12) return;
        dir /= nrm;

        double tmin = 0.0, tmax = 0.0;
        bool firstT = true;
        for (int idx : model.inliers) {
            if (idx < 0 || idx >= (int)allVecPoints.size()) continue;
            Eigen::Vector2d p2 = allVecPoints[idx].head(2).template cast<double>();
            double t = dir.dot(p2 - origin2);
            if (firstT) { tmin = tmax = t; firstT = false; }
            else { tmin = std::min(tmin, t); tmax = std::max(tmax, t); }
        }
        if (firstT) { tmin = -coordExtent; tmax = coordExtent; }

        double span = (tmax - tmin);
        double margin = std::max(0.1 * std::max(1.0, span), 0.1);
        double start = tmin - margin;
        double end = tmax + margin;

        const int SAMPLES = 200;
        std::vector<glm::vec3> linePts; linePts.reserve(SAMPLES+1);
        for (int s = 0; s <= SAMPLES; ++s) {
            double t = start + (end - start) * (double)s / (double)SAMPLES;
            Eigen::Vector2d pos = origin2 + t * dir;
            linePts.emplace_back((float)pos(0), (float)pos(1), 0.0f);
        }
        auto pc = polyscope::registerPointCloud(name + " (line)", linePts);
        if (pc) {
            pc->setPointRadius(0.002f);
            pc->setPointColor(color);   // apply color
        }
        return;
    }

    // 2D: full plane in XY
    if (subDim >= 2) {
        double xmin =  std::numeric_limits<double>::infinity();
        double xmax = -std::numeric_limits<double>::infinity();
        double ymin =  std::numeric_limits<double>::infinity();
        double ymax = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (int idx : model.inliers) {
            if (idx < 0 || idx >= (int)allVecPoints.size()) continue;
            Eigen::Vector2d p2 = allVecPoints[idx].head(2).template cast<double>();
            xmin = std::min(xmin, p2(0)); xmax = std::max(xmax, p2(0));
            ymin = std::min(ymin, p2(1)); ymax = std::max(ymax, p2(1));
            any = true;
        }
        if (!any) { xmin = -coordExtent; xmax = coordExtent; ymin = -coordExtent; ymax = coordExtent; }

        const int NX = 30, NY = 30;
        std::vector<glm::vec3> grid; grid.reserve(NX*NY);
        for (int ix = 0; ix < NX; ++ix) {
            for (int iy = 0; iy < NY; ++iy) {
                double x = xmin + (xmax - xmin) * ix / double(NX-1);
                double y = ymin + (ymax - ymin) * iy / double(NY-1);
                grid.emplace_back((float)x, (float)y, 0.0f);
            }
        }
        auto pc = polyscope::registerPointCloud(name + " (plane)", grid);
        if (pc) {
            pc->setPointRadius(0.005f);
            pc->setPointColor(color);   // apply color
        }
    }
}


// ========================= Clustering Helper =========================
inline std::vector<int> clusterSubspaces(const Eigen::MatrixXd &jaccardMatrix,
                                         double threshold,
                                         int &numClusters) {
    int n = jaccardMatrix.rows();
    std::vector<int> clusterId(n, -1);
    numClusters = 0;

    for (int i = 0; i < n; i++) {
        if (clusterId[i] != -1) continue;
        std::queue<int> q;
        q.push(i);
        clusterId[i] = numClusters;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                if (jaccardMatrix(u, v) >= threshold && clusterId[v] == -1) {
                    clusterId[v] = numClusters;
                    q.push(v);
                }
            }
        }
        numClusters++;
    }
    return clusterId;
}

// ========================= ALGLIB Clustering Helper =========================

// Uses ALGLIB Agglomerative Hierarchical Clustering (AHC)
// Jaccard Matrix: 1.0 = identical, 0.0 = disjoint
// Threshold: Similarity threshold (e.g., 0.5 means merge if sim >= 0.5)
inline std::vector<int> clusterSubspacesAlglib(const Eigen::MatrixXd &jaccardMatrix,
                                               double similarityThreshold,
                                               int &numClusters) {
    int n = jaccardMatrix.rows();
    if (n == 0) return {};
    if (n == 1) { numClusters = 1; return {0}; }

    try {
        // 1. Convert Eigen Jaccard (Similarity) to ALGLIB Distance (1 - Jaccard)
        alglib::real_2d_array distMat;
        distMat.setlength(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Ensure distance is >= 0
                double dist = 1.0 - jaccardMatrix(i, j);
                if (dist < 0) dist = 0;
                distMat(i, j) = dist;
            }
        }

        // 2. Setup ALGLIB Clustering
        alglib::clusterizerstate s;
        alglib::ahcreport rep;

        alglib::clusterizercreate(s);
        alglib::clusterizersetpoints(s, distMat, 2); // 2 = Metric (Euclidean/Explicit Matrix) -> actually we use setdistances
        // Wait, correct call for explicit matrix is clusterizersetdistances
        alglib::clusterizersetdistances(s, distMat, false); // false = not upper triangle only, we gave full symmetric

        // 3. Run Clustering
        // In standard ALGLIB C++, `clusterizerrunahc` takes only (state, report).
        // To set the linkage method (Single, Complete, Average, etc.),
        // you actually use `clusterizersetahcalgo` BEFORE running.

        // Set Algorithm: 2 = Average Linkage (UPGMA)
        // (0 = Single, 1 = Complete, 2 = Average, 3 = Ward, etc.)
        alglib::clusterizersetahcalgo(s, 2);

        // NOW run it
        alglib::clusterizerrunahc(s, rep);

        // 4. Determine K based on Threshold
        // The report contains merge distances. We want to cut where Distance > (1 - Threshold).
        double distThreshold = 1.0 - similarityThreshold;

        // Count how many merges happened with distance <= distThreshold
        // rep.mergedist is an array of size N-1
        int mergesBelowThreshold = 0;
        for (int i = 0; i < rep.mergedist.length(); ++i) {
            if (rep.mergedist(i) <= distThreshold) {
                mergesBelowThreshold++;
            }
        }

        // If we merged M times, we reduced N clusters to N - M clusters.
        int k = n - mergesBelowThreshold;
        if (k < 1) k = 1;
        if (k > n) k = n;

        // 5. Extract Labels
        alglib::integer_1d_array cidx;
        alglib::integer_1d_array cz; // Not used here, hierarchical info
        alglib::clusterizergetkclusters(rep, k, cidx, cz);

        // Convert back to std::vector
        std::vector<int> labels(n);
        int maxId = -1;
        for (int i = 0; i < n; ++i) {
            labels[i] = static_cast<int>(cidx(i));
            if (labels[i] > maxId) maxId = labels[i];
        }

        numClusters = maxId + 1;
        return labels;

    } catch (alglib::ap_error& e) {
        std::cerr << "ALGLIB Error: " << e.msg << "\n";
        // Fallback: every point is its own cluster
        numClusters = n;
        std::vector<int> fallback(n);
        std::iota(fallback.begin(), fallback.end(), 0);
        return fallback;
    }
}
