#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <glm/glm.hpp>
#include <queue>

#include "external/polyscope/include/polyscope/polyscope.h"
#include "external/polyscope/include/polyscope/point_cloud.h"
#include "external/polyscope/include/polyscope/surface_mesh.h"
#include "external/eigen/Eigen/Dense"

#include "flat.h"
#include "ransac_multiD.h"
//#include "experiment_config.h"
#include "experiment_config_2.h"
#include "progressive.h"

#include <limits>
#include <algorithm>

// ------------- 2D visualization helpers ---------------
void visualizeFlatSamples2D(const Eigen::MatrixXd& pts, const std::string& name, float radius = 0.01f) {
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

template<typename ModelT>
void visualizeSubspace2D(const ModelT& model,
                         const std::string& name,
                         const std::vector<Eigen::VectorXd>& allVecPoints) {
    // basic guards
    if (model.origin.size() < 2) return;
    const Eigen::Vector2d origin2 = model.origin.head(2).template cast<double>();

    int subDim = model.basis.cols();
    // 0D: just show origin
    if (subDim == 0) {
        std::vector<glm::vec3> pts;
        pts.emplace_back((float)origin2(0), (float)origin2(1), 0.0f);
        auto pc = polyscope::registerPointCloud(name + " (origin)", pts);
        if (pc) pc->setPointRadius(0.05f);
        return;
    }

    // 1D: draw a long line segment in the direction of basis.col(0)
    if (subDim == 1) {
        Eigen::Vector2d dir = model.basis.col(0).head(2).template cast<double>();
        double nrm = dir.norm();
        if (nrm < 1e-12) return;
        dir /= nrm;

        // compute extent [tmin, tmax] using model.inliers when available
        double tmin = 0.0, tmax = 0.0;
        bool firstT = true;
        for (int idx : model.inliers) {
            if (idx < 0 || idx >= (int)allVecPoints.size()) continue;
            Eigen::Vector2d p2 = allVecPoints[idx].head(2).template cast<double>();
            double t = dir.dot(p2 - origin2);
            if (firstT) { tmin = tmax = t; firstT = false; }
            else { tmin = std::min(tmin, t); tmax = std::max(tmax, t); }
        }
        if (firstT) { // no inliers or couldn't compute => fallback using coordExtent
            tmin = -coordExtent;
            tmax = coordExtent;
        }

        double span = (tmax - tmin);
        double margin = std::max(0.1 * std::max(1.0, span), 0.1);
        double start = tmin - margin;
        double end   = tmax + margin;

        const int SAMPLES = 200;
        std::vector<glm::vec3> linePts; linePts.reserve(SAMPLES+1);
        for (int s = 0; s <= SAMPLES; ++s) {
            double t = start + (end - start) * (double)s / (double)SAMPLES;
            Eigen::Vector2d pos = origin2 + t * dir;
            linePts.emplace_back((float)pos(0), (float)pos(1), 0.0f);
        }
        auto pc = polyscope::registerPointCloud(name + " (line)", linePts);
        if (pc) pc->setPointRadius(0.01f);
        return;
    }

    // 2D (full plane in 2D ambient): draw a grid within the inlier bounding box
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
        std::vector<glm::vec3> grid; grid.reserve(NX * NY);
        for (int ix = 0; ix < NX; ++ix) {
            for (int iy = 0; iy < NY; ++iy) {
                double x = xmin + (xmax - xmin) * ix / double(NX - 1);
                double y = ymin + (ymax - ymin) * iy / double(NY - 1);
                grid.emplace_back((float)x, (float)y, 0.0f);
            }
        }
        auto pc = polyscope::registerPointCloud(name + " (plane)", grid);
        if (pc) pc->setPointRadius(0.005f);
        return;
    }
}


// ===== Clustering function =====
std::vector<int> clusterSubspaces(
    const Eigen::MatrixXd &jaccardMatrix,
    double threshold,
    int &numClusters)
{
    int n = jaccardMatrix.rows();
    std::vector<int> clusterId(n, -1);
    numClusters = 0;

    for (int i = 0; i < n; i++) {
        if (clusterId[i] != -1) continue;
        // new cluster
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

int main() {
    polyscope::init();

    std::vector<Eigen::MatrixXd> allPoints;
    std::vector<int> clusterLabels;
    int clusterId = 0;

    // Generate flats with random flat dimensions [1, ambientDim - 1]
    for (int i = 0; i < numFlats; ++i) {
        int flatDim = flatDimDist(rng);

        // Generate a single random flat with specified ambient and flat dimension
        std::vector<Flat<>> flats = generateRandomFlats(1, ambientDim, flatDim, originSpread, rng);
        Flat<> flat = flats[0];

        Eigen::MatrixXd pts = generateNoisyFlatSamples(flat, numDataPointsPerFlat, coordExtent, noiseStd, rng);
        double err = computeMeanProjectionError(pts, flat);
        std::cout << "Random Flat " << i << " (dim=" << flatDim << "): Mean projection error = " << err << "\n";

        if (ambientDim == 3) {
            visualizeFlatSamples3D(flat, pts, "Random Flat " + std::to_string(i), 1.0, 20);
        } else if (ambientDim == 2) {
            visualizeFlatSamples2D(pts, "Random Flat " + std::to_string(i));
        } else {
            std::cout << "Visualization skipped (ambientDim != 3).\n";
        }

        allPoints.push_back(pts);
        clusterLabels.push_back(clusterId++);
    }

    savePointsToCSV(pointsCSV, allPoints, clusterLabels);

    try {
        // Flatten all Eigen::MatrixXd clusters into a single vector<Eigen::VectorXd>
        std::vector<Eigen::VectorXd> allVecPoints;
        for (const auto& mat : allPoints) {
            for (int i = 0; i < mat.rows(); ++i) {
                allVecPoints.push_back(mat.row(i).transpose());
            }
        }

        auto detectedModels = multiRansacAffine(
                allVecPoints, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_INLIERS, FIXED_DIMENSION);
        std::cout << "Greedy multi-RANSAC detected " << detectedModels.size() << " subspaces\n";


        std::cout << "Detected " << detectedModels.size() << " subspaces\n";

        //Visualization
        for (int i = 0; i < detectedModels.size(); ++i) {
            const auto& model = detectedModels[i];
            std::cout << "Model " << i << ": dim=" << model.basis.cols()
                      << ", inliers=" << model.inliers.size() << "\n";

            if (model.origin.size() == 3) {
                // keep your existing 3D behaviour exactly (draw subspace, register inliers)
                visualizeSubspace3D(model, "RANSAC Subspace " + std::to_string(i));

                std::vector<glm::vec3> inlierPoints;
                for (int idx : model.inliers) {
                    const Eigen::VectorXd& p = allVecPoints[idx];
                    inlierPoints.push_back(glm::vec3(
                        static_cast<float>(p(0)),
                        static_cast<float>(p(1)),
                        static_cast<float>(p(2))
                    ));
                }
                polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);

            } else if (model.origin.size() == 2) {
                // 2D visualizer (subspace geometry)
                visualizeSubspace2D(model, "RANSAC Subspace " + std::to_string(i), allVecPoints);

                // register inliers as a point cloud on the XY plane (Z=0)
                std::vector<glm::vec3> inlierPoints;
                for (int idx : model.inliers) {
                    const Eigen::VectorXd& p = allVecPoints[idx];
                    inlierPoints.push_back(glm::vec3(
                        static_cast<float>(p(0)),
                        static_cast<float>(p(1)),
                        0.0f
                    ));
                }
                polyscope::registerPointCloud("Subspace " + std::to_string(i) + " Inliers", inlierPoints);
            } else {
                std::cout << "Unknown ambient dim for model.origin: " << model.origin.size() << "\n";
            }
        }

        // === Intersection / Jaccard / Normalized Matrices ===
        int n = detectedModels.size();
        Eigen::MatrixXi intersectionMatrix = Eigen::MatrixXi::Zero(n, n);
        Eigen::MatrixXd jaccardMatrix = Eigen::MatrixXd::Zero(n, n);
        Eigen::MatrixXd normalizedMatrix = Eigen::MatrixXd::Zero(n, n);

        int totalPoints = static_cast<int>(allVecPoints.size());

        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                // intersection
                int interCount = 0;
                for (int idx : detectedModels[i].inliers) {
                    if (detectedModels[j].inliers.find(idx) != detectedModels[j].inliers.end()) {
                        ++interCount;
                    }
                }
                intersectionMatrix(i, j) = interCount;
                intersectionMatrix(j, i) = interCount;

                // union
                int unionCount = static_cast<int>(detectedModels[i].inliers.size()
                                   + detectedModels[j].inliers.size() - interCount);

                // jaccard
                double jaccard = (unionCount > 0) ? static_cast<double>(interCount) / unionCount : 0.0;
                jaccardMatrix(i, j) = jaccardMatrix(j, i) = jaccard;

                // normalized by total points
                double normalized = (totalPoints > 0) ? static_cast<double>(interCount) / totalPoints : 0.0;
                normalizedMatrix(i, j) = normalizedMatrix(j, i) = normalized;
            }
        }

        std::cout << "Intersection matrix:\n" << intersectionMatrix << "\n\n";
        std::cout << "Jaccard similarity matrix:\n" << jaccardMatrix << "\n\n";
        std::cout << "Normalized intersection matrix (by total points):\n" << normalizedMatrix << "\n\n";

        // set multiple thresholds
        std::vector<double> thresholds = {0.1};

        for (double thr : thresholds) {
            int numClusters = 0;
            auto clusterIdVec = clusterSubspaces(jaccardMatrix, thr, numClusters);

            std::cout << "\n=== Clustering with threshold " << thr << " ===\n";
            std::cout << "Total clusters: " << numClusters << "\n";

            for (int c = 0; c < numClusters; c++) {
                std::cout << "Cluster " << c << ": ";
                for (int i = 0; i < n; i++) {
                    if (clusterIdVec[i] == c) std::cout << i << " ";
                }
                std::cout << "\n";
            }

            for (int i = 0; i < n; i++) {
                detectedModels[i].clusterId = clusterIdVec[i];
            }

            saveSubspacesToCSV(detectedModels, ransacCSV);

        }

    } catch (const std::exception& e) {
        std::cerr << "RANSAC failed: " << e.what() << "\n";
    }

    polyscope::show();
    return 0;
}
