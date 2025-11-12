#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include <glm/glm.hpp>
#include "color_utils.h"

class MeanQDFLines {
public:
    struct MeanLine {
        int cluster_id;
        Eigen::VectorXd x_m;
        Eigen::VectorXd b_new;
    };

    static void visualizeMeanQDF(const std::string& meanQDFFile) {
        auto meanLines = readMeanQDFFile(meanQDFFile);
        if (meanLines.empty()) {
            std::cerr << "No mean lines parsed — check CSV format.\n";
            return;
        }

        //polyscope::init();
        visualizeMeanLines(meanLines);
        //polyscope::show();
    }

private:
    static std::vector<MeanLine> readMeanQDFFile(const std::string& filename) {
        std::vector<MeanLine> lines;
        std::ifstream in(filename);
        if (!in) {
            std::cerr << "Error: cannot open " << filename << "\n";
            return lines;
        }

        std::string header;
        std::getline(in, header);

        std::string firstDataLine;
        std::getline(in, firstDataLine);
        if (firstDataLine.empty()) {
            std::cerr << "Error: empty mean QDF file.\n";
            return lines;
        }

        // Infer dimension: total = 1 + 2n (x_m + b_new)
        auto tokens = splitCSV(firstDataLine);
        int total = (int)tokens.size();
        int n = (total - 1) / 2;
        if (n <= 0) {
            std::cerr << "Could not infer dimension (found " << total << " columns)\n";
            return lines;
        }

        std::cout << "Detected dimension n = " << n << " in mean QDF visualization\n";

        // Reset file to start reading all lines
        in.clear();
        in.seekg(0);
        std::getline(in, header); // skip header again

        std::string line;
        while (std::getline(in, line)) {
            auto t = splitCSV(line);
            if ((int)t.size() < total) continue;

            MeanLine ml;
            ml.cluster_id = std::stoi(t[0]);

            ml.x_m.resize(n);
            ml.b_new.resize(n);
            for (int i = 0; i < n; ++i)
                ml.x_m(i) = std::stod(t[1 + i]);
            for (int i = 0; i < n; ++i)
                ml.b_new(i) = std::stod(t[1 + n + i]);

            lines.push_back(ml);
        }

        return lines;
    }

    static void visualizeMeanLines(const std::vector<MeanLine>& lines, double lineLength = 1.5) {
        for (const auto& ml : lines) {
            // Convert to 3D (pad with zero if 2D)
            Eigen::Vector3d x_m3(0, 0, 0);
            Eigen::Vector3d b_new3(0, 0, 0);
            for (int i = 0; i < std::min<int>(ml.x_m.size(), 3); ++i) {
                x_m3[i] = ml.x_m[i];
                b_new3[i] = ml.b_new[i];
            }

            Eigen::Vector3d dir = (x_m3 - b_new3).normalized();
            Eigen::Vector3d p1 = b_new3 - dir * (lineLength / 2.0);
            Eigen::Vector3d p2 = b_new3 + dir * (lineLength / 2.0);

            std::vector<glm::vec3> vertices = {
                glm::vec3(p1.x(), p1.y(), p1.z()),
                glm::vec3(p2.x(), p2.y(), p2.z())
            };
            std::vector<std::array<size_t, 2>> edges = {{0, 1}};

            std::string name = "Mean Line Cluster " + std::to_string(ml.cluster_id);
            auto ps_line = polyscope::registerCurveNetwork(name, vertices, edges);

            glm::vec3 color = colorFromClusterId(ml.cluster_id);
            ps_line->setColor(color);
            ps_line->setRadius(0.0025);
        }
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
