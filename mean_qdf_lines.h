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
    struct GeneratedPoint {
        Eigen::Vector3d position;
        int cluster_id;
    };

    struct MeanLine {
        int cluster_id;
        Eigen::Vector3d x_m;
        Eigen::Vector3d b_new;
    };

    static void visualizeMeanQDF( const std::string& meanQDFFile) {
        auto meanLines = readMeanQDFFile(meanQDFFile);
        // Initialize polyscope
        polyscope::init();
        // Visualize mean lines
        visualizeMeanLines(meanLines);
    }

private:
    static std::vector<MeanLine> readMeanQDFFile(const std::string& filename) {
        std::vector<MeanLine> lines;
        std::ifstream in(filename);
        if (!in) {
            std::cerr << "Error: cannot open " << filename << "\n";
            return lines;
        }

        // Skip header
        std::string header;
        std::getline(in, header);

        std::string line;
        while (std::getline(in, line)) {
            auto tokens = splitCSV(line);
            if (tokens.size() < 7) continue;

            MeanLine ml;
            ml.cluster_id = std::stoi(tokens[0]);
            ml.x_m = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
            ml.b_new = Eigen::Vector3d(std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]));
            lines.push_back(ml);
        }
        return lines;
    }

    static void visualizeMeanLines(const std::vector<MeanLine>& lines, double lineLength = 20.0) {
        for (const auto& ml : lines) {
            // Create vertices for line segment
            std::vector<glm::vec3> vertices;
            std::vector<std::array<size_t, 2>> edges;

            // Direction vector from b_new to x_m
            Eigen::Vector3d dir = (ml.x_m - ml.b_new).normalized();

            // Create line segment centered at b_new extending in both directions
            Eigen::Vector3d p1 = ml.b_new - dir * lineLength/2;
            Eigen::Vector3d p2 = ml.b_new + dir * lineLength/2;

            vertices.push_back(glm::vec3(p1.x(), p1.y(), p1.z()));
            vertices.push_back(glm::vec3(p2.x(), p2.y(), p2.z()));
            edges.push_back({0, 1});

            // Create unique name for each line
            std::string name = "Mean Line Cluster " + std::to_string(ml.cluster_id);

            // Register the line with polyscope
            auto ps_line = polyscope::registerCurveNetwork(name, vertices, edges);

            // Set color based on cluster ID
            glm::vec3 color = colorFromClusterId(ml.cluster_id);
            ps_line->setColor(color);
            ps_line->setRadius(0.002);

            // Add points at x_m and b_new
            std::vector<glm::vec3> points = {
                glm::vec3(ml.x_m.x(), ml.x_m.y(), ml.x_m.z()),
                glm::vec3(ml.b_new.x(), ml.b_new.y(), ml.b_new.z())
            };
            //auto ps_points = polyscope::registerPointCloud(name + " Points", points);
            //ps_points->setPointColor(color);
            //ps_points->setPointRadius(0.001);
        }
    }

    static glm::vec3 getColorFromHue(float hue) {
        return glm::vec3(
            0.5f + 0.5f * std::cos(2 * M_PI * hue),
            0.5f + 0.5f * std::cos(2 * M_PI * (hue + 1.0f/3.0f)),
            0.5f + 0.5f * std::cos(2 * M_PI * (hue + 2.0f/3.0f))
        );
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