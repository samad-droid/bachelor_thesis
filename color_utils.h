#pragma once
#include <glm/glm.hpp>
#include <cmath>

inline glm::vec3 colorFromClusterId(int cluster_id) {
    // Map cluster_id deterministically to hue in [0,1)
    float hue = std::fmod(static_cast<float>(cluster_id) * 0.6180339887f, 1.0f); // golden ratio for good spread

    return glm::vec3(
        0.5f + 0.5f * std::cos(2 * M_PI * hue),
        0.5f + 0.5f * std::cos(2 * M_PI * (hue + 1.0f/3.0f)),
        0.5f + 0.5f * std::cos(2 * M_PI * (hue + 2.0f/3.0f))
    );
}
