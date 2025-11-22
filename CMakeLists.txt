cmake_minimum_required(VERSION 3.31)
project(bachelor_thesis)

set(CMAKE_CXX_STANDARD 17)

# Prevent in-source builds
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Run cmake in a separate build directory (e.g. mkdir build && cd build && cmake ..)")
endif()

# Enable OpenGL3 + GLFW backend for Polyscope
set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "")
add_subdirectory(external/polyscope)
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)

add_executable(bachelor_thesis main.cpp
        flat.h
        ransac_multiD.h
        experiment_config.h
        progressive.h
        experiment_config_2.h
        qdf.h
        qdf_analysis.h
        mean_Qdf.h
        mean_qdf_lines.h
        color_utils.h
        merge_csv.h
        "cluster_points_to subspaces.h"
        clustering_accuracy.h
        ransac.h
        median_qdf.h)

target_link_libraries(bachelor_thesis
        polyscope
)
