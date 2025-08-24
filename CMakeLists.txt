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
        ransac_line.h
        ransac_multiD.h
        experiment_config.h
        progressive.h)

target_link_libraries(bachelor_thesis
        polyscope
)
