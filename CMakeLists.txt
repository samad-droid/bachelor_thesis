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

# Define the directory where alglib lives
set(ALGLIB_DIR "${CMAKE_SOURCE_DIR}/external/alglib/alglib-cpp/src/")
# ^^^ CHECK THIS PATH manually in your file browser! ^^^

include_directories(${ALGLIB_DIR})

# List ALGLIB Source Files
# You generally only need these core modules for clustering
set(ALGLIB_SOURCES
        "${ALGLIB_DIR}/alglibinternal.cpp"
        "${ALGLIB_DIR}/alglibmisc.cpp"
        "${ALGLIB_DIR}/ap.cpp"
        "${ALGLIB_DIR}/dataanalysis.cpp"
        "${ALGLIB_DIR}/diffequations.cpp"
        "${ALGLIB_DIR}/fasttransforms.cpp"
        "${ALGLIB_DIR}/integration.cpp"
        "${ALGLIB_DIR}/optimization.cpp"
        "${ALGLIB_DIR}/kernels_avx2.cpp"
        "${ALGLIB_DIR}/kernels_fma.cpp"
        "${ALGLIB_DIR}/linalg.cpp"
        "${ALGLIB_DIR}/minlp.cpp"
        "${ALGLIB_DIR}/solvers.cpp"
        "${ALGLIB_DIR}/specialfunctions.cpp"
        "${ALGLIB_DIR}/statistics.cpp"
        "${ALGLIB_DIR}/alglibinternal.h"
        "${ALGLIB_DIR}/alglibmisc.h"
        "${ALGLIB_DIR}/ap.h"
        "${ALGLIB_DIR}/dataanalysis.h"
        "${ALGLIB_DIR}/diffequations.h"
        "${ALGLIB_DIR}/fasttransforms.h"
        "${ALGLIB_DIR}/integration.h"
        "${ALGLIB_DIR}/optimization.h"
        "${ALGLIB_DIR}/kernels_avx2.h"
        "${ALGLIB_DIR}/kernels_fma.h"
        "${ALGLIB_DIR}/linalg.h"
        "${ALGLIB_DIR}/minlp.h"
        "${ALGLIB_DIR}/solvers.h"
        "${ALGLIB_DIR}/specialfunctions.h"
        "${ALGLIB_DIR}/statistics.h"
)

add_executable(bachelor_thesis main.cpp
        ${ALGLIB_SOURCES}
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
        median_qdf.h
        medianW_qdf.h)

target_link_libraries(bachelor_thesis
        polyscope
)
