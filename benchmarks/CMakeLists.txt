cmake_minimum_required(VERSION 3.15)
project(resfo_benchmarks LANGUAGES CXX)

# This is a standalone CMake project so that benchmarks can be built
# without Python, pybind11, or pip:
#   cmake -S benchmarks -B build_bench -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_bench
# Eigen and Ceres are redeclared here because the root CMakeLists.txt
# requires Python/pybind11 which we want to avoid for benchmark builds.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR
        "CMAKE_BUILD_TYPE is not set. Benchmarks require an explicit build type, e.g.:\n"
        "  cmake -S benchmarks -B build_bench -DCMAKE_BUILD_TYPE=Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c # 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

# Ceres (required by point_in_cell.cpp)
FetchContent_Declare(
    ceres
    GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
    GIT_TAG 85331393dc0dff09f6fb9903ab0c4bfa3e134b01 # 2.2.0
    GIT_SHALLOW TRUE
)
set(MINIGLOG ON CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ceres)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googlebenchmark)

set(CPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src/resfo_utilities/_cpp)

add_executable(bench_grid_search
    bench_grid_search.cpp
    ${CPP_SRC}/column_interval_tree.cpp
    ${CPP_SRC}/point_in_cell.cpp
    ${CPP_SRC}/grid_search.cpp
)

target_include_directories(bench_grid_search PRIVATE ${CPP_SRC} ${CMAKE_CURRENT_SOURCE_DIR})

if(Eigen3_FOUND)
    target_link_libraries(bench_grid_search PRIVATE Eigen3::Eigen ceres benchmark::benchmark)
else()
    target_link_libraries(bench_grid_search PRIVATE eigen ceres benchmark::benchmark)
endif()

add_executable(bench_bounding_boxes
    bench_bounding_boxes.cpp
    ${CPP_SRC}/column_interval_tree.cpp
)

target_include_directories(bench_bounding_boxes PRIVATE ${CPP_SRC} ${CMAKE_CURRENT_SOURCE_DIR})

if(Eigen3_FOUND)
    target_link_libraries(bench_bounding_boxes PRIVATE Eigen3::Eigen benchmark::benchmark)
else()
    target_link_libraries(bench_bounding_boxes PRIVATE eigen benchmark::benchmark)
endif()
