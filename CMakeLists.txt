cmake_minimum_required(VERSION 3.15...3.27)
project(resfo_utilities LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
    # Eigen and ceres creates the same target "unistall"
    set_property(GLOBAL PROPERTY ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)
endif()

FetchContent_Declare(
  ceres
  GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
  GIT_TAG 2.2.0
)
set(MINIGLOG ON CACHE BOOL "Build ceres-solver with miniglog")
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build ceres-solver without benchmarks")
FetchContent_MakeAvailable(ceres)

# Create the Python extension module
pybind11_add_module(_grid_cpp
    src/resfo_utilities/_cpp/bindings.cpp
    src/resfo_utilities/_cpp/grid_search.cpp
    src/resfo_utilities/_cpp/point_in_cell.cpp
)

if(Eigen3_FOUND)
    target_link_libraries(_grid_cpp PRIVATE Eigen3::Eigen ceres)
else()
    target_link_libraries(_grid_cpp PRIVATE eigen ceres)
endif()

if(MSVC)
    target_compile_options(_grid_cpp PRIVATE /W4)
else()
    target_compile_options(_grid_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install the extension module
install(TARGETS _grid_cpp LIBRARY DESTINATION resfo_utilities)
