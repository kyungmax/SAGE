cmake_minimum_required(VERSION 3.16)

project(ada_ef_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  HINTS
    "$ENV{CONDA_PREFIX}/include/eigen3"
    "$ENV{CONDA_PREFIX}/include"
    /usr/include/eigen3
    /usr/local/include/eigen3
)

find_path(BOOST_INCLUDE_DIR
  NAMES boost/math/distributions/normal.hpp
  HINTS
    "$ENV{CONDA_PREFIX}/include"
    /usr/include
    /usr/local/include
)

if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found. Set -DEIGEN3_INCLUDE_DIR=/path/to/eigen3/include.")
endif()

if(NOT BOOST_INCLUDE_DIR)
  message(FATAL_ERROR "Boost headers not found. Set -DBOOST_INCLUDE_DIR=/path/to/boost/include.")
endif()

add_executable(backend_runner
  src/backend_runner.cpp
)

target_include_directories(backend_runner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${EIGEN3_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
)

target_compile_options(backend_runner PRIVATE
  $<$<CONFIG:Release>:-O3>
  -Wall
  -Wextra
  -Wno-unused-parameter
)

target_link_libraries(backend_runner PRIVATE
  OpenMP::OpenMP_CXX
  ${HDF5_CXX_LIBRARIES}
  ${HDF5_LIBRARIES}
)
