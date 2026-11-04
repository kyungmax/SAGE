cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)
