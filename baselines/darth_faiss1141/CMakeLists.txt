cmake_minimum_required(VERSION 3.24)
project(darth_faiss1141 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_MKL OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL avx512 CACHE STRING "" FORCE)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
add_subdirectory(faiss vendor_build EXCLUDE_FROM_ALL)

set(LIGHTGBM_LIB "$ENV{LIGHTGBM_LIB}" CACHE FILEPATH "LightGBM shared library")
if(NOT LIGHTGBM_LIB)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util,pathlib; s=importlib.util.find_spec('lightgbm'); print(pathlib.Path(s.origin).parent/'lib'/'lib_lightgbm.so' if s else '')"
      OUTPUT_VARIABLE LIGHTGBM_PYTHON_LIB OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(EXISTS "${LIGHTGBM_PYTHON_LIB}")
      set(LIGHTGBM_LIB "${LIGHTGBM_PYTHON_LIB}")
    endif()
  endif()
endif()
if(NOT LIGHTGBM_LIB AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/lib/lib_lightgbm.so")
  set(LIGHTGBM_LIB "${CMAKE_CURRENT_SOURCE_DIR}/reference/lib/lib_lightgbm.so")
endif()
if(NOT EXISTS "${LIGHTGBM_LIB}")
  message(FATAL_ERROR "Install lightgbm for the selected Python or set LIGHTGBM_LIB to lib_lightgbm.so")
endif()
add_library(lightgbm_runtime SHARED IMPORTED)
set_target_properties(lightgbm_runtime PROPERTIES
  IMPORTED_LOCATION "${LIGHTGBM_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lightgbm/include")
add_library(darth_predictor STATIC faiss/faiss/impl/DeclarativeRecall.cpp)
target_include_directories(darth_predictor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/faiss")
# Keep the predictor's original floating-point evaluation, including FMA.
target_compile_options(darth_predictor PRIVATE
  -mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt)
target_link_libraries(darth_predictor PUBLIC lightgbm_runtime OpenMP::OpenMP_CXX)

add_library(darth1141 STATIC src/search.cpp)
target_include_directories(darth1141 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(darth1141 PUBLIC faiss_avx512 darth_predictor OpenMP::OpenMP_CXX Threads::Threads)
add_executable(benchmark_new src/benchmark.cpp)
target_compile_definitions(benchmark_new PRIVATE DARTH1141_NEW=1)
target_link_libraries(benchmark_new PRIVATE darth1141)

# Training retains DARTH's original 1.8 collector and checkpoint placement.
# Build it separately because its FAISS target names overlap the online port.
include(ExternalProject)
set(DARTH_TRAIN_BUILD_JOBS 4 CACHE STRING "Parallel jobs for the original training FAISS build")
set(DARTH_TRAIN_BUILD "${CMAKE_CURRENT_BINARY_DIR}/training_faiss18")
ExternalProject_Add(darth_training_faiss18_build
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../darth"
  BINARY_DIR "${DARTH_TRAIN_BUILD}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DFAISS_OPT_LEVEL=generic
    -DLIGHTGBM_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/lightgbm/include
    -DLIGHTGBM_LIB=${LIGHTGBM_LIB}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
    --target faiss --parallel ${DARTH_TRAIN_BUILD_JOBS}
  BUILD_BYPRODUCTS "${DARTH_TRAIN_BUILD}/faiss/libfaiss.so"
  BUILD_ALWAYS TRUE
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL TRUE)
add_library(darth_training_faiss18 SHARED IMPORTED)
set_target_properties(darth_training_faiss18 PROPERTIES
  IMPORTED_LOCATION "${DARTH_TRAIN_BUILD}/faiss/libfaiss.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../darth")
add_dependencies(darth_training_faiss18 darth_training_faiss18_build)

add_library(darth_training_predictor OBJECT faiss/faiss/impl/DeclarativeRecall.cpp)
target_include_directories(darth_training_predictor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../darth")
target_compile_options(darth_training_predictor PRIVATE
  -mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt)
target_link_libraries(darth_training_predictor PUBLIC lightgbm_runtime OpenMP::OpenMP_CXX)
add_executable(train_darth_corrected src/train_darth_corrected.cpp)
target_link_libraries(train_darth_corrected PRIVATE
  darth_training_predictor darth_training_faiss18 Threads::Threads)
# Corrected executable definitions preempt the old collector's shared symbols.
target_link_options(train_darth_corrected PRIVATE -Wl,--export-dynamic)
