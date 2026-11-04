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
target_link_libraries(benchmark_new PRIVATE darth1141)
