cmake_minimum_required(VERSION 3.19)
project(RaBitQLib LANGUAGES CXX)

option(RABITQ_BUILD_PYTHON_BINDINGS "Build the Python extension" ON)
if(RABITQ_BUILD_PYTHON_BINDINGS)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(OpenMP REQUIRED)
  pybind11_add_module(_rabitqlib NO_EXTRAS
    python_bindings/rabitq_bindings.cpp
    python_bindings/hnsw_bindings.cpp
    python_bindings/ivf_bindings.cpp
    python_bindings/symqg_bindings.cpp
    python_bindings/quant_bindings.cpp)
  target_compile_features(_rabitqlib PRIVATE cxx_std_17)
  target_compile_options(_rabitqlib PRIVATE -O3 -march=native)
  target_include_directories(_rabitqlib PRIVATE include)
  target_link_libraries(_rabitqlib PRIVATE OpenMP::OpenMP_CXX)
  set_target_properties(_rabitqlib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/rabitqlib")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python/rabitqlib")
  configure_file(python_bindings/__init__.py
    "${CMAKE_BINARY_DIR}/python/rabitqlib/__init__.py" COPYONLY)
  install(TARGETS _rabitqlib DESTINATION rabitqlib)
  install(FILES python_bindings/__init__.py DESTINATION rabitqlib)
endif()
