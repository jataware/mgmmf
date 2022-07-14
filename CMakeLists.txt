cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pygunrock LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wunused-result -O3 -ffast-math -DNDEBUG -g -fopenmp")

# --
# User defined variables

set(_ANACONDA_PATH $ENV{ANACONDA_PATH})

# --
# python + pytorch

set(PYTHON_INCLUDE_PATH ${ANACONDA_PATH}/include/python3.7m)
set(PYTHON_LIBRARIES    ${ANACONDA_PATH}/lib/libpython3.7m.so)

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./src/pybind11/include
  ./src
)

# --
# _mgmmf_cpp

add_library(
  _mgmmf_cpp 
  SHARED
  ./src/pywrapper.cpp
  ./src/extern/lapjv.cpp
)

target_link_libraries(_mgmmf_cpp PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(_mgmmf_cpp PROPERTIES PREFIX "")
