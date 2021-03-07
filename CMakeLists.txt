cmake_minimum_required(VERSION 3.9)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(GEMM_OPT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -march=native")

enable_testing()
add_subdirectory(src)
