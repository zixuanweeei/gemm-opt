cmake_minimum_required(VERSION 3.9)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(GEMM_OPT)

set(CMAKE_CXX_STANDARD 17)

enable_testing()
add_subdirectory(src)
