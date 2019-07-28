cmake_minimum_required(VERSION 3.0)
project(nurbs)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

# add_executable(main2 gauss_2d_sphere.c)
add_library(gauss SHARED gauss_2d_sphere.c)
set_property(TARGET gauss PROPERTY POSITION_INDEPENDENT_CODE ON)
