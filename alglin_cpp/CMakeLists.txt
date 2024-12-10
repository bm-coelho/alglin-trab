cmake_minimum_required(VERSION 3.15)
project(alglin_cpp LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(alglin_cpp MODULE src/bindings.cpp)
target_include_directories(alglin_cpp PRIVATE include)
target_link_libraries(alglin_cpp PRIVATE pybind11::module Eigen3::Eigen)
set_target_properties(alglin_cpp PROPERTIES PREFIX "" SUFFIX ".so")
