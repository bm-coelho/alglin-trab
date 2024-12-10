cmake_minimum_required(VERSION 3.15)
project(alglin LANGUAGES CXX)

add_subdirectory(alglin_cpp)

install(DIRECTORY python/ DESTINATION alglin)
install(TARGETS alglin_cpp LIBRARY DESTINATION alglin)
