cmake_minimum_required(VERSION 3.10)
project(matty LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(matty src/main.cpp)
target_include_directories(matty PUBLIC include)
target_compile_options(matty PUBLIC -msse4.1 -ftree-vectorize -march=native -finline-functions)
