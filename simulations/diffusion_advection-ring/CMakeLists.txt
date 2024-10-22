cmake_minimum_required(VERSION 3.14)

project(TEST)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O2 -march=x86-64 -std=c++17 -I${CMAKE_CURRENT_SOURCE_DIR}/../../")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(../../fdaPDE/)

add_executable(run_test main.cpp)
target_link_libraries (run_test Eigen3::Eigen)
