cmake_minimum_required(VERSION 3.10)
project(cuda_networks)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Include Google Test
add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)

# Assuming your tests are in the 'tests' folder inside 'cpp'
add_executable(runTests cpp/tests/test_series.cpp cpp/utils.cpp)
target_link_libraries(runTests gtest gtest_main pthread)
