cmake_minimum_required(VERSION 3.10)
project(CPP_Networks_Tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include directories for your project
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/lib/googletest/googletest/include)

# Google Test
add_subdirectory(lib/googletest)

# Define the source files and test files
set(SOURCE_FILES utils.cpp neural_network.cpp)  # Add neural_network.cpp here

# Add executable for the tests
add_executable(unitTests unit_tests.cpp ${SOURCE_FILES})
target_link_libraries(unitTests gtest gtest_main)

# If your project uses pthread, link it as well
find_package(Threads REQUIRED)
target_link_libraries(unitTests Threads::Threads)
