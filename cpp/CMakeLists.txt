cmake_minimum_required(VERSION 3.10)
project(CPP_Networks)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include directories for your project
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/lib/googletest/googletest/include)

# Google Test
add_subdirectory(lib/googletest)

# Define the source files
set(SOURCE_FILES 
    src/linear_algebra.cpp 
    src/neural_network.cpp
)

# Add executable for the unit tests
add_executable(unitTests src/unit_tests.cpp ${SOURCE_FILES})
target_link_libraries(unitTests gtest gtest_main)

# Add executable for the test program
add_executable(test src/test.cpp ${SOURCE_FILES})

# Add executable for the train program
add_executable(train src/train.cpp ${SOURCE_FILES})

# If your project uses pthread, link it for all executables
find_package(Threads REQUIRED)
target_link_libraries(unitTests Threads::Threads)
target_link_libraries(test Threads::Threads)
target_link_libraries(train Threads::Threads)
