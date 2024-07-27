#!/bin/bash

# Navigate to the cpp directory
cd cpp

# Modify CMakeLists.txt to include debug symbols
sed -i '1iset(CMAKE_BUILD_TYPE Debug)' CMakeLists.txt

# Navigate to the build directory
cd build

# Clean the previous builds
echo "Cleaning previous builds..."
rm -rf *

# Run cmake to configure the project
echo "Configuring project with CMake..."
cmake ..

# Build the project
echo "Building the project..."
make

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Starting GDB..."
    gdb -ex run --args ./unitTests
else
    echo "Build failed."
fi

# Return to the original directory
cd ../../

# Restore CMakeLists.txt
sed -i '1d' cpp/CMakeLists.txt
