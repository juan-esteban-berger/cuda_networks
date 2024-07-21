#!/bin/bash

# Navigate to the build directory
cd cpp/build

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
    echo "Running unit tests..."
    ./unitTests  # Adjust this if your executable name is different
else
    echo "Build failed."
fi

# Return to the original directory
cd ../../
