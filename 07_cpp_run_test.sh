#!/bin/bash

cd cpp
mkdir -p build && cd build
cmake ..
make
if [ $? -eq 0 ]; then
    echo "Running test..."
    cd ../..  # Return to project root
    ./cpp/build/test
else
    echo "Build failed."
fi
