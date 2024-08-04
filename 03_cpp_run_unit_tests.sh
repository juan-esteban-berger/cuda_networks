#!/bin/bash

cd cpp
mkdir -p build && cd build
cmake ..
make
if [ $? -eq 0 ]; then
    echo "Running unit tests..."
    ./unitTests
else
    echo "Build failed."
fi
cd ../../
