#!/bin/bash

cd cpp
mkdir -p build && cd build
cmake ..
make
if [ $? -eq 0 ]; then
    echo "Running train..."
    cd ../..  # Return to project root
    ./cpp/build/train
else
    echo "Build failed."
fi
