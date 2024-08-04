#!/bin/bash

cd cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
if [ $? -eq 0 ]; then
    echo "Build successful. Starting GDB..."
    cd ../..  # Return to project root
    gdb -ex run ./cpp/build/train
else
    echo "Build failed."
fi
