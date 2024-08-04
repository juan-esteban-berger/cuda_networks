#!/bin/bash

cd cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
if [ $? -eq 0 ]; then
    echo "Build successful. Starting GDB..."
    gdb -ex run --args ./unitTests
else
    echo "Build failed."
fi
cd ../../
