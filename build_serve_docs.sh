#!/bin/bash

doxygen Doxyfile

# Navigate to the docs/html directory
cd docs/html || exit

# Start the Python HTTP server
python3 -m http.server 8000
