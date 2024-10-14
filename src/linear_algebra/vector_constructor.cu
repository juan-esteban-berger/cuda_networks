/**
 * @file vector_constructor.cu
 * @brief Implementation of the Vector class constructor.
 */
#include "vector.h"
#include <cuda_runtime.h>

Vector::Vector(int rows) : rows(rows) {
    // Allocate memory on the GPU for the vector data
    cudaMalloc(&d_data, rows * sizeof(double));
}
