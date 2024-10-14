/**
 * @file vector_initialize.cu
 * @brief Implementation of the Vector::initialize method.
 */
#include "vector.h"
#include <cuda_runtime.h>

void Vector::initialize() {
    // Use cudaMemset to set all elements of d_data to 0
    cudaMemset(d_data, 0, rows * sizeof(double));
}
