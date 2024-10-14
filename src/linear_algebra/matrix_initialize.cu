/**
 * @file matrix_initialize.cu
 * @brief Implementation of the Matrix::initialize method.
 */
#include "matrix.h"
#include <cuda_runtime.h>

void Matrix::initialize() {
    // Use cudaMemset to set all elements of d_data to 0
    cudaMemset(d_data, 0, rows * cols * sizeof(double));
}
