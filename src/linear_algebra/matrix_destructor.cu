/**
 * @file matrix_destructor.cu
 * @brief Implementation of the Matrix class destructor.
 */
#include "matrix.h"
#include <cuda_runtime.h>

Matrix::~Matrix() {
    // Free the GPU memory allocated for this matrix
    cudaFree(d_data);
}
