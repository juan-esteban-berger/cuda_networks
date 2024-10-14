/**
 * @file matrix_constructor.cu
 * @brief Implementation of the Matrix class constructor.
 */
#include "matrix.h"
#include <cuda_runtime.h>

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    // Allocate memory on the GPU for the matrix data
    // The size is calculated as rows * cols * sizeof(double)
    cudaMalloc(&d_data, rows * cols * sizeof(double));
}
