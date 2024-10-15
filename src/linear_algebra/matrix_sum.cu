/**
 * @file matrix_sum.cu
 * @brief Implementation of the Matrix::sum method for GPU-accelerated summing of all elements in a matrix.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

/**
 * @brief Sums all elements in the matrix.
 * @return The sum of all elements in the matrix.
 */
double Matrix::sum() const {
    // Create a thrust device pointer from the raw CUDA pointer
    thrust::device_ptr<double> d_ptr(d_data);
    
    // Use thrust::reduce to sum all elements
    double result = thrust::reduce(d_ptr, d_ptr + rows * cols);
    
    return result;
}
