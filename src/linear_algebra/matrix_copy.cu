/**
 * @file matrix_copy.cu
 * @brief Implementation of the copy method for matrices.
 */

#include "matrix.h"
#include <cuda_runtime.h>

/**
 * @brief Creates a deep copy of the matrix.
 * @return A new Matrix object with the same content as the original.
 */
Matrix Matrix::copy() const {
    // Create a new matrix with the same dimensions
    Matrix result(rows, cols);
    
    // Calculate the total number of elements
    int size = rows * cols;
    
    // Copy the data from the current matrix to the new matrix
    cudaMemcpy(result.d_data, d_data, size * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Return the new matrix
    return result;
}
