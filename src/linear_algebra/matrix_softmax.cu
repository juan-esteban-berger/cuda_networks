/**
 * @file matrix_softmax.cu
 * @brief Implementation of the softmax function for matrices.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <cfloat>

/**
 * @brief CUDA kernel for applying the softmax function column-wise.
 * @param input Pointer to the input matrix data.
 * @param output Pointer to the output matrix data.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
__global__ void softmaxKernel(const double* input, double* output, int rows, int cols) {
    // Calculate the column index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the column is within bounds
    if (col < cols) {
        // Initialize maximum value to negative infinity
        double max_val = -DBL_MAX;
        
        // Find the maximum value in the column
        for (int row = 0; row < rows; ++row) {
            max_val = fmax(max_val, input[row * cols + col]);
        }
        
        // Initialize sum of exponentials
        double sum_exp = 0.0;
        
        // Calculate the sum of exponentials
        for (int row = 0; row < rows; ++row) {
            sum_exp += exp(input[row * cols + col] - max_val);
        }
        
        // Add a small epsilon to avoid division by zero
        sum_exp += 1e-15;
        
        // Apply softmax function
        for (int row = 0; row < rows; ++row) {
            output[row * cols + col] = exp(input[row * cols + col] - max_val) / sum_exp;
        }
    }
}

/**
 * @brief Applies the softmax function to the matrix column-wise.
 * @return A new Matrix object with softmax applied.
 */
Matrix Matrix::softmax() const {
    // Create a new matrix with the same dimensions
    Matrix result(rows, cols);
    
    // Define the number of threads per block
    int threadsPerBlock = 256;
    
    // Calculate the number of blocks needed
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the CUDA kernel
    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result.d_data, rows, cols);
    
    // Synchronize to ensure the kernel execution is complete
    cudaDeviceSynchronize();
    
    return result;
}
