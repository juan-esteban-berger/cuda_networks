/**
 * @file matrix_relu_derivative.cu
 * @brief Implementation of the ReLU derivative function for matrices.
 */

#include "matrix.h"
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for applying the ReLU derivative function element-wise.
 * @param input Pointer to the input matrix data.
 * @param output Pointer to the output matrix data.
 * @param size Total number of elements in the matrix.
 */
__global__ void reluDerivativeKernel(const double* input, double* output, int size) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread is within the matrix bounds
    if (idx < size) {
        // Apply ReLU derivative: 1 if x > 0, 0 otherwise
        output[idx] = (input[idx] > 0.0) ? 1.0 : 0.0;
    }
}

/**
 * @brief Applies the ReLU derivative function to the matrix.
 * @return A new Matrix object with ReLU derivative applied.
 */
Matrix Matrix::relu_derivative() const {
    // Create a new matrix with the same dimensions
    Matrix result(rows, cols);
    
    // Calculate the total number of elements
    int size = rows * cols;
    
    // Define the number of threads per block
    int threadsPerBlock = 256;
    
    // Calculate the number of blocks needed
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the CUDA kernel
    reluDerivativeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, result.d_data, size);
    
    // Synchronize to ensure the kernel execution is complete
    cudaDeviceSynchronize();
    
    return result;
}
