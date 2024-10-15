/**
 * @file matrix_transpose.cu
 * @brief Implementation of the Matrix::transpose method for GPU-accelerated matrix transposition.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief CUDA kernel for matrix transposition.
 * @param input Pointer to the input matrix data.
 * @param output Pointer to the output (transposed) matrix data.
 * @param rows Number of rows in the input matrix.
 * @param cols Number of columns in the input matrix.
 */
__global__ void matrixTransposeKernel(const double* input, double* output, int rows, int cols) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < rows && col < cols) {
        // Calculate transposed index and assign value
        int transposedIdx = col * rows + row;
        int originalIdx = row * cols + col;
        output[transposedIdx] = input[originalIdx];
    }
}

/**
 * @brief Transposes the matrix and returns a new Matrix object containing the transposed data.
 * @return A new Matrix object with transposed dimensions.
 */
Matrix Matrix::transpose() const {
    // Create a new matrix to hold the transposed data
    Matrix result(cols, rows);

    // Define block dimensions (16x16 is common for matrix operations)
    dim3 threadsPerBlock(16, 16);

    // Calculate grid dimensions to cover the entire matrix
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel to perform transposition
    matrixTransposeKernel<<<numBlocks, threadsPerBlock>>>(d_data, result.d_data, rows, cols);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device to ensure completion
    cudaDeviceSynchronize();

    return result;
}
