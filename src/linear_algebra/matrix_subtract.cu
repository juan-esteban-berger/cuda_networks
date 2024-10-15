/**
 * @file matrix_subtract.cu
 * @brief Implementation of the Matrix::subtract method for GPU-accelerated matrix subtraction.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief CUDA kernel for element-wise matrix subtraction.
 * @param a Pointer to the first input matrix data.
 * @param b Pointer to the second input matrix data.
 * @param c Pointer to the output matrix data.
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 */
__global__ void matrixSubtractKernel(const double* a, const double* b, double* c, int rows, int cols) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < rows && col < cols) {
        // Calculate index of current element
        int index = row * cols + col;

        // Perform element-wise subtraction
        c[index] = a[index] - b[index];
    }
}

/**
 * @brief Subtracts another matrix from this matrix.
 * @param other The matrix to subtract.
 * @return A new Matrix object containing the result of the subtraction.
 * @throws std::invalid_argument if matrix dimensions are not identical.
 */
Matrix Matrix::subtract(const Matrix& other) const {
    // Check if matrices have identical dimensions
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must be identical for subtraction");
    }

    // Create result matrix
    Matrix result(rows, cols);

    // Define block dimensions
    dim3 threadsPerBlock(16, 16);

    // Calculate grid dimensions
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel
    matrixSubtractKernel<<<numBlocks, threadsPerBlock>>>(d_data, other.d_data, result.d_data, rows, cols);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();

    return result;
}
