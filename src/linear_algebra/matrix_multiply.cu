/**
 * @file matrix_multiply.cu
 * @brief Implementation of the Matrix::multiply method for GPU-accelerated matrix multiplication.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>  // For std::invalid_argument and std::runtime_error
#include <string>     // For std::string

/**
 * @brief CUDA kernel for matrix multiplication.
 * @param a Pointer to the first input matrix data.
 * @param b Pointer to the second input matrix data.
 * @param c Pointer to the output matrix data.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A / rows in matrix B.
 * @param k Number of columns in matrix B.
 */
__global__ void matrixMultiplyKernel(const double* a,
                                     const double* b,
                                     double* c,
                                     int m,
                                     int n,
                                     int k) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < m && col < k) {
        // Initialize sum for dot product
        double sum = 0.0;

        // Perform dot product of row from A and column from B
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }

        // Store the result in matrix C
        c[row * k + col] = sum;
    }
}

/**
 * @brief Multiplies this matrix with another matrix.
 * @param other The matrix to multiply with.
 * @return A new Matrix object containing the result of the multiplication.
 * @throws std::invalid_argument if matrix dimensions are incompatible for multiplication.
 */
Matrix Matrix::multiply(const Matrix& other) const {
    // Check if matrices can be multiplied
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    // Create result matrix
    Matrix result(rows, other.cols);

    // Define block dimensions
    dim3 threadsPerBlock(16, 16);

    // Calculate grid dimensions
    dim3 numBlocks((other.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_data,
                                                         other.d_data,
                                                         result.d_data, rows,
                                                         cols,
                                                         other.cols);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();

    return result;
}
