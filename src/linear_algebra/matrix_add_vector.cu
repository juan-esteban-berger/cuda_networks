/**
 * @file matrix_add_vector.cu
 * @brief Implementation of the Matrix::add_vector method for GPU-accelerated addition of a vector to each column of a matrix.
 */

#include "matrix.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief CUDA kernel for adding a vector to each column of a matrix.
 * @param m Pointer to the matrix data.
 * @param v Pointer to the vector data.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
__global__ void addVectorToMatrixKernel(double* m, const double* v, int rows, int cols) {
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within matrix bounds
    if (row < rows && col < cols) {
        // Calculate index of current matrix element
        int index = row * cols + col;

        // Add vector element to matrix element
        m[index] += v[row];
    }
}

/**
 * @brief Adds a vector to each column of the matrix.
 * @param v The vector to add.
 * @throws std::invalid_argument if vector dimension doesn't match matrix rows.
 */
void Matrix::add_vector(const Vector& v) {
    // Check if vector dimension matches matrix rows
    if (rows != v.get_rows()) {
        throw std::invalid_argument("Vector dimension must match matrix rows for addition");
    }

    // Define block dimensions
    dim3 threadsPerBlock(16, 16);

    // Calculate grid dimensions
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel
    addVectorToMatrixKernel<<<numBlocks, threadsPerBlock>>>(d_data, v.get_data(), rows, cols);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
