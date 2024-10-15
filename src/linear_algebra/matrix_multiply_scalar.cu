/**
 * @file matrix_multiply_scalar.cu
 * @brief Implementation of the Matrix::multiply_scalar method for GPU-accelerated multiplication of a matrix by a scalar.
 */

#include "matrix.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

/**
 * @brief CUDA kernel for multiplying matrix elements by a scalar.
 * @param data Pointer to the matrix data.
 * @param scalar The scalar to multiply by.
 * @param size Total number of elements in the matrix.
 */
__global__ void multiplyScalarKernel(double* data, double scalar, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (idx < size) {
        // Perform multiplication
        double result = data[idx] * scalar;

        // Handle overflow
        if (!isfinite(result)) {
            result = (result > 0.0) ? DBL_MAX : -DBL_MAX;
        }

        // Store the result
        data[idx] = result;
    }
}

/**
 * @brief Multiplies all elements in the matrix by a scalar.
 * @param scalar The scalar to multiply by.
 */
void Matrix::multiply_scalar(double scalar) {
    // Calculate total number of elements
    int size = rows * cols;

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    multiplyScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, size);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // Synchronize device
    cudaDeviceSynchronize();
}
